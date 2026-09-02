from agno.agent import Agent
from agno.models.openai import OpenResponses

import os
import asyncio
import re
import sys
import json
import time
import logging
import requests
import cloudscraper
import textwrap
import pandas as pd
from tabulate import tabulate
from pydantic import BaseModel
from typing import List
from bs4 import BeautifulSoup
from openai import OpenAI
from datetime import datetime, date
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

TOPICS = [
    "Middle East Mergers & Acquisitions",
    "Strategic Investments & Partnerships",
    "AI & Generative AI",
    "Cloud & Digital Infrastructure",
    "Digital Transformation",
    "Cybersecurity",
    "Sovereign Wealth & Government Technology Investment",
]

# Bare domains only — the API's allowed_domains filter works at the domain
# level, not on sub-paths. We steer the Reuters "Middle East" section via
# the prompt text instead, since the filter itself can't enforce a path.
SOURCE_DOMAINS = [
    "reuters.com",
    "thenationalnews.com",
    "arabnews.com",
]

# Upper bound on what we *ask* the model for per topic, just to keep each
# response bounded — NOT a cap on the final table. Every same-day article
# that survives the date-verification step is kept; some topics may end up
# with 0 rows on a slow news day, others with more than this number if the
# model's search + our verification both agree there's more real coverage.
REQUEST_LIMIT_PER_TOPIC = 7

# How many days off from "today" a publish date is still allowed to be.
# 0 = must be published on the exact same calendar date as the run.
# Loosen this (e.g. to 1) if same-day-only proves too strict in practice.
DATE_TOLERANCE_DAYS = 0

# If we can't determine an article's publish date at all (fetch blocked,
# no date metadata on the page), should we keep it anyway? Default: no —
# the whole point of this filter is a hard same-day guarantee.
KEEP_UNDATED_ARTICLES = False

MAX_RETRIES_PER_TOPIC = 0  # retries if a call comes back with zero citations
REQUEST_TIMEOUT_SECS = 10
PAGE_FETCH_DELAY_SECS = 0.5  # be polite between page fetches
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

BROWSER_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

DATE_META_CANDIDATES = [
    {"property": "article:published_time"},
    {"property": "og:article:published_time"},
    {"name": "pubdate"},
    {"name": "publishdate"},
    {"name": "publish-date"},
    {"name": "date"},
    {"itemprop": "datePublished"},
    {"name": "sailthru.date"},
]

@dataclass
class Article:
    topic: str
    header: str
    news: str
    source_link: str

class DailyDigest(BaseModel):
    items: List[Article]

LINE_PATTERN = re.compile(r"^\s*\d+\.\s*(.+?)\s*:::\s*(.+?)\s*$")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("news_agent")



def get_client_and_model() -> tuple[OpenAI, str]:
    """
    Builds an OpenAI client pointed at the Azure OpenAI resource, using
    API-key auth. Raises/exits with a clear message if config is missing.
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

    missing = [name for name, val in [
        ("AZURE_OPENAI_ENDPOINT", endpoint),
        ("AZURE_OPENAI_API_KEY", api_key),
        ("AZURE_OPENAI_DEPLOYMENT_NAME", deployment),
    ] if not val]

    if missing:
        log.error(f"Missing required environment variable(s): {', '.join(missing)}")
        log.error("Example endpoint format: https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/")
        sys.exit(1)

    client = OpenAI(base_url=endpoint, api_key=api_key)
    return client, deployment


def ordered_citation_urls(annotations) -> list[str]:
    """
    Returns citation URLs in document order (the order the API emitted them).
    We pair these positionally with our numbered lines rather than trying to
    match by exact character offset: annotation spans often cover only a
    narrow phrase within a sentence (not the whole line), so a strict overlap
    check is brittle and drops real, verified citations for no good reason.
    Position-based pairing is far more robust while still guaranteeing every
    URL traces back to a real annotation, never model-typed text.
    """
    dated = []
    for ann in annotations:
        url = getattr(ann, "url", None)
        start = getattr(ann, "start_index", None)
        if url is None:
            continue
        dated.append((start if start is not None else 0, url))
    dated.sort(key=lambda pair: pair[0])
    return [url for _, url in dated]

def _parse_iso_date(text: str) -> date | None:
    """Best-effort parse of common ISO-ish date/datetime strings to a date."""
    if not text:
        return None
    text = text.strip()
    # Normalize a trailing 'Z' to an explicit UTC offset Python can parse.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for fmt in (None, "%Y-%m-%d"):
        try:
            if fmt is None:
                return datetime.fromisoformat(text).date()
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _search_json_ld_for_date(soup: BeautifulSoup) -> date | None:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        candidates = data if isinstance(data, list) else [data]
        for entry in candidates:
            if not isinstance(entry, dict):
                continue
            for key in ("datePublished", "dateCreated", "uploadDate"):
                if key in entry:
                    parsed = _parse_iso_date(str(entry[key]))
                    if parsed:
                        return parsed
    return None


def get_published_date(url: str) -> date | None:
    """
    Fetches the article page directly and extracts its real publish date from
    standard <meta> tags or JSON-LD structured data. Returns None if the page
    can't be fetched or no date can be found — callers decide how to treat that.
    """
    try:
        resp = _scraper.get(url, headers=BROWSER_HEADERS, timeout=REQUEST_TIMEOUT_SECS)
    except requests.RequestException as e:
        log.warning(f"    Date check: fetch error for {url} ({e})")
        return None

    if resp.status_code == 403:
        log.warning(f"    Date check: BLOCKED (403 — bot protection) for {url}")
        return None
    if not resp.ok:
        log.warning(f"    Date check: HTTP {resp.status_code} for {url}")
        return None
    soup = BeautifulSoup(resp.text, "html.parser")

    for attrs in DATE_META_CANDIDATES:
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            parsed = _parse_iso_date(tag["content"])
            if parsed:
                return parsed
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        parsed = _parse_iso_date(time_tag["datetime"])
        if parsed:
            return parsed

    return _search_json_ld_for_date(soup)


def fetch_articles_for_topic(client: OpenAI, model: str, topic: str,
                              request_limit: int = REQUEST_LIMIT_PER_TOPIC) -> list[Article]:
    """
    One Responses API call per topic (retried up to MAX_RETRIES_PER_TOPIC times
    if zero citations come back — a sign the model answered without actually
    grounding in search results). Search is restricted to SOURCE_DOMAINS via
    the API's native allowed_domains filter. Every returned Article's link
    comes from a real citation annotation, never model-typed text.

    `request_limit` only bounds what we ask the model for — it is NOT a cap
    on the final result count. All same-day, citation-verified articles are
    returned, however many that turns out to be (including zero).
    """
    today_str = date.today().isoformat()
    prompt = (
        f"Today's date is {today_str}."
        f"Find the most relevant news articles published TODAY, {today_str}, about the topic, across the allowed sources."
        f"\"{topic}\" — up to a maximum of {request_limit}. Do not include older "
        f"articles even if they are relevant — only today's news. If there are "
        f"none published today, that's fine — just return an empty list.\n"
        f"From the allowed sources, always prioritize articles from the Middle East section of Reuters "
        f"Middle East section (reuters.com/world/middle-east).\n\n"
        f"You must use the web_search tool to find real articles — do not answer "
        f"from memory. Every line below must be grounded in an actual search result.\n\n"
        f"Output ONLY a numbered list, one article per line, in exactly this format "
        f"(no extra text, no markdown, no headers):\n"
        f"1. HEADLINE ::: ONE-TO-TWO SENTENCE SUMMARY\n"
        f"2. HEADLINE ::: ONE-TO-TWO SENTENCE SUMMARY\n"
        f"...\n\n"
        f"Do not pad the list with older articles just to reach {request_limit} — "
        f"only include ones genuinely published today."
    )

    lines: list[tuple[str, str]] = []
    urls: list[str] = []

    for attempt in range(1, MAX_RETRIES_PER_TOPIC + 2):  # initial try + retries
        try:
            response = client.responses.create(
                model=model,
                tools=[{
                    "type": "web_search",
                    "filters": {"allowed_domains": SOURCE_DOMAINS},
                }],
                tool_choice="auto",
                input=prompt,
            )
        except Exception as e:
            log.error(f"API call failed for topic '{topic}' (attempt {attempt}): {e}")
            continue

        full_text = ""
        annotations = []
        for item in getattr(response, "output", []):
            if getattr(item, "type", None) != "web_search_call":
                continue

            action = getattr(item, "action", None)

            if not action:
                continue

            action_type = getattr(action, "type", None)

            if action_type == "search":
                queries = getattr(action, "queries", None) or []

                log.info(
                    f"[{topic}] Web search queries: {len(queries)}"
                )

                for i, query in enumerate(queries, 1):
                    log.info(
                        f"[{topic}] Query {i}: {query}"
                    )

            elif action_type == "open_page":
                log.info(
                    f"[{topic}] Opened page: "
                    f"{getattr(action, 'url', None)}"
                )

            elif action_type == "find_in_page":
                log.info(
                    f"[{topic}] Find-in-page: "
                    f"{getattr(action, 'pattern', None)} "
                    f"on {getattr(action, 'url', None)}"
                )

        for item in getattr(response, "output", []):
            if getattr(item, "type", None) != "message":
                continue
            for block in getattr(item, "content", []):
                if getattr(block, "type", None) == "output_text":
                    full_text += block.text
                    annotations.extend(getattr(block, "annotations", []) or [])

        lines = []
        for raw_line in full_text.splitlines():
            if not raw_line.strip():
                continue
            match = LINE_PATTERN.match(raw_line)
            if match:
                lines.append((match.group(1).strip(), match.group(2).strip()))

        urls = ordered_citation_urls(annotations)

        if urls:
            break  # got real grounded results, no need to retry

        log.warning(f"  Topic '{topic}' attempt {attempt}: 0 citations returned "
                    f"(model likely didn't ground its answer in search) — "
                    f"{'retrying' if attempt <= MAX_RETRIES_PER_TOPIC else 'giving up'}")

    if len(urls) < len(lines):
        log.warning(f"  Topic '{topic}': {len(lines)} article lines but only "
                    f"{len(urls)} citations returned — extra lines will be dropped")

    candidates = [
        Article(topic=topic, header=header, news=news, source_link=url)
        for (header, news), url in zip(lines, urls)
    ]

    # Hard same-day filter: verify each candidate's real publish date by
    # fetching the page directly, independent of anything the model claimed.
    verified: list[Article] = []
    today = date.today()
    for art in candidates:
        pub_date = get_published_date(art.source_link)
        time.sleep(PAGE_FETCH_DELAY_SECS)

        if pub_date is None:
            log.warning(f"    Dropping (no publish date found): {art.header[:60]!r}")
            if KEEP_UNDATED_ARTICLES:
                verified.append(art)
            continue

        if abs((today - pub_date).days) > DATE_TOLERANCE_DAYS:
            log.warning(f"    Dropping (published {pub_date}, not today): {art.header[:60]!r}")
            continue

        verified.append(art)

    log.info(f"Topic '{topic}': {len(verified)} articles verified as today's news")
    return verified

client, model = get_client_and_model()

def get_verified_articles(topic: str) -> list[dict]:
    """Finds same-day, citation-verified news articles for a topic."""
    articles = fetch_articles_for_topic(client, model, topic)
    return [{"header": a.header, "news": a.news, "source_link": a.source_link} for a in articles]

agent = Agent(
    model=OpenResponses(
            id= os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            base_url= os.getenv("AZURE_OPENAI_ENDPOINT"), 
            api_key= os.getenv("AZURE_OPENAI_API_KEY"),
            parallel_tool_calls=True,
        ),
    tools=[get_verified_articles],
    output_schema=DailyDigest,
    instructions=(
        "For each topic given, call get_verified_articles exactly once. "
        "Call get_verified_articles for all topics in parallel whenever possible. "
        "For each article the tool returns, add one item copying header, news, and "
        "source_link EXACTLY as given — character-for-character, no markdown, no added "
        "links, no reformatting. If the tool returns an empty list for a topic, add one "
        "item: header='N/A', news='No verified same-day articles found', source_link='N/A'."
    ),
)

_CITATION_JUNK = re.compile(r'\s*\(\[[^\]]*\]\(https?://\S+?\)\)')

def clean_text(s: str) -> str:
    return _CITATION_JUNK.sub('', s).strip()

def clean_rows(digest: DailyDigest):
    return [
        {
            "topic": item.topic,
            "header": clean_text(item.header),
            "news": clean_text(item.news),
            "source_link": item.source_link,
        }
        for item in digest.items
    ]
def print_clean_table(df: pd.DataFrame, width: int = 50):
    wrapped = df.copy()
    for col in wrapped.columns:
        if col == "source_link":
            continue  # keep URLs unbroken so terminals can auto-detect them as clickable
        wrapped[col] = wrapped[col].apply(lambda x: "\n".join(textwrap.wrap(str(x), width)) or str(x))
    print(tabulate(wrapped, headers="keys", tablefmt="grid", showindex=False))

async def main():
    start = time.perf_counter()
    response = await agent.arun(f"Topics: {', '.join(TOPICS)}")
    digest: DailyDigest = response.content   
    rows = clean_rows(digest)
    df = pd.DataFrame(rows)
    print_clean_table(df)
    latency = time.perf_counter() - start
    print(f"Latency: {latency:.3f}s")

if __name__ == "__main__":
    asyncio.run(main())