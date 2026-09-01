"""
news_agent.py

Pulls TODAY'S news articles per topic (combined across all sources) using
the AZURE OpenAI Responses API's built-in web_search tool, with the search
scoped to your source domains via the API's native `allowed_domains` filter
(not a "site:" prompt hack). Source URLs are pulled directly from the
response's citation annotations (url_citation), never from model-typed text.

No fixed article count per topic — this is a daily digest, so a topic gets
however many genuinely same-day, verified articles exist (including zero).

Output columns: Topic | Header | News | Source Link

Same-day filter: the web_search tool has no reliable "published date" field, and
Azure's grounding is index-based (not live), so we can't trust the model's own
claim of freshness. Instead, after getting a verified URL from a citation, we
fetch that page directly and read its actual publish-date metadata, then keep
only articles published on the same calendar date the script is run.

Requirements:
    pip install openai requests beautifulsoup4

Environment variables required:
    AZURE_OPENAI_ENDPOINT     e.g. https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/
    AZURE_OPENAI_API_KEY      your Azure OpenAI resource API key
    AZURE_OPENAI_DEPLOYMENT_NAME   the deployment name you gave your model in Azure
                              (must be GPT-4 class or later; e.g. gpt-4o, gpt-4.1, gpt-5.5)
"""

import os
import re
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timezone, date
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()
import requests
import pandas as pd
from bs4 import BeautifulSoup
from openai import OpenAI

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
REQUEST_LIMIT_PER_TOPIC = 15

# How many days off from "today" a publish date is still allowed to be.
# 0 = must be published on the exact same calendar date as the run.
# Loosen this (e.g. to 1) if same-day-only proves too strict in practice.
DATE_TOLERANCE_DAYS = 0

# If we can't determine an article's publish date at all (fetch blocked,
# no date metadata on the page), should we keep it anyway? Default: no —
# the whole point of this filter is a hard same-day guarantee.
KEEP_UNDATED_ARTICLES = False

MAX_RETRIES_PER_TOPIC = 2  # retries if a call comes back with zero citations
REQUEST_TIMEOUT_SECS = 10
PAGE_FETCH_DELAY_SECS = 0.5  # be polite between page fetches
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Defaults to an "output" folder next to this script, wherever it's run from
# (works the same on Windows, macOS, Linux, or an Azure WebJob/Function).
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

LINE_PATTERN = re.compile(r"^\s*\d+\.\s*(.+?)\s*:::\s*(.+?)\s*$")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("news_agent")


@dataclass
class Article:
    topic: str
    header: str
    news: str
    source_link: str


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
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECS)
        if not resp.ok:
            log.warning(f"    Date check: HTTP {resp.status_code} fetching {url}")
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

    except requests.RequestException as e:
        log.warning(f"    Date check: could not fetch {url} ({e})")
        return None


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
        f"Today's date is {today_str}. Find ALL news articles (combined across "
        f"all allowed sources) published TODAY, {today_str}, about the topic: "
        f"\"{topic}\" — up to a maximum of {request_limit}. Do not include older "
        f"articles even if they are relevant — only today's news. If there are "
        f"none published today, that's fine — just return an empty list.\n"
        f"If a result is from reuters.com, prefer articles from its "
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


def run(topics=None, request_limit=REQUEST_LIMIT_PER_TOPIC) -> pd.DataFrame:
    topics = topics or TOPICS
    client, model = get_client_and_model()

    all_articles: list[Article] = []
    for topic in topics:
        all_articles.extend(fetch_articles_for_topic(client, model, topic, request_limit))

    df = pd.DataFrame([
        {"Topic": a.topic, "Header": a.header, "News": a.news, "Source Link": a.source_link}
        for a in all_articles
    ])
    return df


def save_output(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"news_digest_{timestamp}.csv")
    xlsx_path = os.path.join(output_dir, f"news_digest_{timestamp}.xlsx")

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    log.info(f"Saved CSV:   {csv_path}")
    log.info(f"Saved Excel: {xlsx_path}")
    return csv_path, xlsx_path


def main():
    parser = argparse.ArgumentParser(description="Pull today's topic-tagged news articles into a table.")
    parser.add_argument("--request-limit", type=int, default=REQUEST_LIMIT_PER_TOPIC,
                         help="Upper bound on articles requested per topic (default: 15). "
                              "Not a cap on final results — all same-day verified articles are kept.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                         help="Directory to save CSV/Excel output")
    args = parser.parse_args()

    log.info("Starting news pull...")
    df = run(request_limit=args.request_limit)

    if df.empty:
        log.warning("No articles were retrieved. Nothing to save.")
        return

    log.info(f"\nTotal articles retrieved: {len(df)}\n")
    print(df.to_string(index=False, max_colwidth=60))

    save_output(df, args.output_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()