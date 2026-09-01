"""
news_agent.py

Pulls N articles per topic (combined across all sources) using the
AZURE OpenAI Responses API's built-in web_search tool, with the search
scoped to your source domains via the API's native `allowed_domains` filter
(not a "site:" prompt hack). Source URLs are pulled directly from the
response's citation annotations (url_citation), never from model-typed text.

Output columns: Topic | Header | News | Source Link

Requirements:
    pip install openai

Environment variables required:
    AZURE_OPENAI_ENDPOINT     e.g. https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/
    AZURE_OPENAI_API_KEY      your Azure OpenAI resource API key
    AZURE_OPENAI_DEPLOYMENT_NAME   the deployment name you gave your model in Azure
                              (must be GPT-4 class or later; e.g. gpt-4o, gpt-4.1, gpt-5.5)
"""

import os
import re
import sys
import logging
import argparse
from datetime import datetime, timezone
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()  # load .env file if present
import pandas as pd
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

ARTICLES_PER_TOPIC = 5  # combined across ALL sources, not per source
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


def extract_citation_for_span(annotations, start: int, end: int):
    """Return the URL of the first citation annotation overlapping [start, end)."""
    for ann in annotations:
        ann_start = getattr(ann, "start_index", None)
        ann_end = getattr(ann, "end_index", None)
        ann_url = getattr(ann, "url", None)
        if ann_url is None or ann_start is None or ann_end is None:
            continue
        if ann_start < end and ann_end > start:
            return ann_url
    return None


def fetch_articles_for_topic(client: OpenAI, model: str, topic: str,
                              count: int = ARTICLES_PER_TOPIC) -> list[Article]:
    """
    One Responses API call per topic. Search is restricted to SOURCE_DOMAINS
    via the API's native allowed_domains filter. Returns up to `count`
    Articles, each with a Source Link taken from a real citation annotation —
    items without a matching citation are dropped rather than guessed.
    """
    prompt = (
        f"Find the {count} most recent, most relevant news articles (combined "
        f"across all allowed sources) about the topic: \"{topic}\".\n"
        f"If a result is from reuters.com, prefer articles from its "
        f"Middle East section (reuters.com/world/middle-east).\n\n"
        f"Output ONLY a numbered list, one article per line, in exactly this format "
        f"(no extra text, no markdown, no headers):\n"
        f"1. HEADLINE ::: ONE-TO-TWO SENTENCE SUMMARY\n"
        f"2. HEADLINE ::: ONE-TO-TWO SENTENCE SUMMARY\n"
        f"...\n\n"
        f"Only include articles you actually found via search. If fewer than "
        f"{count} genuinely relevant articles exist, output fewer lines."
    )

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
        log.error(f"API call failed for topic '{topic}': {e}")
        return []

    # Walk the response output to find the assistant message's text + annotations.
    full_text = ""
    annotations = []
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", []):
            if getattr(block, "type", None) == "output_text":
                full_text += block.text
                annotations.extend(getattr(block, "annotations", []) or [])

    if not full_text:
        log.warning(f"No output text returned for topic '{topic}' "
                     f"(check that your deployment supports web_search)")
        return []

    articles = []
    cursor = 0
    for raw_line in full_text.splitlines():
        if not raw_line.strip():
            continue
        match = LINE_PATTERN.match(raw_line)
        line_start = full_text.find(raw_line, cursor)
        line_end = line_start + len(raw_line)
        cursor = line_end

        if not match:
            continue

        header, news = match.group(1).strip(), match.group(2).strip()
        url = extract_citation_for_span(annotations, line_start, line_end)

        if not url:
            log.warning(f"  No verified citation for line, dropping: {header[:60]!r}")
            continue

        articles.append(Article(topic=topic, header=header, news=news, source_link=url))

    log.info(f"Topic '{topic}': {len(articles)} articles with verified links")
    return articles[:count]


def run(topics=None, count=ARTICLES_PER_TOPIC) -> pd.DataFrame:
    topics = topics or TOPICS
    client, model = get_client_and_model()

    all_articles: list[Article] = []
    for topic in topics:
        all_articles.extend(fetch_articles_for_topic(client, model, topic, count))

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
    parser = argparse.ArgumentParser(description="Pull topic-tagged news articles into a table.")
    parser.add_argument("--count", type=int, default=ARTICLES_PER_TOPIC,
                         help="Articles per topic, combined across all sources (default: 5)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                         help="Directory to save CSV/Excel output")
    args = parser.parse_args()

    log.info("Starting news pull...")
    df = run(count=args.count)

    if df.empty:
        log.warning("No articles were retrieved. Nothing to save.")
        return

    log.info(f"\nTotal articles retrieved: {len(df)}\n")
    print(df.to_string(index=False, max_colwidth=60))

    save_output(df, args.output_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()