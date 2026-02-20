# WHY THIS: Imports the 74K-row GDELT article dump and deduplicates by URL.
# The raw GDELT CSV has one row per (article, rating_action_window). The same
# article URL can appear in multiple windows — e.g., a DHFL article found in
# both the 2019-06 downgrade window and the 2019-08 default window. We collapse
# these into one row per URL, preserving ALL associated rating windows as a JSON
# list. This avoids double-counting articles in downstream labeling/training.

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from loguru import logger

from src.data.models import ActionType, read_rating_actions_csv

# --- Constants ---
GDELT_CSV = Path("data/raw/gdelt_articles.csv")
RATING_CSV = Path("data/raw/rating_actions_sourced.csv")
OUTPUT_CSV = Path("data/processed/gdelt_deduped.csv")

# 🎓 Outcome classification — matches seed_ratings.py grouping.
# "suspended" is negative because agencies typically suspend when they
# suspect the entity is hiding information (a red flag, not neutral).
NEGATIVE_ACTIONS = {"downgrade", "default", "watchlist_negative", "outlook_negative", "suspended"}
POSITIVE_ACTIONS = {"upgrade", "watchlist_positive", "outlook_positive"}


def classify_outcome(action_type: str) -> str:
    """Map an action_type string to negative/positive/neutral."""
    if action_type in NEGATIVE_ACTIONS:
        return "negative"
    if action_type in POSITIVE_ACTIONS:
        return "positive"
    return "neutral"


def build_action_lookup(rating_path: Path) -> dict[tuple[str, str], str]:
    """Build (entity, date_iso) → action_type lookup from rating actions CSV.

    🎓 WHY a lookup dict: We need to attach the action_type to each GDELT
    window. The GDELT CSV has entity + rating_action_date but NOT the
    action_type. This lookup lets us join the two datasets in O(1) per row.
    """
    actions = read_rating_actions_csv(rating_path)
    lookup: dict[tuple[str, str], str] = {}
    for a in actions:
        key = (a.entity, a.date.isoformat())
        # If multiple agencies rated same entity on same date, keep most severe
        existing = lookup.get(key)
        if existing is None or existing not in NEGATIVE_ACTIONS:
            lookup[key] = a.action_type.value
    logger.info(f"Built action lookup: {len(lookup)} unique (entity, date) pairs")
    return lookup


def deduplicate(
    gdelt_path: Path, action_lookup: dict[tuple[str, str], str]
) -> list[dict[str, str]]:
    """Read GDELT CSV, group by URL, merge metadata + rating windows."""
    # Group all rows by article URL
    url_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    row_count = 0
    with open(gdelt_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url_groups[row["article_url"]].append(row)
            row_count += 1
    logger.info(f"Read {row_count} rows → {len(url_groups)} unique URLs")

    articles: list[dict[str, str]] = []
    for url, rows in url_groups.items():
        first = rows[0]

        # Collect unique (entity, date) windows — avoid duplicates
        seen: set[tuple[str, str]] = set()
        windows: list[dict] = []
        entities: set[str] = set()

        for row in rows:
            entity = row["entity"]
            action_date = row["rating_action_date"]
            window_key = (entity, action_date)
            if window_key in seen:
                continue
            seen.add(window_key)
            entities.add(entity)

            action_type = action_lookup.get(window_key, "unknown")
            outcome = classify_outcome(action_type)
            try:
                days = int(row["days_before_action"]) if row["days_before_action"] else 0
            except ValueError:
                days = 0

            windows.append({
                "entity": entity,
                "date": action_date,
                "action_type": action_type,
                "outcome": outcome,
                "days_before": days,
            })

        articles.append({
            "article_url": url,
            "article_title": first.get("article_title", ""),
            "article_date": first.get("article_date", ""),
            "source_domain": first.get("source_domain", ""),
            "gdelt_tone": first.get("gdelt_tone", ""),
            "entities": ",".join(sorted(entities)),
            "rating_windows": json.dumps(windows, ensure_ascii=False),
        })

    return sorted(articles, key=lambda a: a.get("article_date", ""))


def write_csv(articles: list[dict[str, str]], path: Path) -> None:
    """Write deduped articles to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "article_url", "article_title", "article_date", "source_domain",
        "gdelt_tone", "entities", "rating_windows",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in articles:
            writer.writerow(a)
    logger.info(f"Wrote {len(articles)} deduped articles to {path}")


def print_summary(articles: list[dict[str, str]]) -> None:
    """Print summary table per OUTPUT_STYLE.md."""
    entity_counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()
    outcome_counter: Counter[str] = Counter()
    dates: list[str] = []
    total_windows = 0

    for a in articles:
        domain_counter[a["source_domain"]] += 1
        for e in a["entities"].split(","):
            entity_counter[e.strip()] += 1
        if a["article_date"]:
            dates.append(a["article_date"])
        for w in json.loads(a["rating_windows"]):
            outcome_counter[w["outcome"]] += 1
            total_windows += 1

    date_range = f"{min(dates)} to {max(dates)}" if dates else "N/A"
    top_domains = domain_counter.most_common(5)
    top_entities = entity_counter.most_common(5)

    print("\n┌──────────────────────────────────────────────────────────┐")
    print("│ Dataset: GDELT Articles (Deduped by URL)                │")
    print("├──────────────────────────────────────────────────────────┤")
    print(f"│ Unique articles (URLs):     {len(articles):>7}                     │")
    print(f"│ Total rating windows:       {total_windows:>7}                     │")
    print(f"│ Unique entities:            {len(entity_counter):>7}                     │")
    print(f"│ Unique domains:             {len(domain_counter):>7}                     │")
    print(f"│ Date range:    {date_range:<42}│")
    print("├──────────────────────────────────────────────────────────┤")
    print("│ Windows by outcome:                                     │")
    for outcome in ["negative", "positive", "neutral"]:
        c = outcome_counter.get(outcome, 0)
        pct = c / total_windows * 100 if total_windows else 0
        print(f"│   {outcome:<12} {c:>6}  ({pct:>5.1f}%)                       │")
    unknown = outcome_counter.get("unknown", 0)
    if unknown:
        print(f"│   {'unknown':<12} {unknown:>6}                                │")
    print("├──────────────────────────────────────────────────────────┤")
    print("│ Top entities (articles mentioning):                     │")
    for name, count in top_entities:
        print(f"│   {name:<30} {count:>6}                  │")
    print("├──────────────────────────────────────────────────────────┤")
    print("│ Top sources:                                            │")
    for domain, count in top_domains:
        print(f"│   {domain:<36} {count:>5}          │")
    print("└──────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import and deduplicate GDELT articles for credit signal pipeline"
    )
    parser.add_argument(
        "--gdelt", type=Path, default=GDELT_CSV,
        help=f"Path to raw GDELT CSV (default: {GDELT_CSV})",
    )
    parser.add_argument(
        "--ratings", type=Path, default=RATING_CSV,
        help=f"Path to rating actions CSV (default: {RATING_CSV})",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_CSV,
        help=f"Output path for deduped CSV (default: {OUTPUT_CSV})",
    )
    args = parser.parse_args()

    logger.info("Phase 1.3 Step 1: Import & deduplicate GDELT articles")
    action_lookup = build_action_lookup(args.ratings)
    articles = deduplicate(args.gdelt, action_lookup)
    write_csv(articles, args.output)
    print_summary(articles)
