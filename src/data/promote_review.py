# WHY THIS: The review bucket has 14,002 articles split into two groups:
# (a) "unknown_domain" (2,011) — entity IS in title, just from an unlisted domain.
#     These are auto-promoted since they passed keyword + entity checks.
# (b) "entity_not_in_title" (11,991) — keyword matched but entity missing from title.
#     We check if the entity appears in the scraped BODY text. If yes → promote.
#     If no → drop (the article is genuinely about a different entity).
# This avoids sending ~8,000 garbage articles to the expensive LLM labeler.

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import yaml
from loguru import logger

REVIEW_SCRAPED = Path(
    "/Users/coddiwomplers/Desktop/Python/data_scraping/output/gdelt_review_article_texts.csv"
)
ENTITY_CONFIG = Path("configs/nbfc_entities.yaml")
OUTPUT_DIR = Path("data/processed")


def build_entity_alias_map(yaml_path: Path) -> dict[str, list[str]]:
    """Build mapping from any entity name → all searchable name forms."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    alias_map: dict[str, list[str]] = {}
    for entities in data["subsectors"].values():
        for entity in entities:
            all_names = [entity["name"]] + entity.get("aliases", [])
            all_lower = [n.lower() for n in all_names]
            for name in all_names:
                alias_map[name] = all_lower
    return alias_map


def text_has_entity(text: str, entities: list[str], alias_map: dict[str, list[str]]) -> bool:
    """Check if any entity name/alias appears in text. Word boundary for short names."""
    t = text.lower()
    for entity in entities:
        names = alias_map.get(entity, [entity.lower()])
        for name in names:
            if len(name) <= 3:
                if re.search(r"\b" + re.escape(name) + r"\b", t):
                    return True
            else:
                if name in t:
                    return True
    return False


def get_outcomes(article: dict[str, str]) -> set[str]:
    """Extract unique outcome values from rating windows."""
    return {w["outcome"] for w in json.loads(article["rating_windows"])}


def run_check(scraped_path: Path, alias_map: dict[str, list[str]]) -> tuple[list, list]:
    """Check each review article. Returns (promoted, dropped)."""
    promoted: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    stats: Counter[str] = Counter()

    with open(scraped_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entities = [e.strip() for e in row["entities"].split(",")]
            title = row.get("article_title", "")
            body = row.get("article_text", "") or ""
            scrape_ok = row.get("scrape_status", "") == "ok"

            # Case A: Entity already in title → auto-promote (was unknown_domain)
            if text_has_entity(title, entities, alias_map):
                row["promote_reason"] = "entity_in_title"
                promoted.append(row)
                stats["entity_in_title"] += 1
                continue

            # Case B: Scrape failed → no body to check → drop
            if not scrape_ok or len(body.strip()) < 50:
                row["promote_reason"] = "scrape_failed"
                dropped.append(row)
                stats["scrape_failed"] += 1
                continue

            # Case C: Entity in body → promote
            if text_has_entity(body, entities, alias_map):
                row["promote_reason"] = "entity_in_body"
                promoted.append(row)
                stats["entity_in_body"] += 1
                continue

            # Case D: Entity not in body → drop
            row["promote_reason"] = "entity_not_in_body"
            dropped.append(row)
            stats["entity_not_in_body"] += 1

    return promoted, dropped, stats


def print_summary(
    promoted: list[dict], dropped: list[dict], stats: Counter,
) -> None:
    """Print results with outcome survival rates."""
    total = len(promoted) + len(dropped)

    print("\n┌──────────────────────────────────────────────────────────┐")
    print("│ Review Bucket: Entity-in-Body Check                     │")
    print("├──────────────────────────────────────────────────────────┤")
    print(f"│ Total review articles:      {total:>7}                      │")
    print(f"│ ✅ Promoted to pass:         {len(promoted):>7}  "
          f"({len(promoted)/total*100:>5.1f}%)              │")
    print(f"│ ❌ Dropped:                  {len(dropped):>7}  "
          f"({len(dropped)/total*100:>5.1f}%)              │")
    print("├──────────────────────────────────────────────────────────┤")
    print("│ Breakdown:                                              │")
    for reason, count in stats.most_common():
        label = {
            "entity_in_title": "Entity in title (auto-promote)",
            "entity_in_body": "Entity found in body text",
            "entity_not_in_body": "Entity NOT in body (dropped)",
            "scrape_failed": "Scrape failed, no body (dropped)",
        }.get(reason, reason)
        print(f"│   {label:<40} {count:>5}       │")

    print("├──────────────────────────────────────────────────────────┤")
    print("│ 🎓 Outcome survival (promoted vs dropped):              │")
    for outcome in ["negative", "positive", "neutral"]:
        n_pro = sum(1 for a in promoted if outcome in get_outcomes(a))
        n_drop = sum(1 for a in dropped if outcome in get_outcomes(a))
        n_total = n_pro + n_drop
        if n_total == 0:
            continue
        surv = n_pro / n_total * 100
        print(f"│   {outcome:<10}  total={n_total:>5}  "
              f"promoted={n_pro:>5}  dropped={n_drop:>5}  "
              f"({surv:>5.1f}%) │")
    print("└──────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check review articles for entity mentions in body text"
    )
    parser.add_argument(
        "--input", type=Path, default=REVIEW_SCRAPED,
        help="Path to scraped review CSV",
    )
    parser.add_argument(
        "--entities", type=Path, default=ENTITY_CONFIG,
        help="Path to entity config YAML",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Output directory",
    )
    args = parser.parse_args()

    logger.info("Phase 1.3: Entity-in-body check for review articles")
    alias_map = build_entity_alias_map(args.entities)
    promoted, dropped, stats = run_check(args.input, alias_map)

    # Write promoted articles (these join the pass set)
    out_promoted = args.output_dir / "gdelt_review_promoted.csv"
    out_promoted.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "article_url", "article_title", "article_date", "source_domain",
        "gdelt_tone", "entities", "rating_windows", "article_text",
        "scrape_status", "promote_reason",
    ]
    with open(out_promoted, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for a in promoted:
            writer.writerow(a)
    logger.info(f"Wrote {len(promoted)} promoted articles to {out_promoted}")

    print_summary(promoted, dropped, stats)
