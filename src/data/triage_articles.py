# WHY THIS: Mechanical title-based filter BEFORE expensive LLM labeling.
# GDELT returns many false matches — defence news, sports, stock tips that
# happened to mention a keyword. This triage removes obvious noise using
# three layers: (1) known vs unknown source domains, (2) financial keyword
# presence in title, (3) entity name actually appearing in title.
# We err on the side of KEEPING articles. Borderline → "review" bucket.
# The LLM labeler (Phase 1.3 Step 4) handles the hard calls.

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import yaml
from loguru import logger

# --- Constants ---
DEDUPED_CSV = Path("data/processed/gdelt_deduped.csv")
ENTITY_CONFIG = Path("configs/nbfc_entities.yaml")
TRIAGE_CONFIG = Path("configs/triage_config.yaml")
OUTPUT_DIR = Path("data/processed")


def load_triage_config(path: Path) -> dict:
    """Load triage filter config (domains + keywords)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_entity_alias_map(yaml_path: Path) -> dict[str, list[str]]:
    """Build mapping from any entity name → all searchable name forms.

    🎓 WHY reverse lookup: The CSV entity column might use "PFC" while the
    YAML canonical name is "Power Finance Corporation". We need to look up
    by CSV name and get ALL aliases for title matching. This dict maps
    every name/alias to the full set of lowercased search terms.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    alias_map: dict[str, list[str]] = {}
    for entities in data["subsectors"].values():
        for entity in entities:
            all_names = [entity["name"]] + entity.get("aliases", [])
            all_lower = [n.lower() for n in all_names]
            # Map every form to the full set
            for name in all_names:
                alias_map[name] = all_lower
    return alias_map


def domain_passes(domain: str, allowed: list[str]) -> bool:
    """Check if domain matches any allowed domain by suffix.

    🎓 WHY suffix match: "indiatimes.com" should match both
    "economictimes.indiatimes.com" and "articles.economictimes.indiatimes.com".
    Standard domain suffix check: domain equals OR ends with "." + allowed.
    """
    d = domain.lower().strip()
    return any(d == a or d.endswith("." + a) for a in allowed)


def title_has_keywords(title: str, keywords: list[str]) -> bool:
    """Check if title contains any financial/credit keyword."""
    t = title.lower()
    return any(kw in t for kw in keywords)


def title_has_entity(title: str, entities: list[str], alias_map: dict[str, list[str]]) -> bool:
    """Check if any associated entity name/alias appears in the title.

    🎓 WHY word boundary for short names: "REC" (3 chars) could match
    "RECORD" or "RECOVERY". We use regex word boundaries for names
    ≤3 chars to avoid false positives. Longer names like "Bajaj Finance"
    are unlikely to be substrings of unrelated words.
    """
    t = title.lower()
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


def read_deduped(path: Path) -> list[dict[str, str]]:
    """Read the deduped GDELT CSV."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def triage_all(
    articles: list[dict[str, str]],
    config: dict,
    alias_map: dict[str, list[str]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Apply three-layer filter. Returns (passed, review, dropped)."""
    allowed = [d.lower() for d in config["source_domains"]["allowed"]]
    blocked = [d.lower() for d in config["source_domains"].get("blocked", [])]
    keywords = [k.lower() for k in config["title_keywords"]]

    passed, review, dropped = [], [], []

    for article in articles:
        domain = article["source_domain"].lower().strip()
        title = article.get("article_title", "")
        entities = [e.strip() for e in article["entities"].split(",")]

        # Layer 1: Blocked domain → auto-drop
        if domain_passes(domain, blocked):
            article["triage_reason"] = "blocked_domain"
            dropped.append(article)
            continue

        # Layer 2: Title must have financial keywords OR entity name
        has_kw = title_has_keywords(title, keywords)
        has_ent = title_has_entity(title, entities, alias_map)

        if not has_kw and not has_ent:
            article["triage_reason"] = "no_financial_terms_or_entity"
            dropped.append(article)
            continue

        # Layer 3: Entity not in title → likely false GDELT match
        if not has_ent:
            article["triage_reason"] = "entity_not_in_title"
            review.append(article)
            continue

        # Layer 4: Unknown domain → review even if content looks OK
        if not domain_passes(domain, allowed):
            article["triage_reason"] = "unknown_domain"
            review.append(article)
            continue

        # All checks passed
        article["triage_reason"] = ""
        passed.append(article)

    return passed, review, dropped


def write_bucket(articles: list[dict], path: Path, include_reason: bool = False) -> None:
    """Write a triage bucket to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "article_url", "article_title", "article_date", "source_domain",
        "gdelt_tone", "entities", "rating_windows",
    ]
    fields = base_fields + (["triage_reason"] if include_reason else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for a in articles:
            writer.writerow(a)


def get_outcomes(article: dict) -> set[str]:
    """Extract unique outcome values from an article's rating windows."""
    return {w["outcome"] for w in json.loads(article["rating_windows"])}


def print_summary(
    passed: list[dict], review: list[dict], dropped: list[dict],
) -> None:
    """Print triage summary with outcome survival rates."""
    total = len(passed) + len(review) + len(dropped)
    p_pct = len(passed) / total * 100 if total else 0
    r_pct = len(review) / total * 100 if total else 0
    d_pct = len(dropped) / total * 100 if total else 0

    print("\n┌──────────────────────────────────────────────────────────┐")
    print("│ Triage Results: Title-Based Noise Filter                │")
    print("├──────────────────────────────────────────────────────────┤")
    print(f"│ Total articles:          {total:>7}                        │")
    print(f"│ ✅ Passed:                {len(passed):>7}  ({p_pct:>5.1f}%)              │")
    print(f"│ 🔍 Review:                {len(review):>7}  ({r_pct:>5.1f}%)              │")
    print(f"│ ❌ Dropped:               {len(dropped):>7}  ({d_pct:>5.1f}%)              │")

    # Drop reasons
    drop_reasons: Counter[str] = Counter()
    for a in dropped:
        drop_reasons[a.get("triage_reason", "unknown")] += 1
    review_reasons: Counter[str] = Counter()
    for a in review:
        review_reasons[a.get("triage_reason", "unknown")] += 1

    print("├──────────────────────────────────────────────────────────┤")
    print("│ Drop reasons:                                           │")
    for reason, count in drop_reasons.most_common():
        print(f"│   {reason:<36} {count:>6}          │")
    print("│ Review reasons:                                         │")
    for reason, count in review_reasons.most_common():
        print(f"│   {reason:<36} {count:>6}          │")

    # Outcome survival rates — THE critical check
    print("├──────────────────────────────────────────────────────────┤")
    print("│ 🎓 Outcome survival rates (are we keeping negatives?):  │")
    for outcome in ["negative", "positive", "neutral"]:
        n_pass = sum(1 for a in passed if outcome in get_outcomes(a))
        n_rev = sum(1 for a in review if outcome in get_outcomes(a))
        n_drop = sum(1 for a in dropped if outcome in get_outcomes(a))
        n_total = n_pass + n_rev + n_drop
        if n_total == 0:
            continue
        surv = (n_pass + n_rev) / n_total * 100  # pass + review = not lost
        print(f"│   {outcome:<10}  total={n_total:>5}  "
              f"pass={n_pass:>5}  review={n_rev:>5}  "
              f"drop={n_drop:>5}  │")
        print(f"│              survival (pass+review): {surv:>5.1f}%"
              f"                │")

    # Top 5 entities by pass rate
    print("├──────────────────────────────────────────────────────────┤")
    print("│ Pass rate by entity (top 10 by volume):                 │")
    entity_pass: Counter[str] = Counter()
    entity_total: Counter[str] = Counter()
    for bucket in [passed, review, dropped]:
        for a in bucket:
            for e in a["entities"].split(","):
                e = e.strip()
                entity_total[e] += 1
                if a in passed:
                    entity_pass[e] += 1
    # Re-count pass properly (above is wrong for review/dropped)
    entity_pass = Counter()
    for a in passed:
        for e in a["entities"].split(","):
            entity_pass[e.strip()] += 1

    for entity, total_count in entity_total.most_common(10):
        p = entity_pass.get(entity, 0)
        rate = p / total_count * 100 if total_count else 0
        print(f"│   {entity:<28} {p:>5}/{total_count:>5}  ({rate:>5.1f}%)   │")

    print("└──────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Title-based triage filter for GDELT articles"
    )
    parser.add_argument(
        "--input", type=Path, default=DEDUPED_CSV,
        help=f"Path to deduped GDELT CSV (default: {DEDUPED_CSV})",
    )
    parser.add_argument(
        "--config", type=Path, default=TRIAGE_CONFIG,
        help=f"Path to triage config YAML (default: {TRIAGE_CONFIG})",
    )
    parser.add_argument(
        "--entities", type=Path, default=ENTITY_CONFIG,
        help=f"Path to entity config YAML (default: {ENTITY_CONFIG})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    logger.info("Phase 1.3 Step 2: Title-based triage filter")
    config = load_triage_config(args.config)
    alias_map = build_entity_alias_map(args.entities)
    articles = read_deduped(args.input)
    logger.info(f"Loaded {len(articles)} deduped articles")

    passed, review, dropped = triage_all(articles, config, alias_map)

    write_bucket(passed, args.output_dir / "gdelt_triaged_pass.csv")
    write_bucket(review, args.output_dir / "gdelt_triaged_review.csv", include_reason=True)
    write_bucket(dropped, args.output_dir / "gdelt_triaged_drop.csv", include_reason=True)
    logger.info(
        f"Wrote pass={len(passed)}, review={len(review)}, drop={len(dropped)} "
        f"to {args.output_dir}"
    )

    print_summary(passed, review, dropped)
