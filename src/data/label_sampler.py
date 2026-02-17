# WHY THIS: Deliberate stratified sampling for LLM calibration, not random.
# Random sampling from 17K articles would over-represent Shriram Finance (24%
# of rows) and under-represent stressed entities with actual credit events.
# Instead we pick 100 likely-credit + 100 likely-noise + 100 ambiguous, capped
# at 15 per entity. This ensures the calibration set exercises all the boundary
# rules in our prompt — the hard cases that determine label quality.

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from loguru import logger


# --- Constants ---
INPUT_CSV = Path("data/processed/gdelt_for_labeling.csv")
CONFIG_PATH = Path("configs/labeling_config.yaml")


def load_config(path: Path) -> dict:
    """Load labeling config YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_articles(path: Path) -> list[dict[str, str]]:
    """Load articles from the for-labeling CSV."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def classify_article(
    article: dict[str, str],
    credit_kw: list[str],
    noise_kw: list[str],
) -> str:
    """Classify an article into credit / noise / ambiguous based on keywords.

    🎓 WHY keyword-based pre-classification: We're NOT using these keywords
    to label articles (that's the LLM's job). We're using them to SAMPLE
    articles that will test different aspects of the prompt. A good calibration
    set should include articles the LLM will find easy AND hard.
    """
    text = (article.get("article_title", "") + " " + article.get("article_text", "")).lower()

    has_credit = any(kw in text for kw in credit_kw)
    has_noise = any(kw in text for kw in noise_kw)

    if has_credit and not has_noise:
        return "credit"
    if has_noise and not has_credit:
        return "noise"
    return "ambiguous"


def stratified_sample(
    articles: list[dict[str, str]],
    bucket: str,
    target_n: int,
    max_per_entity: int,
    classified: dict[str, list[int]],
) -> list[int]:
    """Sample up to target_n article indices from a classification bucket.

    Enforces max_per_entity to prevent any single NBFC from dominating.
    Uses reservoir-style random sampling with entity caps.
    """
    pool = classified[bucket]
    random.shuffle(pool)

    selected: list[int] = []
    entity_counts: Counter[str] = Counter()

    for idx in pool:
        article = articles[idx]
        entities = [e.strip() for e in article["entities"].split(",")]

        # Check entity cap for ALL entities this article is associated with
        if any(entity_counts[e] >= max_per_entity for e in entities):
            continue

        selected.append(idx)
        for e in entities:
            entity_counts[e] += 1

        if len(selected) >= target_n:
            break

    return selected


def sample_calibration(
    articles: list[dict[str, str]], config: dict, seed: int = 42
) -> list[dict[str, str]]:
    """Select ~300 articles for Phase 1 calibration labeling.

    Returns articles with an added 'sample_bucket' column.
    """
    random.seed(seed)
    sampling_cfg = config["sampling"]
    per_bucket = sampling_cfg["per_bucket"]
    max_per_entity = sampling_cfg["max_per_entity"]
    credit_kw = [k.lower() for k in sampling_cfg["credit_keywords"]]
    noise_kw = [k.lower() for k in sampling_cfg["noise_keywords"]]

    # Classify all articles
    classified: dict[str, list[int]] = defaultdict(list)
    for i, article in enumerate(articles):
        bucket = classify_article(article, credit_kw, noise_kw)
        classified[bucket] = classified.get(bucket, [])
        classified[bucket].append(i)

    logger.info(
        f"Pre-classification: credit={len(classified['credit'])}, "
        f"noise={len(classified['noise'])}, ambiguous={len(classified['ambiguous'])}"
    )

    # Sample from each bucket
    selected_indices: set[int] = set()
    result: list[dict[str, str]] = []

    for bucket in ["credit", "noise", "ambiguous"]:
        indices = stratified_sample(
            articles, bucket, per_bucket, max_per_entity, classified
        )
        for idx in indices:
            if idx not in selected_indices:
                selected_indices.add(idx)
                article = dict(articles[idx])  # copy
                article["sample_bucket"] = bucket
                result.append(article)

    logger.info(f"Selected {len(result)} articles for calibration")
    return result


def write_sample(articles: list[dict[str, str]], path: Path) -> None:
    """Write the calibration sample to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "article_url", "article_title", "article_date", "source_domain",
        "gdelt_tone", "entities", "rating_windows", "article_text",
        "source_bucket", "sample_bucket",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for a in articles:
            writer.writerow(a)
    logger.info(f"Wrote {len(articles)} calibration samples to {path}")


def print_summary(articles: list[dict[str, str]]) -> None:
    """Print sampling summary per OUTPUT_STYLE.md."""
    bucket_counts = Counter(a["sample_bucket"] for a in articles)
    entity_counts: Counter[str] = Counter()
    for a in articles:
        for e in a["entities"].split(","):
            entity_counts[e.strip()] += 1

    text_lengths = [len(a.get("article_text", "")) for a in articles]
    avg_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0

    print("\n┌──────────────────────────────────────────────────────────┐")
    print("│ Calibration Sample: Phase 1 Deliberate Sampling         │")
    print("├──────────────────────────────────────────────────────────┤")
    print(f"│ Total selected:          {len(articles):>7}                        │")
    print(f"│ Credit-likely:           {bucket_counts.get('credit', 0):>7}                        │")
    print(f"│ Noise-likely:            {bucket_counts.get('noise', 0):>7}                        │")
    print(f"│ Ambiguous:               {bucket_counts.get('ambiguous', 0):>7}                        │")
    print(f"│ Avg text length:         {avg_len:>7.0f} chars                   │")
    print(f"│ Unique entities:         {len(entity_counts):>7}                        │")
    print("├──────────────────────────────────────────────────────────┤")
    print("│ Top entities in sample:                                  │")
    for entity, count in entity_counts.most_common(10):
        print(f"│   {entity:<36} {count:>5}          │")
    print("└──────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample 300 articles for Phase 1 LLM calibration"
    )
    parser.add_argument(
        "--input", type=Path, default=INPUT_CSV,
        help=f"Path to labeling CSV (default: {INPUT_CSV})",
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_PATH,
        help=f"Path to labeling config (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path (default: from config)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = args.output or Path(config["paths"]["calibration_sample"])

    logger.info("Phase 1.3 Step 4a: Calibration sampling")
    articles = load_articles(args.input)
    logger.info(f"Loaded {len(articles)} articles from {args.input}")

    sample = sample_calibration(articles, config, seed=args.seed)
    write_sample(sample, output_path)
    print_summary(sample)

    print(f"\n⏭️  NEXT: Run calibration labeling:")
    print(f"   python -m src.data.label_articles --phase calibration")
