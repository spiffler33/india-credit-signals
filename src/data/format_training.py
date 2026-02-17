# WHY THIS: Converts 17,274 LLM-labeled articles into fine-tuning JSONL.
# Joins labels (from labels_final.jsonl) with article text (from gdelt_for_labeling.csv),
# formats each row as instruction/input/output, then splits temporally.
# This is the bridge between "data we collected" (Phase 1.3) and "data the model
# trains on" (Phase 2). The temporal split is critical — random shuffling would
# let the model "see the future" (data leakage).

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import yaml
from loguru import logger

from src.data.label_models import ArticleLabel, read_labels_jsonl
from src.data.parse_training_output import format_output_text


def load_config(config_path: Path) -> dict:
    """Load training config YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_articles(csv_path: Path) -> dict[str, dict]:
    """Load articles from CSV, keyed by URL.

    Returns dict[url] → {title, date, entities, text}.
    """
    articles: dict[str, dict] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("article_url", "").strip()
            if not url:
                continue
            articles[url] = {
                "title": row.get("article_title", "").strip(),
                "date": row.get("article_date", "").strip(),
                "entities": row.get("entities", "").strip(),
                "text": row.get("article_text", "").strip(),
            }
    return articles


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate article text to max_chars, breaking at word boundary."""
    if len(text) <= max_chars:
        return text
    # Find last space before max_chars to avoid cutting mid-word
    cut = text[:max_chars].rfind(" ")
    if cut < max_chars // 2:
        cut = max_chars  # If no good break point, hard cut
    return text[:cut] + "..."


def format_input(entity: str, article_date: str, title: str, text: str) -> str:
    """Build the input field for a training example."""
    return f"Entity: {entity}\nDate: {article_date}\nTitle: {title}\nArticle: {text}"


def build_example(
    label: ArticleLabel,
    article: dict,
    instruction: str,
    max_chars: int,
    direction_map: dict[int, str],
    confidence_map: dict[str, str],
) -> dict | None:
    """Build one instruction/input/output training example.

    Returns None if article data is missing.
    """
    title = article.get("title", "")
    article_date = article.get("date", "")
    entity = article.get("entities", "")
    text = article.get("text", "")

    if not text:
        return None

    text = truncate_text(text, max_chars)

    input_text = format_input(entity, article_date, title, text)
    output_text = format_output_text(
        credit_relevant=label.credit_relevant,
        signal_direction=label.signal_direction,
        signal_type=label.signal_type,
        sector_wide=label.sector_wide,
        confidence=label.confidence,
        reasoning=label.reasoning,
        direction_map=direction_map,
        confidence_map=confidence_map,
    )

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        # Metadata (not used in training, but useful for analysis)
        "meta": {
            "url": label.url,
            "entity": entity,
            "date": article_date,
            "credit_relevant": label.credit_relevant,
            "signal_direction": label.signal_direction,
            "signal_type": label.signal_type,
        },
    }


def parse_date(date_str: str) -> date | None:
    """Parse YYYY-MM-DD date string."""
    try:
        parts = date_str.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return None


def split_examples(
    examples: list[dict],
    train_end: str,
    val_end: str,
    holdout_entities: list[str],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Split examples into train/val/test + entity holdout.

    Entity holdout: single-entity articles for held-out entities go to
    entity_holdout. Multi-entity articles stay in the main split.

    🎓 WHY temporal split: Random shuffling causes data leakage in financial
    data. A 2019 article about DHFL defaulting would "teach" the model to
    flag a 2018 DHFL liquidity article — but in production the model wouldn't
    have seen 2019 yet. Temporal split = honest forward test.
    """
    train_cutoff = parse_date(train_end)
    val_cutoff = parse_date(val_end)

    if not train_cutoff or not val_cutoff:
        raise ValueError(f"Invalid split dates: train_end={train_end}, val_end={val_end}")

    holdout_set = set(holdout_entities)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    entity_holdout: list[dict] = []
    skipped_no_date = 0

    for ex in examples:
        entity = ex["meta"]["entity"]
        date_str = ex["meta"]["date"]
        d = parse_date(date_str)

        if d is None:
            skipped_no_date += 1
            continue

        # Entity holdout: single-entity articles only
        # 🎓 Multi-entity articles (e.g., "DHFL, Shriram Finance") stay in
        # training because they discuss multiple entities and don't teach
        # entity-specific patterns. Minor contamination, noted in design doc.
        is_single_entity = "," not in entity
        if is_single_entity and entity.strip() in holdout_set:
            entity_holdout.append(ex)
            continue

        # Temporal split
        if d <= train_cutoff:
            train.append(ex)
        elif d <= val_cutoff:
            val.append(ex)
        else:
            test.append(ex)

    if skipped_no_date > 0:
        logger.warning(f"Skipped {skipped_no_date} examples with unparseable dates")

    return train, val, test, entity_holdout


def write_jsonl(examples: list[dict], path: Path) -> None:
    """Write examples to JSONL file (without meta field — training only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            # Write instruction/input/output only (meta is for our analysis,
            # not for the training framework)
            record = {
                "instruction": ex["instruction"],
                "input": ex["input"],
                "output": ex["output"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(examples)} examples to {path}")


def print_summary(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    entity_holdout: list[dict],
    total_labels: int,
    total_articles: int,
    joined: int,
) -> None:
    """Print a summary stats table."""
    def split_stats(name: str, examples: list[dict]) -> None:
        if not examples:
            print(f"│  {name:20s}  {'(empty)':>8s}  {'':>8s}  {'':>8s}  {'':>8s}  │")
            return

        cr_counts = Counter(ex["meta"]["credit_relevant"] for ex in examples)
        dir_counts = Counter(
            ex["meta"]["signal_direction"]
            for ex in examples
            if ex["meta"]["credit_relevant"] == 1
        )

        total = len(examples)
        cr1 = cr_counts.get(1, 0)
        det = dir_counts.get(-1, 0)
        imp = dir_counts.get(1, 0)

        print(f"│  {name:20s}  {total:>8,d}  {cr1:>8,d}  {det:>8,d}  {imp:>8,d}  │")

    # Date ranges
    def date_range(examples: list[dict]) -> str:
        dates = [ex["meta"]["date"] for ex in examples if ex["meta"]["date"]]
        if not dates:
            return "N/A"
        return f"{min(dates)} to {max(dates)}"

    # Signal type distribution (across all splits)
    all_examples = train + val + test + entity_holdout
    st_counts = Counter(
        ex["meta"]["signal_type"]
        for ex in all_examples
        if ex["meta"]["credit_relevant"] == 1
    )

    print()
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│          Phase 1.4 — Training Data Format Summary                  │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Labels loaded:        {total_labels:>8,d}                                  │")
    print(f"│  Articles loaded:      {total_articles:>8,d}                                  │")
    print(f"│  Joined (label+text):  {joined:>8,d}                                  │")
    print(f"│  Final examples:       {len(all_examples):>8,d}                                  │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│  Split               Examples    CR=1     Det.     Imp.    │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    split_stats("Train", train)
    split_stats("Validation", val)
    split_stats("Test", test)
    split_stats("Entity Holdout", entity_holdout)
    print("├─────────────────────────────────────────────────────────────────────┤")

    # Date ranges per split
    print(f"│  Train range:      {date_range(train):>40s}      │")
    print(f"│  Val range:        {date_range(val):>40s}      │")
    print(f"│  Test range:       {date_range(test):>40s}      │")
    print(f"│  Holdout range:    {date_range(entity_holdout):>40s}      │")
    print("├─────────────────────────────────────────────────────────────────────┤")

    # Signal type distribution
    print("│  Signal Type Distribution (CR=1 across all splits)                 │")
    print("│  ─────────────────────────────────────────────────                 │")
    for st, count in st_counts.most_common():
        cr1_total = sum(1 for ex in all_examples if ex["meta"]["credit_relevant"] == 1)
        pct = count / cr1_total * 100 if cr1_total else 0
        print(f"│    {st:20s}  {count:>6,d}  ({pct:5.1f}%)                             │")

    # Entity holdout breakdown
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│  Entity Holdout Breakdown                                          │")
    print("│  ─────────────────────────────────────────────────                 │")
    holdout_by_entity = Counter(ex["meta"]["entity"] for ex in entity_holdout)
    for ent, count in holdout_by_entity.most_common():
        det_count = sum(
            1 for ex in entity_holdout
            if ex["meta"]["entity"] == ent and ex["meta"]["signal_direction"] == -1
        )
        det_pct = det_count / count * 100 if count else 0
        print(f"│    {ent:20s}  {count:>6,d} articles  ({det_pct:5.1f}% deterioration)    │")

    print("└─────────────────────────────────────────────────────────────────────┘")
    print()


def main(config_path: Path | None = None) -> None:
    """Main entry point: load data, format, split, write, summarize."""
    if config_path is None:
        config_path = Path("configs/training_config.yaml")

    logger.info(f"Loading config from {config_path}")
    cfg = load_config(config_path)

    # Extract config values
    instruction = cfg["instruction"]
    max_chars = cfg["text"]["max_chars"]
    train_end = cfg["splits"]["train_end"]
    val_end = cfg["splits"]["val_end"]
    holdout_entities = cfg["entity_holdout"]
    output_dir = Path(cfg["paths"]["output_dir"])

    # Build direction and confidence maps (config uses string keys for YAML compat)
    direction_map = {int(k): v for k, v in cfg["direction_map"].items()}
    confidence_map = cfg["confidence_map"]

    # Load labels
    labels_path = Path(cfg["paths"]["labels"])
    logger.info(f"Loading labels from {labels_path}")
    labels = read_labels_jsonl(labels_path)
    labels_by_url = {label.url: label for label in labels}
    logger.info(f"Loaded {len(labels)} labels ({len(labels_by_url)} unique URLs)")

    # Load articles
    articles_path = Path(cfg["paths"]["articles"])
    logger.info(f"Loading articles from {articles_path}")
    articles = load_articles(articles_path)
    logger.info(f"Loaded {len(articles)} articles")

    # Join and format
    examples: list[dict] = []
    no_article = 0
    no_text = 0

    for url, label in labels_by_url.items():
        article = articles.get(url)
        if article is None:
            no_article += 1
            continue

        ex = build_example(
            label=label,
            article=article,
            instruction=instruction,
            max_chars=max_chars,
            direction_map=direction_map,
            confidence_map=confidence_map,
        )
        if ex is None:
            no_text += 1
            continue

        examples.append(ex)

    joined = len(examples)
    if no_article > 0:
        logger.warning(f"{no_article} labels had no matching article in CSV")
    if no_text > 0:
        logger.warning(f"{no_text} articles had empty body text")
    logger.info(f"Formatted {joined} training examples")

    # Split
    train, val, test, entity_holdout = split_examples(
        examples, train_end, val_end, holdout_entities
    )

    # Write output files
    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(val, output_dir / "val.jsonl")
    write_jsonl(test, output_dir / "test.jsonl")
    write_jsonl(entity_holdout, output_dir / "entity_holdout.jsonl")

    # Print summary
    print_summary(
        train, val, test, entity_holdout,
        total_labels=len(labels),
        total_articles=len(articles),
        joined=joined,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format labeled articles into training JSONL splits"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training_config.yaml"),
        help="Path to training config YAML",
    )
    args = parser.parse_args()
    main(config_path=args.config)
