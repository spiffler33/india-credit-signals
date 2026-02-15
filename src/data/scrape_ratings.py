# WHY THIS: Thin wrapper that loads the sourced rating actions dataset and
# outputs summary stats. The actual scraping was done in a separate project
# (data_scraping/) using CRISIL/ICRA SEBI disclosure scrapers + manual
# curation for CARE/India Ratings. This script is kept as the CLI entry point
# for the data pipeline — it loads, filters, validates, and reports.
#
# 📐 DECISION: Scraper code lives in separate data_scraping/ project
#   PRO: Clean separation — scraping infra != training pipeline
#   PRO: Scraping can be re-run independently when new data arrives
#   CON: Two repos to maintain
#   RISK: None — the CSV is the contract between the two projects

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from src.data.models import write_rating_actions_csv
from src.data.seed_ratings import (
    SOURCED_CSV_PATH,
    load_rating_actions,
    load_training_window,
    print_dataset_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load, filter, and summarize NBFC rating actions dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.data.scrape_ratings                    # Training window (2016-2024)\n"
            "  python -m src.data.scrape_ratings --all              # All records\n"
            "  python -m src.data.scrape_ratings --output out.csv   # Export filtered data\n"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"Input CSV path (default: {SOURCED_CSV_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Export filtered dataset to this CSV path",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include all records, not just the 2016-2024 training window",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info("=" * 55)
    logger.info("[Phase 1.1] Loading NBFC rating actions dataset...")
    logger.info("=" * 55)

    if args.all:
        actions = load_rating_actions(path=args.input)
    else:
        actions = load_training_window(path=args.input)

    print_dataset_summary(actions)

    if args.output:
        write_rating_actions_csv(actions, args.output)
        logger.info(f"Exported {len(actions)} actions to {args.output}")

    print(
        "\n⏭️  NEXT: Phase 1.2 — For each rating action, scrape news articles\n"
        "   from the 6 months BEFORE the action date using GDELT.\n"
        "   This gives us the input text → label pairs for training.\n"
    )
