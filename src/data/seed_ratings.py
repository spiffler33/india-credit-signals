# WHY THIS: Loads the verified rating actions dataset sourced from SEBI
# disclosures (CRISIL/ICRA automated scrapers + CARE/India Ratings manual
# curation). This replaced the original hand-written seed data with 1,654
# verified records covering 39 entities and 6 agencies.
# The sourced data lives at data/raw/rating_actions_sourced.csv and is the
# single source of truth for ground truth labels in the training pipeline.

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

from loguru import logger

from src.data.models import RatingAction, read_rating_actions_csv


# Primary sourced dataset (from data_scraping project)
SOURCED_CSV_PATH = Path("data/raw/rating_actions_sourced.csv")

# Training window — actions outside this range are kept but flagged
TRAIN_DATE_START = date(2016, 1, 1)
TRAIN_DATE_END = date(2024, 12, 31)


def load_rating_actions(
    path: Path | None = None,
    date_start: date | None = None,
    date_end: date | None = None,
) -> list[RatingAction]:
    """Load the verified NBFC rating actions dataset.

    Args:
        path: Path to the CSV. Defaults to data/raw/rating_actions_sourced.csv.
        date_start: Filter to actions on or after this date. None = no lower bound.
        date_end: Filter to actions on or before this date. None = no upper bound.

    Returns:
        List of RatingAction objects, sorted by date.

    🎓 WHY filter by date: The sourced dataset goes back to 2008 for some
    entities. But our training window is 2016-2024 because (a) GDELT coverage
    of Indian media improves dramatically after 2015, and (b) the NBFC
    regulatory regime changed significantly with RBI's 2018 framework.
    Pre-2016 data has different dynamics.
    """
    csv_path = path or SOURCED_CSV_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Rating actions file not found: {csv_path}\n"
            "Run from project root or provide explicit path."
        )

    actions = read_rating_actions_csv(csv_path)
    logger.info(f"Loaded {len(actions)} total rating actions from {csv_path}")

    # Validate: no empty entities or agencies
    for action in actions:
        if not action.entity:
            raise ValueError(f"Empty entity name in row: {action}")
        if not action.agency:
            raise ValueError(f"Empty agency name in row: {action}")

    # Apply date filter
    if date_start or date_end:
        start = date_start or date.min
        end = date_end or date.max
        filtered = [a for a in actions if start <= a.date <= end]
        logger.info(
            f"Filtered to {len(filtered)} actions in "
            f"{start.isoformat()} to {end.isoformat()} window "
            f"(dropped {len(actions) - len(filtered)} outside range)"
        )
        actions = filtered

    return sorted(actions, key=lambda a: a.date)


def load_training_window(path: Path | None = None) -> list[RatingAction]:
    """Convenience: load only the 2016-2024 training window."""
    return load_rating_actions(
        path=path,
        date_start=TRAIN_DATE_START,
        date_end=TRAIN_DATE_END,
    )


def print_dataset_summary(actions: list[RatingAction]) -> None:
    """Print a summary table of the dataset (per OUTPUT_STYLE.md).

    🎓 WHY this format: The OUTPUT_STYLE.md requires data summaries after
    every data operation. This isn't just cosmetic — it forces you to look
    at your data distribution and catch problems (e.g., if 90% of actions
    are from one agency, your model might learn agency bias).
    """
    entity_counts = Counter(a.entity for a in actions)
    agency_counts = Counter(a.agency for a in actions)
    action_counts = Counter(a.action_type.value for a in actions)
    dates = [a.date for a in actions]

    # Signal direction grouping
    negative_types = {"downgrade", "default", "watchlist_negative", "outlook_negative", "suspended"}
    positive_types = {"upgrade", "outlook_positive", "watchlist_positive"}
    neutral_types = {"affirmed", "withdrawn", "initial"}

    negative = sum(1 for a in actions if a.action_type.value in negative_types)
    positive = sum(1 for a in actions if a.action_type.value in positive_types)
    neutral = len(actions) - negative - positive

    # Stressed entities (defaulted)
    stressed = [a for a in actions if a.action_type.value == "default"]
    stressed_entities = set(a.entity for a in stressed)

    top_entities = entity_counts.most_common(5)

    print("\n┌───────────────────────────────────────────────────────────┐")
    print("│ Dataset: NBFC Rating Actions (Sourced from SEBI/CRAs)   │")
    print("├───────────────────────────────────────────────────────────┤")
    print(f"│ Total rating actions:       {len(actions):<30}│")
    print(f"│ Unique entities:            {len(entity_counts):<30}│")
    print(f"│ Rating agencies:            {len(agency_counts):<30}│")
    print(f"│ Date range:        {min(dates).isoformat()} to {max(dates).isoformat():<13}│")
    print("├───────────────────────────────────────────────────────────┤")
    print("│ Signal direction:                                        │")
    print(f"│   Negative (↓ downgrade/default/watch-/outlook-):  {negative:>4}  │")
    print(f"│   Positive (↑ upgrade/outlook+/watch+):            {positive:>4}  │")
    print(f"│   Neutral  (→ affirmed/initial/withdrawn):         {neutral:>4}  │")
    print("├───────────────────────────────────────────────────────────┤")
    print(f"│ Stressed entities (rated D):  {len(stressed_entities):<28}│")
    for ent in sorted(stressed_entities):
        d_count = sum(1 for a in stressed if a.entity == ent)
        print(f"│   {ent:<36}{d_count:>3} default actions  │")
    print("├───────────────────────────────────────────────────────────┤")
    print("│ By agency:                                               │")
    for agency, count in agency_counts.most_common():
        pct = count / len(actions) * 100
        print(f"│   {agency:<20}  {count:>5}  ({pct:>5.1f}%)               │")
    print("├───────────────────────────────────────────────────────────┤")
    print("│ Top entities (most rating actions):                      │")
    for name, count in top_entities:
        print(f"│   {name:<36}{count:>4} actions       │")
    print("├───────────────────────────────────────────────────────────┤")
    print("│ Action type breakdown:                                   │")
    for atype, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / len(actions) * 100
        print(f"│   {atype:<24}  {count:>5}  ({pct:>5.1f}%)               │")
    print("└───────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and summarize NBFC rating actions")
    parser.add_argument("--all", action="store_true", help="Show all records, not just 2016-2024")
    parser.add_argument("--path", type=Path, default=None, help="Path to CSV file")
    args = parser.parse_args()

    if args.all:
        actions = load_rating_actions(path=args.path)
    else:
        actions = load_training_window(path=args.path)

    print_dataset_summary(actions)
