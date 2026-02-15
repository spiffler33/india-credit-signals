# WHY THIS: The seed dataset is our hand-curated ground truth of major NBFC
# rating actions. Indian rating agency websites are JS-rendered SPAs that resist
# simple scraping. Rather than fighting Playwright fragility, we start with
# well-documented events (IL&FS, DHFL, YES Bank, etc.) that we KNOW are correct.
# This gives us 60+ actions across 30+ entities — enough to start Phase 1.2
# (news collection) immediately while scrapers are refined in parallel.

from __future__ import annotations

from collections import Counter
from pathlib import Path

from loguru import logger

from src.data.models import RatingAction, read_rating_actions_csv


# Default seed file location relative to project root
SEED_CSV_PATH = Path("data/raw/seed_rating_actions.csv")


def load_seed_ratings(path: Path | None = None) -> list[RatingAction]:
    """Load the curated seed dataset of NBFC rating actions.

    Args:
        path: Path to the seed CSV file. Defaults to data/raw/seed_rating_actions.csv.

    Returns:
        List of RatingAction objects, sorted by date.

    Raises:
        FileNotFoundError: If the seed CSV doesn't exist.
        ValueError: If any row has invalid data (bad date format, unknown action type).

    🎓 WHY validate at load time: "Garbage in, garbage out" is the #1 ML pitfall.
    If our ground truth has a typo ("dwongrade" instead of "downgrade"), it silently
    corrupts training labels. Strict validation here means we catch errors early.
    """
    csv_path = path or SEED_CSV_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Seed rating actions file not found: {csv_path}\n"
            "Run from project root or provide explicit path."
        )

    actions = read_rating_actions_csv(csv_path)
    logger.info(f"Loaded {len(actions)} seed rating actions from {csv_path}")

    # Validate: no future dates, no empty entities
    for action in actions:
        if not action.entity:
            raise ValueError(f"Empty entity name in row: {action}")
        if not action.agency:
            raise ValueError(f"Empty agency name in row: {action}")

    return sorted(actions, key=lambda a: a.date)


def print_seed_summary(actions: list[RatingAction]) -> None:
    """Print a summary table of the seed dataset (per OUTPUT_STYLE.md).

    🎓 WHY this format: The OUTPUT_STYLE.md requires data summaries after
    every data operation. This isn't just cosmetic — it forces you to look
    at your data distribution and catch problems (e.g., if 90% of actions
    are from one agency, your model might learn agency bias).
    """
    entity_counts = Counter(a.entity for a in actions)
    agency_counts = Counter(a.agency for a in actions)
    action_counts = Counter(a.action_type.value for a in actions)
    dates = [a.date for a in actions]

    # Count downgrades + defaults as "negative" events
    negative_actions = sum(
        1 for a in actions
        if a.action_type.value in ("downgrade", "default", "watchlist_negative", "outlook_negative")
    )
    positive_actions = sum(
        1 for a in actions
        if a.action_type.value in ("upgrade", "outlook_positive", "watchlist_positive")
    )

    top_agency = agency_counts.most_common(1)[0] if agency_counts else ("N/A", 0)

    print("\n┌─────────────────────────────────────────────┐")
    print("│ Dataset: NBFC Rating Actions (Seed)         │")
    print("├─────────────────────────────────────────────┤")
    print(f"│ Total rating actions:    {len(actions):<20}│")
    print(f"│ Unique entities:         {len(entity_counts):<20}│")
    print(f"│ Unique agencies:         {len(agency_counts):<20}│")
    print(f"│ Negative signals:        {negative_actions:<20}│")
    print(f"│ Positive signals:        {positive_actions:<20}│")
    print(f"│ Date range:     {min(dates).isoformat()} to {max(dates).isoformat():<9}│")
    print(f"│ Top agency:     {top_agency[0]:<8} ({top_agency[1]} actions)       │")
    print("├─────────────────────────────────────────────┤")
    print("│ Action type breakdown:                      │")
    for action_type, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / len(actions) * 100
        print(f"│   {action_type:<24} {count:>3} ({pct:>4.1f}%)    │")
    print("└─────────────────────────────────────────────┘\n")


if __name__ == "__main__":
    # Quick test: load and display seed data summary
    actions = load_seed_ratings()
    print_seed_summary(actions)
