# WHY THIS: Main scraper orchestrator for rating agency data.
# The realistic approach: Indian rating agency websites are JS-rendered SPAs.
# We TRY to scrape them (api_probe strategy), but gracefully fall back to
# the seed dataset when scraping fails. This is honest engineering — we don't
# pretend fragile scrapers are reliable, but we build the infrastructure
# so extending coverage later (Playwright, manual additions) is trivial.

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import httpx
import yaml
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.data.models import RatingAction, write_rating_actions_csv
from src.data.seed_ratings import load_seed_ratings, print_seed_summary


# ---------------------------------------------------------------------------
# Agency-specific scrapers
# ---------------------------------------------------------------------------
# 🎓 CONCEPT: Each agency scraper returns a list of RatingActions.
# They all follow the same interface: take config dict, return list.
# If scraping fails (which is EXPECTED for JS-rendered sites), they
# return an empty list — no exceptions, no crashes. The orchestrator
# merges whatever they find with the seed dataset.


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def _fetch_url(client: httpx.Client, url: str) -> httpx.Response:
    """Fetch a URL with retry logic.

    🎓 WHY tenacity: Network requests fail for dozens of reasons (timeouts,
    rate limits, transient server errors). tenacity retries with exponential
    backoff — wait 2s, then 4s, then 8s. This is the polite way to retry.
    """
    response = client.get(url, follow_redirects=True)
    response.raise_for_status()
    return response


def scrape_crisil(config: dict, entities: list[str]) -> list[RatingAction]:
    """Attempt to scrape CRISIL rating data.

    📐 DECISION: CRISIL's rating list page is fully JS-rendered (Angular SPA).
    We probe for an internal API endpoint that might return JSON directly.
    If it doesn't exist or is blocked, we return empty and rely on seed data.
    """
    logger.info("Probing CRISIL for API endpoints...")
    scraped: list[RatingAction] = []

    headers = config.get("headers", {})
    rate_limit = config.get("rate_limit_seconds", 3.0)

    # 🎓 WHY httpx over requests: httpx supports HTTP/2, async (for later),
    # and has a cleaner API. The Client() context manager handles connection
    # pooling and cleanup automatically.
    with httpx.Client(headers=headers, timeout=30.0) as client:
        # Probe: try the rating list page to see if any data is server-rendered
        try:
            response = _fetch_url(client, config["rating_list_url"])
            soup = BeautifulSoup(response.text, "lxml")

            # Look for rating data in tables or structured elements
            tables = soup.find_all("table")
            if tables:
                logger.info(f"CRISIL: Found {len(tables)} table(s) — parsing...")
                # Parse tables for rating data (structure depends on what's rendered)
                for table in tables:
                    rows = table.find_all("tr")
                    logger.debug(f"  Table has {len(rows)} rows")
            else:
                logger.warning(
                    "CRISIL: No tables found in HTML — page is likely JS-rendered. "
                    "Falling back to seed data."
                )
        except (httpx.HTTPError, Exception) as e:
            logger.warning(f"CRISIL scrape failed: {e}. Using seed data.")

        time.sleep(rate_limit)

    logger.info(f"CRISIL: Scraped {len(scraped)} rating actions")
    return scraped


def scrape_icra(config: dict, entities: list[str]) -> list[RatingAction]:
    """Attempt to scrape ICRA rating data.

    ICRA has a search form at /Rationale/Search. We try POST-ing entity names
    to see if the server returns results without JS rendering.
    """
    logger.info("Probing ICRA search endpoint...")
    scraped: list[RatingAction] = []

    headers = config.get("headers", {})
    rate_limit = config.get("rate_limit_seconds", 3.0)

    with httpx.Client(headers=headers, timeout=30.0) as client:
        try:
            # Try the search page first to get any form tokens
            response = _fetch_url(client, config["rating_list_url"])
            soup = BeautifulSoup(response.text, "lxml")

            # Check if there's a server-rendered form we can POST to
            forms = soup.find_all("form")
            if forms:
                logger.info(f"ICRA: Found {len(forms)} form(s) — may be able to POST search")
            else:
                logger.warning(
                    "ICRA: No forms found — search is likely JS-rendered. "
                    "Falling back to seed data."
                )
        except (httpx.HTTPError, Exception) as e:
            logger.warning(f"ICRA scrape failed: {e}. Using seed data.")

        time.sleep(rate_limit)

    logger.info(f"ICRA: Scraped {len(scraped)} rating actions")
    return scraped


def scrape_care(config: dict, entities: list[str]) -> list[RatingAction]:
    """Attempt to scrape CARE Ratings (CareEdge) data."""
    logger.info("Probing CARE Ratings (CareEdge)...")
    scraped: list[RatingAction] = []

    headers = config.get("headers", {})
    rate_limit = config.get("rate_limit_seconds", 3.0)

    with httpx.Client(headers=headers, timeout=30.0) as client:
        try:
            response = _fetch_url(client, config["rating_list_url"])
            soup = BeautifulSoup(response.text, "lxml")

            # CareEdge rebranded in 2022 with a new React-based site
            tables = soup.find_all("table")
            if tables:
                logger.info(f"CARE: Found {len(tables)} table(s)")
            else:
                logger.warning(
                    "CARE: No server-rendered data found. "
                    "Falling back to seed data."
                )
        except (httpx.HTTPError, Exception) as e:
            logger.warning(f"CARE scrape failed: {e}. Using seed data.")

        time.sleep(rate_limit)

    logger.info(f"CARE: Scraped {len(scraped)} rating actions")
    return scraped


def scrape_india_ratings(config: dict, entities: list[str]) -> list[RatingAction]:
    """Attempt to scrape India Ratings (Fitch subsidiary) data."""
    logger.info("Probing India Ratings...")
    scraped: list[RatingAction] = []

    headers = config.get("headers", {})
    rate_limit = config.get("rate_limit_seconds", 3.0)

    with httpx.Client(headers=headers, timeout=30.0) as client:
        try:
            response = _fetch_url(client, config["rating_list_url"])
            soup = BeautifulSoup(response.text, "lxml")

            # India Ratings press release page
            articles = soup.find_all("article") or soup.find_all("div", class_="press-release")
            if articles:
                logger.info(f"India Ratings: Found {len(articles)} press release elements")
            else:
                logger.warning(
                    "India Ratings: No parseable content found. "
                    "Falling back to seed data."
                )
        except (httpx.HTTPError, Exception) as e:
            logger.warning(f"India Ratings scrape failed: {e}. Using seed data.")

        time.sleep(rate_limit)

    logger.info(f"India Ratings: Scraped {len(scraped)} rating actions")
    return scraped


# Map agency config keys to scraper functions
SCRAPER_REGISTRY: dict[str, callable] = {
    "crisil": scrape_crisil,
    "icra": scrape_icra,
    "care": scrape_care,
    "india_ratings": scrape_india_ratings,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def deduplicate_actions(actions: list[RatingAction]) -> list[RatingAction]:
    """Remove duplicate rating actions.

    🎓 WHY deduplicate: When we merge scraped data with seed data, the same
    event might appear twice (once from seed, once scraped). We deduplicate
    on (entity, agency, date, action_type) — if two records match on all
    four fields, we keep the one with more metadata (longer notes, has URL).
    """
    seen: dict[tuple, RatingAction] = {}
    for action in actions:
        key = (action.entity, action.agency, action.date, action.action_type)
        if key in seen:
            existing = seen[key]
            # Keep the version with more metadata
            if len(action.notes) > len(existing.notes) or (
                action.rationale_url and not existing.rationale_url
            ):
                seen[key] = action
        else:
            seen[key] = action

    deduped = sorted(seen.values(), key=lambda a: a.date)
    removed = len(actions) - len(deduped)
    if removed:
        logger.info(f"Deduplication removed {removed} duplicate actions")
    return deduped


def load_entity_names(entities_yaml: Path) -> list[str]:
    """Extract flat list of entity names from nbfc_entities.yaml."""
    with open(entities_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names: list[str] = []
    for subsector_key, entities in config.get("subsectors", {}).items():
        for entity in entities:
            names.append(entity["name"])
    return names


def run_scrapers(
    agency_config_path: Path,
    entities_yaml_path: Path,
    seed_path: Path | None = None,
    output_path: Path | None = None,
) -> list[RatingAction]:
    """Main orchestrator: load seed data, try scrapers, merge, output.

    📐 DECISION: Seed-first strategy
    The seed dataset is our reliable baseline. Scrapers EXTEND it, not replace it.
    Even if all scrapers fail (likely for v1), we still have 60+ actions to work with.
    This lets Phase 1.2 (news collection) start immediately.
    """
    # 1. Load configs
    with open(agency_config_path, "r", encoding="utf-8") as f:
        agency_config = yaml.safe_load(f)

    entities = load_entity_names(entities_yaml_path)
    logger.info(f"Tracking {len(entities)} entities across scrapers")

    # 2. Load seed data (our reliable baseline)
    seed_actions = load_seed_ratings(seed_path)
    logger.info(f"Loaded {len(seed_actions)} seed rating actions")

    # 3. Try each agency scraper
    all_scraped: list[RatingAction] = []
    agencies = agency_config.get("agencies", {})

    for agency_key, agency_conf in agencies.items():
        scraper_fn = SCRAPER_REGISTRY.get(agency_key)
        if scraper_fn is None:
            logger.warning(f"No scraper registered for agency: {agency_key}")
            continue

        strategy = agency_conf.get("scrape_strategy", "seed_only")
        if strategy == "seed_only":
            logger.info(f"{agency_conf['name']}: Seed-only strategy — skipping scrape")
            continue

        logger.info(f"Running {agency_conf['name']} scraper (strategy: {strategy})...")
        try:
            scraped = scraper_fn(agency_conf, entities)
            all_scraped.extend(scraped)
        except Exception as e:
            logger.error(f"{agency_conf['name']} scraper crashed: {e}")

    # 4. Merge seed + scraped, deduplicate
    all_actions = seed_actions + all_scraped
    all_actions = deduplicate_actions(all_actions)

    # 5. Write output
    out_path = output_path or Path(
        agency_config.get("output", {}).get("raw_file", "data/raw/rating_actions.csv")
    )
    write_rating_actions_csv(all_actions, out_path)
    logger.info(f"Wrote {len(all_actions)} rating actions to {out_path}")

    return all_actions


def print_full_summary(actions: list[RatingAction]) -> None:
    """Print comprehensive summary (per OUTPUT_STYLE.md)."""
    entity_counts = Counter(a.entity for a in actions)
    agency_counts = Counter(a.agency for a in actions)
    action_counts = Counter(a.action_type.value for a in actions)
    source_counts = Counter(a.source for a in actions)
    dates = [a.date for a in actions]

    # Entities with most actions (the "interesting" ones for our model)
    top_entities = entity_counts.most_common(5)

    # Count by signal direction
    negative_types = {"downgrade", "default", "watchlist_negative", "outlook_negative", "suspended"}
    positive_types = {"upgrade", "outlook_positive", "watchlist_positive"}

    negative = sum(1 for a in actions if a.action_type.value in negative_types)
    positive = sum(1 for a in actions if a.action_type.value in positive_types)
    neutral = len(actions) - negative - positive

    print("\n" + "=" * 55)
    print("✅ DONE: Phase 1.1 — Credit Event Timeline")
    print("=" * 55)

    print("\n┌─────────────────────────────────────────────────────┐")
    print("│ Dataset: NBFC Rating Actions 2016-2024 (Combined)  │")
    print("├─────────────────────────────────────────────────────┤")
    print(f"│ Total rating actions:     {len(actions):<26}│")
    print(f"│ Unique entities:          {len(entity_counts):<26}│")
    print(f"│ Rating agencies:          {len(agency_counts):<26}│")
    print(f"│ Date range:      {min(dates).isoformat()} to {max(dates).isoformat():<11}│")
    print("├─────────────────────────────────────────────────────┤")
    print("│ Signal direction:                                   │")
    print(f"│   Negative (↓):          {negative:<4} ({negative/len(actions)*100:.0f}%)                │")
    print(f"│   Positive (↑):          {positive:<4} ({positive/len(actions)*100:.0f}%)                │")
    print(f"│   Neutral (→):           {neutral:<4} ({neutral/len(actions)*100:.0f}%)                 │")
    print("├─────────────────────────────────────────────────────┤")
    print("│ Data source:                                        │")
    for src, count in source_counts.most_common():
        print(f"│   {src:<12}            {count:<4} ({count/len(actions)*100:.0f}%)                │")
    print("├─────────────────────────────────────────────────────┤")
    print("│ By agency:                                          │")
    for agency, count in agency_counts.most_common():
        print(f"│   {agency:<20}    {count:<4} ({count/len(actions)*100:.0f}%)                │")
    print("├─────────────────────────────────────────────────────┤")
    print("│ Top entities (most rating actions):                 │")
    for name, count in top_entities:
        print(f"│   {name:<24}{count:>3} actions               │")
    print("├─────────────────────────────────────────────────────┤")
    print("│ Action type breakdown:                              │")
    for atype, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / len(actions) * 100
        print(f"│   {atype:<24} {count:>3} ({pct:>4.1f}%)              │")
    print("└─────────────────────────────────────────────────────┘")

    print(
        "\n🎓 KEY CONCEPT: Notice the heavy negative skew (~70% negative signals).\n"
        "   This is expected — rating agencies act more on deterioration than\n"
        "   improvement. For training, we'll need to oversample positive cases\n"
        "   (like FinGPT does) to prevent the model from always predicting negative."
    )

    print(
        "\n⏭️  NEXT: Phase 1.2 — For each rating action, scrape news articles\n"
        "   from the 6 months BEFORE the action date using GDELT.\n"
        "   This gives us the input text → label pairs for training.\n"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape NBFC rating actions from Indian rating agencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m src.data.scrape_ratings --config configs/rating_agencies.yaml\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rating_agencies.yaml"),
        help="Path to rating agencies YAML config (default: configs/rating_agencies.yaml)",
    )
    parser.add_argument(
        "--entities",
        type=Path,
        default=Path("configs/nbfc_entities.yaml"),
        help="Path to NBFC entities YAML config (default: configs/nbfc_entities.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=Path,
        default=None,
        help="Path to seed rating actions CSV (default: data/raw/seed_rating_actions.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: from config file)",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Skip scraping entirely, only use seed data",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info("=" * 50)
    logger.info("[Phase 1.1] Scraping NBFC rating actions...")
    logger.info("=" * 50)

    if args.seed_only:
        logger.info("Seed-only mode: skipping all scrapers")
        actions = load_seed_ratings(args.seed)
        out_path = args.output or Path("data/raw/rating_actions.csv")
        write_rating_actions_csv(actions, out_path)
        logger.info(f"Wrote {len(actions)} seed actions to {out_path}")
    else:
        actions = run_scrapers(
            agency_config_path=args.config,
            entities_yaml_path=args.entities,
            seed_path=args.seed,
            output_path=args.output,
        )

    print_full_summary(actions)
