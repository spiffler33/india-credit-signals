# WHY THIS: Typed dataclasses give us a single source of truth for data shapes.
# Every scraper, loader, and pipeline step speaks the same "language."
# Using dataclasses (not dicts) means typos like action["ratnig"] blow up at
# creation time, not silently propagate through the pipeline.

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Self


# 🎓 CONCEPT: Enums for categorical fields
# Instead of passing around raw strings like "downgrade" that could be
# misspelled as "downgade", Enums restrict to valid values only.
# This catches errors at data load time, not 3 pipeline steps later.

class ActionType(str, Enum):
    """Types of rating actions an agency can take."""
    DOWNGRADE = "downgrade"
    UPGRADE = "upgrade"
    DEFAULT = "default"           # Rated D or equivalent
    WATCHLIST_NEG = "watchlist_negative"  # Placed on watch with negative implications
    WATCHLIST_POS = "watchlist_positive"
    OUTLOOK_NEG = "outlook_negative"     # Outlook revised to negative
    OUTLOOK_POS = "outlook_positive"
    AFFIRMED = "affirmed"         # Rating confirmed (no change)
    WITHDRAWN = "withdrawn"       # Rating withdrawn (entity stopped paying the agency)
    SUSPENDED = "suspended"       # Rating suspended by agency
    INITIAL = "initial"           # First-time rating assignment


class Subsector(str, Enum):
    """NBFC subsector classification for contagion grouping."""
    HOUSING_FINANCE = "housing_finance"
    INFRASTRUCTURE = "infrastructure_finance"
    DIVERSIFIED = "diversified_nbfc"
    MICROFINANCE = "microfinance"
    VEHICLE_FINANCE = "vehicle_finance"
    GOLD_LOAN = "gold_loan"
    SPECIAL = "special_situations"


@dataclass
class RatingAction:
    """A single credit rating action by an agency on an entity.

    This is the atomic unit of our ground truth dataset. Each row in
    rating_actions_sourced.csv becomes one of these. Data sourced from
    SEBI disclosures via CRISIL/ICRA scrapers + manual curation for
    CARE/India Ratings/Brickwork/Acuite.
    """
    entity: str                    # Short name (e.g., "DHFL")
    entity_full_name: str          # Legal name (e.g., "Dewan Housing Finance Corporation Limited")
    agency: str                    # Rating agency (e.g., "CRISIL", "ICRA", "CARE", etc.)
    date: date                     # Date of rating action
    action_type: ActionType        # What happened (downgrade, default, etc.)
    from_rating: str               # Previous rating in agency-native scale (e.g., "[ICRA] AA+")
    to_rating: str                 # New rating in agency-native scale (e.g., "[ICRA] D")
    instrument_type: str = ""      # What was rated (NCD, CP, bank facilities, etc.)
    rationale_url: str = ""        # URL to agency's rationale document
    notes: str = ""                # Free-text context / source provenance

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> Self:
        """Parse a CSV DictReader row into a RatingAction.

        🎓 WHY: CSV files store everything as strings. We need to convert
        date strings to date objects and action_type strings to ActionType enums.
        Doing it here (at the boundary) means the rest of our code works with
        proper types, not raw strings.
        """
        return cls(
            entity=row["entity"].strip(),
            entity_full_name=row.get("entity_full_name", "").strip(),
            agency=row["agency"].strip(),
            date=datetime.strptime(row["date"].strip(), "%Y-%m-%d").date(),
            action_type=ActionType(row["action_type"].strip().lower()),
            from_rating=row.get("from_rating", "").strip(),
            to_rating=row.get("to_rating", "").strip(),
            instrument_type=row.get("instrument_type", "").strip(),
            rationale_url=row.get("rationale_url", "").strip(),
            notes=row.get("notes", "").strip(),
        )

    def to_dict(self) -> dict[str, str]:
        """Serialize to a flat dict for CSV writing."""
        return {
            "entity": self.entity,
            "entity_full_name": self.entity_full_name,
            "agency": self.agency,
            "date": self.date.isoformat(),
            "action_type": self.action_type.value,
            "from_rating": self.from_rating,
            "to_rating": self.to_rating,
            "instrument_type": self.instrument_type,
            "rationale_url": self.rationale_url,
            "notes": self.notes,
        }


@dataclass
class Entity:
    """An NBFC entity we track for credit signals.

    Loaded from configs/nbfc_entities.yaml. The aliases list is critical
    for fuzzy-matching entity mentions in news articles — "Dewan Housing"
    and "DHFL" both refer to the same entity.
    """
    name: str                           # Short canonical name
    full_name: str                      # Legal name
    aliases: list[str] = field(default_factory=list)
    bse_code: str = ""
    nse_code: str = ""
    subsector: Subsector = Subsector.DIVERSIFIED
    status: str = "active"              # active, defaulted, merged

    @classmethod
    def from_yaml_entry(cls, entry: dict, subsector: str) -> Self:
        """Parse a YAML entity entry into an Entity object."""
        return cls(
            name=entry["name"],
            full_name=entry["full_name"],
            aliases=entry.get("aliases", []),
            bse_code=entry.get("bse_code", ""),
            nse_code=entry.get("nse_code", ""),
            subsector=Subsector(subsector),
            status=entry.get("status", "active"),
        )


# --- I/O Helpers ---

def write_rating_actions_csv(actions: list[RatingAction], path: Path) -> None:
    """Write a list of RatingActions to CSV.

    🎓 WHY csv.DictWriter: We could use pandas, but for simple tabular output
    the stdlib csv module is lighter and has zero import overhead. Pandas is
    overkill when you're just writing rows — save it for analysis.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "entity", "entity_full_name", "agency", "date", "action_type",
        "from_rating", "to_rating", "instrument_type", "rationale_url", "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for action in sorted(actions, key=lambda a: a.date):
            writer.writerow(action.to_dict())


def read_rating_actions_csv(path: Path) -> list[RatingAction]:
    """Read a CSV file into a list of RatingActions."""
    actions: list[RatingAction] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            actions.append(RatingAction.from_csv_row(row))
    return actions


def write_rating_actions_json(actions: list[RatingAction], path: Path) -> None:
    """Write a list of RatingActions to JSON (for downstream pipelines)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [a.to_dict() for a in sorted(actions, key=lambda a: a.date)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
