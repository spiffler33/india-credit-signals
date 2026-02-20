# WHY THIS: Centralized data loading with Streamlit caching. Every dashboard view
# needs the same 6 data files. Loading parquet into memory once and caching with
# @st.cache_data means the second page load is instant (<100ms) instead of re-reading
# from disk. Without this, each view would independently load data = 6x I/O.
#
# Why @st.cache_data and not @st.cache_resource? cache_data is for serializable
# data (DataFrames, dicts). cache_resource is for non-serializable things like
# database connections or ML models. DataFrames are serializable, so cache_data
# gives us automatic invalidation on input changes (e.g., if the parquet file
# is regenerated).

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Default path — relative to repo root (where `streamlit run` is invoked from)
DEFAULT_DATA_DIR = Path("data/dashboard")


def _resolve_data_dir() -> Path:
    """Resolve data directory, checking both relative and absolute paths."""
    data_dir = DEFAULT_DATA_DIR
    if not data_dir.exists():
        # Try absolute path from project root
        project_root = Path(__file__).parent.parent.parent.parent
        data_dir = project_root / "data" / "dashboard"
    return data_dir


def _check_file(path: Path, name: str) -> None:
    """Raise a clear error if a required data file is missing."""
    if not path.exists():
        st.error(
            f"Missing data file: `{path}`\n\n"
            f"Run the export pipeline first:\n"
            f"```\npython -m src.signals.export_dashboard_data\n```"
        )
        st.stop()


# ============================================================
# Parquet Loaders
# ============================================================

@st.cache_data(ttl=3600)  # 🎓 Re-read from disk at most once per hour
def load_entity_scores(data_dir: Path | None = None) -> pd.DataFrame:
    """Load pre-computed entity-day scores (direct + contagion + rolling).

    This is the largest file (~82K rows, 2MB). Powers Entity Timeline,
    Sector Heatmap, and Alert Dashboard.
    """
    data_dir = data_dir or _resolve_data_dir()
    path = data_dir / "entity_scores.parquet"
    _check_file(path, "entity_scores")

    df = pd.read_parquet(path)

    # 🎓 Filter out multi-entity rows (e.g., "Aavas Financiers,Cholamandalam").
    # These are articles mentioning multiple entities — only 0.5% of rows.
    # Each entity already has its own single-entity row from the scoring pipeline,
    # so the compound rows would double-count scores in timeline/heatmap views.
    df = df[~df["entity"].str.contains(",", na=False)].copy()

    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600)
def load_signals(data_dir: Path | None = None) -> pd.DataFrame:
    """Load per-article signals for the Signal Feed view.

    Each row is one article with its model-extracted credit signal:
    direction, signal_type, confidence, sector_wide, etc.
    """
    data_dir = data_dir or _resolve_data_dir()
    path = data_dir / "signals.parquet"
    _check_file(path, "signals")

    df = pd.read_parquet(path)
    df = df[~df["entity"].str.contains(",", na=False)].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600)
def load_contagion_edges(data_dir: Path | None = None) -> pd.DataFrame:
    """Load contagion graph edges for the Network Graph view.

    Each row is an entity pair with edge weight and subsector info.
    """
    data_dir = data_dir or _resolve_data_dir()
    path = data_dir / "contagion_edges.parquet"
    _check_file(path, "contagion_edges")
    return pd.read_parquet(path)


@st.cache_data(ttl=3600)
def load_rating_actions(data_dir: Path | None = None) -> pd.DataFrame:
    """Load rating actions for timeline overlays.

    Vertical red/green lines on Entity Timeline. Each row is one
    downgrade/upgrade/default action by a rating agency.
    """
    data_dir = data_dir or _resolve_data_dir()
    path = data_dir / "rating_actions.parquet"
    _check_file(path, "rating_actions")

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ============================================================
# JSON Loaders
# ============================================================

@st.cache_data(ttl=3600)
def load_entity_metadata(data_dir: Path | None = None) -> dict[str, Any]:
    """Load entity metadata (names, subsectors, peer counts).

    Small file (44 entities). Used for dropdown labels, subsector
    grouping, and color coding across all views.
    """
    data_dir = data_dir or _resolve_data_dir()
    path = data_dir / "entity_metadata.json"
    _check_file(path, "entity_metadata")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_crisis_results(data_dir: Path | None = None) -> list[dict]:
    """Load crisis replay results for the Alert Dashboard context.

    Contains lead times, improvements, and control breach rates
    from the two crisis replays (IL&FS/DHFL 2018, SREI/RelCap 2019).
    """
    data_dir = data_dir or _resolve_data_dir()
    path = data_dir / "crisis_results.json"
    _check_file(path, "crisis_results")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Derived Data Helpers
# ============================================================

def get_entity_list(metadata: dict[str, Any]) -> list[str]:
    """Get sorted list of entity names from metadata."""
    return sorted(e["name"] for e in metadata["entities"])


def get_subsector_map(metadata: dict[str, Any]) -> dict[str, str]:
    """Get entity → subsector mapping from metadata."""
    return {e["name"]: e["subsector"] for e in metadata["entities"]}


def get_entities_by_subsector(metadata: dict[str, Any]) -> dict[str, list[str]]:
    """Group entity names by subsector — for grouped dropdowns."""
    groups: dict[str, list[str]] = {}
    for entity in metadata["entities"]:
        subsector = entity["subsector"]
        if subsector not in groups:
            groups[subsector] = []
        groups[subsector].append(entity["name"])
    for names in groups.values():
        names.sort()
    return groups
