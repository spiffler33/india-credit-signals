# WHY THIS: Tests for the dashboard data export pipeline. Each export function
# is tested with synthetic data to verify: correct column schemas, proper
# subsector assignment, edge deduplication, rating action outcome mapping,
# entity metadata structure, and crisis result serialization.
#
# These tests do NOT require real data files — they build minimal DataFrames
# and EntityGraphs in-memory, following the same pattern as test_propagation.py.

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.signals.entity_graph import ContagionEdge, EntityGraph, EntityNode
from src.signals.export_dashboard_data import (
    _serialize_target_result,
    export_contagion_edges,
    export_entity_metadata,
    export_entity_scores,
    export_rating_actions,
    export_crisis_results,
)


# ============================================================
# Fixtures: Minimal graph + signals for testing
# ============================================================

@pytest.fixture
def mini_graph() -> EntityGraph:
    """Build a 4-entity graph: 2 housing_finance + 2 diversified_nbfc."""
    g = EntityGraph()

    g.add_node(EntityNode(
        name="DHFL", full_name="DHFL", subsector="housing_finance",
        status="defaulted", aliases=["Dewan Housing"],
    ))
    g.add_node(EntityNode(
        name="PNB Housing", full_name="PNB Housing Finance",
        subsector="housing_finance", status="active",
    ))
    g.add_node(EntityNode(
        name="Bajaj Finance", full_name="Bajaj Finance Limited",
        subsector="diversified_nbfc", status="active",
    ))
    g.add_node(EntityNode(
        name="Chola", full_name="Cholamandalam Investment",
        subsector="diversified_nbfc", status="active",
    ))

    # Intra-subsector edges (weight=0.8)
    g.add_edge(ContagionEdge("DHFL", "PNB Housing", 0.8, {"subsector": 0.8}))
    g.add_edge(ContagionEdge("Bajaj Finance", "Chola", 0.8, {"subsector": 0.8}))

    # Cross-subsector edges (weight=0.1)
    g.add_edge(ContagionEdge("DHFL", "Bajaj Finance", 0.1, {"subsector": 0.1}))
    g.add_edge(ContagionEdge("DHFL", "Chola", 0.1, {"subsector": 0.1}))
    g.add_edge(ContagionEdge("PNB Housing", "Bajaj Finance", 0.1, {"subsector": 0.1}))
    g.add_edge(ContagionEdge("PNB Housing", "Chola", 0.1, {"subsector": 0.1}))

    return g


@pytest.fixture
def sample_signals() -> pd.DataFrame:
    """Create minimal signals DataFrame for scoring tests."""
    return pd.DataFrame([
        {
            "entity": "DHFL", "date": pd.Timestamp("2018-09-01"),
            "credit_relevant": 1, "direction": "Deterioration",
            "signal_type": "asset_quality", "sector_wide": False,
            "confidence": "High", "signal_source": "model",
        },
        {
            "entity": "DHFL", "date": pd.Timestamp("2018-09-05"),
            "credit_relevant": 1, "direction": "Deterioration",
            "signal_type": "liquidity", "sector_wide": True,
            "confidence": "High", "signal_source": "model",
        },
        {
            "entity": "DHFL", "date": pd.Timestamp("2018-09-10"),
            "credit_relevant": 0, "direction": "Neutral",
            "signal_type": "other", "sector_wide": False,
            "confidence": "Low", "signal_source": "model",
        },
        {
            "entity": "Bajaj Finance", "date": pd.Timestamp("2018-09-15"),
            "credit_relevant": 1, "direction": "Improvement",
            "signal_type": "funding", "sector_wide": False,
            "confidence": "Medium", "signal_source": "label",
        },
    ])


@pytest.fixture
def default_config() -> dict:
    """Minimal contagion config for tests."""
    return {
        "direction_multiplier": {
            "Deterioration": 1.0, "Improvement": -0.5, "Neutral": 0.0,
        },
        "confidence_weights": {"High": 1.0, "Medium": 0.6, "Low": 0.3},
        "sector_wide_multiplier": 1.5,
        "contagion_window_days": 30,
        "peer_signal_discount": 0.5,
        "normalize_by_peers": True,
        "rolling_windows": [7, 30],
        "score_thresholds": {"warning": 4.0, "critical": 10.0},
    }


# ============================================================
# Tests: export_entity_scores
# ============================================================

class TestExportEntityScores:
    """Tests for the entity scores export function."""

    def test_returns_expected_columns(
        self, sample_signals: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """Scores DataFrame must have all required columns."""
        scores = export_entity_scores(sample_signals, mini_graph, default_config)

        required = {
            "entity", "date", "direct_score", "contagion_score",
            "total_score", "n_signals", "n_sources", "top_source",
            "subsector",
        }
        assert required.issubset(set(scores.columns))

    def test_subsector_assigned(
        self, sample_signals: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """Each entity should have a subsector from the graph."""
        scores = export_entity_scores(sample_signals, mini_graph, default_config)

        dhfl_rows = scores[scores["entity"] == "DHFL"]
        if len(dhfl_rows) > 0:
            assert dhfl_rows["subsector"].iloc[0] == "housing_finance"

        bajaj_rows = scores[scores["entity"] == "Bajaj Finance"]
        if len(bajaj_rows) > 0:
            assert bajaj_rows["subsector"].iloc[0] == "diversified_nbfc"

    def test_credit_irrelevant_excluded_from_direct(
        self, sample_signals: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """Articles with credit_relevant=0 should not contribute to direct scores."""
        scores = export_entity_scores(sample_signals, mini_graph, default_config)

        # DHFL has 2 credit-relevant signals on Sep 1 and Sep 5, plus 1 non-relevant on Sep 10
        # Sep 10 should NOT appear as a direct score day for DHFL
        dhfl_direct = scores[
            (scores["entity"] == "DHFL") & (scores["direct_score"] > 0)
        ]
        # Only 2 days should have direct scores (Sep 1 and Sep 5)
        assert len(dhfl_direct) == 2

    def test_contagion_propagates_to_peers(
        self, sample_signals: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """PNB Housing (same subsector as DHFL) should get contagion."""
        scores = export_entity_scores(sample_signals, mini_graph, default_config)

        pnb_rows = scores[scores["entity"] == "PNB Housing"]
        if len(pnb_rows) > 0:
            assert pnb_rows["contagion_score"].max() > 0, (
                "PNB Housing should receive contagion from DHFL"
            )

    def test_empty_signals_returns_empty(
        self, mini_graph: EntityGraph, default_config: dict,
    ) -> None:
        """Empty signals should return empty DataFrame (no crash)."""
        empty = pd.DataFrame(columns=[
            "entity", "date", "credit_relevant", "direction",
            "signal_type", "sector_wide", "confidence",
        ])
        scores = export_entity_scores(empty, mini_graph, default_config)
        assert len(scores) == 0


# ============================================================
# Tests: export_contagion_edges
# ============================================================

class TestExportContagionEdges:
    """Tests for the contagion edge export."""

    def test_returns_expected_columns(self, mini_graph: EntityGraph) -> None:
        """Edge DataFrame must have source, target, weight, subsector info."""
        edges = export_contagion_edges(mini_graph)

        required = {
            "source", "target", "weight", "source_subsector",
            "target_subsector", "same_subsector",
        }
        assert required.issubset(set(edges.columns))

    def test_no_duplicate_edges(self, mini_graph: EntityGraph) -> None:
        """Symmetric graph should export each edge only once."""
        edges = export_contagion_edges(mini_graph)

        # Create sorted pair for dedup check
        pairs = set()
        for _, row in edges.iterrows():
            pair = tuple(sorted([row["source"], row["target"]]))
            assert pair not in pairs, f"Duplicate edge: {pair}"
            pairs.add(pair)

    def test_edge_count(self, mini_graph: EntityGraph) -> None:
        """4-entity graph: 2 intra + 4 cross = 6 unique edges."""
        edges = export_contagion_edges(mini_graph)
        assert len(edges) == 6

    def test_intra_vs_cross_weights(self, mini_graph: EntityGraph) -> None:
        """Intra-subsector edges should have weight=0.8, cross=0.1."""
        edges = export_contagion_edges(mini_graph)

        intra = edges[edges["same_subsector"]]
        cross = edges[~edges["same_subsector"]]

        assert all(intra["weight"] == 0.8)
        assert all(cross["weight"] == 0.1)

    def test_same_subsector_flag(self, mini_graph: EntityGraph) -> None:
        """same_subsector should be True for intra, False for cross."""
        edges = export_contagion_edges(mini_graph)

        # DHFL-PNB Housing: both housing_finance → True
        dhfl_pnb = edges[
            ((edges["source"] == "DHFL") & (edges["target"] == "PNB Housing")) |
            ((edges["source"] == "PNB Housing") & (edges["target"] == "DHFL"))
        ]
        assert len(dhfl_pnb) == 1
        assert dhfl_pnb["same_subsector"].iloc[0] == True  # noqa: E712 — numpy bool


# ============================================================
# Tests: export_rating_actions
# ============================================================

class TestExportRatingActions:
    """Tests for rating actions export."""

    def test_outcome_mapping(self, tmp_path: Path) -> None:
        """Downgrade→negative, upgrade→positive, other→neutral."""
        csv = tmp_path / "ratings.csv"
        csv.write_text(
            "entity,date,agency,action_type,from_rating,to_rating\n"
            "DHFL,2019-02-03,CRISIL,downgrade,AA,A\n"
            "DHFL,2021-06-01,CARE,default,D,D\n"
            "Aavas,2020-03-15,ICRA,upgrade,A,AA\n"
            "Bajaj,2019-08-01,CRISIL,reaffirmation,AAA,AAA\n"
        )
        df = export_rating_actions(csv)

        assert df.loc[df["entity"] == "DHFL", "outcome"].tolist() == ["negative", "negative"]
        assert df.loc[df["entity"] == "Aavas", "outcome"].iloc[0] == "positive"
        assert df.loc[df["entity"] == "Bajaj", "outcome"].iloc[0] == "neutral"

    def test_date_parsing(self, tmp_path: Path) -> None:
        """Dates should be parsed as datetime."""
        csv = tmp_path / "ratings.csv"
        csv.write_text(
            "entity,date,agency,action_type,from_rating,to_rating\n"
            "DHFL,2019-02-03,CRISIL,downgrade,AA,A\n"
        )
        df = export_rating_actions(csv)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_invalid_dates_dropped(self, tmp_path: Path) -> None:
        """Rows with unparseable dates should be dropped."""
        csv = tmp_path / "ratings.csv"
        csv.write_text(
            "entity,date,agency,action_type,from_rating,to_rating\n"
            "DHFL,2019-02-03,CRISIL,downgrade,AA,A\n"
            "DHFL,not_a_date,CRISIL,downgrade,A,BBB\n"
        )
        df = export_rating_actions(csv)
        assert len(df) == 1


# ============================================================
# Tests: export_entity_metadata
# ============================================================

class TestExportEntityMetadata:
    """Tests for entity metadata export."""

    def test_structure(self, mini_graph: EntityGraph) -> None:
        """Metadata must have entities list, subsectors dict, counts."""
        meta = export_entity_metadata(mini_graph)

        assert "entities" in meta
        assert "subsectors" in meta
        assert "n_entities" in meta
        assert "generated_at" in meta
        assert meta["n_entities"] == 4

    def test_entity_fields(self, mini_graph: EntityGraph) -> None:
        """Each entity should have name, subsector, status, peer counts."""
        meta = export_entity_metadata(mini_graph)

        dhfl = next(e for e in meta["entities"] if e["name"] == "DHFL")
        assert dhfl["subsector"] == "housing_finance"
        assert dhfl["status"] == "defaulted"
        assert dhfl["n_peers"] == 3  # PNB Housing + Bajaj + Chola
        assert dhfl["n_intra_peers"] == 1  # PNB Housing
        assert dhfl["n_cross_peers"] == 2  # Bajaj + Chola

    def test_subsector_counts(self, mini_graph: EntityGraph) -> None:
        """Subsector summary should match node counts."""
        meta = export_entity_metadata(mini_graph)

        assert meta["subsectors"]["housing_finance"] == 2
        assert meta["subsectors"]["diversified_nbfc"] == 2


# ============================================================
# Tests: _serialize_target_result
# ============================================================

class TestSerializeTargetResult:
    """Tests for target result JSON serialization."""

    def test_timestamps_become_strings(self) -> None:
        """Pandas Timestamps should serialize to date strings."""
        target = {
            "entity": "PNB Housing",
            "is_control": False,
            "first_action": pd.Timestamp("2020-02-21"),
            "first_breach_date": pd.Timestamp("2018-11-15"),
            "lead_time_days": 463,
            "direct_only_lead_time": None,
            "lead_time_improvement": 463,
            "peak_score": 12.345,
            "peak_contagion": 8.765,
            "n_warning_breaches": 0,
        }
        s = _serialize_target_result(target)

        assert s["first_action"] == "2020-02-21"
        assert s["first_breach_date"] == "2018-11-15"
        assert s["peak_score"] == 12.35  # rounded to 2dp
        assert s["peak_contagion"] == 8.77

    def test_none_dates_stay_none(self) -> None:
        """None values should serialize as None (not 'None' string)."""
        target = {
            "entity": "Chola",
            "is_control": True,
            "first_action": None,
            "first_breach_date": None,
            "lead_time_days": None,
            "direct_only_lead_time": None,
            "lead_time_improvement": None,
            "peak_score": 3.5,
            "peak_contagion": 2.1,
            "n_warning_breaches": 15,
        }
        s = _serialize_target_result(target)

        assert s["first_action"] is None
        assert s["first_breach_date"] is None
        assert s["n_warning_breaches"] == 15

    def test_json_serializable(self) -> None:
        """Serialized result must be JSON-serializable (no Timestamps/NaT)."""
        target = {
            "entity": "Can Fin",
            "first_action": pd.Timestamp("2019-05-06"),
            "first_breach_date": pd.NaT,
            "lead_time_days": None,
            "direct_only_lead_time": None,
            "lead_time_improvement": None,
            "peak_score": 0.0,
            "peak_contagion": 0.0,
        }
        s = _serialize_target_result(target)

        # This should not raise
        json_str = json.dumps(s)
        assert isinstance(json_str, str)
