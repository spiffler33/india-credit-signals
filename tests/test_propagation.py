# WHY THIS: Unit tests for signal propagation and contagion scoring using synthetic
# data. Verifies the math: direction × confidence × sector_wide for direct scores,
# edge_weight × rolling_direct × peer_discount for contagion, and rolling windows.

from __future__ import annotations

import pytest
import pandas as pd
import yaml
from pathlib import Path

from src.signals.entity_graph import EntityGraph, EntityNode, ContagionEdge, load_entity_graph
from src.signals.propagation import (
    compute_all_scores,
    compute_contagion_scores,
    compute_direct_scores,
    compute_rolling_scores,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def default_config() -> dict:
    """Standard contagion config for testing."""
    return {
        "direction_multiplier": {"Deterioration": 1.0, "Improvement": -0.5, "Neutral": 0.0},
        "confidence_weights": {"High": 1.0, "Medium": 0.6, "Low": 0.3},
        "sector_wide_multiplier": 1.5,
        "contagion_window_days": 30,
        "peer_signal_discount": 0.5,
        "rolling_windows": [7, 30],
        "edge_weights": {"intra_subsector": 0.8, "cross_subsector": 0.1},
    }


@pytest.fixture
def mini_graph(tmp_path: Path) -> EntityGraph:
    """Small graph: 3 housing_finance + 1 diversified_nbfc."""
    data = {
        "subsectors": {
            "housing_finance": [
                {"name": "DHFL", "full_name": "DHFL", "aliases": ["DHFL"], "status": "defaulted"},
                {"name": "Indiabulls HF", "full_name": "Indiabulls HF", "aliases": ["Indiabulls HF"], "status": "active"},
                {"name": "PNB Housing", "full_name": "PNB Housing", "aliases": ["PNB Housing"], "status": "active"},
            ],
            "diversified_nbfc": [
                {"name": "Chola", "full_name": "Chola", "aliases": ["Chola"], "status": "active"},
            ],
        }
    }
    yaml_path = tmp_path / "entities.yaml"
    yaml_path.write_text(yaml.dump(data, default_flow_style=False))
    config = {"edge_weights": {"intra_subsector": 0.8, "cross_subsector": 0.1}}
    return load_entity_graph(yaml_path, config)


@pytest.fixture
def single_signal_df() -> pd.DataFrame:
    """One high-confidence deterioration signal for DHFL."""
    return pd.DataFrame([{
        "entity": "DHFL",
        "date": pd.Timestamp("2018-11-15"),
        "credit_relevant": 1,
        "direction": "Deterioration",
        "signal_type": "liquidity",
        "sector_wide": False,
        "confidence": "High",
    }])


@pytest.fixture
def multi_signal_df() -> pd.DataFrame:
    """Multiple signals across entities and days."""
    return pd.DataFrame([
        # DHFL: 3 deterioration signals in Nov 2018
        {"entity": "DHFL", "date": pd.Timestamp("2018-11-01"), "credit_relevant": 1,
         "direction": "Deterioration", "signal_type": "liquidity", "sector_wide": False,
         "confidence": "High"},
        {"entity": "DHFL", "date": pd.Timestamp("2018-11-10"), "credit_relevant": 1,
         "direction": "Deterioration", "signal_type": "asset_quality", "sector_wide": True,
         "confidence": "Medium"},
        {"entity": "DHFL", "date": pd.Timestamp("2018-11-20"), "credit_relevant": 1,
         "direction": "Deterioration", "signal_type": "funding", "sector_wide": False,
         "confidence": "High"},
        # DHFL: one improvement signal
        {"entity": "DHFL", "date": pd.Timestamp("2018-11-05"), "credit_relevant": 1,
         "direction": "Improvement", "signal_type": "funding", "sector_wide": False,
         "confidence": "Low"},
        # DHFL: not credit-relevant (should be ignored)
        {"entity": "DHFL", "date": pd.Timestamp("2018-11-15"), "credit_relevant": 0,
         "direction": "Neutral", "signal_type": "other", "sector_wide": False,
         "confidence": "High"},
        # Indiabulls: 1 signal (for testing it also gets contagion from DHFL)
        {"entity": "Indiabulls HF", "date": pd.Timestamp("2018-12-01"), "credit_relevant": 1,
         "direction": "Deterioration", "signal_type": "funding", "sector_wide": False,
         "confidence": "Medium"},
    ])


# ============================================================
# Test: Direct Scoring
# ============================================================

class TestDirectScoring:
    def test_single_high_det(self, single_signal_df: pd.DataFrame, default_config: dict) -> None:
        """High-confidence deterioration, not sector-wide → score = 1.0 × 1.0 × 1.0 = 1.0."""
        result = compute_direct_scores(single_signal_df, default_config)
        assert len(result) == 1
        assert result.iloc[0]["direct_score"] == pytest.approx(1.0)
        assert result.iloc[0]["n_signals"] == 1
        assert result.iloc[0]["entity"] == "DHFL"

    def test_sector_wide_bonus(self, default_config: dict) -> None:
        """Sector-wide signal should get 1.5× multiplier."""
        df = pd.DataFrame([{
            "entity": "DHFL", "date": pd.Timestamp("2018-11-15"),
            "credit_relevant": 1, "direction": "Deterioration",
            "signal_type": "contagion", "sector_wide": True, "confidence": "High",
        }])
        result = compute_direct_scores(df, default_config)
        # 1.0 (direction) × 1.0 (confidence) × 1.5 (sector_wide) = 1.5
        assert result.iloc[0]["direct_score"] == pytest.approx(1.5)

    def test_medium_confidence(self, default_config: dict) -> None:
        """Medium confidence → 0.6 weight."""
        df = pd.DataFrame([{
            "entity": "DHFL", "date": pd.Timestamp("2018-11-15"),
            "credit_relevant": 1, "direction": "Deterioration",
            "signal_type": "liquidity", "sector_wide": False, "confidence": "Medium",
        }])
        result = compute_direct_scores(df, default_config)
        assert result.iloc[0]["direct_score"] == pytest.approx(0.6)

    def test_improvement_negative(self, default_config: dict) -> None:
        """Improvement signals should have negative score (-0.5 × confidence)."""
        df = pd.DataFrame([{
            "entity": "DHFL", "date": pd.Timestamp("2018-11-15"),
            "credit_relevant": 1, "direction": "Improvement",
            "signal_type": "funding", "sector_wide": False, "confidence": "High",
        }])
        result = compute_direct_scores(df, default_config)
        assert result.iloc[0]["direct_score"] == pytest.approx(-0.5)

    def test_neutral_zero(self, default_config: dict) -> None:
        """Neutral signals should score 0."""
        df = pd.DataFrame([{
            "entity": "DHFL", "date": pd.Timestamp("2018-11-15"),
            "credit_relevant": 1, "direction": "Neutral",
            "signal_type": "other", "sector_wide": False, "confidence": "High",
        }])
        result = compute_direct_scores(df, default_config)
        assert result.iloc[0]["direct_score"] == pytest.approx(0.0)

    def test_not_credit_relevant_excluded(self, default_config: dict) -> None:
        """Non-credit-relevant articles should be excluded entirely."""
        df = pd.DataFrame([{
            "entity": "DHFL", "date": pd.Timestamp("2018-11-15"),
            "credit_relevant": 0, "direction": "Deterioration",
            "signal_type": "liquidity", "sector_wide": False, "confidence": "High",
        }])
        result = compute_direct_scores(df, default_config)
        assert len(result) == 0

    def test_multi_signal_aggregation(self, default_config: dict) -> None:
        """Multiple signals on same day should sum."""
        df = pd.DataFrame([
            {"entity": "DHFL", "date": pd.Timestamp("2018-11-15"),
             "credit_relevant": 1, "direction": "Deterioration",
             "signal_type": "liquidity", "sector_wide": False, "confidence": "High"},
            {"entity": "DHFL", "date": pd.Timestamp("2018-11-15"),
             "credit_relevant": 1, "direction": "Deterioration",
             "signal_type": "asset_quality", "sector_wide": False, "confidence": "Medium"},
        ])
        result = compute_direct_scores(df, default_config)
        assert len(result) == 1
        # 1.0 + 0.6 = 1.6
        assert result.iloc[0]["direct_score"] == pytest.approx(1.6)
        assert result.iloc[0]["n_signals"] == 2

    def test_empty_input(self, default_config: dict) -> None:
        """Empty DataFrame should return empty result."""
        df = pd.DataFrame(columns=["entity", "date", "credit_relevant", "direction",
                                    "signal_type", "sector_wide", "confidence"])
        result = compute_direct_scores(df, default_config)
        assert len(result) == 0


# ============================================================
# Test: Contagion Propagation
# ============================================================

class TestContagionPropagation:
    def test_intra_subsector_contagion(
        self, single_signal_df: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """DHFL signal should propagate to Indiabulls HF (same subsector, weight=0.8)."""
        direct_df = compute_direct_scores(single_signal_df, default_config)
        contagion_df = compute_contagion_scores(direct_df, mini_graph, default_config)

        # Indiabulls should have contagion from DHFL
        indiabulls = contagion_df[contagion_df["entity"] == "Indiabulls HF"]
        assert len(indiabulls) > 0
        # 0.8 (edge) × 1.0 (DHFL direct) × 0.5 (peer discount, not sector-wide) = 0.4
        assert indiabulls.iloc[0]["contagion_score"] == pytest.approx(0.4)

    def test_cross_subsector_lower(
        self, single_signal_df: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """Cross-subsector contagion (DHFL → Chola) should be weaker than intra."""
        direct_df = compute_direct_scores(single_signal_df, default_config)
        contagion_df = compute_contagion_scores(direct_df, mini_graph, default_config)

        chola = contagion_df[contagion_df["entity"] == "Chola"]
        assert len(chola) > 0
        # 0.1 (edge) × 1.0 (DHFL direct) × 0.5 (peer discount) = 0.05
        assert chola.iloc[0]["contagion_score"] == pytest.approx(0.05)

    def test_sector_wide_no_discount(
        self, mini_graph: EntityGraph, default_config: dict,
    ) -> None:
        """Sector-wide signals should propagate without peer discount."""
        df = pd.DataFrame([{
            "entity": "DHFL", "date": pd.Timestamp("2018-11-15"),
            "credit_relevant": 1, "direction": "Deterioration",
            "signal_type": "contagion", "sector_wide": True, "confidence": "High",
        }])
        direct_df = compute_direct_scores(df, default_config)
        contagion_df = compute_contagion_scores(direct_df, mini_graph, default_config)

        indiabulls = contagion_df[contagion_df["entity"] == "Indiabulls HF"]
        assert len(indiabulls) > 0
        # 0.8 (edge) × 1.5 (sector-wide direct score) × 1.0 (no discount) = 1.2
        assert indiabulls.iloc[0]["contagion_score"] == pytest.approx(1.2)

    def test_zero_direct_gets_contagion(
        self, single_signal_df: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """Indiabulls has 0 direct signals but should still get contagion."""
        direct_df = compute_direct_scores(single_signal_df, default_config)
        contagion_df = compute_contagion_scores(direct_df, mini_graph, default_config)

        # Indiabulls has no direct signals
        assert "Indiabulls HF" not in direct_df["entity"].values
        # But does have contagion
        indiabulls = contagion_df[contagion_df["entity"] == "Indiabulls HF"]
        assert len(indiabulls) > 0
        assert indiabulls.iloc[0]["contagion_score"] > 0

    def test_contagion_window(
        self, mini_graph: EntityGraph, default_config: dict,
    ) -> None:
        """Signal outside the contagion window should not propagate."""
        # Signal 60 days before the check date
        df = pd.DataFrame([{
            "entity": "DHFL", "date": pd.Timestamp("2018-09-01"),
            "credit_relevant": 1, "direction": "Deterioration",
            "signal_type": "liquidity", "sector_wide": False, "confidence": "High",
        }])
        # Set contagion window to 30 days
        config = dict(default_config)
        config["contagion_window_days"] = 30

        direct_df = compute_direct_scores(df, config)
        contagion_df = compute_contagion_scores(direct_df, mini_graph, config)

        # Contagion should only appear on dates within 30 days of Sep 1
        # (i.e., on Sep 1 itself since there's only one date)
        indiabulls = contagion_df[contagion_df["entity"] == "Indiabulls HF"]
        if len(indiabulls) > 0:
            # The contagion date should be Sep 1 (the only date in the system)
            assert indiabulls.iloc[0]["date"] == pd.Timestamp("2018-09-01")

    def test_top_source_tracked(
        self, single_signal_df: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """Top source should identify the peer with highest contribution."""
        direct_df = compute_direct_scores(single_signal_df, default_config)
        contagion_df = compute_contagion_scores(direct_df, mini_graph, default_config)

        indiabulls = contagion_df[contagion_df["entity"] == "Indiabulls HF"]
        assert indiabulls.iloc[0]["top_source"] == "DHFL"

    def test_empty_direct_no_contagion(
        self, mini_graph: EntityGraph, default_config: dict,
    ) -> None:
        """No direct signals → no contagion."""
        empty = pd.DataFrame(columns=["entity", "date", "direct_score", "n_signals",
                                       "n_det", "n_imp", "has_sector_wide"])
        contagion_df = compute_contagion_scores(empty, mini_graph, default_config)
        assert len(contagion_df) == 0


# ============================================================
# Test: v2 Normalization (peer-count normalization)
# ============================================================

class TestContagionNormalization:
    """Tests for v2 contagion normalization by contributing peer count.

    # 🎓 WHY THIS: v1 contagion accumulated raw sums from all contributing peers.
    # With 44 entities, cross-sector controls (Chola, Bajaj) breached the warning
    # threshold on 85% of crisis days — not because of real spillover, but because
    # they had ~30 contributing peers each adding small amounts.
    # v2 divides by n_contributing_peers to get "average contagion per peer."
    """

    def test_single_source_unchanged(
        self, single_signal_df: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """With 1 contributing source, normalization is a no-op (x / 1 = x)."""
        direct_df = compute_direct_scores(single_signal_df, default_config)
        # Default config has normalize_by_peers=True (or unset → defaults True)
        contagion_df = compute_contagion_scores(direct_df, mini_graph, default_config)

        indiabulls = contagion_df[contagion_df["entity"] == "Indiabulls HF"]
        # 0.8 × 1.0 × 0.5 = 0.4, divided by 1 source = 0.4
        assert indiabulls.iloc[0]["contagion_score"] == pytest.approx(0.4)
        assert indiabulls.iloc[0]["n_sources"] == 1

    def test_multi_source_normalized(self, mini_graph: EntityGraph, default_config: dict) -> None:
        """With 2 contributing sources, contagion is averaged across them.

        # 🎓 This is the core normalization test. If DHFL and PNB Housing both
        # have direct signals, Indiabulls gets contagion from both. v1 would sum
        # both contributions. v2 divides by 2.
        """
        df = pd.DataFrame([
            {"entity": "DHFL", "date": pd.Timestamp("2018-11-15"),
             "credit_relevant": 1, "direction": "Deterioration",
             "signal_type": "liquidity", "sector_wide": False, "confidence": "High"},
            {"entity": "PNB Housing", "date": pd.Timestamp("2018-11-15"),
             "credit_relevant": 1, "direction": "Deterioration",
             "signal_type": "asset_quality", "sector_wide": False, "confidence": "High"},
        ])
        direct_df = compute_direct_scores(df, default_config)

        # With normalization ON (default)
        config_norm = dict(default_config)
        config_norm["normalize_by_peers"] = True
        contagion_df = compute_contagion_scores(direct_df, mini_graph, config_norm)

        indiabulls = contagion_df[contagion_df["entity"] == "Indiabulls HF"]
        assert len(indiabulls) > 0
        # DHFL → Indiabulls: 0.8 × 1.0 × 0.5 = 0.4
        # PNB  → Indiabulls: 0.8 × 1.0 × 0.5 = 0.4
        # Raw sum = 0.8, n_sources = 2
        # Normalized = 0.8 / 2 = 0.4
        assert indiabulls.iloc[0]["contagion_score"] == pytest.approx(0.4)
        assert indiabulls.iloc[0]["n_sources"] == 2

    def test_multi_source_unnormalized(self, mini_graph: EntityGraph, default_config: dict) -> None:
        """With normalization OFF, raw sum is preserved (v1 behavior)."""
        df = pd.DataFrame([
            {"entity": "DHFL", "date": pd.Timestamp("2018-11-15"),
             "credit_relevant": 1, "direction": "Deterioration",
             "signal_type": "liquidity", "sector_wide": False, "confidence": "High"},
            {"entity": "PNB Housing", "date": pd.Timestamp("2018-11-15"),
             "credit_relevant": 1, "direction": "Deterioration",
             "signal_type": "asset_quality", "sector_wide": False, "confidence": "High"},
        ])
        direct_df = compute_direct_scores(df, default_config)

        # With normalization OFF (v1 behavior)
        config_v1 = dict(default_config)
        config_v1["normalize_by_peers"] = False
        contagion_df = compute_contagion_scores(direct_df, mini_graph, config_v1)

        indiabulls = contagion_df[contagion_df["entity"] == "Indiabulls HF"]
        # DHFL → Indiabulls: 0.4, PNB → Indiabulls: 0.4, raw sum = 0.8
        assert indiabulls.iloc[0]["contagion_score"] == pytest.approx(0.8)

    def test_normalization_preserves_ratio(
        self, mini_graph: EntityGraph, default_config: dict,
    ) -> None:
        """Intra/cross ratio should be preserved (or improved) by normalization.

        # 🎓 The whole point: relative differentiation should survive normalization.
        # Indiabulls (intra, weight=0.8) should still score higher than Chola (cross, 0.1).
        """
        df = pd.DataFrame([
            {"entity": "DHFL", "date": pd.Timestamp("2018-11-15"),
             "credit_relevant": 1, "direction": "Deterioration",
             "signal_type": "liquidity", "sector_wide": False, "confidence": "High"},
        ])
        direct_df = compute_direct_scores(df, default_config)

        config_norm = dict(default_config)
        config_norm["normalize_by_peers"] = True
        contagion_df = compute_contagion_scores(direct_df, mini_graph, config_norm)

        indiabulls = contagion_df[contagion_df["entity"] == "Indiabulls HF"]
        chola = contagion_df[contagion_df["entity"] == "Chola"]

        # Both have 1 source (DHFL), so normalization is x/1 = x
        # Indiabulls: 0.8 × 1.0 × 0.5 / 1 = 0.4
        # Chola: 0.1 × 1.0 × 0.5 / 1 = 0.05
        # Ratio: 0.4 / 0.05 = 8× (same as v1)
        assert indiabulls.iloc[0]["contagion_score"] > chola.iloc[0]["contagion_score"]
        ratio = indiabulls.iloc[0]["contagion_score"] / chola.iloc[0]["contagion_score"]
        assert ratio >= 2.0  # Must maintain at least 2× differentiation


# ============================================================
# Test: Rolling Windows
# ============================================================

class TestRollingWindows:
    def test_single_day(self) -> None:
        """Single day → rolling should equal the value itself."""
        df = pd.DataFrame([{
            "entity": "DHFL", "date": pd.Timestamp("2018-11-15"),
            "total_score": 2.0,
        }])
        result = compute_rolling_scores(df, windows=[7])
        assert result.iloc[0]["rolling_7d"] == pytest.approx(2.0)

    def test_multi_day_average(self) -> None:
        """Multiple days → rolling 7d should be average over window."""
        df = pd.DataFrame([
            {"entity": "DHFL", "date": pd.Timestamp("2018-11-15"), "total_score": 1.0},
            {"entity": "DHFL", "date": pd.Timestamp("2018-11-16"), "total_score": 3.0},
        ])
        result = compute_rolling_scores(df, windows=[7])
        # Day 1: rolling = 1.0/1 = 1.0 (min_periods=1)
        # Day 2: rolling = (0+1.0+3.0)/2... no, reindexed daily with fill_value=0
        # Day 15 = 1.0, Day 16 = 3.0, avg over 2 days present = (1.0+3.0)/2 = 2.0
        assert result[result["date"] == pd.Timestamp("2018-11-16")]["rolling_7d"].iloc[0] == pytest.approx(2.0)

    def test_empty_input(self) -> None:
        """Empty DataFrame should return empty with rolling columns."""
        df = pd.DataFrame(columns=["entity", "date", "total_score"])
        result = compute_rolling_scores(df, windows=[7, 30])
        assert "rolling_7d" in result.columns
        assert "rolling_30d" in result.columns


# ============================================================
# Test: End-to-End Pipeline
# ============================================================

class TestComputeAllScores:
    def test_end_to_end(
        self, multi_signal_df: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """Full pipeline should produce scores for DHFL, Indiabulls, PNB Housing, Chola."""
        result = compute_all_scores(multi_signal_df, mini_graph, default_config)

        # DHFL should have direct scores
        dhfl = result[result["entity"] == "DHFL"]
        assert len(dhfl) > 0
        assert dhfl["direct_score"].sum() > 0

        # Indiabulls should have contagion from DHFL
        indiabulls = result[result["entity"] == "Indiabulls HF"]
        assert len(indiabulls) > 0

        # Result should have rolling columns
        assert "rolling_7d" in result.columns
        assert "rolling_30d" in result.columns

    def test_total_is_direct_plus_contagion(
        self, multi_signal_df: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """total_score should equal direct_score + contagion_score."""
        result = compute_all_scores(multi_signal_df, mini_graph, default_config)
        for _, row in result.iterrows():
            assert row["total_score"] == pytest.approx(
                row["direct_score"] + row["contagion_score"]
            )
