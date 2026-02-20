# WHY THIS: Unit tests for the backtest module with synthetic data and known
# real-world dates (DHFL first downgrade Feb 3 2019, first default Jun 5 2019).
# These tests verify that lead time calculations, alert metrics, and entity
# matching all produce correct results before running on actual model outputs.

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.training.backtest import (
    build_entity_alias_map,
    compute_alert_metrics,
    compute_entity_timeline,
    compute_lead_time,
    compute_naive_baselines,
    extract_date_from_input,
    extract_entity_from_input,
    extract_title_from_input,
    load_predictions_with_metadata,
    load_rating_actions,
    normalize_entity,
    sweep_alert_thresholds,
)


# ============================================================
# Fixtures: Synthetic Data
# ============================================================

@pytest.fixture
def sample_predictions_df() -> pd.DataFrame:
    """Synthetic predictions DataFrame mimicking holdout output."""
    return pd.DataFrame([
        # DHFL: deterioration signals before Feb 2019 downgrade
        {"entity": "DHFL", "date": pd.Timestamp("2018-09-15"), "title": "DHFL faces liquidity issues",
         "predicted_cr": True, "predicted_direction": "Deterioration",
         "predicted_signal_type": "liquidity", "predicted_confidence": "High",
         "expected_cr": True, "expected_direction": "Deterioration", "parse_ok": True},
        {"entity": "DHFL", "date": pd.Timestamp("2018-11-20"), "title": "DHFL NPA concerns grow",
         "predicted_cr": True, "predicted_direction": "Deterioration",
         "predicted_signal_type": "asset_quality", "predicted_confidence": "High",
         "expected_cr": True, "expected_direction": "Deterioration", "parse_ok": True},
        {"entity": "DHFL", "date": pd.Timestamp("2019-01-10"), "title": "DHFL funding pressures",
         "predicted_cr": True, "predicted_direction": "Deterioration",
         "predicted_signal_type": "funding", "predicted_confidence": "Medium",
         "expected_cr": True, "expected_direction": "Deterioration", "parse_ok": True},
        # DHFL: non-deterioration articles
        {"entity": "DHFL", "date": pd.Timestamp("2018-06-01"), "title": "DHFL stock price target",
         "predicted_cr": False, "predicted_direction": None,
         "predicted_signal_type": None, "predicted_confidence": None,
         "expected_cr": False, "expected_direction": None, "parse_ok": True},
        # Reliance Capital: signals before Mar 2019 downgrade
        {"entity": "Reliance Capital", "date": pd.Timestamp("2019-01-15"), "title": "RelCap stress signs",
         "predicted_cr": True, "predicted_direction": "Deterioration",
         "predicted_signal_type": "funding", "predicted_confidence": "Medium",
         "expected_cr": True, "expected_direction": "Deterioration", "parse_ok": True},
        # Cholamandalam: false positive — deterioration on stable entity
        {"entity": "Cholamandalam", "date": pd.Timestamp("2019-03-01"), "title": "Chola routine business",
         "predicted_cr": True, "predicted_direction": "Deterioration",
         "predicted_signal_type": "other", "predicted_confidence": "Low",
         "expected_cr": False, "expected_direction": None, "parse_ok": True},
        # Cholamandalam: correctly identified as not credit-relevant
        {"entity": "Cholamandalam", "date": pd.Timestamp("2019-04-01"), "title": "Chola stock target",
         "predicted_cr": False, "predicted_direction": None,
         "predicted_signal_type": None, "predicted_confidence": None,
         "expected_cr": False, "expected_direction": None, "parse_ok": True},
    ])


@pytest.fixture
def sample_actions_df() -> pd.DataFrame:
    """Synthetic rating actions for DHFL and Reliance Capital."""
    return pd.DataFrame([
        {"entity": "DHFL", "date": pd.Timestamp("2019-02-03"), "action_type": "downgrade",
         "from_rating": "AAA", "to_rating": "AA+", "agency": "CARE"},
        {"entity": "DHFL", "date": pd.Timestamp("2019-06-05"), "action_type": "default",
         "from_rating": "A", "to_rating": "D", "agency": "CARE"},
        {"entity": "Reliance Capital", "date": pd.Timestamp("2019-03-05"), "action_type": "downgrade",
         "from_rating": "A1+", "to_rating": "A1", "agency": "ICRA"},
    ])


@pytest.fixture
def alias_map() -> dict[str, str]:
    """Entity alias map for testing."""
    return build_entity_alias_map({
        "DHFL": ["DHFL", "Dewan Housing"],
        "Reliance Capital": ["Reliance Capital"],
        "Cholamandalam": ["Cholamandalam"],
    })


# ============================================================
# Test: Entity/Date/Title Extraction
# ============================================================

class TestExtractors:
    def test_extract_entity(self) -> None:
        text = "Entity: DHFL\nDate: 2019-06-01\nTitle: Some title\nArticle: Body text"
        assert extract_entity_from_input(text) == "DHFL"

    def test_extract_entity_with_spaces(self) -> None:
        text = "Entity: Reliance Capital\nDate: 2019-06-01\nTitle: Some title"
        assert extract_entity_from_input(text) == "Reliance Capital"

    def test_extract_entity_missing(self) -> None:
        text = "Date: 2019-06-01\nTitle: Some title"
        assert extract_entity_from_input(text) == "UNKNOWN"

    def test_extract_date(self) -> None:
        text = "Entity: DHFL\nDate: 2019-06-01\nTitle: Some title"
        assert extract_date_from_input(text) == "2019-06-01"

    def test_extract_date_missing(self) -> None:
        text = "Entity: DHFL\nTitle: Some title"
        assert extract_date_from_input(text) == ""

    def test_extract_title(self) -> None:
        text = "Entity: DHFL\nDate: 2019-06-01\nTitle: DHFL faces defaults\nArticle: Body"
        assert extract_title_from_input(text) == "DHFL faces defaults"


# ============================================================
# Test: Entity Aliases
# ============================================================

class TestEntityAliases:
    def test_build_alias_map(self) -> None:
        aliases = {"DHFL": ["DHFL", "Dewan Housing"], "Reliance Capital": ["Reliance Capital"]}
        am = build_entity_alias_map(aliases)
        assert am["dhfl"] == "DHFL"
        assert am["dewan housing"] == "DHFL"
        assert am["reliance capital"] == "Reliance Capital"

    def test_normalize_known_entity(self, alias_map: dict[str, str]) -> None:
        assert normalize_entity("DHFL", alias_map) == "DHFL"
        assert normalize_entity("Dewan Housing", alias_map) == "DHFL"

    def test_normalize_unknown_entity(self, alias_map: dict[str, str]) -> None:
        # Unknown entity returns itself unchanged
        assert normalize_entity("Unknown Corp", alias_map) == "Unknown Corp"

    def test_case_insensitive_matching(self, alias_map: dict[str, str]) -> None:
        assert normalize_entity("dhfl", alias_map) == "DHFL"
        assert normalize_entity("dewan housing", alias_map) == "DHFL"


# ============================================================
# Test: Load Predictions with Metadata
# ============================================================

class TestLoadPredictions:
    def test_index_matching(self, tmp_path: Path) -> None:
        """Test that index-based matching correctly joins predictions to source."""
        # Create prediction file
        preds = [
            {"entity": "DHFL", "expected": "CREDIT_RELEVANT: No\nREASONING: Stock.\nEND",
             "generated": "CREDIT_RELEVANT: No\nREASONING: Stock price.\nEND"},
        ]
        pred_path = tmp_path / "preds.jsonl"
        pred_path.write_text(
            "\n".join(json.dumps(p) for p in preds), encoding="utf-8"
        )

        # Create source file
        source = [
            {"instruction": "Assess...", "input": "Entity: DHFL\nDate: 2019-01-15\nTitle: DHFL stock\nArticle: ...",
             "output": "CREDIT_RELEVANT: No\nREASONING: Stock.\nEND"},
        ]
        source_path = tmp_path / "source.jsonl"
        source_path.write_text(
            "\n".join(json.dumps(s) for s in source), encoding="utf-8"
        )

        df = load_predictions_with_metadata(pred_path, source_path, match_by="index")
        assert len(df) == 1
        assert df.iloc[0]["entity"] == "DHFL"
        assert df.iloc[0]["predicted_cr"] == False  # noqa: E712
        assert pd.notna(df.iloc[0]["date"])

    def test_expected_matching(self, tmp_path: Path) -> None:
        """Test that expected-field matching correctly finds source records."""
        output_text = "CREDIT_RELEVANT: Yes\nDIRECTION: Deterioration\nSIGNAL_TYPE: liquidity\nSECTOR_WIDE: No\nCONFIDENCE: High\nREASONING: Unique reasoning text here.\nEND"
        preds = [
            {"expected": output_text,
             "generated": "CREDIT_RELEVANT: Yes\nDIRECTION: Deterioration\nSIGNAL_TYPE: liquidity\nSECTOR_WIDE: No\nCONFIDENCE: High\nREASONING: Model output.\nEND"},
        ]
        pred_path = tmp_path / "preds.jsonl"
        pred_path.write_text(json.dumps(preds[0]), encoding="utf-8")

        source = [
            {"instruction": "Assess...", "input": "Entity: DHFL\nDate: 2019-02-01\nTitle: DHFL liquidity\nArticle: ...",
             "output": output_text},
            {"instruction": "Assess...", "input": "Entity: Cholamandalam\nDate: 2019-03-01\nTitle: Chola stock\nArticle: ...",
             "output": "CREDIT_RELEVANT: No\nREASONING: Stock.\nEND"},
        ]
        source_path = tmp_path / "source.jsonl"
        source_path.write_text(
            "\n".join(json.dumps(s) for s in source), encoding="utf-8"
        )

        df = load_predictions_with_metadata(pred_path, source_path, match_by="expected")
        assert len(df) == 1
        assert df.iloc[0]["entity"] == "DHFL"


# ============================================================
# Test: Lead Time Computation
# ============================================================

class TestLeadTime:
    def test_dhfl_first_downgrade_lead_time(
        self, sample_predictions_df: pd.DataFrame,
        sample_actions_df: pd.DataFrame, alias_map: dict[str, str]
    ) -> None:
        """DHFL first downgrade is Feb 3, 2019. Earliest signal is Sep 15, 2018.
        Lead time should be 141 days."""
        lt_df = compute_lead_time(
            sample_predictions_df, sample_actions_df,
            lookback_days=180, alias_map=alias_map,
        )
        dhfl_first = lt_df[
            (lt_df["entity"] == "DHFL") &
            (lt_df["action_date"] == pd.Timestamp("2019-02-03"))
        ].iloc[0]
        assert dhfl_first["lead_time_days"] == 141
        assert dhfl_first["n_signals_before"] == 3  # Sep 15, Nov 20, Jan 10

    def test_relcap_lead_time(
        self, sample_predictions_df: pd.DataFrame,
        sample_actions_df: pd.DataFrame, alias_map: dict[str, str]
    ) -> None:
        """Reliance Capital downgrade Mar 5, 2019. Signal on Jan 15, 2019. Lead = 49 days."""
        lt_df = compute_lead_time(
            sample_predictions_df, sample_actions_df,
            lookback_days=180, alias_map=alias_map,
        )
        relcap = lt_df[lt_df["entity"] == "Reliance Capital"].iloc[0]
        assert relcap["lead_time_days"] == 49
        assert relcap["n_signals_before"] == 1

    def test_no_signals_before_action(self, alias_map: dict[str, str]) -> None:
        """If no deterioration signals exist before a rating action, lead_time should be None."""
        preds = pd.DataFrame([
            {"entity": "SomeEntity", "date": pd.Timestamp("2020-01-01"),
             "predicted_cr": False, "predicted_direction": None,
             "predicted_signal_type": None, "predicted_confidence": None,
             "expected_cr": False, "expected_direction": None, "parse_ok": True},
        ])
        actions = pd.DataFrame([
            {"entity": "SomeEntity", "date": pd.Timestamp("2020-03-01"),
             "action_type": "downgrade", "agency": "CRISIL"},
        ])
        lt_df = compute_lead_time(preds, actions, lookback_days=180, alias_map=alias_map)
        assert len(lt_df) == 1
        assert pd.isna(lt_df.iloc[0]["lead_time_days"])

    def test_signal_outside_lookback_window(self, alias_map: dict[str, str]) -> None:
        """Signal more than 180 days before action should not count."""
        preds = pd.DataFrame([
            {"entity": "TestEntity", "date": pd.Timestamp("2018-06-01"),
             "predicted_cr": True, "predicted_direction": "Deterioration",
             "predicted_signal_type": "liquidity", "predicted_confidence": "High",
             "expected_cr": True, "expected_direction": "Deterioration", "parse_ok": True},
        ])
        actions = pd.DataFrame([
            {"entity": "TestEntity", "date": pd.Timestamp("2019-06-01"),
             "action_type": "downgrade", "agency": "CARE"},
        ])
        lt_df = compute_lead_time(preds, actions, lookback_days=180, alias_map=alias_map)
        assert pd.isna(lt_df.iloc[0]["lead_time_days"])

    def test_multiple_actions_same_entity(self, alias_map: dict[str, str]) -> None:
        """Two rating actions for same entity should get independent lead times."""
        preds = pd.DataFrame([
            {"entity": "DHFL", "date": pd.Timestamp("2019-01-10"),
             "predicted_cr": True, "predicted_direction": "Deterioration",
             "predicted_signal_type": "funding", "predicted_confidence": "High",
             "expected_cr": True, "expected_direction": "Deterioration", "parse_ok": True},
        ])
        actions = pd.DataFrame([
            {"entity": "DHFL", "date": pd.Timestamp("2019-02-03"),
             "action_type": "downgrade", "agency": "CARE"},
            {"entity": "DHFL", "date": pd.Timestamp("2019-06-05"),
             "action_type": "default", "agency": "CARE"},
        ])
        lt_df = compute_lead_time(preds, actions, lookback_days=180, alias_map=alias_map)
        assert len(lt_df) == 2
        # First action: signal 24 days before
        assert lt_df.iloc[0]["lead_time_days"] == 24
        # Second action: same signal is 146 days before
        assert lt_df.iloc[1]["lead_time_days"] == 146


# ============================================================
# Test: Alert Metrics
# ============================================================

class TestAlertMetrics:
    def test_basic_alert_precision_recall(
        self, sample_predictions_df: pd.DataFrame,
        sample_actions_df: pd.DataFrame, alias_map: dict[str, str]
    ) -> None:
        """Basic alert test: with threshold N=1, window=30, we should get alerts."""
        metrics = compute_alert_metrics(
            sample_predictions_df, sample_actions_df,
            n_threshold=1, window_days=30, lookahead_days=180,
            alias_map=alias_map,
        )
        # Should have some alerts (DHFL and RelCap both have det signals)
        assert metrics["n_alerts"] > 0
        # Should have some true positives (signals before real downgrades)
        assert metrics["n_true_positives"] > 0

    def test_high_threshold_fewer_alerts(
        self, sample_predictions_df: pd.DataFrame,
        sample_actions_df: pd.DataFrame, alias_map: dict[str, str]
    ) -> None:
        """Higher N threshold should produce fewer or equal alerts."""
        metrics_low = compute_alert_metrics(
            sample_predictions_df, sample_actions_df,
            n_threshold=1, window_days=30, lookahead_days=180,
            alias_map=alias_map,
        )
        metrics_high = compute_alert_metrics(
            sample_predictions_df, sample_actions_df,
            n_threshold=5, window_days=30, lookahead_days=180,
            alias_map=alias_map,
        )
        assert metrics_high["n_alerts"] <= metrics_low["n_alerts"]


# ============================================================
# Test: Threshold Sweep
# ============================================================

class TestThresholdSweep:
    def test_grid_produces_expected_rows(
        self, sample_predictions_df: pd.DataFrame,
        sample_actions_df: pd.DataFrame, alias_map: dict[str, str]
    ) -> None:
        """Grid search should produce N × M × K rows."""
        config = {
            "alert_thresholds": {
                "n_signals": [1, 2],
                "window_days": [30, 60],
                "lookahead_days": [90],
            }
        }
        df = sweep_alert_thresholds(
            sample_predictions_df, sample_actions_df,
            config=config, alias_map=alias_map,
        )
        # 2 × 2 × 1 = 4 rows
        assert len(df) == 4
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "f1" in df.columns

    def test_full_default_grid(
        self, sample_predictions_df: pd.DataFrame,
        sample_actions_df: pd.DataFrame, alias_map: dict[str, str]
    ) -> None:
        """Default grid (4 × 4 × 2 = 32 rows) should all run without errors."""
        config = {
            "alert_thresholds": {
                "n_signals": [1, 2, 3, 5],
                "window_days": [14, 30, 60, 90],
                "lookahead_days": [90, 180],
            }
        }
        df = sweep_alert_thresholds(
            sample_predictions_df, sample_actions_df,
            config=config, alias_map=alias_map,
        )
        assert len(df) == 32


# ============================================================
# Test: Entity Timeline
# ============================================================

class TestEntityTimeline:
    def test_dhfl_timeline(
        self, sample_predictions_df: pd.DataFrame,
        sample_actions_df: pd.DataFrame, alias_map: dict[str, str]
    ) -> None:
        """DHFL timeline should have daily signal counts and rating actions."""
        result = compute_entity_timeline(
            sample_predictions_df, sample_actions_df,
            entity="DHFL", alias_map=alias_map,
        )
        daily = result["daily_signals"]
        actions = result["rating_actions"]

        assert len(daily) > 0
        assert "n_det" in daily.columns
        assert "cumulative_det" in daily.columns
        assert len(actions) == 2  # downgrade + default

    def test_cholamandalam_no_actions(
        self, sample_predictions_df: pd.DataFrame,
        sample_actions_df: pd.DataFrame, alias_map: dict[str, str]
    ) -> None:
        """Cholamandalam should have articles but zero rating actions."""
        result = compute_entity_timeline(
            sample_predictions_df, sample_actions_df,
            entity="Cholamandalam", alias_map=alias_map,
        )
        assert len(result["daily_signals"]) > 0
        assert len(result["rating_actions"]) == 0


# ============================================================
# Test: Naive Baselines
# ============================================================

class TestNaiveBaselines:
    def test_always_deterioration_baseline(
        self, sample_predictions_df: pd.DataFrame,
        sample_actions_df: pd.DataFrame, alias_map: dict[str, str]
    ) -> None:
        """Always-deterioration should cover all actions with signals."""
        baselines = compute_naive_baselines(
            sample_predictions_df, sample_actions_df, alias_map=alias_map,
        )
        assert "always_deterioration" in baselines
        always_det = baselines["always_deterioration"]
        # Every article is a "signal", so all actions should have coverage
        assert always_det["n_with_signal"] == always_det["n_total_actions"]

    def test_ground_truth_labels_baseline(
        self, sample_predictions_df: pd.DataFrame,
        sample_actions_df: pd.DataFrame, alias_map: dict[str, str]
    ) -> None:
        """Ground truth labels baseline should exist and have valid lead times."""
        baselines = compute_naive_baselines(
            sample_predictions_df, sample_actions_df, alias_map=alias_map,
        )
        assert "ground_truth_labels" in baselines


# ============================================================
# Test: Load Rating Actions
# ============================================================

class TestLoadRatingActions:
    def test_load_with_filter(self, tmp_path: Path) -> None:
        """Loading with action_types filter should only return matching rows."""
        csv_content = (
            "entity,entity_full_name,agency,date,action_type,from_rating,to_rating,instrument_type,rationale_url,notes\n"
            "DHFL,DHFL Ltd,CARE,2019-02-03,downgrade,AAA,AA+,LT,,test\n"
            "DHFL,DHFL Ltd,CARE,2019-06-05,default,A,D,LT,,test\n"
            "DHFL,DHFL Ltd,ICRA,2019-01-15,affirmed,AAA,AAA,NCD,,test\n"
        )
        csv_path = tmp_path / "actions.csv"
        csv_path.write_text(csv_content, encoding="utf-8")

        df = load_rating_actions(csv_path, action_types=["downgrade", "default"])
        assert len(df) == 2

    def test_load_without_filter(self, tmp_path: Path) -> None:
        """Loading without filter should return all rows."""
        csv_content = (
            "entity,entity_full_name,agency,date,action_type,from_rating,to_rating,instrument_type,rationale_url,notes\n"
            "DHFL,DHFL Ltd,CARE,2019-02-03,downgrade,AAA,AA+,LT,,test\n"
            "DHFL,DHFL Ltd,ICRA,2019-01-15,affirmed,AAA,AAA,NCD,,test\n"
        )
        csv_path = tmp_path / "actions.csv"
        csv_path.write_text(csv_content, encoding="utf-8")

        df = load_rating_actions(csv_path)
        assert len(df) == 2


# ============================================================
# Test: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_predictions(self, alias_map: dict[str, str]) -> None:
        """Lead time with empty predictions should return results for each action."""
        preds = pd.DataFrame(columns=[
            "entity", "date", "title", "predicted_cr", "predicted_direction",
            "predicted_signal_type", "predicted_confidence",
            "expected_cr", "expected_direction", "parse_ok",
        ])
        actions = pd.DataFrame([
            {"entity": "DHFL", "date": pd.Timestamp("2019-02-03"),
             "action_type": "downgrade", "agency": "CARE"},
        ])
        lt_df = compute_lead_time(preds, actions, lookback_days=180, alias_map=alias_map)
        assert len(lt_df) == 1
        assert pd.isna(lt_df.iloc[0]["lead_time_days"])

    def test_empty_actions(
        self, sample_predictions_df: pd.DataFrame, alias_map: dict[str, str]
    ) -> None:
        """Lead time with no rating actions should return empty DataFrame."""
        actions = pd.DataFrame(columns=["entity", "date", "action_type", "agency"])
        lt_df = compute_lead_time(
            sample_predictions_df, actions,
            lookback_days=180, alias_map=alias_map,
        )
        assert len(lt_df) == 0

    def test_same_day_signal_and_action(self, alias_map: dict[str, str]) -> None:
        """Signal on the same day as action should have 0-day lead time."""
        preds = pd.DataFrame([
            {"entity": "DHFL", "date": pd.Timestamp("2019-02-03"),
             "predicted_cr": True, "predicted_direction": "Deterioration",
             "predicted_signal_type": "funding", "predicted_confidence": "High",
             "expected_cr": True, "expected_direction": "Deterioration", "parse_ok": True},
        ])
        actions = pd.DataFrame([
            {"entity": "DHFL", "date": pd.Timestamp("2019-02-03"),
             "action_type": "downgrade", "agency": "CARE"},
        ])
        lt_df = compute_lead_time(preds, actions, lookback_days=180, alias_map=alias_map)
        assert lt_df.iloc[0]["lead_time_days"] == 0
