# WHY THIS: Unit tests for the contagion backtest using synthetic data.
# Verifies: signal loading, crisis replay, contagion lead time computation,
# and the "lead time improvement" metric that measures contagion's value.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.signals.contagion_backtest import (
    compute_contagion_lead_time,
    load_holdout_signals,
    load_label_signals,
    run_crisis_replay,
)
from src.signals.entity_graph import EntityGraph, load_entity_graph
from src.signals.propagation import compute_all_scores


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def default_config() -> dict:
    """Standard contagion config."""
    return {
        "direction_multiplier": {"Deterioration": 1.0, "Improvement": -0.5, "Neutral": 0.0},
        "confidence_weights": {"High": 1.0, "Medium": 0.6, "Low": 0.3},
        "sector_wide_multiplier": 1.5,
        "contagion_window_days": 30,
        "peer_signal_discount": 0.5,
        "rolling_windows": [7, 30],
        "edge_weights": {"intra_subsector": 0.8, "cross_subsector": 0.1},
        "score_thresholds": {"warning": 2.0, "critical": 5.0},
    }


@pytest.fixture
def mini_graph(tmp_path: Path) -> EntityGraph:
    """Small graph: DHFL + Indiabulls (housing), Chola (diversified)."""
    data = {
        "subsectors": {
            "housing_finance": [
                {"name": "DHFL", "full_name": "DHFL", "aliases": ["DHFL", "Dewan Housing"], "status": "defaulted"},
                {"name": "Indiabulls HF", "full_name": "Indiabulls HF", "aliases": ["Indiabulls HF", "Indiabulls"], "status": "active"},
            ],
            "diversified_nbfc": [
                {"name": "Chola", "full_name": "Chola", "aliases": ["Chola", "Cholamandalam"], "status": "active"},
            ],
        }
    }
    yaml_path = tmp_path / "entities.yaml"
    yaml_path.write_text(yaml.dump(data, default_flow_style=False))
    config = {"edge_weights": {"intra_subsector": 0.8, "cross_subsector": 0.1}}
    return load_entity_graph(yaml_path, config)


@pytest.fixture
def crisis_signals_df() -> pd.DataFrame:
    """Synthetic signals mimicking the DHFL 2018 crisis.

    DHFL: multiple deterioration signals in Nov 2018
    Indiabulls: 0 direct signals (tests that contagion provides its ONLY warning)
    Chola: 1 mild signal (tests cross-subsector FP control)
    """
    return pd.DataFrame([
        # DHFL crisis signals — 5 high-confidence deteriorations
        {"entity": "DHFL", "date": pd.Timestamp("2018-11-01"), "credit_relevant": 1,
         "direction": "Deterioration", "signal_type": "liquidity", "sector_wide": False,
         "confidence": "High"},
        {"entity": "DHFL", "date": pd.Timestamp("2018-11-05"), "credit_relevant": 1,
         "direction": "Deterioration", "signal_type": "asset_quality", "sector_wide": True,
         "confidence": "High"},
        {"entity": "DHFL", "date": pd.Timestamp("2018-11-10"), "credit_relevant": 1,
         "direction": "Deterioration", "signal_type": "funding", "sector_wide": False,
         "confidence": "High"},
        {"entity": "DHFL", "date": pd.Timestamp("2018-11-15"), "credit_relevant": 1,
         "direction": "Deterioration", "signal_type": "governance", "sector_wide": False,
         "confidence": "Medium"},
        {"entity": "DHFL", "date": pd.Timestamp("2018-11-20"), "credit_relevant": 1,
         "direction": "Deterioration", "signal_type": "liquidity", "sector_wide": False,
         "confidence": "High"},
        # Chola: one low-confidence signal (FP check)
        {"entity": "Chola", "date": pd.Timestamp("2018-11-12"), "credit_relevant": 1,
         "direction": "Deterioration", "signal_type": "other", "sector_wide": False,
         "confidence": "Low"},
    ])


# ============================================================
# Test: Contagion Lead Time
# ============================================================

class TestContagionLeadTime:
    def test_dhfl_contagion_reaches_indiabulls(
        self, crisis_signals_df: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """Indiabulls HF should get contagion score > 0 from DHFL signals."""
        scores = compute_all_scores(crisis_signals_df, mini_graph, default_config)

        indiabulls = scores[scores["entity"] == "Indiabulls HF"]
        assert len(indiabulls) > 0, "Indiabulls should have contagion scores"
        assert indiabulls["contagion_score"].sum() > 0

    def test_indiabulls_zero_direct(
        self, crisis_signals_df: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """Indiabulls has no direct signals — only contagion."""
        scores = compute_all_scores(crisis_signals_df, mini_graph, default_config)

        indiabulls = scores[scores["entity"] == "Indiabulls HF"]
        assert all(indiabulls["direct_score"] == 0.0)
        assert all(indiabulls["contagion_score"] > 0)

    def test_lead_time_with_contagion(
        self, crisis_signals_df: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """Contagion should give Indiabulls a positive lead time before its 'downgrade'."""
        scores = compute_all_scores(crisis_signals_df, mini_graph, default_config)

        # Assume Indiabulls gets downgraded Aug 30, 2019
        result = compute_contagion_lead_time(
            scores_df=scores,
            entity="Indiabulls HF",
            first_action_date="2019-08-30",
            threshold=0.1,  # Low threshold to ensure breach
            graph=mini_graph,
        )
        assert result["lead_time_days"] is not None
        assert result["lead_time_days"] > 0

    def test_lead_time_improvement_when_no_direct(
        self, crisis_signals_df: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """When direct-only has no breach, improvement = full contagion lead time."""
        scores = compute_all_scores(crisis_signals_df, mini_graph, default_config)

        result = compute_contagion_lead_time(
            scores_df=scores,
            entity="Indiabulls HF",
            first_action_date="2019-08-30",
            threshold=0.1,
            graph=mini_graph,
        )

        # Since Indiabulls has 0 direct signals, direct-only lead time should be None
        assert result["direct_only_lead_time"] is None
        # And improvement should equal the full contagion lead time
        assert result["lead_time_improvement"] == result["lead_time_days"]

    def test_chola_lower_contagion(
        self, crisis_signals_df: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """Chola (cross-subsector) should get less contagion than Indiabulls (intra)."""
        scores = compute_all_scores(crisis_signals_df, mini_graph, default_config)

        indiabulls = scores[scores["entity"] == "Indiabulls HF"]
        chola = scores[scores["entity"] == "Chola"]

        if len(indiabulls) > 0 and len(chola) > 0:
            # Compare peak contagion scores
            indiabulls_peak = indiabulls["contagion_score"].max()
            chola_peak = chola["contagion_score"].max()
            assert indiabulls_peak > chola_peak

    def test_entity_not_in_scores(self, default_config: dict) -> None:
        """Entity not found should return None lead times."""
        empty_scores = pd.DataFrame(columns=[
            "entity", "date", "direct_score", "contagion_score", "total_score",
        ])
        result = compute_contagion_lead_time(
            scores_df=empty_scores,
            entity="Nonexistent Corp",
            first_action_date="2019-08-30",
            threshold=2.0,
        )
        assert result["lead_time_days"] is None
        assert result["peak_score"] == 0.0


# ============================================================
# Test: Crisis Replay
# ============================================================

class TestCrisisReplay:
    def test_run_crisis_replay(
        self, crisis_signals_df: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """Full crisis replay should return structured results."""
        crisis_config = {
            "name": "Test DHFL Crisis",
            "start_date": "2018-06-01",
            "end_date": "2020-06-30",
            "source_entities": ["DHFL"],
            "target_entities": [
                {"name": "Indiabulls HF", "first_action": "2019-08-30"},
                {"name": "Chola", "first_action": None},  # Stable control
            ],
        }

        result = run_crisis_replay(
            signals_df=crisis_signals_df,
            graph=mini_graph,
            crisis_config=crisis_config,
            config=default_config,
        )

        assert result["crisis_name"] == "Test DHFL Crisis"
        assert len(result["source_results"]) == 1
        assert result["source_results"][0]["entity"] == "DHFL"
        assert len(result["target_results"]) == 2

    def test_source_entity_has_signals(
        self, crisis_signals_df: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """Source entity (DHFL) should have direct scores in crisis replay."""
        crisis_config = {
            "name": "Test",
            "start_date": "2018-06-01",
            "end_date": "2020-06-30",
            "source_entities": ["DHFL"],
            "target_entities": [],
        }
        result = run_crisis_replay(crisis_signals_df, mini_graph, crisis_config, default_config)
        assert result["source_results"][0]["peak_direct"] > 0

    def test_control_entity_no_lead_time(
        self, crisis_signals_df: pd.DataFrame, mini_graph: EntityGraph,
        default_config: dict,
    ) -> None:
        """Control entity (first_action=null) should not have lead time computed."""
        crisis_config = {
            "name": "Test",
            "start_date": "2018-06-01",
            "end_date": "2020-06-30",
            "source_entities": ["DHFL"],
            "target_entities": [
                {"name": "Chola", "first_action": None},
            ],
        }
        result = run_crisis_replay(crisis_signals_df, mini_graph, crisis_config, default_config)
        chola = result["target_results"][0]
        assert chola["is_control"] is True
        assert chola["lead_time_days"] is None


# ============================================================
# Test: Load Holdout Signals
# ============================================================

class TestLoadHoldoutSignals:
    def test_loads_valid_predictions(self, tmp_path: Path) -> None:
        """Should parse valid model predictions into signal rows."""
        # Create prediction JSONL
        preds = [
            {
                "entity": "DHFL",
                "expected": "CREDIT_RELEVANT: Yes\nDIRECTION: Deterioration\nSIGNAL_TYPE: liquidity\nSECTOR_WIDE: No\nCONFIDENCE: High\nREASONING: Test.\nEND",
                "generated": "CREDIT_RELEVANT: Yes\nDIRECTION: Deterioration\nSIGNAL_TYPE: liquidity\nSECTOR_WIDE: No\nCONFIDENCE: High\nREASONING: Test output.\nEND",
            },
        ]
        pred_path = tmp_path / "preds.jsonl"
        pred_path.write_text(json.dumps(preds[0]), encoding="utf-8")

        # Create source JSONL
        source = [
            {
                "instruction": "Assess...",
                "input": "Entity: DHFL\nDate: 2018-11-15\nTitle: Test\nArticle: Body",
                "output": "...",
            },
        ]
        source_path = tmp_path / "source.jsonl"
        source_path.write_text(json.dumps(source[0]), encoding="utf-8")

        df = load_holdout_signals(pred_path, source_path)
        assert len(df) == 1
        assert df.iloc[0]["entity"] == "DHFL"
        assert df.iloc[0]["direction"] == "Deterioration"
        assert df.iloc[0]["signal_source"] == "model"

    def test_skips_unparseable(self, tmp_path: Path) -> None:
        """Unparseable predictions should be skipped."""
        preds = [{"entity": "DHFL", "expected": "...", "generated": "Just a free text essay."}]
        pred_path = tmp_path / "preds.jsonl"
        pred_path.write_text(json.dumps(preds[0]), encoding="utf-8")

        source = [{"instruction": "...", "input": "Entity: DHFL\nDate: 2018-11-15\nTitle: T\nArticle: B", "output": "..."}]
        source_path = tmp_path / "source.jsonl"
        source_path.write_text(json.dumps(source[0]), encoding="utf-8")

        df = load_holdout_signals(pred_path, source_path)
        assert len(df) == 0


# ============================================================
# Test: Load Label Signals
# ============================================================

class TestLoadLabelSignals:
    def test_loads_labels(self, tmp_path: Path) -> None:
        """Should load labels and match to articles."""
        # Create labels JSONL
        labels_path = tmp_path / "labels.jsonl"
        label = {
            "url": "http://example.com/article1",
            "credit_relevant": 1,
            "signal_direction": -1,
            "signal_type": "liquidity",
            "sector_wide": 0,
            "confidence": "high",
            "reasoning": "Test",
        }
        labels_path.write_text(json.dumps(label), encoding="utf-8")

        # Create articles CSV
        articles_path = tmp_path / "articles.csv"
        articles_path.write_text(
            "article_url,article_title,article_date,source_domain,gdelt_tone,entities,rating_windows,article_text,source_bucket\n"
            "http://example.com/article1,Test Article,2018-11-15,example.com,,Test Entity,,Body,pass\n",
            encoding="utf-8",
        )

        df = load_label_signals(labels_path, articles_path)
        assert len(df) == 1
        assert df.iloc[0]["entity"] == "Test Entity"
        assert df.iloc[0]["direction"] == "Deterioration"
        assert df.iloc[0]["signal_source"] == "label"

    def test_excludes_holdout_entities(self, tmp_path: Path) -> None:
        """Should exclude specified holdout entities."""
        labels_path = tmp_path / "labels.jsonl"
        label = {
            "url": "http://example.com/dhfl1",
            "credit_relevant": 1,
            "signal_direction": -1,
            "signal_type": "liquidity",
            "sector_wide": 0,
            "confidence": "high",
            "reasoning": "Test",
        }
        labels_path.write_text(json.dumps(label), encoding="utf-8")

        articles_path = tmp_path / "articles.csv"
        articles_path.write_text(
            "article_url,article_title,article_date,source_domain,gdelt_tone,entities,rating_windows,article_text,source_bucket\n"
            "http://example.com/dhfl1,DHFL Article,2018-11-15,example.com,,DHFL,,Body,pass\n",
            encoding="utf-8",
        )

        df = load_label_signals(labels_path, articles_path, exclude_entities=["DHFL"])
        assert len(df) == 0
