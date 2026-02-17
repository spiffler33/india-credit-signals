# WHY THIS: The JSON parser is the most fragile part of the pipeline — it handles
# every possible way an LLM can mangle JSON output. These tests ensure coercion
# (yes→1, markdown fences, unknown signal types) works correctly. If a test here
# fails, labels in the pipeline will silently be wrong.

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.data.label_models import (
    ArticleLabel,
    get_completed_urls,
    parse_llm_response,
    read_labels_jsonl,
    write_label_jsonl,
)


class TestParseValidJSON:
    """Test parsing of well-formed LLM responses."""

    def test_clean_json(self):
        raw = json.dumps({
            "credit_relevant": 1,
            "signal_direction": -1,
            "signal_type": "liquidity",
            "sector_wide": 0,
            "confidence": "high",
            "reasoning": "Severe liquidity crunch reported",
        })
        label = parse_llm_response(raw, url="http://example.com/1")
        assert label.credit_relevant == 1
        assert label.signal_direction == -1
        assert label.signal_type == "liquidity"
        assert label.sector_wide == 0
        assert label.confidence == "high"
        assert label.reasoning == "Severe liquidity crunch reported"
        assert label.parse_error is None

    def test_not_credit_relevant(self):
        raw = json.dumps({
            "credit_relevant": 0,
            "signal_direction": 0,
            "signal_type": "other",
            "sector_wide": 0,
            "confidence": "high",
            "reasoning": "CSR activity, not credit-related",
        })
        label = parse_llm_response(raw, url="http://example.com/2")
        assert label.credit_relevant == 0
        assert label.signal_direction == 0
        assert label.signal_type == "other"


class TestParseMarkdownFences:
    """Test stripping of markdown code fences — very common LLM behavior."""

    def test_json_fence(self):
        raw = '```json\n{"credit_relevant": 1, "signal_direction": -1, "signal_type": "asset_quality", "sector_wide": 0, "confidence": "medium", "reasoning": "Rising NPAs"}\n```'
        label = parse_llm_response(raw, url="http://example.com/3")
        assert label.credit_relevant == 1
        assert label.signal_type == "asset_quality"
        assert label.parse_error is None

    def test_plain_fence(self):
        raw = '```\n{"credit_relevant": 0, "signal_direction": 0, "signal_type": "other", "sector_wide": 0, "confidence": "high", "reasoning": "Sports news"}\n```'
        label = parse_llm_response(raw, url="http://example.com/4")
        assert label.credit_relevant == 0
        assert label.parse_error is None


class TestBoolCoercion:
    """Test yes/no/true/false → 1/0 coercion."""

    def test_yes_no(self):
        raw = json.dumps({
            "credit_relevant": "yes",
            "signal_direction": "deterioration",
            "signal_type": "governance",
            "sector_wide": "no",
            "confidence": "medium",
            "reasoning": "Board reshuffled amid fraud probe",
        })
        label = parse_llm_response(raw, url="http://example.com/5")
        assert label.credit_relevant == 1
        assert label.signal_direction == -1
        assert label.sector_wide == 0
        assert label.parse_error is None

    def test_true_false_strings(self):
        raw = json.dumps({
            "credit_relevant": "true",
            "signal_direction": "positive",
            "signal_type": "funding",
            "sector_wide": "false",
            "confidence": "low",
            "reasoning": "Successfully raised capital",
        })
        label = parse_llm_response(raw, url="http://example.com/6")
        assert label.credit_relevant == 1
        assert label.signal_direction == 1


class TestSignalTypeCoercion:
    """Test mapping of invented signal_type values to valid enums."""

    def test_unknown_maps_to_other(self):
        raw = json.dumps({
            "credit_relevant": 1,
            "signal_direction": -1,
            "signal_type": "financial_stress",
            "sector_wide": 0,
            "confidence": "medium",
            "reasoning": "Test",
        })
        label = parse_llm_response(raw, url="http://example.com/7")
        # "financial_stress" doesn't match any prefix → other
        assert label.signal_type == "other"

    def test_liquid_prefix(self):
        raw = json.dumps({
            "credit_relevant": 1,
            "signal_direction": -1,
            "signal_type": "liquidity_crisis",
            "sector_wide": 0,
            "confidence": "high",
            "reasoning": "Test",
        })
        label = parse_llm_response(raw, url="http://example.com/8")
        assert label.signal_type == "liquidity"

    def test_regulatory_variant(self):
        raw = json.dumps({
            "credit_relevant": 1,
            "signal_direction": -1,
            "signal_type": "RBI_regulation",
            "sector_wide": 1,
            "confidence": "high",
            "reasoning": "Test",
        })
        label = parse_llm_response(raw, url="http://example.com/9")
        assert label.signal_type == "regulatory"


class TestConsistencyEnforcement:
    """Test that non-credit-relevant labels get zeroed out."""

    def test_zero_out_signals_when_not_relevant(self):
        """If LLM says credit_relevant=0 but signal_direction=-1, zero it."""
        raw = json.dumps({
            "credit_relevant": 0,
            "signal_direction": -1,
            "signal_type": "liquidity",
            "sector_wide": 1,
            "confidence": "medium",
            "reasoning": "Contradictory output",
        })
        label = parse_llm_response(raw, url="http://example.com/10")
        assert label.credit_relevant == 0
        assert label.signal_direction == 0
        assert label.signal_type == "other"
        assert label.sector_wide == 0


class TestUnparseableResponses:
    """Test graceful handling of garbage LLM output."""

    def test_complete_garbage(self):
        label = parse_llm_response(
            "I cannot analyze this article.", url="http://example.com/11"
        )
        assert label.parse_error is not None
        assert label.credit_relevant == 0  # safe default

    def test_partial_json(self):
        label = parse_llm_response(
            '{"credit_relevant": 1, "signal_direction":', url="http://example.com/12"
        )
        assert label.parse_error is not None

    def test_empty_string(self):
        label = parse_llm_response("", url="http://example.com/13")
        assert label.parse_error is not None

    def test_json_with_surrounding_text(self):
        """LLM wraps JSON in explanation text — parser should find the JSON."""
        raw = 'Here is my analysis:\n{"credit_relevant": 1, "signal_direction": -1, "signal_type": "governance", "sector_wide": 0, "confidence": "high", "reasoning": "Fraud detected"}\nHope this helps!'
        label = parse_llm_response(raw, url="http://example.com/14")
        assert label.credit_relevant == 1
        assert label.signal_type == "governance"
        assert label.parse_error is None


class TestJSONLIO:
    """Test JSONL read/write and resume support."""

    def test_write_and_read_roundtrip(self, tmp_path: Path):
        path = tmp_path / "test.jsonl"
        label1 = ArticleLabel(url="http://a.com", credit_relevant=1, signal_direction=-1)
        label2 = ArticleLabel(url="http://b.com", credit_relevant=0)

        write_label_jsonl(label1, path)
        write_label_jsonl(label2, path)

        labels = read_labels_jsonl(path)
        assert len(labels) == 2
        assert labels[0].url == "http://a.com"
        assert labels[0].credit_relevant == 1
        assert labels[1].url == "http://b.com"

    def test_get_completed_urls(self, tmp_path: Path):
        path = tmp_path / "test.jsonl"
        write_label_jsonl(ArticleLabel(url="http://a.com"), path)
        write_label_jsonl(ArticleLabel(url="http://b.com"), path)

        urls = get_completed_urls(path)
        assert urls == {"http://a.com", "http://b.com"}

    def test_missing_file_returns_empty(self, tmp_path: Path):
        path = tmp_path / "nonexistent.jsonl"
        assert read_labels_jsonl(path) == []
        assert get_completed_urls(path) == set()

    def test_corrupt_line_skipped(self, tmp_path: Path):
        path = tmp_path / "test.jsonl"
        write_label_jsonl(ArticleLabel(url="http://good.com"), path)
        # Append corrupt line
        with open(path, "a") as f:
            f.write("this is not json\n")
        write_label_jsonl(ArticleLabel(url="http://also-good.com"), path)

        labels = read_labels_jsonl(path)
        assert len(labels) == 2
        assert labels[0].url == "http://good.com"
        assert labels[1].url == "http://also-good.com"
