# WHY THIS: The parser is the contract between what we train the model to produce
# and what we can actually use in production. If the parser accepts invalid output
# or rejects valid output, the whole pipeline breaks silently. These tests cover
# every edge case we've seen (and some we invented) to catch regressions early.
# Formatter tests verify the training data structure is correct before a $50+ run.

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.data.parse_training_output import (
    ParsedOutput,
    format_output_text,
    parse_training_output,
)
from src.data.format_training import (
    build_example,
    format_input,
    load_config,
    parse_date,
    split_examples,
    truncate_text,
)
from src.data.label_models import ArticleLabel


# ============================================================
# Parser Tests
# ============================================================


class TestParseFullForm:
    """Test parsing of credit-relevant (full-form) outputs."""

    def test_correct_full_output(self):
        text = (
            "CREDIT_RELEVANT: Yes\n"
            "DIRECTION: Deterioration\n"
            "SIGNAL_TYPE: asset_quality\n"
            "SECTOR_WIDE: No\n"
            "CONFIDENCE: High\n"
            "REASONING: 11.4% of PFC's loan book entering insolvency.\n"
            "END"
        )
        result = parse_training_output(text)
        assert result.parse_ok is True
        assert result.credit_relevant is True
        assert result.direction == "Deterioration"
        assert result.signal_type == "asset_quality"
        assert result.sector_wide is False
        assert result.confidence == "High"
        assert "insolvency" in result.reasoning
        assert result.has_end_token is True
        assert result.error_field == ""

    def test_improvement_signal(self):
        text = (
            "CREDIT_RELEVANT: Yes\n"
            "DIRECTION: Improvement\n"
            "SIGNAL_TYPE: funding\n"
            "SECTOR_WIDE: No\n"
            "CONFIDENCE: Medium\n"
            "REASONING: Successful Rs 500 crore NCD issuance.\n"
            "END"
        )
        result = parse_training_output(text)
        assert result.parse_ok is True
        assert result.direction == "Improvement"
        assert result.signal_type == "funding"
        assert result.confidence == "Medium"

    def test_neutral_sector_wide(self):
        text = (
            "CREDIT_RELEVANT: Yes\n"
            "DIRECTION: Neutral\n"
            "SIGNAL_TYPE: regulatory\n"
            "SECTOR_WIDE: Yes\n"
            "CONFIDENCE: Low\n"
            "REASONING: RBI guidelines for all NBFCs.\n"
            "END"
        )
        result = parse_training_output(text)
        assert result.parse_ok is True
        assert result.direction == "Neutral"
        assert result.sector_wide is True

    def test_all_signal_types(self):
        """Every valid signal_type should parse correctly."""
        for st in ["liquidity", "asset_quality", "regulatory", "contagion",
                    "governance", "funding", "operational", "other"]:
            text = (
                f"CREDIT_RELEVANT: Yes\n"
                f"DIRECTION: Deterioration\n"
                f"SIGNAL_TYPE: {st}\n"
                f"SECTOR_WIDE: No\n"
                f"CONFIDENCE: High\n"
                f"REASONING: Test for {st}.\n"
                f"END"
            )
            result = parse_training_output(text)
            assert result.parse_ok is True, f"Failed for signal_type={st}"
            assert result.signal_type == st


class TestParseShortForm:
    """Test parsing of non-credit-relevant (short-form) outputs."""

    def test_correct_short_output(self):
        text = (
            "CREDIT_RELEVANT: No\n"
            "REASONING: Stock price movements with no credit information.\n"
            "END"
        )
        result = parse_training_output(text)
        assert result.parse_ok is True
        assert result.credit_relevant is False
        assert "Stock price" in result.reasoning
        assert result.direction == ""
        assert result.signal_type == ""
        assert result.has_end_token is True

    def test_short_form_missing_reasoning(self):
        text = "CREDIT_RELEVANT: No\nEND"
        result = parse_training_output(text)
        assert result.parse_ok is False
        assert result.error_field == "REASONING"


class TestParseErrors:
    """Test that the parser correctly rejects invalid outputs."""

    def test_missing_end_token(self):
        text = (
            "CREDIT_RELEVANT: Yes\n"
            "DIRECTION: Deterioration\n"
            "SIGNAL_TYPE: liquidity\n"
            "SECTOR_WIDE: No\n"
            "CONFIDENCE: High\n"
            "REASONING: Severe liquidity crunch."
        )
        result = parse_training_output(text)
        assert result.parse_ok is False
        assert result.error_field == "END"
        assert result.has_end_token is False
        # Fields should still be parsed correctly
        assert result.credit_relevant is True
        assert result.direction == "Deterioration"

    def test_wrong_direction_vocabulary(self):
        """'Negative' is not in our vocab — should be 'Deterioration'."""
        text = (
            "CREDIT_RELEVANT: Yes\n"
            "DIRECTION: Negative\n"
            "SIGNAL_TYPE: liquidity\n"
            "SECTOR_WIDE: No\n"
            "CONFIDENCE: High\n"
            "REASONING: Test.\n"
            "END"
        )
        result = parse_training_output(text)
        assert result.parse_ok is False
        assert result.error_field == "DIRECTION"
        assert "Negative" in result.error_detail

    def test_wrong_signal_type_vocabulary(self):
        """'financial_stress' is not a valid signal type."""
        text = (
            "CREDIT_RELEVANT: Yes\n"
            "DIRECTION: Deterioration\n"
            "SIGNAL_TYPE: financial_stress\n"
            "SECTOR_WIDE: No\n"
            "CONFIDENCE: High\n"
            "REASONING: Test.\n"
            "END"
        )
        result = parse_training_output(text)
        assert result.parse_ok is False
        assert result.error_field == "SIGNAL_TYPE"

    def test_wrong_confidence_vocabulary(self):
        """'high' (lowercase) is not valid — should be 'High'."""
        text = (
            "CREDIT_RELEVANT: Yes\n"
            "DIRECTION: Deterioration\n"
            "SIGNAL_TYPE: liquidity\n"
            "SECTOR_WIDE: No\n"
            "CONFIDENCE: high\n"
            "REASONING: Test.\n"
            "END"
        )
        result = parse_training_output(text)
        assert result.parse_ok is False
        assert result.error_field == "CONFIDENCE"

    def test_missing_credit_relevant(self):
        text = (
            "DIRECTION: Deterioration\n"
            "SIGNAL_TYPE: liquidity\n"
            "END"
        )
        result = parse_training_output(text)
        assert result.parse_ok is False
        assert result.error_field == "CREDIT_RELEVANT"

    def test_invalid_credit_relevant_value(self):
        text = (
            "CREDIT_RELEVANT: Maybe\n"
            "REASONING: Unclear.\n"
            "END"
        )
        result = parse_training_output(text)
        assert result.parse_ok is False
        assert result.error_field == "CREDIT_RELEVANT"
        assert "Maybe" in result.error_detail

    def test_missing_direction_for_cr_yes(self):
        text = (
            "CREDIT_RELEVANT: Yes\n"
            "SIGNAL_TYPE: liquidity\n"
            "SECTOR_WIDE: No\n"
            "CONFIDENCE: High\n"
            "REASONING: Test.\n"
            "END"
        )
        result = parse_training_output(text)
        assert result.parse_ok is False
        assert result.error_field == "DIRECTION"

    def test_empty_input(self):
        result = parse_training_output("")
        assert result.parse_ok is False

    def test_garbled_input(self):
        result = parse_training_output("This is just random text with no fields.")
        assert result.parse_ok is False
        assert result.error_field == "CREDIT_RELEVANT"


class TestParseEdgeCases:
    """Test parser robustness on edge cases."""

    def test_extra_whitespace(self):
        """Extra spaces around field values should still parse."""
        text = (
            "CREDIT_RELEVANT:   Yes  \n"
            "DIRECTION:  Deterioration  \n"
            "SIGNAL_TYPE:  asset_quality  \n"
            "SECTOR_WIDE:  No  \n"
            "CONFIDENCE:  High  \n"
            "REASONING:  Some text here.  \n"
            "END"
        )
        result = parse_training_output(text)
        assert result.parse_ok is True
        assert result.credit_relevant is True
        assert result.direction == "Deterioration"

    def test_reasoning_with_colons(self):
        """REASONING field can contain colons (e.g., 'NPA ratio: 11.4%')."""
        text = (
            "CREDIT_RELEVANT: Yes\n"
            "DIRECTION: Deterioration\n"
            "SIGNAL_TYPE: asset_quality\n"
            "SECTOR_WIDE: No\n"
            "CONFIDENCE: High\n"
            "REASONING: NPA ratio: 11.4% vs 8.2% last quarter: deteriorating trend.\n"
            "END"
        )
        result = parse_training_output(text)
        assert result.parse_ok is True
        assert "11.4%" in result.reasoning
        assert "8.2%" in result.reasoning

    def test_blank_lines_between_fields(self):
        """Blank lines between fields should be ignored."""
        text = (
            "CREDIT_RELEVANT: No\n"
            "\n"
            "REASONING: Not relevant.\n"
            "\n"
            "END"
        )
        result = parse_training_output(text)
        assert result.parse_ok is True
        assert result.credit_relevant is False

    def test_multiline_reasoning(self):
        """Reasoning that spans multiple lines should be concatenated."""
        text = (
            "CREDIT_RELEVANT: Yes\n"
            "DIRECTION: Deterioration\n"
            "SIGNAL_TYPE: governance\n"
            "SECTOR_WIDE: No\n"
            "CONFIDENCE: Medium\n"
            "REASONING: CEO forced out by RBI.\n"
            "Board reshuffle indicates regulatory concern.\n"
            "END"
        )
        result = parse_training_output(text)
        assert result.parse_ok is True
        assert "CEO forced out" in result.reasoning
        assert "Board reshuffle" in result.reasoning


# ============================================================
# Format Output Text Tests
# ============================================================


class TestFormatOutputText:
    """Test the inverse operation: label fields → structured text."""

    def test_full_form_deterioration(self):
        text = format_output_text(
            credit_relevant=1,
            signal_direction=-1,
            signal_type="asset_quality",
            sector_wide=0,
            confidence="high",
            reasoning="NPA ratio rising.",
        )
        assert "CREDIT_RELEVANT: Yes" in text
        assert "DIRECTION: Deterioration" in text
        assert "SIGNAL_TYPE: asset_quality" in text
        assert "SECTOR_WIDE: No" in text
        assert "CONFIDENCE: High" in text
        assert "REASONING: NPA ratio rising." in text
        assert text.endswith("END")

    def test_short_form_not_relevant(self):
        text = format_output_text(
            credit_relevant=0,
            signal_direction=0,
            signal_type="other",
            sector_wide=0,
            confidence="high",
            reasoning="Stock price noise.",
        )
        assert "CREDIT_RELEVANT: No" in text
        assert "REASONING: Stock price noise." in text
        assert "DIRECTION" not in text
        assert "SIGNAL_TYPE" not in text
        assert text.endswith("END")

    def test_roundtrip_full_form(self):
        """format_output_text → parse_training_output should roundtrip."""
        text = format_output_text(
            credit_relevant=1,
            signal_direction=1,
            signal_type="funding",
            sector_wide=1,
            confidence="medium",
            reasoning="Successful NCD issuance.",
        )
        result = parse_training_output(text)
        assert result.parse_ok is True
        assert result.credit_relevant is True
        assert result.direction == "Improvement"
        assert result.signal_type == "funding"
        assert result.sector_wide is True
        assert result.confidence == "Medium"

    def test_roundtrip_short_form(self):
        """Short-form should also roundtrip."""
        text = format_output_text(
            credit_relevant=0,
            signal_direction=0,
            signal_type="other",
            sector_wide=0,
            confidence="low",
            reasoning="Award ceremony coverage.",
        )
        result = parse_training_output(text)
        assert result.parse_ok is True
        assert result.credit_relevant is False
        assert result.reasoning == "Award ceremony coverage."


# ============================================================
# Formatter Utility Tests
# ============================================================


class TestTruncateText:
    def test_short_text_unchanged(self):
        assert truncate_text("hello world", 100) == "hello world"

    def test_long_text_truncated(self):
        text = "word " * 1000  # 5000 chars
        result = truncate_text(text, 50)
        assert len(result) <= 54  # 50 + "..."
        assert result.endswith("...")

    def test_word_boundary_break(self):
        text = "The quick brown fox jumps over"
        result = truncate_text(text, 15)
        # Should break at a word boundary, not mid-word
        assert "..." in result
        assert len(result) <= 18


class TestFormatInput:
    def test_format_input_structure(self):
        result = format_input("YES Bank", "2018-09-19", "CEO steps down", "Article body here.")
        assert "Entity: YES Bank" in result
        assert "Date: 2018-09-19" in result
        assert "Title: CEO steps down" in result
        assert "Article: Article body here." in result


class TestParseDate:
    def test_valid_date(self):
        from datetime import date
        assert parse_date("2021-12-31") == date(2021, 12, 31)

    def test_invalid_date(self):
        assert parse_date("not-a-date") is None

    def test_empty_date(self):
        assert parse_date("") is None


class TestBuildExample:
    def test_builds_correct_structure(self):
        label = ArticleLabel(
            url="http://example.com/1",
            credit_relevant=1,
            signal_direction=-1,
            signal_type="liquidity",
            sector_wide=0,
            confidence="high",
            reasoning="Liquidity crunch reported.",
        )
        article = {
            "title": "NBFC faces liquidity crunch",
            "date": "2019-03-15",
            "entities": "Test NBFC",
            "text": "Full article body here.",
        }
        ex = build_example(
            label=label,
            article=article,
            instruction="Assess credit quality.",
            max_chars=3000,
            direction_map={-1: "Deterioration", 0: "Neutral", 1: "Improvement"},
            confidence_map={"low": "Low", "medium": "Medium", "high": "High"},
        )
        assert ex is not None
        assert ex["instruction"] == "Assess credit quality."
        assert "Entity: Test NBFC" in ex["input"]
        assert "CREDIT_RELEVANT: Yes" in ex["output"]
        assert "DIRECTION: Deterioration" in ex["output"]
        assert ex["meta"]["url"] == "http://example.com/1"
        assert ex["meta"]["credit_relevant"] == 1

    def test_returns_none_for_empty_text(self):
        label = ArticleLabel(url="http://example.com/2")
        article = {"title": "Test", "date": "2020-01-01", "entities": "X", "text": ""}
        ex = build_example(
            label=label, article=article, instruction="Test",
            max_chars=3000,
            direction_map={-1: "Deterioration", 0: "Neutral", 1: "Improvement"},
            confidence_map={"low": "Low", "medium": "Medium", "high": "High"},
        )
        assert ex is None


# ============================================================
# Split Tests
# ============================================================


class TestSplitExamples:
    """Test temporal splitting and entity holdout logic."""

    def _make_example(self, entity: str, date_str: str, cr: int = 0, sd: int = 0) -> dict:
        return {
            "instruction": "test",
            "input": "test",
            "output": "test",
            "meta": {
                "url": f"http://example.com/{entity}/{date_str}",
                "entity": entity,
                "date": date_str,
                "credit_relevant": cr,
                "signal_direction": sd,
                "signal_type": "other",
            },
        }

    def test_temporal_split_boundaries(self):
        examples = [
            self._make_example("Bajaj Finance", "2021-12-31"),  # train (on boundary)
            self._make_example("Bajaj Finance", "2022-01-01"),  # val
            self._make_example("Bajaj Finance", "2023-06-30"),  # val (on boundary)
            self._make_example("Bajaj Finance", "2023-07-01"),  # test
        ]
        train, val, test, holdout = split_examples(
            examples, "2021-12-31", "2023-06-30", []
        )
        assert len(train) == 1
        assert len(val) == 2
        assert len(test) == 1
        assert len(holdout) == 0

    def test_entity_holdout_single_entity(self):
        examples = [
            self._make_example("DHFL", "2019-06-01"),
            self._make_example("Bajaj Finance", "2019-06-01"),
        ]
        train, val, test, holdout = split_examples(
            examples, "2021-12-31", "2023-06-30", ["DHFL"]
        )
        assert len(holdout) == 1
        assert holdout[0]["meta"]["entity"] == "DHFL"
        assert len(train) == 1

    def test_multi_entity_stays_in_main_split(self):
        """Multi-entity articles mentioning held-out entities stay in training."""
        examples = [
            self._make_example("DHFL,Shriram Finance", "2019-06-01"),
        ]
        train, val, test, holdout = split_examples(
            examples, "2021-12-31", "2023-06-30", ["DHFL"]
        )
        # Multi-entity article should stay in train, NOT go to holdout
        assert len(holdout) == 0
        assert len(train) == 1

    def test_all_three_holdout_entities(self):
        examples = [
            self._make_example("DHFL", "2019-01-01"),
            self._make_example("Reliance Capital", "2020-01-01"),
            self._make_example("Cholamandalam", "2021-01-01"),
            self._make_example("Bajaj Finance", "2019-01-01"),
        ]
        train, val, test, holdout = split_examples(
            examples, "2021-12-31", "2023-06-30",
            ["DHFL", "Reliance Capital", "Cholamandalam"]
        )
        assert len(holdout) == 3
        assert len(train) == 1
        holdout_entities = {ex["meta"]["entity"] for ex in holdout}
        assert holdout_entities == {"DHFL", "Reliance Capital", "Cholamandalam"}

    def test_unparseable_date_skipped(self):
        examples = [
            self._make_example("Bajaj Finance", "bad-date"),
            self._make_example("Bajaj Finance", "2020-01-01"),
        ]
        train, val, test, holdout = split_examples(
            examples, "2021-12-31", "2023-06-30", []
        )
        assert len(train) == 1  # Only the valid-date example
