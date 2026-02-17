# WHY THIS: Strict parser for the structured text output format that the fine-tuned
# model will produce. Unlike the JSON parser in label_models.py (which is forgiving —
# coerces "yes"→1, maps invented signal types), this parser REJECTS anything outside
# the strict vocabulary. Why strict? Because this validates *model* output, not LLM
# labeling output. If the fine-tuned model drifts from the vocabulary, we want to
# know immediately — not silently coerce it into something plausible.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# --- Load vocabulary from config ---

def _load_vocab(config_path: Path | None = None) -> dict[str, set[str]]:
    """Load strict vocabulary sets from training_config.yaml."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "training_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    vocab = cfg.get("vocabulary", {})
    return {k: set(v) for k, v in vocab.items()}


# --- Parsed Output Dataclass ---

@dataclass
class ParsedOutput:
    """Result of parsing a model's structured text output.

    If parse_ok is True, all fields are populated and valid.
    If parse_ok is False, error_field and error_detail explain what broke.
    Partial fields may still be populated up to the point of failure.
    """
    credit_relevant: bool = False
    direction: str = ""
    signal_type: str = ""
    sector_wide: bool = False
    confidence: str = ""
    reasoning: str = ""
    has_end_token: bool = False

    parse_ok: bool = False
    error_field: str = ""         # Which field caused the error
    error_detail: str = ""        # What went wrong


# 🎓 CONCEPT: Why a strict parser separate from the forgiving one?
#
# label_models.py parser = FORGIVING. It handles messy LLM output during
# the labeling phase (Phase 1.3). "Yes"→1, "financial_stress"→"asset_quality".
# That's appropriate when you're building training data and want to salvage
# as much as possible from 17K API calls.
#
# This parser = STRICT. It validates what the fine-tuned model produces.
# If the model says "Negative" instead of "Deterioration", that's a training
# failure we need to catch, not silently fix. Strict parsing is your early
# warning system for model quality issues.


def parse_training_output(
    text: str,
    vocab: dict[str, set[str]] | None = None,
) -> ParsedOutput:
    """Parse structured text output into a ParsedOutput.

    Handles both full-form (credit-relevant) and short-form (not credit-relevant).

    Full-form:
        CREDIT_RELEVANT: Yes
        DIRECTION: Deterioration
        SIGNAL_TYPE: asset_quality
        SECTOR_WIDE: No
        CONFIDENCE: High
        REASONING: Some explanation here.
        END

    Short-form:
        CREDIT_RELEVANT: No
        REASONING: Not credit-relevant because...
        END
    """
    if vocab is None:
        vocab = _load_vocab()

    result = ParsedOutput()
    text = text.strip()

    # Check for END token
    if text.endswith("END"):
        result.has_end_token = True
        text = text[:-3].strip()
    else:
        result.error_field = "END"
        result.error_detail = "Missing END token"
        # Continue parsing — we want to know what else is wrong too,
        # but we'll record this error if nothing worse is found

    # Parse line by line into a dict of field → value
    # 🎓 We use a simple key: value split on ": " rather than regex because
    # the REASONING field can contain colons (e.g., "NPA ratio: 11.4%")
    fields: dict[str, str] = {}
    current_key: str | None = None
    current_val: str = ""

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Try to match a FIELD: VALUE pattern
        # Only match known field names to avoid splitting REASONING on colons
        matched = False
        for field_name in ("CREDIT_RELEVANT", "DIRECTION", "SIGNAL_TYPE",
                           "SECTOR_WIDE", "CONFIDENCE", "REASONING"):
            prefix = field_name + ":"
            if line.upper().startswith(prefix.upper()):
                # Save previous field
                if current_key is not None:
                    fields[current_key] = current_val.strip()
                current_key = field_name
                current_val = line[len(prefix):].strip()
                matched = True
                break

        if not matched and current_key is not None:
            # Continuation of previous field (multi-line REASONING)
            current_val += " " + line

    # Save last field
    if current_key is not None:
        fields[current_key] = current_val.strip()

    # --- Validate CREDIT_RELEVANT (required) ---
    cr_val = fields.get("CREDIT_RELEVANT", "").strip()
    if not cr_val:
        result.error_field = "CREDIT_RELEVANT"
        result.error_detail = "Missing CREDIT_RELEVANT field"
        return result

    if cr_val not in vocab.get("credit_relevant", {"Yes", "No"}):
        result.error_field = "CREDIT_RELEVANT"
        result.error_detail = f"Invalid value '{cr_val}', expected Yes/No"
        return result

    result.credit_relevant = (cr_val == "Yes")

    # --- Short-form path (not credit-relevant) ---
    if not result.credit_relevant:
        reasoning = fields.get("REASONING", "").strip()
        if not reasoning:
            result.error_field = "REASONING"
            result.error_detail = "Missing REASONING for non-credit-relevant article"
            return result
        result.reasoning = reasoning
        # Short-form is valid if we have CR=No + REASONING (+ END checked above)
        if result.has_end_token:
            result.parse_ok = True
        else:
            # END was missing — that's the only error
            pass
        return result

    # --- Full-form path (credit-relevant) ---
    # Validate DIRECTION
    dir_val = fields.get("DIRECTION", "").strip()
    if not dir_val:
        result.error_field = "DIRECTION"
        result.error_detail = "Missing DIRECTION field for credit-relevant article"
        return result
    if dir_val not in vocab.get("direction", set()):
        result.error_field = "DIRECTION"
        result.error_detail = f"Invalid value '{dir_val}', expected one of: {sorted(vocab.get('direction', set()))}"
        return result
    result.direction = dir_val

    # Validate SIGNAL_TYPE
    st_val = fields.get("SIGNAL_TYPE", "").strip()
    if not st_val:
        result.error_field = "SIGNAL_TYPE"
        result.error_detail = "Missing SIGNAL_TYPE field for credit-relevant article"
        return result
    if st_val not in vocab.get("signal_type", set()):
        result.error_field = "SIGNAL_TYPE"
        result.error_detail = f"Invalid value '{st_val}', expected one of: {sorted(vocab.get('signal_type', set()))}"
        return result
    result.signal_type = st_val

    # Validate SECTOR_WIDE
    sw_val = fields.get("SECTOR_WIDE", "").strip()
    if not sw_val:
        result.error_field = "SECTOR_WIDE"
        result.error_detail = "Missing SECTOR_WIDE field for credit-relevant article"
        return result
    if sw_val not in vocab.get("sector_wide", {"Yes", "No"}):
        result.error_field = "SECTOR_WIDE"
        result.error_detail = f"Invalid value '{sw_val}', expected Yes/No"
        return result
    result.sector_wide = (sw_val == "Yes")

    # Validate CONFIDENCE
    conf_val = fields.get("CONFIDENCE", "").strip()
    if not conf_val:
        result.error_field = "CONFIDENCE"
        result.error_detail = "Missing CONFIDENCE field for credit-relevant article"
        return result
    if conf_val not in vocab.get("confidence", set()):
        result.error_field = "CONFIDENCE"
        result.error_detail = f"Invalid value '{conf_val}', expected one of: {sorted(vocab.get('confidence', set()))}"
        return result
    result.confidence = conf_val

    # Validate REASONING
    reasoning = fields.get("REASONING", "").strip()
    if not reasoning:
        result.error_field = "REASONING"
        result.error_detail = "Missing REASONING field for credit-relevant article"
        return result
    result.reasoning = reasoning

    # All fields valid — check END
    if result.has_end_token:
        result.parse_ok = True
    # If END was missing, error_field/error_detail already set above

    return result


# --- Format output text from label data ---

def format_output_text(
    credit_relevant: int,
    signal_direction: int,
    signal_type: str,
    sector_wide: int,
    confidence: str,
    reasoning: str,
    direction_map: dict[int, str] | None = None,
    confidence_map: dict[str, str] | None = None,
) -> str:
    """Convert label fields into structured text training output.

    This is the inverse of parse_training_output — it produces the text
    that the model will learn to generate.
    """
    if direction_map is None:
        direction_map = {-1: "Deterioration", 0: "Neutral", 1: "Improvement"}
    if confidence_map is None:
        confidence_map = {"low": "Low", "medium": "Medium", "high": "High"}

    if credit_relevant == 0:
        # Short-form: just CR=No + reasoning + END
        return f"CREDIT_RELEVANT: No\nREASONING: {reasoning}\nEND"

    # Full-form
    lines = [
        f"CREDIT_RELEVANT: Yes",
        f"DIRECTION: {direction_map.get(signal_direction, 'Neutral')}",
        f"SIGNAL_TYPE: {signal_type}",
        f"SECTOR_WIDE: {'Yes' if sector_wide else 'No'}",
        f"CONFIDENCE: {confidence_map.get(confidence, 'Low')}",
        f"REASONING: {reasoning}",
        "END",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick demo: parse a sample output
    sample_full = """CREDIT_RELEVANT: Yes
DIRECTION: Deterioration
SIGNAL_TYPE: asset_quality
SECTOR_WIDE: No
CONFIDENCE: High
REASONING: 11.4% of PFC's loan book entering insolvency with write-off risk.
END"""

    sample_short = """CREDIT_RELEVANT: No
REASONING: Stock price movements with no credit quality information.
END"""

    for label, text in [("Full-form", sample_full), ("Short-form", sample_short)]:
        result = parse_training_output(text)
        print(f"\n{label}:")
        print(f"  parse_ok={result.parse_ok}")
        print(f"  credit_relevant={result.credit_relevant}")
        if result.credit_relevant:
            print(f"  direction={result.direction}")
            print(f"  signal_type={result.signal_type}")
            print(f"  confidence={result.confidence}")
        print(f"  reasoning={result.reasoning[:60]}...")
        if not result.parse_ok:
            print(f"  ERROR: {result.error_field}: {result.error_detail}")
