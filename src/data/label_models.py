# WHY THIS: Typed dataclasses + bulletproof JSON parsing for LLM label output.
# LLMs are unreliable JSON generators — they wrap in markdown fences, output
# "yes" instead of 1, invent signal_type values, etc. This module absorbs ALL
# that messiness so the rest of the pipeline works with clean, typed objects.
# Separate from models.py because these are labeling-specific, not rating data.

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# --- Enums ---

class SignalType(str, Enum):
    """Credit signal categories matching PLAN.md taxonomy."""
    LIQUIDITY = "liquidity"
    ASSET_QUALITY = "asset_quality"
    REGULATORY = "regulatory"
    CONTAGION = "contagion"
    GOVERNANCE = "governance"
    FUNDING = "funding"
    OPERATIONAL = "operational"
    OTHER = "other"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# --- Label Dataclass ---

@dataclass
class ArticleLabel:
    """One LLM-generated label for a GDELT article.

    Every field has a safe default so we can always construct a label,
    even from partially-parsed LLM output. The parse_error field captures
    what went wrong (if anything) for debugging.
    """
    url: str
    credit_relevant: int = 0               # 0 or 1
    signal_direction: int = 0              # -1, 0, +1
    signal_type: str = "other"             # SignalType value
    sector_wide: int = 0                   # 0 or 1
    confidence: str = "low"                # Confidence value
    reasoning: str = ""
    model: str = ""                        # Which Claude model produced this
    phase: str = ""                        # calibration, bulk, audit
    parse_error: str | None = None         # None = clean parse
    raw_response: str = ""                 # Full LLM response for debugging

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- JSON Parsing (the hard part) ---

# 🎓 LLMs produce "almost-JSON" more often than you'd expect:
#   - Wrapped in ```json ... ``` markdown fences
#   - "yes"/"no" instead of 1/0
#   - "deterioration" instead of -1
#   - Invented signal_type values like "financial_stress"
# This parser handles ALL of those without raising exceptions.

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_VALID_SIGNAL_TYPES = {t.value for t in SignalType}
_VALID_CONFIDENCES = {c.value for c in Confidence}

# Coercion maps for common LLM deviations
_BOOL_COERCE = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
_DIRECTION_COERCE = {
    "deterioration": -1, "negative": -1, "decline": -1, "worsening": -1,
    "improvement": 1, "positive": 1, "upgrade": 1,
    "neutral": 0, "mixed": 0, "stable": 0, "none": 0,
    "-1": -1, "0": 0, "1": 1,
}


def _coerce_int(val: Any, coerce_map: dict[str, int] | None = None) -> int | None:
    """Try to make val an int, using coercion map for strings."""
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if coerce_map and s in coerce_map:
            return coerce_map[s]
        try:
            return int(s)
        except ValueError:
            return None
    return None


def _coerce_signal_type(val: Any) -> str:
    """Map LLM signal_type to valid enum, defaulting to 'other'."""
    if not isinstance(val, str):
        return "other"
    s = val.strip().lower().replace(" ", "_").replace("-", "_")
    if s in _VALID_SIGNAL_TYPES:
        return s
    # Common LLM inventions → closest match
    if "liquid" in s:
        return "liquidity"
    if "asset" in s or "npa" in s or "quality" in s:
        return "asset_quality"
    if "regulat" in s or "rbi" in s or "compliance" in s:
        return "regulatory"
    if "contag" in s or "sector" in s or "systemic" in s:
        return "contagion"
    if "govern" in s or "management" in s or "fraud" in s:
        return "governance"
    if "fund" in s or "debt" in s or "borrow" in s or "capital" in s:
        return "funding"
    if "operat" in s:
        return "operational"
    return "other"


def _coerce_confidence(val: Any) -> str:
    """Map LLM confidence to valid enum."""
    if not isinstance(val, str):
        return "low"
    s = val.strip().lower()
    if s in _VALID_CONFIDENCES:
        return s
    return "low"


def parse_llm_response(raw: str, url: str, model: str = "", phase: str = "") -> ArticleLabel:
    """Parse raw LLM text into an ArticleLabel. NEVER raises — always returns a label.

    🎓 WHY never-raise: In a 17K-article pipeline, one bad LLM response shouldn't
    crash the whole run. We record the error in parse_error and move on.
    After the run, we can filter for parse_error != None and inspect failures.
    """
    label = ArticleLabel(url=url, model=model, phase=phase, raw_response=raw)

    # Step 1: Strip markdown fences if present
    text = raw.strip()
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1)

    # Step 2: Try JSON parse
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # Last resort: find the first { ... } in the text
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                data = json.loads(text[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                label.parse_error = f"JSON decode failed: {e}"
                return label
        else:
            label.parse_error = f"No JSON object found: {e}"
            return label

    if not isinstance(data, dict):
        label.parse_error = f"Expected dict, got {type(data).__name__}"
        return label

    # Step 3: Extract and coerce each field
    cr = _coerce_int(data.get("credit_relevant"), _BOOL_COERCE)
    if cr is not None:
        label.credit_relevant = 1 if cr else 0

    sd = _coerce_int(data.get("signal_direction"), _DIRECTION_COERCE)
    if sd is not None:
        label.signal_direction = max(-1, min(1, sd))  # clamp to [-1, 1]

    label.signal_type = _coerce_signal_type(data.get("signal_type"))
    label.confidence = _coerce_confidence(data.get("confidence"))

    sw = _coerce_int(data.get("sector_wide"), _BOOL_COERCE)
    if sw is not None:
        label.sector_wide = 1 if sw else 0

    reasoning = data.get("reasoning", "")
    if isinstance(reasoning, str):
        label.reasoning = reasoning.strip()

    # Step 4: Enforce consistency — if not credit-relevant, zero out signals
    if label.credit_relevant == 0:
        label.signal_direction = 0
        label.signal_type = "other"
        label.sector_wide = 0

    return label


# --- JSONL I/O ---
# 🎓 JSONL (JSON Lines) = one JSON object per line. Crash-safe because each
# line is independently valid. If the process dies mid-write, you lose at most
# the last line, not the whole file. This is standard for ML data pipelines.

def write_label_jsonl(label: ArticleLabel, path: Path) -> None:
    """Append one label to a JSONL file. Creates file if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(label.to_dict(), ensure_ascii=False) + "\n")


def read_labels_jsonl(path: Path) -> list[ArticleLabel]:
    """Read all labels from a JSONL file. Skips malformed lines."""
    if not path.exists():
        return []
    labels: list[ArticleLabel] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                labels.append(ArticleLabel(**{
                    k: v for k, v in data.items()
                    if k in ArticleLabel.__dataclass_fields__
                }))
            except (json.JSONDecodeError, TypeError):
                # Skip corrupt lines — log but don't crash
                pass
    return labels


def get_completed_urls(path: Path) -> set[str]:
    """Get URLs already labeled in a JSONL file (for resume support)."""
    labels = read_labels_jsonl(path)
    return {label.url for label in labels}
