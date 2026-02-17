# WHY THIS: Canonical evaluation module for both base-model robustness testing (Phase 2.1)
# and post-training model evaluation (Phase 2.4). Lives here so both the Colab notebook
# and production eval scripts import from the same source of truth.
#
# Key design decision: per-entity reporting is mandatory for holdout evaluation.
# DHFL (1,243 articles, 91% deterioration) dominates aggregate metrics and hides
# Cholamandalam (1,372 articles, 12% deterioration) performance. If you only report
# aggregate F1, a model that just says "Deterioration" for everything gets ~65% accuracy
# on holdout but fails completely on the false-positive-control entity.

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ============================================================
# Strict Output Parser (copied from src/data/parse_training_output.py)
# 🎓 SOURCE OF TRUTH: src/data/parse_training_output.py
# If the parser logic changes, update it there first, then here.
# The notebook also copies this — three places to keep in sync.
# ============================================================

# --- Vocabulary (hardcoded here to avoid YAML dependency on Colab) ---
# These MUST match configs/training_config.yaml → vocabulary

VOCAB: dict[str, set[str]] = {
    "credit_relevant": {"Yes", "No"},
    "direction": {"Deterioration", "Improvement", "Neutral"},
    "signal_type": {
        "liquidity", "asset_quality", "regulatory", "contagion",
        "governance", "funding", "operational", "other",
    },
    "sector_wide": {"Yes", "No"},
    "confidence": {"Low", "Medium", "High"},
}


@dataclass
class ParsedOutput:
    """Result of parsing a model's structured text output."""
    credit_relevant: bool = False
    direction: str = ""
    signal_type: str = ""
    sector_wide: bool = False
    confidence: str = ""
    reasoning: str = ""
    has_end_token: bool = False

    parse_ok: bool = False
    error_field: str = ""
    error_detail: str = ""


def parse_model_output(
    text: str,
    vocab: dict[str, set[str]] | None = None,
) -> ParsedOutput:
    """Parse structured text output into a ParsedOutput.

    This is functionally identical to parse_training_output() in
    src/data/parse_training_output.py, but uses hardcoded vocab
    so it works standalone on Colab without YAML config access.
    """
    if vocab is None:
        vocab = VOCAB

    result = ParsedOutput()
    text = text.strip()

    # Check for END token
    if text.endswith("END"):
        result.has_end_token = True
        text = text[:-3].strip()
    else:
        result.error_field = "END"
        result.error_detail = "Missing END token"

    # Parse line by line into field → value dict
    fields: dict[str, str] = {}
    current_key: str | None = None
    current_val: str = ""

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        matched = False
        for field_name in ("CREDIT_RELEVANT", "DIRECTION", "SIGNAL_TYPE",
                           "SECTOR_WIDE", "CONFIDENCE", "REASONING"):
            prefix = field_name + ":"
            if line.upper().startswith(prefix.upper()):
                if current_key is not None:
                    fields[current_key] = current_val.strip()
                current_key = field_name
                current_val = line[len(prefix):].strip()
                matched = True
                break

        if not matched and current_key is not None:
            current_val += " " + line

    if current_key is not None:
        fields[current_key] = current_val.strip()

    # --- Validate CREDIT_RELEVANT ---
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

    # --- Short-form path ---
    if not result.credit_relevant:
        reasoning = fields.get("REASONING", "").strip()
        if not reasoning:
            result.error_field = "REASONING"
            result.error_detail = "Missing REASONING for non-credit-relevant article"
            return result
        result.reasoning = reasoning
        if result.has_end_token:
            result.parse_ok = True
        return result

    # --- Full-form path ---
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

    reasoning = fields.get("REASONING", "").strip()
    if not reasoning:
        result.error_field = "REASONING"
        result.error_detail = "Missing REASONING field for credit-relevant article"
        return result
    result.reasoning = reasoning

    if result.has_end_token:
        result.parse_ok = True

    return result


# ============================================================
# Failure Mode Taxonomy
# ============================================================

# 🎓 CONCEPT: Why categorize failures instead of just counting them?
#
# A 30% parse failure rate means very different things depending on WHY:
# - 30% "wrong vocab" (model says "Negative" not "Deterioration") → fixable
#   with more training data showing correct vocab, or a relaxed parser
# - 30% "totally unstructured" (model writes a paragraph instead of fields)
#   → format is too unfamiliar, might need format simplification
# - 30% "missing END" but otherwise correct → trivially fixable, just strip
#
# The failure taxonomy tells you WHAT to fix, not just WHETHER to fix.

FAILURE_BUCKETS = [
    "missing_field",       # Required field absent (CREDIT_RELEVANT, DIRECTION, etc.)
    "wrong_vocab",         # Field present but value not in vocabulary
    "missing_end",         # Everything correct except no END token
    "extra_content",       # Model added extra text/commentary beyond the format
    "refusal",             # Model refused or added meta-commentary ("As an AI...")
    "totally_unstructured", # No recognizable field structure at all
    "partial_format",      # Some fields present, some missing (structural issue)
]


def classify_failure(raw_output: str, parsed: ParsedOutput) -> str:
    """Classify a parse failure into a taxonomy bucket.

    Only call this when parsed.parse_ok is False.
    """
    text = raw_output.strip()

    # Check for AI refusals/meta-commentary
    refusal_patterns = [
        r"(?i)as an (ai|language model|assistant)",
        r"(?i)i cannot",
        r"(?i)i'm not able to",
        r"(?i)i don't have",
        r"(?i)i apologize",
        r"(?i)sorry,?\s+(?:but\s+)?i",
    ]
    for pattern in refusal_patterns:
        if re.search(pattern, text):
            return "refusal"

    # Check if ANY known field names appear at all
    known_fields = {"CREDIT_RELEVANT", "DIRECTION", "SIGNAL_TYPE",
                    "SECTOR_WIDE", "CONFIDENCE", "REASONING"}
    found_fields = set()
    for field_name in known_fields:
        if field_name + ":" in text.upper():
            found_fields.add(field_name)

    if not found_fields:
        return "totally_unstructured"

    # If only error is missing END, that's a specific bucket
    if parsed.error_field == "END" and parsed.error_detail == "Missing END token":
        # Check if fields were otherwise valid by re-parsing with END appended
        test_parsed = parse_model_output(text + "\nEND")
        if test_parsed.parse_ok:
            return "missing_end"

    # Wrong vocab: field exists but value is invalid
    if "Invalid value" in parsed.error_detail:
        return "wrong_vocab"

    # Missing field: field expected but not found
    if "Missing" in parsed.error_detail and parsed.error_field in known_fields:
        # Distinguish between "field totally absent" vs "partial format"
        if len(found_fields) >= 2:
            return "partial_format"
        return "missing_field"

    # Check for extra content after END or between fields
    lines = text.split("\n")
    non_field_lines = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped or line_stripped == "END":
            continue
        is_field = any(line_stripped.upper().startswith(f + ":") for f in known_fields)
        if not is_field:
            non_field_lines.append(line_stripped)
    if non_field_lines and len(found_fields) >= 3:
        return "extra_content"

    # Default: partial format (some structure but not enough)
    return "partial_format"


# ============================================================
# Per-field Accuracy
# ============================================================

@dataclass
class FieldAccuracy:
    """Accuracy metrics for a single field across multiple examples."""
    total: int = 0
    correct: int = 0
    confusion: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


def compute_field_accuracy(
    predictions: list[ParsedOutput],
    ground_truth: list[dict],
) -> dict[str, FieldAccuracy]:
    """Compute per-field accuracy for successfully parsed outputs.

    Args:
        predictions: List of ParsedOutput from parse_model_output().
        ground_truth: List of dicts with expected output text (the 'output' field
                     from training JSONL). These get parsed with the same parser
                     to extract ground truth values.

    Returns:
        Dict mapping field name → FieldAccuracy with accuracy and confusion matrix.
    """
    # 🎓 We parse ground truth through the same parser to get structured values.
    # This ensures we're comparing apples to apples — no manual mapping errors.
    metrics: dict[str, FieldAccuracy] = {
        "credit_relevant": FieldAccuracy(),
        "direction": FieldAccuracy(),
        "signal_type": FieldAccuracy(),
        "sector_wide": FieldAccuracy(),
        "confidence": FieldAccuracy(),
    }

    for pred, gt_dict in zip(predictions, ground_truth):
        if not pred.parse_ok:
            continue

        # Parse ground truth output text
        gt_text = gt_dict.get("output", "")
        gt = parse_model_output(gt_text)
        if not gt.parse_ok:
            continue

        # credit_relevant (bool → str for confusion matrix)
        pred_cr = "Yes" if pred.credit_relevant else "No"
        gt_cr = "Yes" if gt.credit_relevant else "No"
        metrics["credit_relevant"].total += 1
        if pred_cr == gt_cr:
            metrics["credit_relevant"].correct += 1
        metrics["credit_relevant"].confusion[gt_cr][pred_cr] += 1

        # Only compare other fields if both say credit_relevant
        if pred.credit_relevant and gt.credit_relevant:
            # direction
            metrics["direction"].total += 1
            if pred.direction == gt.direction:
                metrics["direction"].correct += 1
            metrics["direction"].confusion[gt.direction][pred.direction] += 1

            # signal_type
            metrics["signal_type"].total += 1
            if pred.signal_type == gt.signal_type:
                metrics["signal_type"].correct += 1
            metrics["signal_type"].confusion[gt.signal_type][pred.signal_type] += 1

            # sector_wide
            pred_sw = "Yes" if pred.sector_wide else "No"
            gt_sw = "Yes" if gt.sector_wide else "No"
            metrics["sector_wide"].total += 1
            if pred_sw == gt_sw:
                metrics["sector_wide"].correct += 1
            metrics["sector_wide"].confusion[gt_sw][pred_sw] += 1

            # confidence
            metrics["confidence"].total += 1
            if pred.confidence == gt.confidence:
                metrics["confidence"].correct += 1
            metrics["confidence"].confusion[gt.confidence][pred.confidence] += 1

    return metrics


# ============================================================
# Per-Entity Holdout Evaluation
# ============================================================

@dataclass
class EntityMetrics:
    """Evaluation metrics for a single holdout entity."""
    entity: str
    total: int = 0
    parsed: int = 0
    parse_rate: float = 0.0

    # Per-field accuracy (only for parsed examples)
    cr_accuracy: float = 0.0
    dir_accuracy: float = 0.0
    st_accuracy: float = 0.0

    # Direction-specific recall
    # 🎓 For DHFL (91% deterioration), overall accuracy is misleading.
    # What matters: does the model catch deterioration signals specifically?
    det_recall: float = 0.0    # Of actual deterioration, how many caught?
    det_precision: float = 0.0  # Of predicted deterioration, how many correct?
    imp_recall: float = 0.0
    imp_precision: float = 0.0

    failure_modes: dict[str, int] = field(default_factory=dict)


def evaluate_holdout_per_entity(
    predictions: list[ParsedOutput],
    raw_outputs: list[str],
    ground_truth: list[dict],
    entities: list[str],
) -> dict[str, EntityMetrics]:
    """Evaluate model performance per holdout entity.

    Args:
        predictions: ParsedOutput for each example.
        raw_outputs: Raw model output strings (for failure classification).
        ground_truth: Dicts with 'output' and 'input' fields from JSONL.
        entities: List of entity names (extracted from input text).

    Returns:
        Dict mapping entity name → EntityMetrics.

    🎓 WHY per-entity: DHFL has 1,243 articles (91% deterioration), Cholamandalam
    has 1,372 (12% deterioration). Aggregate metrics are dominated by DHFL's volume
    and extreme skew. A model that always says "Deterioration" would look great on
    aggregate holdout (65%+ accuracy) but catastrophically fail on Cholamandalam
    (88% false positive rate). Per-entity reporting catches this.
    """
    # Group by entity
    entity_groups: dict[str, list[int]] = defaultdict(list)
    for i, entity in enumerate(entities):
        entity_groups[entity].append(i)

    results: dict[str, EntityMetrics] = {}

    for entity, indices in entity_groups.items():
        em = EntityMetrics(entity=entity)
        em.total = len(indices)

        # Parse rate
        parsed_indices = [i for i in indices if predictions[i].parse_ok]
        em.parsed = len(parsed_indices)
        em.parse_rate = em.parsed / em.total if em.total > 0 else 0.0

        # Failure modes
        failure_counts: Counter = Counter()
        for i in indices:
            if not predictions[i].parse_ok:
                bucket = classify_failure(raw_outputs[i], predictions[i])
                failure_counts[bucket] += 1
        em.failure_modes = dict(failure_counts)

        # Per-field accuracy on parsed examples
        if parsed_indices:
            cr_correct = 0
            dir_correct = 0
            dir_total = 0
            st_correct = 0
            st_total = 0

            det_tp = 0  # True positive: predicted det, actually det
            det_fp = 0  # False positive: predicted det, actually not det
            det_fn = 0  # False negative: actually det, predicted not det
            imp_tp = 0
            imp_fp = 0
            imp_fn = 0

            for i in parsed_indices:
                gt_text = ground_truth[i].get("output", "")
                gt = parse_model_output(gt_text)
                if not gt.parse_ok:
                    continue

                # credit_relevant
                if predictions[i].credit_relevant == gt.credit_relevant:
                    cr_correct += 1

                # Fields only comparable when both say credit-relevant
                if predictions[i].credit_relevant and gt.credit_relevant:
                    dir_total += 1
                    if predictions[i].direction == gt.direction:
                        dir_correct += 1
                    st_total += 1
                    if predictions[i].signal_type == gt.signal_type:
                        st_correct += 1

                # Direction-specific precision/recall
                pred_det = (predictions[i].credit_relevant and
                           predictions[i].direction == "Deterioration")
                gt_det = (gt.credit_relevant and gt.direction == "Deterioration")
                pred_imp = (predictions[i].credit_relevant and
                           predictions[i].direction == "Improvement")
                gt_imp = (gt.credit_relevant and gt.direction == "Improvement")

                if pred_det and gt_det:
                    det_tp += 1
                elif pred_det and not gt_det:
                    det_fp += 1
                elif not pred_det and gt_det:
                    det_fn += 1

                if pred_imp and gt_imp:
                    imp_tp += 1
                elif pred_imp and not gt_imp:
                    imp_fp += 1
                elif not pred_imp and gt_imp:
                    imp_fn += 1

            em.cr_accuracy = cr_correct / len(parsed_indices) if parsed_indices else 0.0
            em.dir_accuracy = dir_correct / dir_total if dir_total > 0 else 0.0
            em.st_accuracy = st_correct / st_total if st_total > 0 else 0.0
            em.det_recall = det_tp / (det_tp + det_fn) if (det_tp + det_fn) > 0 else 0.0
            em.det_precision = det_tp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 0.0
            em.imp_recall = imp_tp / (imp_tp + imp_fn) if (imp_tp + imp_fn) > 0 else 0.0
            em.imp_precision = imp_tp / (imp_tp + imp_fp) if (imp_tp + imp_fp) > 0 else 0.0

        results[entity] = em

    return results


def extract_entity_from_input(input_text: str) -> str:
    """Extract entity name from training input text.

    Input format: "Entity: DHFL\\nDate: 2019-06-01\\nTitle: ...\\nArticle: ..."
    """
    for line in input_text.split("\n"):
        if line.startswith("Entity:"):
            return line[len("Entity:"):].strip()
    return "UNKNOWN"


# ============================================================
# Reporting
# ============================================================

def print_parse_report(
    predictions: list[ParsedOutput],
    raw_outputs: list[str],
) -> dict[str, Any]:
    """Print parse success report and return summary stats.

    Returns dict with: parse_rate, failure_counts, failure_examples.
    """
    total = len(predictions)
    parsed = sum(1 for p in predictions if p.parse_ok)
    failed = total - parsed
    parse_rate = parsed / total if total > 0 else 0.0

    print("=" * 70)
    print("PARSE RESULTS")
    print("=" * 70)
    print(f"  Total examples:     {total:>6,d}")
    print(f"  Successfully parsed:{parsed:>6,d}  ({parse_rate:.1%})")
    print(f"  Parse failures:     {failed:>6,d}  ({1 - parse_rate:.1%})")
    print()

    # Failure taxonomy
    failure_counts: Counter = Counter()
    failure_examples: dict[str, list[str]] = defaultdict(list)

    for pred, raw in zip(predictions, raw_outputs):
        if not pred.parse_ok:
            bucket = classify_failure(raw, pred)
            failure_counts[bucket] += 1
            if len(failure_examples[bucket]) < 3:
                # Truncate for display
                truncated = raw[:200] + "..." if len(raw) > 200 else raw
                failure_examples[bucket].append(truncated)

    if failure_counts:
        print("FAILURE MODE TAXONOMY:")
        print("-" * 70)
        for bucket, count in failure_counts.most_common():
            pct = count / failed * 100 if failed > 0 else 0
            print(f"  {bucket:25s}  {count:>5d}  ({pct:5.1f}% of failures)")
            for j, example in enumerate(failure_examples[bucket], 1):
                print(f"    Example {j}: {example}")
            print()

    return {
        "parse_rate": parse_rate,
        "total": total,
        "parsed": parsed,
        "failed": failed,
        "failure_counts": dict(failure_counts),
        "failure_examples": dict(failure_examples),
    }


def print_field_accuracy_report(metrics: dict[str, FieldAccuracy]) -> None:
    """Print per-field accuracy table."""
    print("=" * 70)
    print("PER-FIELD ACCURACY (on successfully parsed outputs)")
    print("=" * 70)
    print(f"  {'Field':20s}  {'Total':>6s}  {'Correct':>7s}  {'Accuracy':>8s}")
    print("-" * 50)
    for field_name, fa in metrics.items():
        if fa.total > 0:
            print(f"  {field_name:20s}  {fa.total:>6d}  {fa.correct:>7d}  {fa.accuracy:>8.1%}")
        else:
            print(f"  {field_name:20s}  {'N/A':>6s}  {'N/A':>7s}  {'N/A':>8s}")
    print()


def print_holdout_report(entity_metrics: dict[str, EntityMetrics]) -> None:
    """Print per-entity holdout evaluation table."""
    print("=" * 70)
    print("HOLDOUT ENTITY EVALUATION")
    print("=" * 70)

    for entity, em in entity_metrics.items():
        print(f"\n  {entity} ({em.total} articles)")
        print(f"  {'─' * 50}")
        print(f"    Parse rate:         {em.parse_rate:>6.1%}  ({em.parsed}/{em.total})")
        print(f"    CR accuracy:        {em.cr_accuracy:>6.1%}")
        print(f"    Direction accuracy:  {em.dir_accuracy:>6.1%}")
        print(f"    Signal type acc:    {em.st_accuracy:>6.1%}")
        print(f"    Det. precision:     {em.det_precision:>6.1%}")
        print(f"    Det. recall:        {em.det_recall:>6.1%}")
        print(f"    Imp. precision:     {em.imp_precision:>6.1%}")
        print(f"    Imp. recall:        {em.imp_recall:>6.1%}")
        if em.failure_modes:
            print(f"    Failure modes: {em.failure_modes}")
    print()


def print_decision(parse_rate: float) -> str:
    """Print GO/NO-GO decision based on parse success rate.

    Returns: "GO", "INVESTIGATE", or "NO-GO".
    """
    print("=" * 70)
    print("DECISION")
    print("=" * 70)

    if parse_rate >= 0.80:
        decision = "GO"
        print(f"  Parse rate: {parse_rate:.1%} >= 80%")
        print(f"  DECISION: GO — Proceed to LoRA training.")
        print(f"  The base model can already produce our output format most of the time.")
        print(f"  Fine-tuning will improve both format compliance and content accuracy.")
    elif parse_rate >= 0.20:
        decision = "INVESTIGATE"
        print(f"  Parse rate: {parse_rate:.1%} (between 20% and 80%)")
        print(f"  DECISION: INVESTIGATE — Review failure modes before proceeding.")
        print(f"  Check the failure taxonomy above:")
        print(f"    - If mostly 'missing_end' or 'wrong_vocab' → fixable, proceed with caution")
        print(f"    - If mostly 'totally_unstructured' → consider format simplification")
        print(f"    - If mostly 'refusal' → adjust the instruction prompt")
    else:
        decision = "NO-GO"
        print(f"  Parse rate: {parse_rate:.1%} < 20%")
        print(f"  DECISION: NO-GO — Format is too unfamiliar for the base model.")
        print(f"  Options:")
        print(f"    1. Simplify to fewer fields (just CREDIT_RELEVANT + DIRECTION)")
        print(f"    2. Switch to JSON output (models are more familiar with JSON)")
        print(f"    3. Try a different base model (LLaMA 3.1 or Mistral)")
        print(f"    4. Add output format examples to the instruction prompt")

    print()
    return decision


# ============================================================
# Data Loading Utilities
# ============================================================

def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stratified_sample(
    examples: list[dict],
    n: int,
    seed: int = 42,
) -> list[dict]:
    """Sample n examples, stratified by credit_relevant in the output field.

    🎓 WHY stratified: If we randomly sample 1,000 from train.jsonl, we'd get
    ~540 credit-relevant and ~460 not (matching the 54% base rate). That's fine
    for parse testing. But stratified 50/50 gives us equal coverage of both the
    full-form and short-form output formats, which is what we actually care about
    for format robustness testing.
    """
    import random
    rng = random.Random(seed)

    # Split by whether output starts with "CREDIT_RELEVANT: Yes" or "No"
    cr_yes = [ex for ex in examples if ex.get("output", "").startswith("CREDIT_RELEVANT: Yes")]
    cr_no = [ex for ex in examples if ex.get("output", "").startswith("CREDIT_RELEVANT: No")]

    half = n // 2
    # If one group is smaller than half, take all of it and fill from the other
    if len(cr_yes) < half:
        sample_yes = cr_yes[:]
        sample_no = rng.sample(cr_no, min(n - len(sample_yes), len(cr_no)))
    elif len(cr_no) < half:
        sample_no = cr_no[:]
        sample_yes = rng.sample(cr_yes, min(n - len(sample_no), len(cr_yes)))
    else:
        sample_yes = rng.sample(cr_yes, half)
        sample_no = rng.sample(cr_no, n - half)

    combined = sample_yes + sample_no
    rng.shuffle(combined)
    return combined


if __name__ == "__main__":
    # Quick self-test: parse a sample output and classify a failure
    sample = "CREDIT_RELEVANT: Yes\nDIRECTION: Deterioration\nSIGNAL_TYPE: asset_quality\nSECTOR_WIDE: No\nCONFIDENCE: High\nREASONING: Test.\nEND"
    result = parse_model_output(sample)
    print(f"Parse OK: {result.parse_ok}")
    assert result.parse_ok

    bad = "I cannot assess credit risk as I am an AI language model."
    bad_result = parse_model_output(bad)
    bucket = classify_failure(bad, bad_result)
    print(f"Failure bucket: {bucket}")
    assert bucket == "refusal"

    print("Self-test passed.")
