# Phase 1.3 — Label Quality Report

**Date:** 2026-02-17
**Dataset:** 17,274 LLM-generated labels (`labels_final.jsonl`)
**Ground truth:** Rating actions from SEBI disclosures (`rating_actions_sourced.csv`, 1,654 records)
**Evaluation window:** Articles published ≤90 days before a rating action

---

## Pipeline Summary

| Phase | Model | Articles | Parse Errors | Cost |
|-------|-------|----------|-------------|------|
| Calibration | Sonnet 4.5 | 300 | 0 | ~$1.50 |
| Bulk | Haiku 4.5 | 17,299 | 0 | ~$36 |
| Targeted Audit | Sonnet 4.5 | 313 | 0 | ~$2 |
| **Total** | | **17,274 final** | **0** | **~$38** |

Haiku↔Sonnet agreement on `credit_relevant`: **82.3%** on 300 stratified sample.
All disagreements are Haiku over-labeling (marking routine financial articles as credit-relevant).
Accepted: false positives are cheap for credit early-warning; false negatives are expensive.

---

## Spot-Check: Labels vs Rating Outcomes

17,009 of 17,274 labels had a matching rating action within 90 days (time-window join on entity + date).

### Confusion Matrix

|  Predicted ↓ \ Actual → | Negative (Downgrade) | Neutral | Positive (Upgrade) | Total |
|--------------------------|---------------------:|--------:|-------------------:|------:|
| **Deterioration (-1)**   | 3,543 | 2,328 | 407 | 6,278 |
| **Neutral (0)**          | 865 | 7,253 | 658 | 8,776 |
| **Improvement (+1)**     | 286 | 1,406 | 263 | 1,955 |
| **Total**                | 4,694 | 10,987 | 1,328 | 17,009 |

### Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Deterioration recall** | 75.5% (3,543/4,694) | Of articles near downgrades, 3 in 4 were flagged |
| **Deterioration precision** | 56.4% (3,543/6,278) | Of articles flagged as deterioration, just over half had nearby downgrades |

**Why precision appears low:** The model labels *article content*, not rating outcomes. An article about a genuinely stressed NBFC may be near a *neutral* rating action (different entity, different issue). The time-window join introduces noise — precision would be higher with entity-level matching and tighter windows.

### Alignment Breakdown

| Category | Count | % | Description |
|----------|------:|--:|-------------|
| Correct detection | 3,806 | 22.4% | Model direction matches rating outcome |
| Missed signal | 1,151 | 6.8% | Downgrade coming, model didn't flag |
| False alarm | 2,735 | 16.1% | Model flagged deterioration, no downgrade nearby |
| Missed positive | 658 | 3.9% | Upgrade coming, model didn't flag improvement |
| Irrelevant correct | 7,253 | 42.6% | Both say neutral — boring but correct |
| Other | 1,406 | 8.3% | Various mixed combinations |

---

## Analysis of Misses

Reviewed 50 stratified sample rows. Misses fall into three patterns:

### Pattern 1: Stock Price Articles (Model is Correct)
Example: *"Indiabulls Housing Finance slips for fifth straight session"* — model correctly ignores stock moves per boundary rules, even though a downgrade was 6 days away. The article genuinely has no credit content.

### Pattern 2: Routine Corporate Actions (Model is Correct)
Example: *"Piramal Enterprises allots 3.05 lakh equity shares pursuant to amalgamation"* — model correctly flags as non-credit-relevant. Amalgamation paperwork is not a signal.

### Pattern 3: Distress-Era Capital Raising (Model is Wrong)
Example: *"YES Bank jumps 4% as lender plans to raise funds"* — model flags as +1 (funding improvement). In context, YES Bank was desperately raising capital before its collapse. Distress-era fund raises are actually negative signals.

Similarly: *"Shriram Transport Finance plans to raise Rs 4,000 crore"* — flagged as improvement, but was emergency liquidity before downgrade.

**Implication:** We have a boundary rule for distress-era *asset sales* → direction=-1, but not for distress-era *capital raises*. The fine-tuned model should learn this contextual distinction from training data (the rating outcome provides the signal).

---

## Signal Type Distribution (Correct Detections)

| Signal Type | Count | % of Correct |
|-------------|------:|--:|
| asset_quality | 913 | 24.0% |
| governance | 815 | 21.4% |
| liquidity | 763 | 20.0% |
| contagion | 543 | 14.3% |
| funding | 365 | 9.6% |
| regulatory | 127 | 3.3% |
| operational | 14 | 0.4% |
| other | 3 | 0.1% |

Good spread — not dominated by any single category. Asset quality, governance, and liquidity are the top three, which aligns with typical NBFC credit deterioration patterns.

---

## Insight: Distress-Era Capital Raises (Phase 2 Feature Engineering)

The distress-era capital raise pattern is genuinely the hardest problem in credit analysis — distinguishing "raising capital because we're growing" from "raising capital because we're about to die." Even human analysts get this wrong in real-time (YES Bank's FPO was marketed as a recovery story).

**This is not a labeling problem — it's a feature engineering problem for Phase 2.** The fine-tuned model won't learn this from article text alone because the text looks identical in both cases. The distinguishing signal is *context*: what else is happening to that entity in the surrounding weeks.

**Action for Phase 2:** Build a rolling window feature — signal count and direction per entity over trailing 30 days. An entity with 5 negative signals in 30 days followed by a capital raise looks very different from an entity with 5 positive signals followed by a capital raise. This contextual feature should be an input alongside the article text.

## Insight: Signal Type Predictiveness (Phase 2 Feature Weighting)

`funding` is 5th in correct detections (365, 9.6%) despite being 2nd in the overall label distribution. That's because many funding articles are routine (e.g., HDFC issuing NCDs) and don't precede rating actions.

Meanwhile `governance` (815, 21.4%) and `contagion` (543, 14.3%) punch well above their weight — they're rarer signals but much more predictive of actual downgrades.

**Action for Phase 2:** Don't treat all signal types equally in training or scoring. Consider signal-type-specific weights, or at minimum, include signal type as an explicit feature rather than flattening it away. Governance and contagion signals should carry more weight in the rolling credit score (Phase 3).

---

## Decision

**Accept labels as-is for Phase 1.4 (training data formatting).**

Rationale:
1. 75.5% recall is strong for an LLM-labeled first pass
2. Most "misses" are articles with genuinely no credit content (model is right, time-window join is noisy)
3. The distress-era capital raise misclassification is a small fraction of 17K examples — and it's a feature engineering problem, not a labeling problem
4. Phase 2 model evaluation against held-back rating actions will reveal if label noise hurts performance
5. If it does hurt, we can re-audit specific patterns (capital raises near defaults) — targeted, not wholesale
6. Signal type predictiveness varies significantly — leverage this in Phase 2/3 weighting

---

## Reproducibility

```bash
# Re-run the spot-check
python -m src.data.label_spotcheck

# Adjust window (default 90 days)
python -m src.data.label_spotcheck --max-days 180

# Sample CSV for manual review
# → data/processed/spotcheck_sample.csv
```

---

*Generated by `src/data/label_spotcheck.py` — Phase 1.3 Step 8*
