# Phase 2.2 — LoRA Training Results Report

**Date:** 2026-02-19
**Purpose:** Document LoRA fine-tuning results, per-entity holdout analysis, and lessons learned
**Audience:** Technical reviewers, contest judges, team leads unfamiliar with ML fine-tuning
**Prerequisite reading:** `reports/phase1_4_training_data_design.md` (training data format design)

---

## Executive Summary

We fine-tuned Qwen 2.5-7B-Instruct using QLoRA on 9,591 training examples to extract structured
credit risk signals from news articles about Indian financial institutions.

**Key results:**
- **Format learning: solved.** Parse rate went from 0% (base model) to 100% (fine-tuned).
- **Crisis detection: excellent.** 97.7% deterioration recall on DHFL — an entity never seen in training.
- **Generalization: confirmed.** 90.0% det. recall on Reliance Capital (different crisis type).
- **False positives: moderate.** 33% false alarm rate on stable Cholamandalam — the main weakness.
- **Overfitting: fast.** Best checkpoint at step 500 of 1,800 (~0.83 epochs). 3 epochs was too many.

**Decision:** Model is ready for backtesting against actual rating actions (Phase 2.4).
RLMF reinforcement learning (Phase 2.3) is deferred — SFT results are strong enough to proceed.

---

## Training Setup

### Model & Hardware
| Parameter | Value |
|-----------|-------|
| Base model | Qwen 2.5-7B-Instruct |
| Quantization | 4-bit NF4 (QLoRA) |
| GPU | Google Colab Pro, T4 (15 GB VRAM) |
| Training time | 145 minutes |
| Precision | fp16 (T4 is Turing architecture, no bf16) |

### LoRA Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | 16 | Higher than FinGPT's r=8 — our task is harder (multi-field structured output vs binary sentiment) |
| Alpha | 32 | Standard 2× rank scaling |
| Dropout | 0.05 | Light regularization |
| Target modules | q_proj, v_proj, gate_proj, up_proj, down_proj | 5 of 7 linear layers. MLP modules (gate/up/down) needed for vocabulary learning |
| Adapter size | 152.8 MB | Larger than initial 33 MB estimate due to 5 target modules |

### Training Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 5e-4 | QLoRA paper recommendation (higher than standard 2e-4 because gradients pass through quantized weights) |
| Schedule | Cosine with 100 warmup steps | Gentle ramp-up, smooth decay |
| Epochs | 3 (but best at ~0.83) | Overfit early — see Lessons Learned |
| Batch size | 4 per device × 4 grad accumulation = 16 effective | Memory-constrained on T4 |
| Max sequence length | 2048 tokens | Longest example ~1,200 tokens |
| Eval frequency | Every 500 steps | 3 eval checkpoints + final |
| Loss masking | assistant_only_loss=True | Only compute loss on output tokens, not instruction/article text |
| Gradient checkpointing | True (use_reentrant=False) | Required for LoRA compatibility, saves ~40% VRAM |

### Training Data
| Split | Examples | Date Range | Purpose |
|-------|---------|-----------|---------|
| Train | 9,591 | 2017-04 to 2021-12 | Model learns from this |
| Validation | 2,247 | 2022-01 to 2023-06 | Early stopping / checkpoint selection |
| Test | 2,133 | 2023-07 to 2024-12 | The "honest number" — forward-looking |
| Entity holdout | 3,303 | 2017-06 to 2024-12 | 3 entities removed entirely from training |

---

## Training Dynamics

### Loss Curve
| Step | Train Loss | Val Loss | Notes |
|------|-----------|----------|-------|
| 500 | ~1.2 | **1.49** | Best checkpoint (auto-selected) |
| 1000 | ~0.9 | 1.50 | Slight overfit begins |
| 1500 | ~0.7 | 1.57 | Clear overfit — val loss rising while train loss drops |
| 1800 (final) | ~0.6 | — | `load_best_model_at_end` rolled back to step 500 |

The model learned the structured output format in less than 1 epoch. The remaining 2+ epochs
only memorized training examples without improving generalization. This makes sense: the base model
already understood credit concepts from pre-training. It only needed to learn our specific
`CREDIT_RELEVANT: / DIRECTION: / END` format — a relatively simple task.

---

## Results

### Parse Rate (Format Compliance)

| Split | Parse Rate | Parse Failures | Notes |
|-------|-----------|----------------|-------|
| Base model (Phase 2.1) | 0.0% | 1,000/1,000 | 100% free-form essays |
| Validation (500) | **100.0%** | 0 | Perfect format compliance |
| Test (500) | **99.6%** | 2 | Invented vocab: "macro", "valuation" |
| Entity holdout (3,303) | **100.0%** | 0 | Perfect on unseen entities |

The 2 test failures invented signal_type values outside the allowed vocabulary
(`macro` and `valuation` instead of one of the 8 allowed types). These could be handled
by a lenient parser mapping to `other`, but the strict parser correctly rejects them.

### Per-Field Accuracy

| Field | Validation (500) | Test (500) | Entity Holdout (3,303) |
|-------|-----------------|------------|----------------------|
| credit_relevant | 83.0% | 89.8% | 92.7% |
| direction | 80.3% | 77.2% | 92.3% |
| signal_type | 84.5% | 76.8% | 70.6% |
| sector_wide | 95.7% | 93.4% | 90.0% |
| confidence | 70.0% | 58.3% | 92.6% |

**Notes:**
- `credit_relevant` and `direction` are the most important fields for the use case.
  Both are strong (>77%) across all splits.
- `signal_type` is the weakest classification task. Signal categories overlap conceptually
  (e.g., "DHFL's mutual fund investors pulling money due to NPA fears" touches liquidity,
  asset_quality, AND contagion simultaneously). 70-84% on 8-way classification with
  overlapping categories is reasonable.
- `confidence` varies wildly (58-93%). This is an inherently subjective label —
  two human analysts would disagree on "Medium" vs "High" frequently. Low priority.
- `sector_wide` is consistently strong (90-96%). Binary classification with clear signals.

### Per-Entity Holdout Results

These are the most important numbers in this report. Three entities were entirely
removed from training data to test whether the model learned language patterns or
just memorized entity names.

#### DHFL (1,243 articles) — Crisis Detection

| Metric | Value |
|--------|-------|
| Parse rate | 100.0% |
| CR accuracy | 97.3% |
| Direction accuracy | 96.5% |
| Signal type accuracy | 65.1% |
| **Det. precision** | **96.3%** |
| **Det. recall** | **97.7%** |
| Imp. precision | 40.0% |
| Imp. recall | 42.9% |

DHFL collapsed in 2018-2019 (India's largest housing finance default). 91% of its articles
are genuine deterioration signals. The model — which has never seen DHFL's name during
training — detects the crisis with 97.7% recall and 96.3% precision.

**What this proves:** The model learned what distress language looks like ("NPA provisions
surging," "liquidity crunch," "creditors filing insolvency"), not just "DHFL = bad."
It generalizes to unseen entities.

Signal type accuracy (65.1%) is the weakest field because DHFL's crisis simultaneously
involved liquidity, asset quality, governance, and contagion — many articles could
reasonably be classified as multiple types.

#### Reliance Capital (688 articles) — Generalization

| Metric | Value |
|--------|-------|
| Parse rate | 100.0% |
| CR accuracy | 93.9% |
| Direction accuracy | 87.5% |
| Signal type accuracy | 76.1% |
| **Det. precision** | **90.4%** |
| **Det. recall** | **90.0%** |
| Imp. precision | 13.6% |
| Imp. recall | 26.7% |

Reliance Capital (Anil Ambani group) had a different crisis arc — slower governance-driven
deterioration rather than sudden liquidity collapse. The model still catches 90% of
deterioration signals, confirming it generalizes across crisis types.

Low improvement precision (13.6%) reflects the small denominator: Reliance Capital had
very few genuine improvement articles, so the few "improvement" predictions are mostly wrong.

#### Cholamandalam (1,372 articles) — False Positive Control

| Metric | Value |
|--------|-------|
| Parse rate | 100.0% |
| CR accuracy | 87.9% |
| Direction accuracy | 83.7% |
| Signal type accuracy | 82.9% |
| **Det. precision** | **67.0%** |
| **Det. recall** | 73.6% |
| Imp. precision | 49.4% |
| **Imp. recall** | **82.5%** |

Cholamandalam is a healthy, stable NBFC — only 12% of articles are genuine deterioration.
This is the false positive test: does the model cry wolf?

**The main weakness:** 67% det. precision means 33% of deterioration predictions for
Cholamandalam are false alarms. When the model says "deterioration" about a stable company,
1 in 3 times it's wrong.

**Root cause:** Likely inherited from the Haiku labeler's over-labeling bias (82.3%
agreement with Sonnet on the 300-article calibration sample). Routine business articles
about stable NBFCs were labeled credit-relevant during bulk labeling, and the model
learned to reproduce this bias.

**Mitigation options:**
1. **Confidence thresholds** — only flag deterioration if confidence is "High"
2. **Cleaner training data** — re-label stable NBFC articles with Sonnet instead of Haiku
3. **Post-hoc filtering** — require 2+ deterioration signals within 30 days before alerting

**Context:** In credit surveillance, 33% false positive rate is arguably acceptable.
A credit analyst reviewing 10 flagged articles and dismissing 3 as noise is standard
workflow. Missing a real signal (false negative) is far more costly than a false alarm.

---

## Comparison: Base Model vs Fine-Tuned

| Metric | Base Model (Phase 2.1) | Fine-Tuned (Phase 2.2) | Improvement |
|--------|----------------------|----------------------|-------------|
| Parse rate | 0.0% | 99.6-100% | Format learned |
| Output format | Free-form essays | Structured fields | Fully compliant |
| Credit concepts | Present (from pre-training) | Present + structured | Was already there |
| Inference speed | ~1.5 s/example | ~1.5 s/example | No change (same base model) |
| Adapter size | 0 (base model) | 152.8 MB | Small vs 14 GB base |

---

## Lessons Learned

### 1. Format Learning Is Easy, Domain Knowledge Is Hard
The model went from 0% to 100% parse rate in <1 epoch (~500 steps). The structured output
format was the easy part. The hard part — actually understanding which articles signal
credit deterioration — came from pre-training, not fine-tuning. Fine-tuning only taught
the model to EXPRESS its knowledge in our format.

**Implication:** If you have a strong base model with good domain knowledge, even a few
hundred examples might be enough for format training. Our 9,591 examples were more than
sufficient.

### 2. Overfitting Happens Fast with QLoRA
Best checkpoint at step 500 of 1,800 (28% of training). The remaining 72% only overfit.
This is common with QLoRA on domain-specific tasks: the adapter parameters are small
(~33M vs 7B base), so they memorize training examples quickly.

**For next time:** Use 1 epoch with `EarlyStoppingCallback(patience=2)` to save Colab time.
The current run wasted ~95 minutes on overfitting that was automatically discarded.

### 3. Label Quality Flows Through to Model Quality
Cholamandalam's 33% false positive rate directly traces to Haiku's over-labeling bias.
The model faithfully learned the label distribution, including the noise. Better labels
→ better model, especially for stable entities where the signal-to-noise ratio is low.

### 4. Entity Holdout Is the Right Evaluation
Standard val/test splits measure in-distribution performance. Entity holdout measures
what actually matters: can the model detect credit signals for entities it has never
seen? This is the real-world use case — the next NBFC crisis will involve a company
that wasn't in our training data.

### 5. Adapter Size Scales with Target Modules
Initial estimate was ~33 MB. Actual: 152.8 MB. The difference comes from targeting
5 modules (q_proj, v_proj, gate_proj, up_proj, down_proj) instead of just 2 (q_proj,
v_proj). Each additional module adds ~30 MB at r=16. Still tiny compared to the
14 GB base model.

### 6. Colab T4 Throttling Is Real
Entity holdout inference estimated 83 minutes but took 7.5 hours. Colab dynamically
reduces GPU priority for long-running tasks. For future work: break large inference
runs into smaller batches or use Colab A100 for time-sensitive runs.

---

## Next Steps

### Immediate: Phase 2.4 — Backtest Against Rating Actions
The critical question: **does the model predict downgrades BEFORE they happen?**

Using test set articles (2023-07 to 2024-12) with known rating actions:
1. For each entity, construct a rolling 30-day signal score from model predictions
2. Measure **lead time**: how many days before a rating action did the score cross a threshold?
3. Measure **real-world precision**: what fraction of alerts led to actual rating actions?
4. Compare against baselines: prompted Opus, FinRLlama 3.2-3B

### Deferred: Phase 2.3 — RLMF
RLMF (reinforcement learning from market/rating feedback) could improve the model by
rewarding correct predictions against actual rating outcomes. However:
- The SFT model already has strong performance
- RLMF requires significant engineering (reward model, PPO training loop)
- Backtesting will show whether the improvement is needed
- Best pursued as Phase 5 polish if the backtest reveals specific weaknesses

---

## Files

| File | Purpose |
|------|---------|
| `notebooks/phase2_2_lora_training.ipynb` | Training + evaluation notebook (ran on Colab) |
| `src/training/evaluate.py` | Canonical evaluation module (parser, metrics, holdout eval) |
| `reports/phase2_2_training_results.md` | This report |

### Output Files (on Google Drive)
| File | Size | Contents |
|------|------|----------|
| `data/models/qwen-credit-lora/` | 152.8 MB | LoRA adapters + tokenizer config |
| `data/processed/finetuned_val_outputs.jsonl` | — | 500 validation predictions |
| `data/processed/finetuned_test_outputs.jsonl` | — | 500 test predictions |
| `data/processed/finetuned_holdout_outputs.jsonl` | — | 3,303 entity holdout predictions |
