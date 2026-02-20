# Phase 1.4 — Training Data Design Report

**Date:** 2026-02-17
**Purpose:** Document the design decisions for converting 17,274 LLM-labeled articles into a fine-tuning dataset
**Audience:** Technical reviewers, contest judges, team leads unfamiliar with ML fine-tuning
**Prerequisite reading:** `reports/phase1_label_quality.md` (label quality spot-check)

---

## Executive Summary

We have 17,274 news articles about Indian financial institutions, each labeled by an LLM for credit risk signals. This report documents how we convert those labels into a training dataset for fine-tuning a language model.

Five key design decisions were made:

1. **Output format:** Structured text with strict vocabulary and an END stop token (not JSON)
2. **Class balancing:** Keep the natural deterioration skew (70% of credit signals); adjust the model, not the data
3. **Train/test split:** Temporal (date-based) to prevent the model from "seeing the future"
4. **Entity memorization diagnostic:** Hold out 3 entities entirely to test whether the model learns language patterns or just entity names
5. **Output robustness check:** Test the parser before training to catch format problems early

Each decision is explained below in plain language, with the technical reasoning for reviewers who want it.

---

## What Is "Training Data Format" and Why Does It Matter?

Fine-tuning a language model means showing it thousands of examples of the task you want it to perform. Each example has three parts:

| Part | What It Is | Analogy |
|------|-----------|---------|
| **Instruction** | "Here's what I want you to do" | The job description |
| **Input** | "Here's the information to work with" | The document on your desk |
| **Output** | "Here's what a good answer looks like" | The model answer from a senior analyst |

The model learns by seeing thousands of these instruction → input → output triplets. After training, you give it a new instruction + input, and it generates the output on its own.

**Why the format matters:** The output format you choose during training is the format the model will produce forever. If you train it to output messy, ambiguous text, that's what you get in production. If you train it to output clean, structured, parseable text, that's what you get. There are no do-overs — retraining costs time and money.

---

## Decision 1: Output Format — Structured Text with Strict Vocabulary

### The Problem

The model needs to produce output that a computer can parse into structured fields (credit-relevant yes/no, direction, signal type, etc.). Two approaches:

**Option A — JSON:**
```json
{"credit_relevant": 1, "signal_direction": -1, "signal_type": "asset_quality", ...}
```

**Option B — Structured text:**
```
CREDIT_RELEVANT: Yes
DIRECTION: Deterioration
SIGNAL_TYPE: asset_quality
SECTOR_WIDE: No
CONFIDENCE: High
REASONING: 11.4% of PFC's loan book entering insolvency with write-off risk.
END
```

### The Decision: Structured text (Option B)

**Why not JSON?** Language models are unreliable JSON generators. They frequently produce malformed JSON — missing closing braces, extra commas, unescaped quotes. When JSON parsing fails, you lose the *entire* output. With structured text, if the model garbles one field, the other fields are still recoverable. Each line is independent.

**Why strict vocabulary?** Every field has a fixed set of allowed values:

| Field | Allowed Values | Why Fixed |
|-------|---------------|-----------|
| CREDIT_RELEVANT | `Yes`, `No` | Binary decision — no ambiguity |
| DIRECTION | `Deterioration`, `Improvement`, `Neutral` | Three-way classification, human-readable labels |
| SIGNAL_TYPE | `liquidity`, `asset_quality`, `regulatory`, `contagion`, `governance`, `funding`, `operational`, `other` | Matches our credit taxonomy from Phase 1.3 |
| SECTOR_WIDE | `Yes`, `No` | Binary — does this affect one entity or the whole sector? |
| CONFIDENCE | `Low`, `Medium`, `High` | Three levels, self-explanatory |
| REASONING | Free text (one sentence) | The model's explanation — useful for human review |

If the model outputs a word not in this vocabulary (e.g., "Negative" instead of "Deterioration"), the parser rejects it immediately. This catches drift before it enters downstream analysis.

**Why the END token?** Language models don't inherently know when to stop generating text. Without a clear stop signal, the model may repeat itself, hallucinate extra fields, or generate a second "example." The `END` token teaches the model a hard stop during training. During inference, we configure `END` as a stop sequence — generation halts the moment it appears. This saves inference cost and prevents garbage output.

**Short-form for noise articles:** When an article is not credit-relevant, the output is just:
```
CREDIT_RELEVANT: No
REASONING: Stock price movements with no credit quality information.
END
```
No DIRECTION, SIGNAL_TYPE, etc. This teaches the model to reject quickly — it learns a shorter code path for noise, saving inference tokens (and cost) on the ~47% of articles that are irrelevant.

### What This Looks Like in Practice

**Example: Credit deterioration signal**
```
Instruction: Assess whether this news article contains signals relevant to
             the credit quality of the mentioned Indian financial institution.

Input:       Entity: PFC
             Date: 2018-07-15
             Title: PFC power projects with 14,000 Mw capacity to go down insolvency alley
             Article: Around 14,000 Mw of projects, involving Rs 300 bn of PFC's
             loan book, under IBC route and over half may have to be written off...

Output:      CREDIT_RELEVANT: Yes
             DIRECTION: Deterioration
             SIGNAL_TYPE: asset_quality
             SECTOR_WIDE: No
             CONFIDENCE: High
             REASONING: 11.4% of PFC's loan book entering insolvency with write-off risk.
             END
```

**Example: Not credit-relevant (stock price noise)**
```
Instruction: [same as above]

Input:       Entity: Shriram Finance
             Date: 2023-05-10
             Title: Shriram Finance stock hit high of Rs 3088
             Article: Shriram Finance stock hit high of Rs 3088.55 and low of
             Rs 3000.00. ROE at 15.04%. Traded volume 10,000 shares...

Output:      CREDIT_RELEVANT: No
             REASONING: Stock price movements and trading metrics with no credit information.
             END
```

---

## Decision 2: Keep the Natural Deterioration Skew

### The Data Distribution

| Category | Count | % of Total |
|----------|------:|---:|
| Not credit-relevant (cr=0) | 8,054 | 46.6% |
| Credit-relevant: Deterioration | 6,456 | 37.4% |
| Credit-relevant: Improvement | 1,965 | 11.4% |
| Credit-relevant: Neutral/mixed | 799 | 4.6% |
| **Total** | **17,274** | **100%** |

Deterioration signals dominate the credit-relevant articles (70% of cr=1). Should we artificially balance this by duplicating improvement and neutral examples?

### The Decision: No. Keep the natural distribution.

**Why?** Three reasons:

1. **The skew reflects reality.** In credit markets, bad news travels faster and louder than good news. There are more articles about NPAs, defaults, and liquidity crises than about rating upgrades. A model trained on an artificially balanced dataset would see the world as 33% deterioration / 33% improvement / 33% neutral — that's not how financial news works.

2. **Our primary metric is deterioration recall.** This model is an early-warning system for credit deterioration. We care most about: "When a downgrade is coming, does the model flag it?" (recall). We care less about: "When the model flags deterioration, is it always right?" (precision). The natural skew gives the model more deterioration examples to learn from, which directly helps recall.

3. **We can adjust the model without distorting the data.** If the model underperforms on improvement signals, two clean levers exist:
   - **Loss weighting:** Tell the training algorithm "mistakes on improvement examples cost 2x more than mistakes on deterioration examples." The model pays more attention to rare classes without changing the data.
   - **Threshold tuning:** At inference time, adjust how confident the model must be before it calls something "Improvement." Tune this on the validation set.

Both are Phase 2 concerns. Phase 1.4 formats the data faithfully.

---

## Decision 3: Temporal Train/Val/Test Split

### The Problem: Why Not Just Shuffle and Split?

In most ML tutorials, you randomly shuffle your data and split 70/15/15. For financial time-series data, this is **dangerous**.

Consider: Your dataset has articles from 2017-2024. If you randomly shuffle, your training set might contain a December 2019 article about DHFL's default, while your test set contains a November 2019 article about DHFL's liquidity crisis. The model has effectively "seen the future" — it knows DHFL defaults, so of course it flags the earlier article correctly. The test score looks great, but the model hasn't actually learned anything useful.

This is called **temporal data leakage**, and it's the #1 cause of over-optimistic backtests in financial ML. The model appears to predict the future, but it's actually just memorizing the past.

### The Decision: Strict date-based splits

| Split | Date Range | Est. Articles | Purpose |
|-------|-----------|:---:|---------|
| **Train** | 2017-04 to 2021-12 | ~10,600 (70%) | Learn from crisis + recovery periods |
| **Validation** | 2022-01 to 2023-06 | ~2,100 (14%) | Tune hyperparameters, detect overfitting |
| **Test** | 2023-07 to 2024-12 | ~2,700 (16%) | Final evaluation — never touched until the end |

**Why these specific cutoffs?**

- **Train (2017-2021):** Contains the NBFC liquidity crisis (2018-2019), the peak of deterioration signals. This is where the model learns what credit distress looks like. Also includes the 2020-2021 recovery, so it sees improvement signals too.

- **Validation (2022 to mid-2023):** A "quiet" period with fewer crisis signals. Used to tune model hyperparameters (learning rate, LoRA rank, etc.) and detect overfitting. If the model performs well on crisis-era training data but poorly on quiet-period validation, it's memorizing rather than learning.

- **Test (mid-2023 to 2024):** The "forward test." This simulates production: the model was trained on historical data and must perform on articles it has never seen, from a time period it has never encountered. This is the number we report in the contest submission.

### What about deterioration density in the test set?

The test period (mid-2023 to 2024) is quieter than the crisis era — fewer deterioration signals. This is **by design**. In production, the model mostly encounters "normal" articles and must correctly ignore them. A test set dominated by crisis articles would overstate the model's real-world accuracy.

However, we also need to know: "Can the model detect deterioration when it IS present?" That's what the entity holdout diagnostic (Decision 4) addresses.

### Backtesting

The test set doubles as the backtest set. Every article in the dataset has `rating_windows` metadata — the actual rating action that happened for that entity within 90 days. By comparing the model's predictions on test set articles against actual rating outcomes, we get a direct measure of: "Would this model have warned you before the downgrade?"

No separate backtest hold-out needed. The test split already contains the ground truth.

---

## Decision 4: Entity Memorization Diagnostic

### The Problem

Our dataset has 33 unique entities. Some appear thousands of times:

| Entity | Articles | Deterioration % |
|--------|:---:|:---:|
| YES Bank | 1,590 | 71% |
| DHFL | 1,243 | 91% |
| Reliance Capital | 688 | 80% |
| Cholamandalam | 1,372 | 12% |
| Bajaj Finance | 739 | 10% |

A model could achieve high accuracy by simply learning: "If the article mentions YES Bank or DHFL, output Deterioration." This is called **entity memorization** — the model learns names, not language patterns.

This would be useless in production. When a *new* NBFC starts showing stress signals, the model has never seen its name before and would miss it entirely.

### The Decision: Hold out 3 entities entirely from training

| Held-Out Entity | Articles | Why This Entity |
|-----------------|:---:|-----------------|
| **DHFL** | 1,243 | Crisis poster child (91% deterioration). If the model correctly identifies DHFL deterioration signals without ever training on DHFL articles, it proves the model learned "the language of distress," not "the word DHFL." |
| **Reliance Capital** | 688 | Different crisis arc — Anil Ambani conglomerate unwind, not NBFC liquidity crisis. Tests generalization across different types of financial distress. |
| **Cholamandalam** | 1,372 | Stable entity (only 22% credit-relevant). Tests false positive control: can the model correctly say "No" for a healthy entity it has never seen? |

**How it works:**
1. Remove all single-entity articles for these 3 entities from training (~3,303 articles)
2. Train the model on the remaining ~13,971 articles
3. Evaluate on the held-out articles separately
4. Compare: Does the model's accuracy on held-out entities match its accuracy on in-training entities?

**What the results tell us:**
- **Accuracy matches** → The model learned text patterns. Entity names are incidental.
- **Accuracy drops significantly** → Entity memorization problem. Mitigation: mask entity names during training (replace with "[ENTITY]"), or add more diverse entities to the training data.

**Caveat:** Some multi-entity articles (e.g., "Cholamandalam, Shriram Finance") mention held-out entities alongside in-training entities. These remain in the training set, since they primarily discuss multiple entities and don't teach entity-specific patterns. This minor contamination is noted in the diagnostic results.

**Important:** This is a *diagnostic*, not the main evaluation. The main model is trained on ALL entities (including DHFL, Reliance Capital, and Cholamandalam). The entity holdout is a separate training run used only to answer the memorization question. The production model benefits from seeing all available data.

---

## Decision 5: Output Robustness Check

### The Problem

We're designing a format that a model will learn to produce. But what if the model fundamentally struggles with our format? If we discover parsing failures *after* a $50-100 training run, we've wasted time and money.

### The Decision: Two-phase robustness testing

**Phase 1 (now, before training):**
- Build a strict parser for the structured text format
- Test it against synthetic outputs: correct, partially correct, garbled, extra whitespace, missing END, wrong vocabulary, mixed case, etc.
- Ensure the parser fails cleanly and reports *which* field broke
- Write comprehensive unit tests (part of the training data formatter code)

**Phase 2 (at start of training, on Colab with GPU):**
- Take the chosen base model (Qwen 2.5-7B or LLaMA 3.1-8B)
- Feed it 1,000 training examples using our exact instruction + input format
- Attempt to parse every output
- Measure: % parseable, which fields fail most often, common failure patterns

**Decision threshold:** If >20% of base model outputs fail to parse, we tighten the format before training. Options include:
- Simplifying field names
- Reducing the number of fields
- Switching to a more constrained format (e.g., multiple-choice instead of free generation)

This follows the same "calibrate before bulk" philosophy we used in Phase 1.3 labeling. Small sample → fix problems → full run.

---

## Input Format

Each training example's input includes four fields from the original article:

```
Entity: YES Bank
Date: 2018-09-19
Title: YES Bank CEO Rana Kapoor to step down by January 2019
Article: YES Bank CEO Rana Kapoor reiterates commitment after RBI trimmed his term...
```

### What's included and why

| Field | Why Included |
|-------|-------------|
| **Entity name** | Critical context. "Capital raise" means different things for YES Bank (desperation) vs HDFC (routine growth). The model needs to know who the article is about. |
| **Date** | Temporal context. An NPA article in 2018 (peak crisis) has different systemic implications than the same article in 2023 (recovery period). |
| **Title** | Often the strongest signal — many articles have the credit event right in the headline. Also much shorter than the full text, so it anchors the model's attention. |
| **Article text** | The full content, truncated at 3,000 characters. This matches the truncation used during our LLM labeling pipeline (Phase 1.3), ensuring consistency between labels and training inputs. |

### What's excluded and why

| Field | Why Excluded |
|-------|-------------|
| **Source domain** | Weakly predictive. Including it risks the model learning "articles from Economic Times are more credit-relevant" — a spurious correlation that won't hold in production. |
| **GDELT tone score** | Known to be unreliable for credit signals (see Phase 1.3 analysis). General article positivity/negativity ≠ credit relevance. Including it would teach the model the wrong signal. |
| **Rating windows** | This is the ground truth — the actual rating action that happened after the article was published. Including it in the input would be **data leakage**: the model would learn to parrot the answer rather than analyzing the text. Rating windows are used only for post-hoc evaluation, never shown to the model. |

### The Instruction Text

Fixed across all examples:

> "Assess whether this news article contains signals relevant to the credit quality of the mentioned Indian financial institution."

**Why "Indian financial institution" and not "Indian NBFC"?** Our dataset includes entities that are not technically NBFCs:
- **Banks:** YES Bank (1,590 articles), Lakshmi Vilas Bank (231), PMC Bank (59) — together 10.9% of the dataset
- **Infrastructure finance:** IL&FS (94 articles)
- **Diversified financial companies:** Reliance Capital (688), Piramal Enterprises (946)

Saying "NBFC" would be factually wrong for ~11% of training examples. Worse, during inference on a bank article, the model might force-fit an NBFC framing. "Financial institution" is accurate and broad enough to cover all current and future entity types.

---

## Dataset Summary

### Overall Numbers

| Metric | Value |
|--------|------:|
| Total training examples | 17,274 |
| Unique entities | 33 (single-entity) + multi-entity combinations |
| Date range | 2017-04-07 to 2024-12-24 |
| Median article length | 1,891 characters |
| Mean article length | 2,645 characters |
| Text truncation | 3,000 characters |

### Label Distribution

| Category | Count | % |
|----------|------:|---:|
| Not credit-relevant | 8,054 | 46.6% |
| Deterioration | 6,456 | 37.4% |
| Improvement | 1,965 | 11.4% |
| Neutral/mixed | 799 | 4.6% |

### Credit Signal Type Distribution (within credit-relevant)

| Signal Type | Count | % of CR=1 | Predictiveness Note |
|-------------|------:|---:|---------------------|
| asset_quality | 2,801 | 30.4% | Most common; strong predictor of downgrades |
| funding | 2,172 | 23.6% | High volume but many routine issuances dilute signal |
| liquidity | 1,335 | 14.5% | Strong predictor when present |
| governance | 1,245 | 13.5% | Rarer but disproportionately predictive (see Phase 1.3 report) |
| contagion | 881 | 9.6% | Rarer but disproportionately predictive |
| regulatory | 522 | 5.7% | Infrequent; usually high-impact |
| operational | 234 | 2.5% | Rare |
| other | 30 | 0.3% | Catch-all |

### Temporal Distribution

| Year | Articles | Deterioration | Improvement | % Credit-Relevant |
|------|------:|------:|------:|---:|
| 2017 | 904 | 91 | 48 | 33.5% |
| 2018 | 1,981 | 902 | 149 | 59.9% |
| 2019 | 5,044 | 3,136 | 600 | 74.1% |
| 2020 | 2,415 | 1,008 | 278 | 61.6% |
| 2021 | 1,786 | 514 | 182 | 46.5% |
| 2022 | 1,617 | 295 | 157 | 37.4% |
| 2023 | 1,896 | 222 | 228 | 27.9% |
| 2024 | 1,631 | 288 | 323 | 33.1% |

Note the clear crisis signature: 2018-2020 has 74% of all deterioration signals. 2022-2024 is a quieter period. The temporal split ensures the model trains on the crisis and is tested on the recovery/quiet period — simulating real-world forward deployment.

### Planned Split

| Split | Date Range | Est. Articles | Purpose |
|-------|-----------|------:|---------|
| Train | 2017-04 to 2021-12 | ~10,600 | Crisis + recovery, richest signal density |
| Validation | 2022-01 to 2023-06 | ~2,100 | Hyperparameter tuning, overfitting detection |
| Test | 2023-07 to 2024-12 | ~2,700 | Forward evaluation + backtest against rating actions |

### Entity Holdout Diagnostic

| Entity | Articles | Type | Purpose |
|--------|------:|------|---------|
| DHFL | 1,243 | Crisis (91% deterioration) | Can the model detect distress in an entity it's never seen? |
| Reliance Capital | 688 | Crisis (80% deterioration) | Different crisis arc — generalization test |
| Cholamandalam | 1,372 | Stable (22% credit-relevant) | Can the model correctly say "No" for an unseen healthy entity? |

---

## Relationship to Prior and Future Work

### Builds On
- **Phase 1.3 labels** (`labels_final.jsonl`): The 17,274 LLM-generated labels are the "answers" in our training examples
- **Phase 1.3 quality report** (`reports/phase1_label_quality.md`): Confirmed labels are high enough quality for training (75.5% deterioration recall vs rating actions)
- **FinGPT/FinRLlama research** (Phase 0): The instruction/input/output three-field format follows their proven approach

### Feeds Into
- **Phase 2.1 (Base Model Selection):** Training data format determines which models are compatible
- **Phase 2.2 (Training):** The JSONL files are the direct input to the fine-tuning script
- **Phase 2.3 (RLMF):** Rating windows (held back from training) become the reward signal
- **Phase 2.4 (Evaluation):** Test split + entity holdout provide the evaluation benchmarks

### Key Insight for Phase 2: Distress-Era Capital Raises

From the Phase 1.3 quality report: the model's biggest weakness is distinguishing "raising capital because we're growing" from "raising capital because we're about to die." The text looks identical in both cases. This is not a training data problem — it's a **feature engineering** problem. In Phase 2, we plan to add a rolling-window context feature (signal count and direction per entity over trailing 30 days) to help the model distinguish these cases.

---

## Files Produced by Phase 1.4

| File | Description |
|------|-------------|
| `configs/training_config.yaml` | Split cutoffs, instruction text, field vocabulary, held-out entities |
| `src/data/format_training.py` | Joins labels + articles, formats, splits, writes JSONL |
| `src/data/parse_training_output.py` | Strict parser for the structured text output format |
| `tests/test_format_training.py` | Tests for formatter + parser |
| `data/processed/train.jsonl` | Training split |
| `data/processed/val.jsonl` | Validation split |
| `data/processed/test.jsonl` | Test split |
| `data/processed/entity_holdout.jsonl` | Diagnostic: held-out entities for memorization test |

---

*Report generated during Phase 1.4 design — prior to code implementation.*
*Training data formatting code to follow.*
