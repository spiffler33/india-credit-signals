# India NBFC Credit Signal Engine — Master Plan

## What We're Building
A fine-tuned LLM that extracts credit deterioration signals from English-language news/filings for Indian NBFCs, with regulator-to-sector contagion logic. Backtested against actual rating actions.

**End goal:** FinRL/FinAI 2026 contest entry + workplace tool + learning vehicle for ML fine-tuning.

## Learning Philosophy
**Learn by building existing things first, then adapt.** We clone FinRLlama, run it, understand it, THEN swap in our credit risk data. Every Claude Code output teaches you what it did and why.

---

## Phase 0: Foundation (Days 1-2)
**Goal:** Run someone else's fine-tuning pipeline end-to-end. Understand every step.

### 0.1 Clone & Run FinRLlama ✅ DONE
```
Repo: https://github.com/Arnav-Gr0ver/ICAIF_FinRL-2024
Model: https://huggingface.co/Arnav-Gr0ver/FinRLlama-3.2-3B-Instruct
```
- ✅ Cloned repo, read every file, documented the full pipeline architecture
- ✅ Ran Qwen 2.5-3B baseline on Colab — generic sentiment OK, credit signals poor
- ✅ Meta LLaMA 3.2 access approved (2026-02-13). FinRLlama inference deferred to Phase 2.4 as baseline.
- **Learning checkpoint:** You should be able to explain LoRA rank, learning rate schedules, and what RLMF does differently from standard SFT

### 0.2 Clone & Read FinGPT ✅ DONE
```
Repo: https://github.com/AI4Finance-Foundation/FinGPT
Focus: FinGPT v3 series (LoRA fine-tuning on sentiment)
```
- ✅ Cloned repo, read v3 data pipeline + Benchmark training/eval scripts
- ✅ Key takeaways documented:
  - Three-field format: instruction/input/output (we adopt this)
  - Dataset balancing via oversampling small datasets (critical for our imbalanced credit data)
  - LoRA r=8, alpha=32 for sentiment; we use r=16 for harder credit signal task
  - Prompt masking: loss only on answer tokens, not instruction/input
  - Their eval is substring matching — we need structured multi-field evaluation
- **Learning checkpoint:** You should understand the difference between SFT, RLHF, and RLMF

### 0.3 Set Up Development Environment ✅ DONE
- ✅ Python 3.12.10 (native ARM), PyTorch 2.10, transformers 4.57, peft 0.18
- ✅ Colab Pro ($12/mo) for GPU work (A100/T4)
- ✅ GitHub repo: `spiffler33/india-credit-signals` (private)

**Environment split:**
| Task | Where |
|------|-------|
| Code editing, data scraping, git | Local Mac (Claude Code) |
| Model inference, training | Google Colab (GPU) |

**Discuss in Claude Chat (not Code):**
- Review your understanding of LoRA, RLMF, SFT after reading the repos
- Clarify anything confusing about the training pipeline architecture

---

## Phase 1: Data Collection (Days 3-8)
**Goal:** Build the labeled training dataset. This is 60% of the project.

### 1.1 Credit Event Timeline ✅ DONE
**Source:** CRISIL/ICRA SEBI disclosure scrapers + CARE/India Ratings manual curation.
Built in separate data sourcing project (`/Users/coddiwomplers/Desktop/Python/data_scraping/`).

**Result:**
- 1,654 total records (1,311 in 2016-2024 training window)
- 39 entities, 6 agencies (CRISIL, ICRA, CARE, India Ratings, Brickwork, Acuite)
- 10 stressed entities with 38 default events
- All dates verified from SEBI disclosures, agency-native rating scales preserved
- CSV: `data/raw/rating_actions_sourced.csv` (tracked in git)

**Known gaps (acceptable for now):**
- CARE/India Ratings coverage is thin (25 + 9 records vs 807 + 463 for CRISIL/ICRA)
- Watchlist/outlook actions are underrepresented (9 total)
- Can backfill later by adding CARE scraper — merge pipeline deduplicates automatically

### 1.2 News Data Collection ✅ DONE
**Source:** GDELT DOC 2.0 API (free, no API key).
Built in separate project (`/Users/coddiwomplers/Desktop/Python/data_scraping/gdelt_news/`).

**Result:**
- 74,028 article rows (32,570 unique URLs) across 756 query windows
- 38/39 entities covered (IL&FS Financial Services got 0 hits — articles are in parent IL&FS results)
- Date range: 2017-04-07 to 2024-12-25
- Top sources: Economic Times, Business Standard, Moneycontrol, Hindu Business Line, Financial Express
- Imported into main project and processed through Steps 1-3 below

### 1.3 Data Processing & Label Construction

Phase 1.3 has 4 sub-steps. Steps 1-3 clean and filter raw GDELT data. Step 4 is the LLM labeling pipeline.

#### Step 1: Import & Deduplicate ✅ DONE
- `src/data/import_gdelt.py` — collapsed 74K rows → 32,570 unique URLs
- Attached rating windows (entity, date, action_type, outcome, days_before) as JSON
- Output: `data/processed/gdelt_deduped.csv`

#### Step 2: Title-Based Triage ✅ DONE
- `src/data/triage_articles.py` + `configs/triage_config.yaml`
- Three-layer filter: blocked domains → financial keywords → entity-in-title
- Result: 14,688 pass, 4,098 review, 13,784 drop
- Critical: 97.5% survival rate for negative-outcome articles
- Output: `data/processed/gdelt_triaged_pass.csv`, `gdelt_triaged_review.csv`, `gdelt_triaged_drop.csv`

#### Step 3: Body Text Scraping + Review Promotion ✅ DONE
- Body text scraped for pass + review articles (done in data_scraping project)
- `src/data/promote_review.py` — promoted review articles with body text to pass
- Combined into final labeling input: 17,299 articles with body text
- Output: `data/processed/gdelt_for_labeling.csv`

#### Step 4: LLM Labeling Pipeline ✅ COMPLETE
Three-phase Calibrate → Bulk → Audit pipeline (~$30-35 total).

**Strategy:** Cheap model (Haiku) for volume, expensive model (Sonnet) for quality, merge for best of both.
This is standard in ML data pipelines — you don't need GPT-4-level accuracy on every single article,
just on the ones that matter (credit-relevant, borderline, low-confidence).

**Architecture — 5 files + 1 config:**

| File | Lines | Purpose |
|------|-------|---------|
| `configs/labeling_config.yaml` | 136 | All prompts, model IDs, thresholds, sampling keywords, few-shot examples (initially empty) |
| `src/data/label_models.py` | 178 | `SignalType`/`Confidence` enums, `ArticleLabel` dataclass, bulletproof JSON parser (handles markdown fences, yes/no→1/0, invented signal types), JSONL I/O |
| `src/data/label_sampler.py` | 218 | Deliberate 300-article calibration sample: 100 credit-likely + 100 noise-likely + 100 ambiguous, max 15 per entity per bucket |
| `src/data/label_articles.py` | 322 | Async labeling engine: asyncio + semaphore (10 concurrent), exponential backoff retry, progressive JSONL writes (crash-safe), full resume support |
| `src/data/label_audit.py` | 237 | select: identifies audit candidates, run: relabels with Sonnet, merge: Sonnet overrides Haiku on disagreement |
| `tests/test_label_models.py` | 198 | 18 tests: JSON parse, markdown fence stripping, bool coercion, signal type mapping, consistency enforcement, JSONL I/O, corrupt line handling |

Dependency: `anthropic` SDK v0.79.0 installed via `uv add anthropic`.

**Key design decisions:**

1. **1 article per API call** (not batched). Batching 5 per call saves ~$1.50 total but one
   bad article corrupts the whole batch's JSON parsing. At Haiku prices (~$0.0003/article),
   the simplicity of 1-per-call is worth the marginal cost. Each call: ~900 input + ~120 output tokens.

2. **Text truncation at 3,000 chars.** Article bodies average ~2,646 chars. First paragraphs
   contain the credit signal 90% of the time. 3,000 covers 75% of articles in full. This
   keeps per-article cost down while capturing nearly all signal content.

3. **Models:** Calibration + Audit use `claude-sonnet-4-5-20250929` (accurate, $3/M input).
   Bulk uses `claude-haiku-4-5-20251001` (fast/cheap, $0.25/M input). Config-driven — swap
   model IDs in `configs/labeling_config.yaml` without code changes.

4. **Resumable pipeline.** Each label is appended to JSONL immediately after the API returns.
   On restart, `get_completed_urls()` reads existing JSONL, builds a set of done URLs, skips them.
   A crash at article 8,000 loses zero work.

5. **Audit selection criteria:** Re-check with Sonnet if ANY of:
   `credit_relevant == 1` OR `signal_direction != 0` OR `confidence == "low"` OR `parse_error`.
   This focuses the expensive model on "interesting" articles (~4-6K estimated).

**Prompt design — boundary rules (critical for label quality):**

The system prompt anchors the model as a credit analyst with 8 explicit boundary rules:
- General business expansion is NOT credit-relevant unless it implies leverage/funding pressure
- Management changes are NOT credit-relevant unless tied to governance concerns/distress
- Regulatory actions ARE credit-relevant if they restrict operations/capital adequacy
- Revenue/profit growth is NOT credit-relevant unless it affects debt servicing
- Stock price movements alone are NOT credit-relevant
- Awards, CSR, brand marketing are NEVER credit-relevant
- Legal/court cases ARE credit-relevant if they involve debt recovery/insolvency/large penalties
- Asset sales during financial stress → direction=-1 (distress signal despite temporary liquidity)

Output format: strict JSON with 7 fields. If credit_relevant=0, other fields forced to defaults.
Full prompt text lives in `configs/labeling_config.yaml` → `prompt.system`.

**Data flow:**
```
gdelt_for_labeling.csv (17,299)
    │
    ├─ label_sampler.py ──→ labels_calibration_sample.csv (300) ✅ DONE
    │                              │
    │                     label_articles.py --phase calibration ✅ DONE (300, 0 errors)
    │                              │
    │                     labels_calibration.jsonl (300 labels) ✅ DONE
    │                              │
    │                     11 few-shot examples → config YAML ✅ DONE
    │
    ├─ label_articles.py --phase bulk ✅ DONE (17,299, 0 errors after retry)
    │                              ──→ labels_bulk.jsonl (17,299)
    │                                                          │
    │                                               label_audit.py select --targeted ✅ DONE
    │                                                          │
    │                                               audit_candidates.csv (313: 300 sample + 13 low-conf)
    │                                                          │
    │                                               label_audit.py run (Sonnet, ~$2) ✅ DONE
    │                                                          │
    │                                               labels_audit.jsonl (313)
    │                                                          │
    │                                               label_audit.py merge ✅ DONE
    │                                                          │ 82.3% agreement (accepted)
    └──────────────────────────────────────────→ labels_final.jsonl (17,274 merged)
                                                 labels_final.csv (for Excel review)
```

**Execution steps:**
1. ✅ Create `.env` with `ANTHROPIC_API_KEY=sk-ant-...` (gitignored)
2. ✅ Run calibration: 300 articles with Sonnet, 0 parse errors, 50/50 credit split
   - Output: `data/processed/labels_calibration.jsonl`
3. ✅ Reviewed 300 labels, added 11 few-shot examples to `configs/labeling_config.yaml`
   - 5 deterioration (asset_quality, liquidity, funding, governance, contagion)
   - 4 not-credit-relevant (stock price, routine mgmt, awards, business expansion)
   - 2 improvement (rating upgrade, capital raise)
   - Added boundary rule: distress-era asset sales → direction=-1
4. ✅ Run bulk: 17,299 articles with Haiku, 0 errors after retry
   - 9,299 credit-relevant (53.8%), 6,506 deterioration, 1,973 improvement
   - Output: `data/processed/labels_bulk.jsonl`
   - Cost: ~$36 USD (higher than $5 estimate due to 11 few-shot examples tripling input tokens)
5. ✅ Targeted audit select: `python -m src.data.label_audit select --targeted`
   - 313 candidates: 300 stratified sample + 13 low-confidence + 0 parse errors
   - Full audit (9,299) would cost ~$58 — targeted costs ~$2
   - Output: `data/processed/audit_candidates.csv`
6. ✅ Targeted audit run: `python -m src.data.label_audit run`
   - 313 articles re-labeled with Sonnet, 0 parse errors, 104 seconds
   - Output: `data/processed/labels_audit.jsonl`
7. ✅ Merge: `python -m src.data.label_audit merge`
   - Haiku↔Sonnet agreement on credit_relevant: **82.3%** (247/300 sample)
   - Below 90% threshold, but all disagreements are Haiku over-labeling (cr=1→cr=0)
   - Decision: ACCEPT AS-IS. For a credit early-warning system, false positives
     (routine articles labeled credit-relevant) are cheap; false negatives (missed
     real signals) are expensive. ~17.7% noise is tolerable — model evaluation
     against held-back rating actions will reveal if it actually hurts.
   - Sonnet overrides applied to all 313 audited articles
   - Output: `data/processed/labels_final.jsonl` (17,274) + `labels_final.csv`
8. ✅ Spot-check labels against rating_windows ground truth
   - 17,009 articles matched to rating actions within 90 days
   - **Deterioration recall: 75.5%** (3,543/4,694 articles near downgrades were flagged)
   - **Deterioration precision: 56.4%** (time-window noise inflates false alarms — expected)
   - Misses fall into 3 patterns: stock price articles (model correct), routine corporate actions (model correct), distress-era capital raises (model wrong — labels as improvement)
   - Full report: `reports/phase1_label_quality.md`
   - Sample for review: `data/processed/spotcheck_sample.csv` (50 stratified rows)

**Verification results:**
- Parse error rate: **0.0%** across all phases (calibration, bulk, audit)
- Haiku/Sonnet agreement on `credit_relevant`: **82.3%** on 300 stratified sample
  - All disagreements are Haiku over-labeling (cr=1→cr=0): broker reports, routine
    issuances, small debt recovery cases, business performance metrics
  - Accepted: false positives are cheap for credit early-warning; evaluation against
    rating actions will reveal if noise hurts model performance
- Total labeling cost: ~$38 USD (calibration ~$1.50 + bulk ~$36 + audit ~$2)
- Spot-check against rating_windows: 75.5% recall, 56.4% precision (see `reports/phase1_label_quality.md`)

**Label fields:** credit_relevant (0/1), signal_direction (-1/0/+1), signal_type (liquidity, asset_quality, regulatory, contagion, governance, funding, operational, other), sector_wide (0/1), confidence (low/medium/high), reasoning (one sentence)

**Ground truth:** rating_windows held back from LLM prompt — did entity get downgraded within 6 months? This is never shown to the labeling model (would be data leakage). Used only for post-hoc evaluation.

### 1.4 Training Data Format ✅ COMPLETE

**Design report:** `reports/phase1_4_training_data_design.md` (full rationale for all decisions)

**Format:** Three-field JSONL (instruction / input / output) following FinGPT/FinRLlama convention.

**Instruction** (fixed across all examples):
> "Assess whether this news article contains signals relevant to the credit quality of the mentioned Indian financial institution."

Why "financial institution" not "NBFC": corpus includes banks (YES Bank 1,590 articles, Lakshmi
Vilas Bank 231, PMC Bank 59) and infrastructure finance (IL&FS 94). ~11% of articles are non-NBFC.

**Input fields:** Entity + Date + Title + Article text (3,000 char truncation)

**Output format:** Structured text with strict vocabulary and END stop token (not JSON).
- JSON is all-or-nothing: one malformed brace loses the entire output
- Structured text: each field on its own line, independently recoverable
- END token teaches the model a hard stop; set as stop sequence during inference

Credit-relevant article:
```
CREDIT_RELEVANT: Yes
DIRECTION: Deterioration
SIGNAL_TYPE: asset_quality
SECTOR_WIDE: No
CONFIDENCE: High
REASONING: 11.4% of PFC's loan book entering insolvency with write-off risk.
END
```

Not credit-relevant (short-form — model learns to reject quickly):
```
CREDIT_RELEVANT: No
REASONING: Stock price movements with no credit quality information.
END
```

Strict vocabulary (parser rejects anything outside these):
- CREDIT_RELEVANT: Yes / No
- DIRECTION: Deterioration / Improvement / Neutral
- SIGNAL_TYPE: liquidity / asset_quality / regulatory / contagion / governance / funding / operational / other
- SECTOR_WIDE: Yes / No
- CONFIDENCE: Low / Medium / High

**Class balancing:** Keep natural distribution (deterioration 70% of cr=1). Don't distort data.
Adjust via loss weighting or threshold tuning in Phase 2 if needed.

**Split strategy:** Temporal (date-based) to prevent data leakage.

**Actual split results:**

| Split | Date Range | Articles | CR=1 | Det. | Imp. |
|-------|-----------|------:|------:|------:|------:|
| Train | 2017-04 to 2021-12 | 9,591 | 5,622 | 3,886 | 1,210 |
| Validation | 2022-01 to 2023-06 | 2,247 | 812 | 334 | 331 |
| Test | 2023-07 to 2024-12 | 2,133 | 721 | 386 | 263 |
| Entity Holdout | 2017-06 to 2024-12 | 3,303 | 2,065 | 1,850 | 161 |

**Entity holdout diagnostic:** 3 entities removed entirely from training (single-entity articles
only) to test whether the model learned text patterns or just entity names.

| Entity | Articles | Profile | Purpose |
|--------|------:|---------|---------|
| DHFL | 1,243 | 91.3% deterioration | Crisis detection on unseen entity |
| Reliance Capital | 688 | 80.2% deterioration | Different crisis arc — generalization |
| Cholamandalam | 1,372 | 11.9% deterioration | False positive control on stable entity |

Multi-entity articles mentioning held-out entities stay in training (minor contamination, noted).

**Output robustness check:** Two-phase approach.
- ✅ Phase 1.4: strict parser + 37 tests against synthetic edge cases (all pass)
- ✅ Phase 2.1 (Colab): 0% parse rate on base model — 100% totally_unstructured (free-form essays).
  Format is unfamiliar but model understands credit concepts → proceed to LoRA training.

**Files created:**

| File | Purpose |
|------|---------|
| `configs/training_config.yaml` | Split cutoffs, instruction text, field vocab, held-out entities |
| `src/data/format_training.py` | Join labels + articles → instruction/input/output JSONL |
| `src/data/parse_training_output.py` | Strict parser for structured text output format + format_output_text() |
| `tests/test_format_training.py` | 37 tests for formatter + parser (55 total across test suite) |

**Output files:**
- `data/processed/train.jsonl` (9,591), `val.jsonl` (2,247), `test.jsonl` (2,133)
- `data/processed/entity_holdout.jsonl` (3,303)

**Total training examples:** 17,274 (all articles, cr=0 included as negative examples, 100% join rate)

---

## Phase 2: Model Training (Days 9-14)
**Goal:** Fine-tune a model that distinguishes credit signals from generic sentiment.

### 2.1 Base Model Evaluation ✅ COMPLETE
**Primary:** Qwen 2.5-7B-Instruct (strong on financial text, multilingual for future Hindi extension)
**Fallback:** LLaMA 3.1-8B-Instruct (more community support, FinRLlama was built on LLaMA)

**What was built:**
- `src/training/evaluate.py` — Canonical evaluation module: strict parser (copied from `parse_training_output.py`),
  failure mode taxonomy (7 buckets: missing_field, wrong_vocab, missing_end, extra_content, refusal,
  totally_unstructured, partial_format), per-field accuracy, per-entity holdout evaluation with
  direction-specific precision/recall, stratified sampling, decision reporting.
- `notebooks/phase2_1_base_model_eval.ipynb` — Self-contained Colab notebook:
  1. Setup: mount Drive, pip install transformers+bitsandbytes+accelerate, load Qwen 2.5-7B in 4-bit NF4
  2. Sample: 1,000 stratified from train.jsonl (500 CR=Yes, 500 CR=No)
  3. Inference: sequential with tqdm, greedy decoding, saves to `base_model_outputs.jsonl`
  4. Parse eval: strict parser + failure taxonomy with 3 examples per bucket
  5. Field accuracy: credit_relevant, direction, signal_type, sector_wide, confidence
  6. Holdout scaffold: per-entity eval function ready for post-training use
  7. Decision: GO (>80%) / INVESTIGATE (20-80%) / NO-GO (<20%)

**Results (2026-02-17):**
- Parse success rate: **0.0%** (0/1,000)
- Failure mode: 100% `totally_unstructured` — model writes free-form analyst paragraphs
- Content quality: decent (catches NPAs, downgrades, credit ratings) but zero format compliance
- Decision: **PROCEED TO TRAINING** — model understands credit concepts, just needs format training
- Raw outputs saved: `drive/MyDrive/india-credit-signals/data/processed/base_model_outputs.jsonl`

🎓 **Why 0% is fine:** The model has never seen our `CREDIT_RELEVANT: / DIRECTION: / END` format.
It defaulted to what instruct models do — write helpful essays. LoRA fine-tuning on 9,591 examples
will teach the format. The fact that it already understands credit concepts (from pre-training)
means the training only needs to teach structure, not domain knowledge.

### 2.2 LoRA Training ✅ COMPLETE
**Notebook:** `notebooks/phase2_2_lora_training.ipynb` — Self-contained Colab notebook (10 cells).
**Training report:** `reports/phase2_2_training_results.md`

**Architecture decisions:**

📐 **TRL SFTTrainer** (not custom training loop):
- Handles Qwen's `<|im_start|>/<|im_end|>` chat template automatically
- `assistant_only_loss=True` → loss only on assistant (output) tokens — model doesn't waste
  gradient updates learning to reproduce the instruction/article text (~80% of token budget)
- Integrates with HuggingFace Trainer for checkpointing, eval, best-model selection

📐 **QLoRA on T4**: 4-bit NF4 base model + fp16/bf16 LoRA adapters.
- Peak VRAM: ~10-12 GB, fits T4 (15GB) with gradient checkpointing
- Gradient checkpointing trades ~30% training speed for ~40% VRAM savings

📐 **lr=5e-4** (not 2e-4 as originally planned):
- QLoRA needs higher lr because gradients pass through quantized weights with reduced precision
- 5e-4 is the QLoRA paper's recommended starting point
- Cosine schedule with 100 warmup steps ramps up gently

**Training configuration:**
- **LoRA:** r=16, alpha=32, dropout=0.05
- **Target modules:** q_proj, v_proj (attention) + gate_proj, up_proj, down_proj (MLP) — 5 of 7
- **MLP rationale:** gate/up/down control vocabulary and output distribution, critical for
  learning our structured format (CREDIT_RELEVANT: Yes/No, specific vocab words)
- **Training:** 3 epochs, batch=4, grad_accum=4 (effective batch=16), cosine scheduler
- **Eval:** every 500 steps, save best model on val loss, load best at end
- **Sequence length:** 2048 tokens (longest example ~1,200 tokens, generous headroom)
- **Precision:** bf16 on Ampere+ (A100), fp16 on Turing (T4) — automatic detection
- **Adapter size:** 152.8 MB saved to Drive (5 target modules × r=16)

**Data format:**
- instruction/input/output JSONL → messages format (system/user/assistant roles)
- SFTTrainer applies Qwen chat template and masks instruction+input tokens from loss

**Qwen-specific gotchas handled:**
1. `pad_token (151643) ≠ eos_token (151645)` — prevents loss masking corruption
2. No bos_token override — prevents double-bot artifact
3. `padding_side="right"` — causal LMs need right-padding
4. `gradient_checkpointing_kwargs={"use_reentrant": False}` — required for LoRA compat
5. `max_seq_length=2048` — covers all examples with headroom

**Evaluation cells (same notebook):**
- Cell 8: 500 validation examples (stratified) → parse rate + field accuracy
- Cell 9: 500 test examples (forward-looking 2023-07 to 2024-12) → the honest number
- Cell 10: Full entity holdout (3,303 examples) → per-entity precision/recall
  - DHFL (1,243, 91% det.) — crisis detection on unseen entity
  - Reliance Capital (688, 80% det.) — generalization across crisis types
  - Cholamandalam (1,372, 12% det.) — false positive control on stable NBFC

**Estimated cost:** ~$0 on Colab Pro subscription (T4)

**Training results (2026-02-19):**
- 145 min training on T4, 1,800 steps (3 epochs)
- Best checkpoint at **step 500** (~0.83 epochs) — overfitting after that
  - Val loss: 1.49 (step 500) → 1.50 (step 1000) → 1.57 (step 1500)
  - `load_best_model_at_end` auto-rolled back to step 500
- Format learned perfectly in <1 epoch (0% → 100% parse rate)

| Split | Parse Rate | CR Acc | Direction | Signal Type | Sector Wide | Confidence |
|-------|-----------|--------|-----------|-------------|-------------|------------|
| Base model (2.1) | 0.0% | — | — | — | — | — |
| Validation (500) | 100.0% | 83.0% | 80.3% | 84.5% | 95.7% | 70.0% |
| Test (500) | 99.6% | 89.8% | 77.2% | 76.8% | 93.4% | 58.3% |
| Entity holdout (3,303) | 100.0% | 92.7% | 92.3% | 70.6% | 90.0% | 92.6% |

**Per-entity holdout results:**

| Entity | CR Acc | Dir Acc | Det. Precision | Det. Recall | Verdict |
|--------|--------|---------|----------------|-------------|---------|
| DHFL (1,243) | 97.3% | 96.5% | 96.3% | 97.7% | Excellent — spots unseen crisis |
| Reliance Capital (688) | 93.9% | 87.5% | 90.4% | 90.0% | Strong — generalizes across crisis types |
| Cholamandalam (1,372) | 87.9% | 83.7% | 67.0% | 73.6% | Moderate — 33% false positive rate on stable NBFC |

**Key findings:**
1. **Format learning is solved.** 0% → 100% parse rate. Structured text output > JSON confirmed.
2. **Crisis detection works on unseen entities.** DHFL 97.7% recall proves the model learned
   distress language patterns, not entity names.
3. **Generalization confirmed.** Reliance Capital (different crisis arc) still gets 90% det. recall.
4. **False positives on stable NBFCs** are the main weakness. Cholamandalam's 33% det. false positive
   rate likely inherited from Haiku over-labeling (82.3% agreement). Manageable with confidence
   thresholds or cleaner training data.
5. **Signal type is inherently ambiguous** (70.6%). Categories overlap (liquidity vs asset_quality
   vs contagion). May not be improvable — and may not need to be for the use case.
6. **Overfitting after <1 epoch.** Best checkpoint was step 500 of 1,800. For next fine-tune:
   use 1 epoch + early stopping to save Colab time.

**Decision: PROCEED TO BACKTESTING.** The SFT model is strong enough. RLMF (Phase 2.3) is
optional polish, not a prerequisite. Priority is Phase 2.4: backtest against actual rating actions
to measure lead time and real-world false positive rates.

**Files:**

| File | Purpose |
|------|---------|
| `notebooks/phase2_2_lora_training.ipynb` | Self-contained training + evaluation notebook |
| `reports/phase2_2_training_results.md` | Full training results report |

**Output files (saved to Drive):**
- `data/models/qwen-credit-lora/` — LoRA adapters + tokenizer (152.8 MB)
- `data/processed/finetuned_val_outputs.jsonl` — val set predictions
- `data/processed/finetuned_test_outputs.jsonl` — test set predictions
- `data/processed/finetuned_holdout_outputs.jsonl` — entity holdout predictions

### 2.3 RLMF Adaptation (DEFERRED)
Deferred — SFT model is strong enough. RLMF is optional polish for a future iteration.

### 2.4 Backtest Against Rating Actions ✅ COMPLETE
**Full plan:** `BACKTEST_PLAN.md`
**Report:** `reports/phase2_4_backtest_results.md`

Backtested the LoRA model's predictions against actual CRISIL/ICRA/CARE rating actions.
This answers the hardest question: does the model predict real downgrades, and how early?

**Headline results:**

| Entity | Rating Actions | Signal Coverage | Mean Lead Time | First Signal |
|--------|---------------|----------------|----------------|-------------|
| DHFL | 23 (11 downgrades, 6 defaults) | **23/23 (100%)** | **160 days** | Nov 4, 2018 |
| Reliance Capital | 15 (8 downgrades, 3 defaults) | **15/15 (100%)** | **156 days** | Nov 20, 2018 |
| Cholamandalam | 0 (clean record) | — | — | 13% false positive rate |

**Best alert threshold:** N≥5 signals in 14-day window, 90-day lookahead
→ Precision: 79%, Recall: 73%, F1: 0.760

**Key insight:** The model flagged every DHFL and RelCap downgrade 3-6 months early.
But the always-deterioration baseline gets similar lead times because crisis entities
have ~80-93% negative articles. The model's real value is:
1. **Precision on stable entities** — 13% Chola FP vs 100% for always-det
2. **Alert-level precision** — 79% at N≥5 threshold
3. **Structured metadata** — signal type, confidence for actionable triage

**Test set:** 500 predictions from quiet period (2023-07 to 2024-12). Only 5 matching
rating actions. Not statistically meaningful — confirms holdout is the real backtest story.

**Files:**

| File | Purpose |
|------|---------|
| `src/training/backtest.py` | All analysis: lead time, alerts, baselines, report generation |
| `configs/backtest_config.yaml` | Thresholds, windows, entity aliases |
| `tests/test_backtest.py` | 30 unit tests (extractors, lead time, alerts, edges) |
| `reports/phase2_4_backtest_results.md` | Full backtest report |
| `BACKTEST_PLAN.md` | Durable plan reference |

### 2.4b Baseline Comparisons (DEFERRED)
Deferred — not needed for building the system. Useful later for contest paper or
build-vs-buy decision (prompted Opus vs LoRA vs FinRLlama).

---

## Phase 3: Contagion Layer ✅ COMPLETE
**Goal:** When one NBFC shows distress signals, propagate warnings to similar entities.
This is what makes the system a *sector-level* early warning tool, not just a single-entity
classifier. The pitch to the global head: "The system didn't just catch DHFL. It would have
flagged the entire housing finance sector 3 months before the cascade."

**Why this matters for work:** You don't monitor one NBFC — you monitor a sector/portfolio.
The IL&FS/DHFL crisis didn't stay contained. It spread through housing finance (PNB Housing,
Indiabulls, Piramal), then infrastructure (SREI), then broader NBFCs. A contagion layer
captures this cascade effect automatically.

**Full plan:** `CONTAGION_PLAN.md`
**Report:** `reports/phase3_contagion_results.md`

### 3.1 Entity Graph ✅ DONE
- 44 NBFC entities across 6 subsectors → weighted adjacency graph (946 edges)
- Edge weights: intra-subsector=0.8, cross-subsector=0.1 (subsector-only v1)
- Alias resolution for entity name matching across data sources
- `src/signals/entity_graph.py` + `tests/test_entity_graph.py` (23 tests)

### 3.2 Signal Propagation ✅ DONE
- Direct scoring: `direction_multiplier × confidence_weight × sector_wide_bonus`
- Contagion: `edge_weight(P,E) × rolling_direct(P, D, window) × peer_discount`
- v2: contagion normalized by n_contributing_peers (`normalize_by_peers: true`)
- Additive (not multiplicative) — entities with zero direct signals still get contagion
- `src/signals/propagation.py` + `tests/test_propagation.py` (24 tests)

### 3.3 Contagion Backtest ✅ DONE (v2 normalized)
Two crisis replays validated. Results below are v2 (normalized, threshold=4.0).

**IL&FS/DHFL 2018-19 (housing finance):**

| Target Entity | First Action | Lead Time | Improvement | Notes |
|---------------|-------------|-----------|-------------|-------|
| PNB Housing | 2020-02-21 | 511d | **+483d** | Contagion is what pushes past threshold |
| Indiabulls HF | 2019-08-30 | 343d | **+7d** | Direct + contagion |
| Can Fin Homes | 2019-05-06 | 220d | **+220d** | Zero direct — contagion only |
| Piramal | 2019-05-07 | 210d | **+210d** | Zero direct — contagion only |
| Reliance Home Finance | 2019-04-03 | 206d | **+12d** | Direct + contagion |

**SREI/RelCap 2019-22 (infrastructure):**

| Target Entity | First Action | Lead Time | Improvement |
|---------------|-------------|-----------|-------------|
| SREI Equipment | 2020-06-03 | 356d | **+112d** |

**Key results (v2):**
- Intra-subsector entities get 3.7× more contagion than cross-subsector (improved from 3.5×)
- Can Fin Homes and Piramal had ZERO direct signals — contagion was their ONLY early warning
- Cross-sector controls (Chola 6.3%, Bajaj 8.0%) now breach warning threshold <10% of days
  (was 85% in v1 — fixed by peer-count normalization + threshold recalibration)

**Post-demo improvements (see CONTAGION_PLAN.md):**
1. ~~Score normalization~~ ✅ DONE
2. ~~Threshold recalibration~~ ✅ DONE
3. Funding profile edges — wholesale/retail similarity (highest remaining priority)
4. Asymmetric weights — large entity stress hits small entities harder
5. Exponential decay, full-corpus inference (lower priority)

### Files

| File | Purpose |
|------|---------|
| `src/signals/entity_graph.py` | Entity graph: 44 nodes, 946 edges, subsector-based weights |
| `src/signals/propagation.py` | Direct scoring + contagion propagation (v2: peer-count normalization) |
| `src/signals/contagion_backtest.py` | Crisis replay engine + report generation |
| `configs/contagion_config.yaml` | Weights, windows, thresholds, 2 crisis definitions |
| `tests/test_entity_graph.py` | 23 tests |
| `tests/test_propagation.py` | 24 tests (20 original + 4 normalization) |
| `tests/test_contagion_backtest.py` | 13 tests |
| `reports/phase3_contagion_results.md` | Full backtest report (v2) |
| `CONTAGION_PLAN.md` | Durable plan reference |
| **Total: 145 tests pass** (141 existing + 4 new) | |

---

## Phase 4: Dashboard & Demo ✅ COMPLETE
**Goal:** Visual demo for global head + usable internal tool.
Built AFTER contagion layer so the dashboard can show sector-level views.

### 4.1 Tech Stack ✅ DONE
📐 **Streamlit for V1** (not React). Python-only, fast to build, DS team already knows it.
Move to React only if the global head says "build this for the desk" and it needs to scale.
- **Frontend:** Streamlit + Plotly (interactive charts with hover/zoom)
- **Data:** Parquet + JSON files from export pipeline (no database needed)
- **Theme:** Light mode, white backgrounds, indigo/purple/amber/orange/crimson/teal palette
- **Run:** `streamlit run src/dashboard/app.py`

### 4.2 Data Export Pipeline ✅ DONE
- `src/signals/export_dashboard_data.py` — runs full scoring pipeline → 6 files in `data/dashboard/`
- `tests/test_export_dashboard_data.py` — 19 tests
- Output: entity_scores (82,781 rows), signals (17,293), edges (946), ratings (1,654), metadata (44), crisis results (2)
- Runtime: ~21 min on M1 (contagion propagation bottleneck)

### 4.3 Dashboard Views ✅ DONE (5 views, 2,362 lines)

1. **Entity Timeline** (the money shot): Rolling 30d deterioration score vs time, with
   threshold crossing markers and rating action vertical lines. Toggle direct/contagion/total.
   DHFL shows 160-day lead time gap. Can Fin Homes shows contagion-only warning.
2. **Sector Heatmap:** 44 entities grouped by subsector, colored by rolling score (green→red).
   Animate through 2018 to see housing finance light up while vehicle finance stays green.
3. **Contagion Network:** Force-directed graph (Plotly). Nodes sized by score, colored by subsector.
   Edges show contagion flow with thickness proportional to weight × signal strength.
4. **Signal Feed:** Sortable/filterable table of individual article signals. Filter by entity,
   direction, date, confidence. Shows title, signal type, reasoning.
5. **Alert Dashboard:** Active threshold breaches with precision context from backtest.
   "79% of similar alerts preceded a downgrade within 90 days."

### 4.4 Dashboard Files

| File | Lines | Purpose |
|------|-------|---------|
| `.streamlit/config.toml` | 12 | Theme: light mode, white backgrounds |
| `src/dashboard/app.py` | 138 | Main app: sidebar nav + page routing |
| `src/dashboard/utils/data_loader.py` | 176 | Load parquet/JSON with `@st.cache_data` |
| `src/dashboard/utils/styling.py` | 150 | Color scales, theme constants, CSS |
| `src/dashboard/views/entity_timeline.py` | 755 | Rolling score timeline + threshold crossings |
| `src/dashboard/views/sector_heatmap.py` | 234 | Subsector heatmap by rolling score |
| `src/dashboard/views/contagion_network.py` | 332 | Network graph of signal propagation |
| `src/dashboard/views/signal_feed.py` | 246 | Article-level signal table with filters |
| `src/dashboard/views/alert_dashboard.py` | 319 | Active alerts with precision context |

### 4.5 Demo Script for Global Head
See `DASHBOARD_PLAN.md` → Demo Script section for the full 6-step walkthrough.
Key beats: DHFL 160d lead time → Can Fin contagion-only → sector heatmap animation →
network graph → alert precision → "your DS team can replicate this for any sector."

---

## Phase 5: Production & Extension (Future)
**Goal:** Make the system operational and extensible to other sectors.

### 5.1 Inference Pipeline
- Ingest new articles daily (GDELT, firm news feeds, or Reuters/Bloomberg)
- Run model inference (GPU cluster or Colab batch)
- Update dashboard, send alerts on threshold breaches

### 5.2 Sector Extension
- Architecture is sector-agnostic: news → LLM → structured signals → contagion → dashboard
- DS team swaps in new entity graph + training data for other sectors
- Document the pipeline for replicability

### 5.3 Contest Submission (Optional)
If timing works, package for FinRL/FinAI 2026 or SecureFinAI 2026.
The work tool is the priority; the contest is a bonus.

---

## What to Discuss in Claude Chat vs Build in Claude Code

| Topic | Where |
|-------|-------|
| Architecture decisions, tradeoffs | Claude Chat (this project) |
| Understanding LoRA, RLMF, training concepts | Claude Chat |
| "Why did this fail?" debugging strategy | Claude Chat |
| Writing actual code, scripts, pipelines | Claude Code |
| Data scraping, cleaning, labeling | Claude Code |
| Model training, evaluation scripts | Claude Code |
| Frontend/dashboard building | Claude Code |
| Git workflow, deployment | Claude Code |

---

## Key Reference Links
| Resource | URL |
|----------|-----|
| FinRLlama Code | https://github.com/Arnav-Gr0ver/ICAIF_FinRL-2024 |
| FinRLlama Model | https://huggingface.co/SesameStreet/FinRLlama-3.2-3B-Instruct |
| FinGPT Framework | https://github.com/AI4Finance-Foundation/FinGPT |
| FinRL Contest 2025 | https://open-finance-lab.github.io/FinRL_Contest_2025/ |
| FinAI Contest 2025 | https://open-finance-lab.github.io/FinAI_Contest_2025/ |
| Benchmarking Paper | https://arxiv.org/abs/2504.02281 |
| LLM Credit Risk Survey (60 papers) | https://arxiv.org/abs/2506.04290 |
| Graph ML for Credit | https://pubsonline.informs.org/doi/10.1287/ijds.2022.00018 |
| GDELT API | https://api.gdeltproject.org/api/v2/doc/doc |
| CRISIL Ratings | https://www.crisil.com/en/home/our-analysis/ratings/rating-list.html |
| ICRA Rationale Search | https://www.icra.in/Rationale/Search |
| RBI NBFC Regulations | https://www.rbi.org.in/Scripts/NotificationUser.aspx |
| NSE Corporate Filings | https://www.nseindia.com/companies-listing/corporate-filings-announcements |
