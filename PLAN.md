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
- ⏳ FinRLlama inference blocked — Meta LLaMA access pending (not a blocker, moving on)
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
- Phase 2.1 (Colab): run 1,000 base-model outputs through parser, iterate format if >20% fail

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

### 2.1 Base Model Selection
**Primary:** Qwen 2.5-7B-Instruct (strong on financial text, multilingual for future Hindi extension)
**Fallback:** LLaMA 3.1-8B-Instruct (more community support, FinRLlama was built on LLaMA)

### 2.2 Training Configuration
Adapt FinRLlama's training script:
- LoRA: rank=16, alpha=32, target_modules=["q_proj", "v_proj"]
- Learning rate: 2e-4 with cosine scheduler
- Batch size: 4 (gradient accumulation 8)
- Epochs: 3 (with early stopping on val loss)
- Estimated cost: ~$50-100 on Colab Pro or Lambda Labs

### 2.3 RLMF Adaptation (Advanced — Week 3)
Instead of market feedback (price movements), use **rating feedback**:
- Reward = model's signal prediction vs actual rating action 6 months later
- Positive reward: model flagged deterioration AND entity was downgraded
- Negative reward: model flagged deterioration BUT entity maintained/upgraded
- This is the novel contribution — RLMF with credit outcomes instead of trading returns

### 2.4 Evaluation Metrics
- **Precision/Recall on credit events:** Can it correctly flag entities that will be downgraded?
- **Lead time:** How many days/weeks before the rating action did the model flag it?
- **False positive rate:** How many entities flagged but NOT downgraded?
- **Baseline comparison:** Prompted Opus on same test set (is fine-tuning actually better?)

---

## Phase 3: Contagion Layer (Days 15-20)
**Goal:** When RBI issues a regulatory change, propagate signals across all exposed NBFCs.

### 3.1 Entity Graph
Build a simple graph of Indian NBFCs:
- Nodes: Each NBFC (name, type, size, primary asset class)
- Edges: Shared characteristics (funding profile, asset class exposure, geography)
- Weight: Similarity score (two housing finance NBFCs are more connected than a housing NBFC and a gold loan NBFC)

### 3.2 Contagion Rules
```python
# When entity X gets a credit signal:
if signal.sector_wide:
    for entity in same_subsector(X):
        entity.score += signal.strength * edge_weight(X, entity)

# When RBI issues regulatory change:
for entity in affected_entities(regulation):
    entity.score += regulation.impact * entity.exposure
```

### 3.3 Rolling Scores
- Per-entity: 7-day, 30-day, 90-day rolling average of signal scores
- Per-subsector: Housing finance, infrastructure, microfinance, vehicle finance
- Threshold alerts: When rolling score crosses -0.5 (warning) or -0.8 (critical)

---

## Phase 4: Dashboard & Interface (Days 15-20, parallel with Phase 3)
**Goal:** Visual interface to see signals, entities, scores in real-time.

### 4.1 Tech Stack
- **Frontend:** React + Tailwind + Recharts (or Plotly)
- **Backend:** FastAPI (you already know this)
- **Database:** Supabase (you already know this) or SQLite for simplicity
- **Deployment:** Vercel (frontend) + Railway/Fly.io (backend)

### 4.2 Key Views
1. **Entity Dashboard:** List of NBFCs with current rolling credit scores, sparklines, last 5 signals
2. **Signal Feed:** Real-time feed of articles with credit relevance scores and signal types
3. **Contagion Map:** Network graph showing how signals propagate (use D3 or vis.js)
4. **Backtest View:** Timeline showing model predictions vs actual rating actions
5. **Model Comparison:** Side-by-side: fine-tuned model vs prompted Opus vs generic sentiment

### 4.3 Backtest Visualization
This is the money shot for any demo/hackathon:
- X-axis: time
- Y-axis: model's rolling credit score for entity
- Vertical lines: actual rating actions
- Show that the score drops BEFORE the vertical line = early warning works

---

## Phase 5: Contest Preparation (Days 21-30)
**Goal:** Package for FinRL/FinAI 2026 or SecureFinAI 2026 submission.

### 5.1 Watch These Repos
```
https://github.com/Open-Finance-Lab/FinRL_Contest_2025
https://github.com/Open-Finance-Lab/FinAI_Contest_2025
https://github.com/Open-Finance-Lab/SecureFinAI_Contest_2025
```
SecureFinAI 2026 repo exists but dates not announced. Check weekly.

### 5.2 Submission Requirements (based on past contests)
- Open-source code on GitHub
- Trained model weights on HuggingFace
- Technical report (4-6 pages, IEEE format)
- Reproducible results with clear instructions

### 5.3 What Makes This Win
Past winners won because of novel data pipelines, not model architecture. Your edge:
- **Credit-specific signals** (not generic sentiment) — nobody else is doing this
- **Regulator-to-sector contagion** — novel contribution to the field
- **EM focus with English-language data** — accessible and reproducible
- **Backtested against real rating actions** — concrete, verifiable results

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
