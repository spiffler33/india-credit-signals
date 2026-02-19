# CLAUDE.md — India NBFC Credit Signal Engine

## Project Context
Building a fine-tuned LLM to extract credit risk deterioration signals from news for Indian NBFCs. The user is a FICC markets professional learning ML/fine-tuning for the first time. Every task is both a build step and a learning opportunity.

## CRITICAL RULES

### Learning-First Output
**YOU MUST follow @.claude/output-styles/vats_output_style.md for ALL responses.** The user is learning ML fine-tuning, LoRA, RLMF, and data engineering. Every code block, every script, every pipeline step must include a brief explanation of WHY you made each decision. This is non-negotiable.

### Code Style
- Python 3.11+. Type hints on all functions.
- Use `uv` for package management (not pip directly).
- Prefer pathlib over os.path.
- All scripts must have `if __name__ == "__main__":` with argparse.
- Logging over print statements. Use `loguru`.
- Config in YAML files, not hardcoded.

### Architecture
- `src/` — all source code
- `src/data/` — scrapers, cleaners, labelers
- `src/training/` — fine-tuning scripts, evaluation
- `src/signals/` — inference, contagion logic, scoring
- `src/dashboard/` — React frontend
- `src/api/` — FastAPI backend
- `data/raw/` — raw scraped data (gitignored)
- `data/processed/` — cleaned, labeled data
- `data/models/` — saved model weights (gitignored)
- `configs/` — YAML config files
- `notebooks/` — Jupyter for exploration only, never production code
- `tests/` — pytest tests

### Git Workflow
- Never commit directly to main.
- Branch naming: `phase-{N}/{description}` (e.g., `phase-1/gdelt-scraper`)
- Commit messages: `[Phase N] Brief description`
- Commit after each working unit, not at the end of a session.

### What NOT To Do
- Never refactor code unless explicitly asked.
- Never install packages without explaining why they're needed.
- Never skip error handling on network requests (scrapers will fail).
- Never train a model without first showing the training data statistics.
- Never write a >200 line file without breaking it into modules.

## Key References
- Master plan: @PLAN.md
- Output style rules: @.claude/OUTPUT_STYLE.md
- Research briefing: @research/EM_Credit_Risk_LLM_Briefing.docx

## Environment Split
- **Local Mac (8GB M1):** Code editing, data scraping, git, dashboard dev — no GPU work
- **Google Colab Pro:** Model inference, training, anything needing GPU memory
- **GitHub:** `spiffler33/india-credit-signals` (private) — bridge between local and Colab
- Local uses `uv` for deps; Colab uses `pip` (Colab's pre-built env conflicts with uv venvs)

## Common Commands
```bash
# Run tests
pytest tests/ -v

# Load and summarize rating actions (training window 2016-2024)
python -m src.data.scrape_ratings

# Load all rating actions (including pre-2016)
python -m src.data.scrape_ratings --all

# Start API server (Phase 4)
uvicorn src.api.main:app --reload

# Start frontend (Phase 4)
cd src/dashboard && npm run dev

# Run model inference (on Colab, not local — Phase 2+)
python -m src.signals.predict --model data/models/latest --input data/processed/test.jsonl
```

## Current Phase
Phase 4 — Dashboard & Demo. Phases 0–3 COMPLETE.

**WHERE WE ARE NOW:** Phase 4 Step 3 (data export pipeline) COMPLETE.
`src/signals/export_dashboard_data.py` runs the full scoring pipeline and saves 6 files
to `data/dashboard/` (parquet + JSON). 164 tests pass (145 existing + 19 new).
Dashboard data: 82,781 entity-day scores, 17,293 signals, 946 edges, 1,654 rating actions.

**Project direction:** This is being built as a **real work tool** — not just a contest entry.
Goal: demonstrate to global head that LLM-based credit signal extraction + sector contagion
is the direction of travel for risk alerting systems. Data science team may extend to other
sectors. Priority order: ~~(1) contagion layer~~, ~~(2) contagion v2 fix~~,
(1) dashboard/demo, (2) inference pipeline.
Post-demo: funding profile edges + asymmetric weights (see CONTAGION_PLAN.md v2 items 3-4).

**Immediate next action:** Phase 4 Step 4 — Streamlit dashboard skeleton.
Build `src/dashboard/app.py` (main app + sidebar nav), `utils/data_loader.py` (parquet loading
with `@st.cache_data`), `utils/styling.py` (color scales, theme). Then Step 5: Entity Timeline
view (the money shot for the demo). See `DASHBOARD_PLAN.md`.

**Data sourcing workflow:** Complex scraping tasks are done in a separate project at
`/Users/coddiwomplers/Desktop/Python/data_scraping/`. Output CSVs are imported into this project.

**Key data files:**
- Rating actions: `data/raw/rating_actions_sourced.csv` (1,654 records, tracked in git)
- GDELT for labeling: `data/processed/gdelt_for_labeling.csv` (17,299 articles with body text)
- Labeling config: `configs/labeling_config.yaml` (prompts, models, thresholds, 11 few-shot examples)
- Calibration labels: `data/processed/labels_calibration.jsonl` (300 Sonnet labels, 0 errors)
- Bulk labels: `data/processed/labels_bulk.jsonl` (17,299 Haiku labels, 0 errors)
- Audit labels: `data/processed/labels_audit.jsonl` (313 Sonnet audit labels)
- **Final labels: `data/processed/labels_final.jsonl` (17,274 merged, 0 parse errors)**
- Final labels CSV: `data/processed/labels_final.csv` (for Excel review)
- Training config: `configs/training_config.yaml` (split dates, vocab, holdout entities)
- **Training splits: `data/processed/train.jsonl` (9,591), `val.jsonl` (2,247), `test.jsonl` (2,133)**
- **Entity holdout: `data/processed/entity_holdout.jsonl` (3,303 — DHFL/Reliance Capital/Cholamandalam)**

**Presentation / Reports:**
The `reports/` directory collects analysis artifacts for demos, presentations, and potential contest submission.
Add a new report at each major milestone (training eval, backtest results, contagion analysis, etc.).
Current reports:
- `reports/phase1_label_quality.md` — label quality spot-check, confusion matrix, miss patterns
- `reports/phase1_4_training_data_design.md` — training data format design decisions, rationale, dataset stats
- `reports/phase2_2_training_results.md` — LoRA training results, per-entity holdout, lessons learned
- `reports/phase2_4_backtest_results.md` — backtest vs actual rating actions, lead times, alert thresholds
- `reports/phase2_4_backtest_test_results.md` — test set backtest (limited, quiet period)
- `reports/phase3_contagion_results.md` — contagion backtest: 2 crises, 6 targets, lead time improvements

**Key labeling scripts:**
```bash
# Sample 300 articles for calibration (already run, output exists)
python -m src.data.label_sampler

# Label articles (calibration=Sonnet, bulk=Haiku)
python -m src.data.label_articles --phase calibration
python -m src.data.label_articles --phase bulk
python -m src.data.label_articles --phase calibration --dry-run  # test prompts

# Audit workflow (targeted mode is default)
python -m src.data.label_audit select --targeted  # 313 candidates (low-conf + 300 sample)
python -m src.data.label_audit select --full       # 9,299 candidates (all credit-relevant)
python -m src.data.label_audit run                 # re-label candidates with Sonnet
python -m src.data.label_audit merge               # combine into labels_final.jsonl

# Format training data (Phase 1.4)
python -m src.data.format_training               # produces train/val/test/entity_holdout JSONL
```

**Key Phase 1.4 scripts:**
- `src/data/format_training.py` — joins labels + articles, formats, temporal split, writes JSONL
- `src/data/parse_training_output.py` — strict parser for structured text output + format_output_text()
- `tests/test_format_training.py` — 37 tests (parser edge cases, formatter structure, split logic)

**Key Phase 2.1 files:**
- `src/training/evaluate.py` — canonical evaluation module (parser, failure taxonomy, per-entity holdout metrics)
- `notebooks/phase2_1_base_model_eval.ipynb` — Colab notebook: Qwen 2.5-7B 4-bit → 1,000 examples → parse eval → GO/NO-GO

**Key Phase 2.2 files:**
- `notebooks/phase2_2_lora_training.ipynb` — Colab notebook: QLoRA training + val/test/holdout eval in one session
  - LoRA: r=16, alpha=32, 5 target modules (q_proj, v_proj, gate_proj, up_proj, down_proj)
  - SFTTrainer with assistant_only_loss, cosine lr=5e-4, 3 epochs, effective batch=16
  - Qwen gotchas: pad≠eos, right-padding, no bos override, reentrant=False for grad checkpointing
  - Results: 100% parse rate, 97.7% det. recall on unseen DHFL, best checkpoint at step 500
- `reports/phase2_2_training_results.md` — full training results report with per-entity analysis

**Phase 2.2 output files (on Google Drive, also copied to local):**
- `data/models/qwen-credit-lora/` — LoRA adapters + tokenizer (152.8 MB)
- `data/processed/finetuned_val_outputs.jsonl` — val set predictions (500 examples)
- `data/processed/finetuned_test_outputs.jsonl` — test set predictions (500 examples)
- `data/processed/finetuned_holdout_outputs.jsonl` — entity holdout predictions (3,303 examples)

**Key Phase 2.4 files:**
- `src/training/backtest.py` — backtest analysis: lead time, alert precision/recall, baselines, report gen
- `configs/backtest_config.yaml` — thresholds, entity aliases, lookback windows
- `tests/test_backtest.py` — 30 unit tests (synthetic data + known DHFL dates)
- `reports/phase2_4_backtest_results.md` — holdout backtest results (DHFL 100% coverage, 160d mean lead)
- `BACKTEST_PLAN.md` — durable plan reference for Phase 2.4

**Key Phase 3 files:**
- `src/signals/entity_graph.py` — entity graph: 44 nodes, 946 edges, subsector-based weights
- `src/signals/propagation.py` — direct scoring + contagion propagation + rolling windows
- `src/signals/contagion_backtest.py` — crisis replay engine + report generation
- `configs/contagion_config.yaml` — weights, windows, thresholds, 2 crisis definitions
- `tests/test_entity_graph.py` — 23 tests
- `tests/test_propagation.py` — 20 tests
- `tests/test_contagion_backtest.py` — 13 tests
- `CONTAGION_PLAN.md` — durable plan reference with lessons learned

**Key Phase 4 files (so far):**
- `src/signals/export_dashboard_data.py` — runs scoring pipeline, saves 6 files to `data/dashboard/`
- `tests/test_export_dashboard_data.py` — 19 tests for export functions
- `data/dashboard/entity_scores.parquet` — 82,781 entity-day scores (gitignored, regenerated)
- `data/dashboard/signals.parquet` — 17,293 article-level signals
- `data/dashboard/contagion_edges.parquet` — 946 graph edges
- `data/dashboard/rating_actions.parquet` — 1,654 rating actions with outcome coloring
- `data/dashboard/entity_metadata.json` — 44 entities with subsector/peer info
- `data/dashboard/crisis_results.json` — 2 crisis replays with lead times

**Phase 3 headline results:**
- IL&FS/DHFL crisis: 5/5 housing finance targets get contagion lead times (280-587d)
- Can Fin Homes (+296d) and Piramal (+334d) had ZERO direct signals — contagion only
- SREI/RelCap crisis: SREI Equipment +71d improvement
- Intra-subsector gets 2.3-3.5× more contagion than cross-subsector
- v2 needed: threshold recalibration for dense graph (cross-sector controls breach 85% of days)

**Phase 2.4 headline results:**
- DHFL: 23/23 rating actions had prior signals, mean 160-day lead time, first signal Nov 4 2018
- Reliance Capital: 15/15 actions, mean 156-day lead time, first signal Nov 20 2018
- Cholamandalam: 0 downgrades, 13% false positive rate (179/1,372 articles)
- Best alert: N≥5 in 14-day window → 79% precision, 73% recall, F1=0.760
