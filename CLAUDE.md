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
Phase 1 — Data Collection. Phase 0 complete. Phase 1.1, 1.2, and 1.3 Steps 1-4 (calibration + bulk) complete.

**WHERE WE ARE NOW:** Phase 1.3 Step 4 — Calibration (300, Sonnet) and Bulk (17,299, Haiku) labeling DONE.
11 few-shot examples added to config. Audit pipeline COMPLETE (targeted: 313 articles, 82.3% agreement).
Final merged labels at `data/processed/labels_final.jsonl` (17,274 labels, 0 parse errors).
See `PLAN.md` → Phase 1.3 → Step 4 execution checklist (steps 1-7 done, step 8 remains).

**Immediate next action:** Spot-check 50 labels against rating_windows ground truth (Step 8),
then move to Phase 1.4 (training data formatting).

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
```
