# Phase 2.4: Backtest Against Actual Rating Actions

> **Durable copy:** Save to `BACKTEST_PLAN.md` at project root (Step 0 below).
> Reference this document across context clears. Each step below is independently executable.

---

## Context

Phase 2.2 LoRA training produced a model with 100% parse rate and strong holdout metrics
(DHFL 97.7% det. recall, RelCap 90%, Chola 33% FP). But those metrics compare model
predictions to **LLM-generated labels** (Haiku/Sonnet). Phase 2.4 answers the harder question:
**does the model predict actual rating agency downgrades, and how early?**

### Critical Data Reality

| Window | Articles | Downgrades | Defaults | Backtest Value |
|--------|---------|------------|----------|----------------|
| Test (2023-07 to 2024-12) | 2,133 (500 predicted) | **1** (Indiabulls Nov 2023) | 0 | Low — quiet period |
| Entity Holdout (2017 to 2024) | 3,303 (all predicted) | **19** (DHFL 11, RelCap 8) | **9** (DHFL 6, RelCap 3) | **High** — two full crisis arcs |

The real backtest story is the **holdout entities**:
- **DHFL:** Catastrophic collapse Feb–Jul 2019 (11 downgrades, 6 defaults across 3 agencies).
  First downgrade Feb 3, first default Jun 5. 1,243 articles spanning 2017–2024.
- **Reliance Capital:** Parallel collapse Feb–Nov 2019 (8 downgrades, 3 defaults).
  First downgrade Mar 5, first default Sep 20. 688 articles.
- **Cholamandalam:** Zero downgrades ever. 1,372 articles. Perfect false-positive control.

### Existing Predictions (no re-run needed)

| File (on Google Drive) | Records | Has Entity? | Has Date? | Matching Strategy |
|------------------------|---------|------------|-----------|-------------------|
| `finetuned_holdout_outputs.jsonl` | 3,303 | Yes (field) | No | Match by index to `entity_holdout.jsonl` (same order, confirmed) |
| `finetuned_test_outputs.jsonl` | 500 | No | No | Match `expected` field to `test.jsonl` `output` field (unique REASONING text) |

No Colab inference re-run needed. Download 2 files from Drive → run all analysis locally.

---

## What to Build

### 1. `src/training/backtest.py` — Analysis Module (~250 lines)

Core functions:

```
load_predictions_with_metadata(pred_path, source_path, match_by="index"|"expected")
  → DataFrame[entity, date, title, predicted_cr, predicted_direction, predicted_confidence,
               expected_cr, expected_direction, ground_truth_direction]

load_rating_actions(path, action_types=["downgrade", "default"])
  → DataFrame[entity, date, action_type, from_rating, to_rating, agency]

compute_lead_time(predictions_df, actions_df, lookback_days=180)
  → For each rating action: earliest signal date, signal count, lead time in days
  → DataFrame[entity, action_date, action_type, first_signal_date, lead_time_days,
               n_signals_before, n_articles_before]

compute_alert_metrics(predictions_df, actions_df, n_threshold, window_days, lookahead_days)
  → Precision/recall at given threshold
  → dict[precision, recall, f1, n_alerts, n_true_positives]

sweep_alert_thresholds(predictions_df, actions_df)
  → Grid search over N ∈ {1,2,3,5}, M ∈ {14,30,60,90}, K ∈ {90,180}
  → DataFrame of all threshold combos with precision/recall

compute_entity_timeline(predictions_df, actions_df, entity)
  → Daily signal counts + rating action markers for one entity
  → For visualization: x=date, y=cumulative_det_signals, vertical_lines=rating_actions

compute_naive_baselines(predictions_df, actions_df)
  → Always-deterioration baseline, random baseline, label-agreement baseline
  → Same metrics as model for comparison

generate_backtest_report(all_results, output_path)
  → Write reports/phase2_4_backtest_results.md
```

Reuses from existing code:
- `src/data/parse_training_output.py` → `parse_training_output()` for parsing `generated` text
- `src/training/evaluate.py` → `extract_entity_from_input()` for getting entity/date from input field

### 2. `configs/backtest_config.yaml` — Parameters

```yaml
# Rating action types that count as "negative events" for backtest
negative_actions: [downgrade, default, watchlist_negative, outlook_negative]

# Lead time analysis
lookback_days: 180  # How far back to look for signals before a rating action

# Alert threshold grid search
alert_thresholds:
  n_signals: [1, 2, 3, 5]        # Minimum deterioration signals to trigger alert
  window_days: [14, 30, 60, 90]   # Rolling window size
  lookahead_days: [90, 180]        # How far forward to check for rating action after alert

# Entity matching (handle name variations)
entity_aliases:
  "DHFL": ["DHFL", "Dewan Housing"]
  "Reliance Capital": ["Reliance Capital"]
  "Cholamandalam": ["Cholamandalam"]
  "Indiabulls Housing Finance": ["Indiabulls Housing Finance", "Indiabulls Housing"]
```

### 3. `tests/test_backtest.py` — Unit Tests (~20 tests)

- Test `load_predictions_with_metadata()` — index matching and expected-field matching
- Test `compute_lead_time()` — known DHFL dates → expected lead times
- Test `compute_alert_metrics()` — synthetic data with known precision/recall
- Test `sweep_alert_thresholds()` — grid produces expected number of rows
- Test entity name matching with aliases
- Test edge cases: no signals before action, multiple actions same entity same day

### 4. `reports/phase2_4_backtest_results.md` — Generated Report

Sections:
1. **Executive Summary** — headline lead times, best threshold, false positive rates
2. **Lead Time Analysis** — per-entity tables with signal-before-downgrade timelines
3. **DHFL Deep Dive** — timeline: when did signals start vs. Feb 2019 first downgrade?
4. **Reliance Capital Deep Dive** — same analysis
5. **Cholamandalam False Positive Analysis** — signal rate on a stable NBFC
6. **Alert Threshold Grid** — precision/recall table across all thresholds
7. **Test Set Analysis** — limited (1 downgrade), but shows quiet-period signal rates
8. **Naive Baselines** — always-deterioration, random, label-agreement
9. **Limitations & Next Steps** — test window sparsity, Haiku bias, baseline comparisons

---

## Execution Plan

### Step 0: Save this plan to project (first action after approval)
Save this document to `BACKTEST_PLAN.md` at project root so it persists across sessions.

### Step 1: Download prediction files from Drive (manual, ~2 min)
User copies from Google Drive to local:
- `finetuned_holdout_outputs.jsonl` → `data/processed/finetuned_holdout_outputs.jsonl`
- `finetuned_test_outputs.jsonl` → `data/processed/finetuned_test_outputs.jsonl`

### Step 2: Write `configs/backtest_config.yaml`

### Step 3: Write `src/training/backtest.py`

### Step 4: Write `tests/test_backtest.py`

### Step 5: Run tests locally
```bash
pytest tests/test_backtest.py -v
```

### Step 6: Run backtest analysis locally
```bash
python -m src.training.backtest \
  --predictions data/processed/finetuned_holdout_outputs.jsonl \
  --source data/processed/entity_holdout.jsonl \
  --rating-actions data/raw/rating_actions_sourced.csv \
  --config configs/backtest_config.yaml \
  --output reports/phase2_4_backtest_results.md
```

### Step 7: Review report, commit

### Deferred to Phase 2.4b: Baseline Comparisons
- Prompted Opus on holdout articles (requires API calls, ~$5-10)
- FinRLlama 3.2-3B on holdout articles (requires Colab)
