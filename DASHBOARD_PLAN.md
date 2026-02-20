# Phase 3v2 + Phase 4: Contagion Fix & Dashboard — Implementation Plan

> **Durable copy:** Reference this document across context clears.
> Each step below is independently executable.

---

## Context

Phase 3 contagion layer is built and validated. The core propagation works — intra-subsector
entities get 2.3-3.5× more contagion than cross-subsector during crises, and Can Fin Homes /
Piramal got their ONLY early warnings through contagion (zero direct signals).

**Problem:** The absolute contagion scores are too high in a 44-entity dense graph. Cross-sector
controls (Chola, Bajaj) breach the warning threshold on 85% of crisis days. The system can
*rank* entities correctly but cannot generate *actionable alerts*.

**Solution:** Two-part fix:
1. **Contagion v2** — normalize scores by peer count, recalibrate thresholds
2. **Dashboard** — Streamlit app with 5 views for the global head demo

**Demo narrative for global head:**
1. "The model flagged DHFL 160 days before the first downgrade" → Entity Timeline
2. "Housing finance lit up red while vehicle finance stayed green" → Sector Heatmap
3. "Stress propagated from DHFL to 5 peers — 2 had zero direct signals" → Contagion Network
4. "Here's what the model extracted from each article" → Signal Feed
5. "Alerts fire for housing finance, stay quiet for diversified" → Alert Dashboard
6. "Your DS team can replicate this for any sector with the same pipeline" → Architecture slide

**Replicability goal:** DS team should be able to swap in a new entity graph + training data
for another sector (e.g., Indian banks, EM sovereign, European utilities) and get the same
pipeline: news → LLM extraction → contagion propagation → dashboard.

---

## Part 1: Contagion v2 (Fix Alerting)

### What's broken
- 44 entities × 43 peers each = contagion accumulates from entire sector
- Chola has 11 intra-subsector peers at 0.8 + 32 cross-subsector at 0.1
- 73% of Chola's contagion comes from its OWN diversified_nbfc peers, not from DHFL
- Warning threshold (2.0) calibrated for per-article scoring, not graph-based

### v2 Fixes (in priority order)

**Fix 1: Score normalization** (highest impact)
- Divide contagion_score by number of contributing peers
- `normalized_contagion(E, D) = contagion_score(E, D) / n_contributing_peers(E, D)`
- This controls for graph density: an entity with 40 peers shouldn't accumulate 40× more
  contagion than one with 5 peers
- Change in: `src/signals/propagation.py` → `compute_contagion_scores()`

**Fix 2: Threshold recalibration**
- After normalization, sweep new warning/critical thresholds
- Method: compute normalized scores for entire crisis period, find threshold where:
  - Housing finance targets breach > 80% of days
  - Cross-sector controls breach < 20% of days
- Update: `configs/contagion_config.yaml` → `score_thresholds`

**Fix 3: Re-run backtest**
- Verify: intra/cross ratio still holds (should improve with normalization)
- Verify: Chola/Bajaj warning breaches drop from 85% to < 20%
- Update: `reports/phase3_contagion_results.md`

### v2 Files Changed

| File | Change |
|------|--------|
| `src/signals/propagation.py` | Add normalization option to `compute_contagion_scores()` |
| `configs/contagion_config.yaml` | Add `normalize_by_peers: true`, update thresholds |
| `tests/test_propagation.py` | Add normalization tests |
| `reports/phase3_contagion_results.md` | Update with v2 results |
| `CONTAGION_PLAN.md` | Update v2 status |

### v2 Deferred (not needed for demo)
- Funding profile edges — needs manual data curation
- Asymmetric weights — needs entity size data
- Exponential decay — marginal improvement over hard window
- Full-corpus inference — needs Colab GPU time

---

## Part 2: Data Export Pipeline

The dashboard needs pre-computed data. Currently the backtest computes scores in memory
and writes only a markdown report. We need to save intermediate DataFrames.

### Export Script: `src/signals/export_dashboard_data.py`

Runs the full pipeline and saves:

| Output File | Contents | Used By |
|-------------|----------|---------|
| `data/dashboard/entity_scores.parquet` | Daily scores: entity, date, direct, contagion, total, rolling_7d/30d/90d | Timeline, Heatmap, Alerts |
| `data/dashboard/signals.parquet` | Per-article signals: entity, date, direction, signal_type, confidence, sector_wide, title, url | Signal Feed |
| `data/dashboard/contagion_edges.parquet` | Entity pairs with contagion flow: source, target, weight, contagion_contributed | Network Graph |
| `data/dashboard/rating_actions.parquet` | Rating actions: entity, date, agency, action_type, outcome | Timeline overlay |
| `data/dashboard/entity_metadata.json` | Entity info: name, subsector, status, peer count | Filters, labels |
| `data/dashboard/crisis_results.json` | Crisis replay summaries: lead times, improvements | Alert context |

Command: `python -m src.signals.export_dashboard_data --config configs/contagion_config.yaml`

---

## Part 3: Streamlit Dashboard

### Tech Stack
- **Streamlit** — pure Python, DS team already knows it
- **Plotly** — interactive charts (hover, zoom, click)
- **streamlit-agraph** or **pyvis** — network graph (evaluate which is simpler)
- **Data:** Parquet files from export pipeline (no database needed)

### File Structure (as built)
```
.streamlit/
    config.toml            # Streamlit theme: light mode, white backgrounds
src/dashboard/
    __init__.py
    app.py                 # Main Streamlit app (sidebar nav + page routing)
    views/
        __init__.py
        entity_timeline.py # View 1: Rolling score timeline + threshold crossings (755 lines)
        sector_heatmap.py  # View 2: Subsector heatmap by rolling score (234 lines)
        contagion_network.py # View 3: Network graph of signal propagation (332 lines)
        signal_feed.py     # View 4: Article-level signal table with filters (246 lines)
        alert_dashboard.py # View 5: Active alerts with precision context (319 lines)
    utils/
        __init__.py
        data_loader.py     # Load parquet/JSON, cache with @st.cache_data (176 lines)
        styling.py         # Color scales, theme constants, CSS (150 lines)
```
Total: 2,362 lines across 9 code files + 3 `__init__.py` + theme config.

### View 1: Entity Timeline (the money shot)

**What it shows:** For a selected entity, X=time, Y=cumulative deterioration score.
Vertical red lines = actual rating actions (downgrades/defaults).

**Controls:**
- Entity dropdown (default: DHFL)
- Date range slider
- Toggle: direct only vs direct+contagion

**Key visual:** The gap between the signal line rising and the red vertical line. That gap
is the lead time. For DHFL it's 160 days. For Can Fin Homes (contagion only), it's 296 days.

**Demo moment:** "See this gap? That's 5 months of early warning. And for Can Fin Homes,
the system had zero direct signals — this entire line is contagion from DHFL."

### View 2: Sector Heatmap

**What it shows:** Grid of all 44 entities, colored by rolling_30d score.
Green (low risk) → Yellow (elevated) → Red (high risk). Grouped by subsector.

**Controls:**
- Date slider (animate through time to show crisis spreading)
- Rolling window selector (7d / 30d / 90d)
- Score type: total, direct-only, contagion-only

**Key visual:** During 2018-Q4, housing_finance subsector turns red while
vehicle_finance stays green. That's the sector differentiation working.

**Demo moment:** "Watch what happens in November 2018. Housing finance goes red.
Diversified stays yellow. Vehicle finance is green. The system identified the right sector."

### View 3: Contagion Network

**What it shows:** Force-directed graph. Nodes = entities (sized by score, colored by
subsector). Edges = contagion flow (thickness = weight × signal strength).

**Controls:**
- Date selector (show network state on specific day)
- Subsector filter
- Min edge weight slider (declutter)

**Key visual:** DHFL node is large and red, with thick edges to housing finance peers
and thin edges to diversified entities.

**Demo moment:** "DHFL is the epicenter. Thick lines to housing finance. Thin lines to
diversified. The system knows the difference."

### View 4: Signal Feed

**What it shows:** Sortable, filterable table of individual article signals.
Columns: date, entity, title, direction, signal_type, confidence, sector_wide.
Click to expand article text + model reasoning.

**Controls:**
- Entity filter
- Direction filter (Deterioration / Improvement / All)
- Date range
- Confidence filter

**Demo moment:** "Here's what the model actually reads. This article about DHFL's
commercial paper rollover failure — the model tagged it as liquidity deterioration,
high confidence. Three months later, the first downgrade."

### View 5: Alert Dashboard

**What it shows:** Active alerts based on threshold breaches.
Table: entity, alert level (warning/critical), days since breach, peak score,
subsector, precision context ("79% of similar alerts preceded a downgrade within 90d").

**Controls:**
- Date selector
- Alert level filter
- Subsector filter

**Depends on:** Contagion v2 normalization (alerts are meaningless without it).

**Demo moment:** "Five housing finance entities are in warning. Zero diversified entities.
Based on backtesting, 79% of these warnings preceded actual downgrades."

---

## Implementation Order

| Step | What | Files | Commit |
|------|------|-------|--------|
| ~~1~~ | ~~Contagion v2: normalize + threshold recal~~ | ~~`propagation.py`, config, tests~~ | ✅ `a15f2a5` |
| ~~2~~ | ~~Re-run backtest with v2~~ | ~~Report update~~ | ✅ (included in Step 1 commit) |
| ~~3~~ | ~~Data export pipeline~~ | ~~`export_dashboard_data.py` + tests~~ | ✅ `[Phase 4] Add dashboard data export pipeline` |
| ~~4~~ | ~~Dashboard skeleton + data loader~~ | ~~`app.py`, `data_loader.py`, `styling.py`~~ | ✅ `20a6c59` |
| ~~5~~ | ~~Entity Timeline view (v3: rolling score + threshold crossings)~~ | ~~`entity_timeline.py`~~ | ✅ `20a6c59` |
| ~~6~~ | ~~Sector Heatmap view~~ | ~~`sector_heatmap.py`~~ | ✅ `20a6c59` |
| ~~7~~ | ~~Signal Feed view~~ | ~~`signal_feed.py`~~ | ✅ `20a6c59` |
| ~~8~~ | ~~Contagion Network view~~ | ~~`contagion_network.py`~~ | ✅ `20a6c59` |
| ~~9~~ | ~~Alert Dashboard view~~ | ~~`alert_dashboard.py`~~ | ✅ `20a6c59` |
| 10 | Polish + demo script | README updates, demo walkthrough | `[Phase 4] Dashboard complete: 5 views for demo` |

---

## Dependencies to Install

```bash
uv add streamlit plotly pyarrow
# Evaluate one of these for network graph:
uv add streamlit-agraph   # lightweight, Streamlit-native
# OR
uv add pyvis              # more features, generates HTML
```

---

## Success Criteria

| Criterion | How to verify |
|-----------|--------------|
| v2 contagion: cross-sector controls breach < 20% of days | Re-run backtest |
| v2 contagion: intra/cross ratio ≥ 2× | Re-run backtest |
| Dashboard loads in < 3 seconds | Parquet files, @st.cache_data |
| Entity Timeline shows DHFL 160d lead time visually | Manual check |
| Heatmap animates through 2018 crisis | Date slider |
| Network graph shows subsector clustering | Visual check |
| Signal feed loads 17K articles without lag | Pagination or virtual scroll |
| Alerts fire for housing finance, quiet for diversified | v2 thresholds |
| `streamlit run src/dashboard/app.py` works from repo root | Manual test |
| DS team can understand structure in 30 min | Clean code + comments |

---

## Demo Script (Updated with Contagion Story)

1. **Open Entity Timeline → DHFL**
   "Our fine-tuned model analyzed 17,000 news articles about Indian NBFCs. For DHFL,
   it flagged credit deterioration signals 160 days before the first downgrade. Every
   single one of DHFL's 23 rating actions had prior signals."

2. **Switch to Can Fin Homes**
   "Can Fin Homes had zero articles flagged by the model directly. But because our
   contagion layer knows Can Fin is a housing finance NBFC — same subsector as DHFL —
   it propagated the warning automatically. 296 days of early warning, purely from
   sector contagion."

3. **Show Sector Heatmap, animate through 2018**
   "Watch November 2018. Housing finance lights up red. Infrastructure turns yellow.
   Vehicle finance stays green. The system correctly identified which sector was
   at risk — without any manual analyst input."

4. **Show Contagion Network**
   "Here's how stress propagated. DHFL is the epicenter. Thick connections to housing
   finance peers. Thin connections to diversified NBFCs. The edge weights contain
   cross-sector spillover."

5. **Show Alert Dashboard**
   "Five housing finance entities are in warning status. Zero diversified entities.
   Based on our backtest, 79% of these alerts preceded actual downgrades within 90 days."

6. **The pitch**
   "This pipeline is sector-agnostic. News → LLM extraction → contagion graph → dashboard.
   Your DS team can replicate this for [European banks / EM sovereigns / any sector]
   by swapping in a new entity graph and training data. The architecture is the same."
