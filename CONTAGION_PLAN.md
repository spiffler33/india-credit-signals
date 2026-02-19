# Phase 3: Contagion Layer — Implementation Plan

> **Durable copy:** Save to `CONTAGION_PLAN.md` at project root.
> Reference this document across context clears. Each step below is independently executable.

---

## Context

Phase 2.4 backtest confirmed the fine-tuned model catches every DHFL and Reliance Capital
downgrade 3-6 months early. But it only detects signals for entities that have *their own*
news articles flagged. When DHFL collapsed in Nov 2018, the system said nothing about
Indiabulls, PNB Housing, Can Fin Homes — all housing finance NBFCs that got downgraded
6-14 months later in the same crisis cascade.

**Solution:** A contagion layer that propagates stress signals from one entity to its peers
based on sector similarity. When DHFL shows distress, the system automatically raises
risk scores for other housing finance NBFCs.

**Why this matters:** This is the novel contribution. It turns a single-entity classifier
into a *sector-level* early warning tool. Demo moment: "The system didn't just catch DHFL.
It would have flagged the entire housing finance sector."

---

## Architecture

### Three Modules

| Module | File | Purpose |
|--------|------|---------|
| Entity Graph | `src/signals/entity_graph.py` | Load YAML → weighted adjacency graph → peer queries |
| Propagation | `src/signals/propagation.py` | Daily scores → contagion propagation → rolling windows |
| Backtest | `src/signals/contagion_backtest.py` | Crisis replay → lead time improvement → report |

### Config
- `configs/contagion_config.yaml` — all weights, windows, thresholds, crisis definitions

### Data Strategy
- **Holdout predictions** (3,303): DHFL, RelCap, Cholamandalam — actual model outputs
- **LLM labels as proxy** (remaining ~14,000): all other entities — Haiku/Sonnet labels
  treated as if the model predicted them
- A `signal_source` column ('model' vs 'label') tracks which is which

---

## Edge Weight Rules (v1)

```
Same subsector → 0.8 (intra_subsector_weight)
Different subsector → 0.1 (cross_subsector_weight)
No self-edges.
```

## Scoring Math

```
direct_score(E, D) = SUM over articles about E on day D:
    direction_multiplier × confidence_weight × sector_wide_bonus

contagion_score(E, D) = SUM over peers P of E:
    edge_weight(P, E) × rolling_direct(P, D, window) × peer_discount

total_score(E, D) = direct_score(E, D) + contagion_score(E, D)
```

Why additive: entities with ZERO direct signals (Indiabulls) must still get contagion.

---

## Crisis Scenarios

| Crisis | Source Entities | Targets | Controls | Period |
|--------|----------------|---------|----------|--------|
| IL&FS/DHFL 2018-19 | IL&FS, DHFL | Indiabulls HF, PNB Housing, Can Fin Homes, Piramal, Reliance Home Finance | Cholamandalam (cross-sector), Bajaj Finance (cross-sector) | 2018-06 to 2020-06 |
| SREI/RelCap 2019-22 | Reliance Capital, SREI Infrastructure Finance | SREI Equipment Finance | L&T Finance (intra-sector stable), Bajaj Finance (cross-sector) | 2019-06 to 2022-01 |

**Note on SREI window:** Start date shifted from 2020-01 to 2019-06 because Reliance Capital's
688 holdout articles end at 2019-11-15 (entity fully defaulted). The earlier start captures
RelCap's distress signals as an actual contagion source.

---

## Success Criteria & Results

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Contagion raises scores before downgrades | **PASS** | 5/5 housing + 1/1 infra targets get lead times |
| Lead time improvement > 0 for ≥2 entities | **PASS** | All 6 targets: +69d to +334d |
| Zero-direct entities get contagion-only warning | **PASS** | Can Fin +296d, Piramal +334d |
| Intra > cross-subsector contagion (relative) | **PASS** | 2.3–3.5× ratio in both crises |
| Cross-sector FP contained at absolute level | **NEEDS v2** | See Lessons Learned below |
| All tests pass | **PASS** | 141/141 (85 existing + 56 new) |

---

## Implementation Steps

| Step | Files | Commit |
|------|-------|--------|
| 0 | `CONTAGION_PLAN.md` | `[Phase 3] Add contagion layer plan` |
| 1 | `entity_graph.py` + `test_entity_graph.py` + `contagion_config.yaml` | `[Phase 3] Add entity contagion graph with subsector-based edges` |
| 2 | `propagation.py` + `test_propagation.py` | `[Phase 3] Add signal propagation with contagion scoring` |
| 3 | `contagion_backtest.py` + `test_contagion_backtest.py` | `[Phase 3] Add contagion backtest with crisis replay` |
| 4 | Run backtest, generate report | `[Phase 3] Phase 3 complete: contagion backtest validated` |

---

## Lessons Learned

1. **Dense graphs amplify contagion.** 44 entities × 43 peers each = every entity gets cumulative
   contagion from the entire sector. Cross-subsector control entities (Chola, Bajaj at weight=0.1)
   breach the warning threshold on 85% of crisis days — not because of DHFL spillover (27% of their
   contagion) but because their OWN intra-sector peers (11 diversified_nbfc × 0.8) are noisy.

2. **Relative differentiation works, absolute thresholds don't.** Intra-sector entities get 2.3–3.5×
   more contagion than cross-sector entities. The edge weight differentiation is effective. But the
   warning threshold (2.0) was calibrated for per-article alerting (Phase 2.4), not graph-based scoring.
   v2 needs: normalize scores by peer count, or use percentile-based thresholds.

3. **"Control" entity selection matters.** L&T Finance (same subsector as SREI) is NOT a cross-sector
   control — it tests intra-sector stability ("did L&T survive while SREI collapsed?"). Use entities
   from different subsectors (Bajaj, Sundaram) for true FP spillover tests.

4. **Defaulted entities stop generating signals.** Reliance Capital fully defaulted by Nov 2019.
   Zero articles after that date. If your crisis window starts after the entity's death, it
   contributes nothing as a source. Match windows to actual article coverage.

5. **Systemic crises are inherently noisy.** During 2018-2019, even stable NBFCs had deterioration-
   labeled articles (market fear, sector-wide reporting). This is realistic — it's what a credit
   analyst would see. The system's job is not to eliminate noise, but to RANK risk correctly
   (housing > diversified > vehicle during a housing crisis). The 3.5× ratio confirms this.

---

## v2 Upgrade Path

1. **Contagion score normalization** — divide by peer count to control graph density effects (highest priority)
2. **Threshold recalibration** — percentile-based or entity-specific baselines instead of absolute cutoffs
3. **Funding profile edges** — wholesale/retail similarity (actual crisis transmission mechanism)
4. **Asymmetric weights** — large entity stress hits small entities harder
5. **Exponential decay** — replace hard window cutoff
6. **Full-corpus inference** — run all 17K articles through fine-tuned model on Colab
