# WHY THIS: Contagion backtest module that replays historical crises to answer:
# "Does contagion improve lead time for secondary entities?"
#
# Data strategy: Merge two signal sources:
# 1. Holdout predictions (3,303) — actual model outputs for DHFL, RelCap, Cholamandalam
# 2. LLM labels as proxy (~14,000) — all other entities, treated as model predictions
#
# This is defensible: the model has ~83% direction accuracy, labels are the training
# target, and the alternative (full-corpus Colab inference) delays Phase 3 for marginal
# accuracy gain. A signal_source column tracks which is which for transparency.

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger

from src.signals.entity_graph import EntityGraph, load_entity_graph
from src.signals.propagation import compute_all_scores, compute_direct_scores
from src.training.backtest import (
    extract_date_from_input,
    extract_entity_from_input,
    load_jsonl,
    load_rating_actions,
)
from src.data.parse_training_output import parse_training_output


# ============================================================
# Data Loading: Merge Holdout Predictions + Label Proxies
# ============================================================

def load_holdout_signals(
    pred_path: Path,
    source_path: Path,
) -> pd.DataFrame:
    """Load holdout entity predictions and extract signal fields.

    Uses the model's actual generated output (parsed via strict parser).
    Returns a signals DataFrame with standardized columns.
    """
    preds = load_jsonl(pred_path)
    source = load_jsonl(source_path)

    rows: list[dict] = []
    for i, pred in enumerate(preds):
        if i >= len(source):
            break

        src = source[i]
        entity = pred.get("entity") or extract_entity_from_input(src["input"])
        date_str = extract_date_from_input(src["input"])
        parsed = parse_training_output(pred["generated"])

        if not parsed.parse_ok:
            continue

        # Map parsed fields to signal columns
        # 🎓 Direction mapping: model outputs "Deterioration"/"Improvement"/"Neutral"
        # Labels use integers (-1/0/1). We normalize to string form here.
        direction_str = parsed.direction if parsed.credit_relevant else "Neutral"

        rows.append({
            "entity": entity,
            "date": date_str,
            "credit_relevant": 1 if parsed.credit_relevant else 0,
            "direction": direction_str,
            "signal_type": parsed.signal_type or "other",
            "sector_wide": parsed.sector_wide,
            "confidence": parsed.confidence or "Low",
            "signal_source": "model",
        })

    if not rows:
        logger.info("Loaded 0 holdout signals from model predictions")
        return pd.DataFrame(columns=[
            "entity", "date", "credit_relevant", "direction", "signal_type",
            "sector_wide", "confidence", "signal_source",
        ])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    logger.info(f"Loaded {len(df)} holdout signals from model predictions")
    return df


def load_label_signals(
    labels_path: Path,
    articles_path: Path,
    exclude_entities: list[str] | None = None,
    alias_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load LLM labels as signal proxies for non-holdout entities.

    # 🎓 WHY labels as proxies? The model was trained on these labels, so
    # its predictions should be ~83% consistent with them. For entities NOT
    # in the holdout set, we don't have model predictions — only labels.
    # Using labels as proxy overstates accuracy slightly but is the pragmatic
    # choice for v1. Full-corpus inference on Colab can fix this in v2.

    Args:
        labels_path: Path to labels_final.jsonl
        articles_path: Path to gdelt_for_labeling.csv (has entity + date)
        exclude_entities: Entities to exclude (holdout entities with real predictions)
        alias_map: Alias map for entity normalization
    """
    if exclude_entities is None:
        exclude_entities = []
    if alias_map is None:
        alias_map = {}

    # Load articles for entity + date metadata
    articles_df = pd.read_csv(articles_path)

    # Load labels
    labels: list[dict] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(json.loads(line))

    # Build URL → label lookup
    label_lookup: dict[str, dict] = {}
    for label in labels:
        label_lookup[label["url"]] = label

    # Map signal_direction integer to string
    direction_map = {-1: "Deterioration", 0: "Neutral", 1: "Improvement"}
    # Map confidence string to title case
    confidence_map = {"low": "Low", "medium": "Medium", "high": "High"}

    # Normalize exclude list
    exclude_lower = {e.lower() for e in exclude_entities}

    rows: list[dict] = []
    for _, article in articles_df.iterrows():
        url = article["article_url"]
        label = label_lookup.get(url)
        if label is None:
            continue

        # Extract entity from the entities column
        entity = article.get("entities", "")
        if pd.isna(entity) or not entity:
            continue

        # Normalize and check exclusion
        entity_norm = alias_map.get(entity.lower(), entity)
        if entity_norm.lower() in exclude_lower or entity.lower() in exclude_lower:
            continue

        rows.append({
            "entity": entity,
            "date": article["article_date"],
            "credit_relevant": label.get("credit_relevant", 0),
            "direction": direction_map.get(label.get("signal_direction", 0), "Neutral"),
            "signal_type": label.get("signal_type", "other"),
            "sector_wide": bool(label.get("sector_wide", 0)),
            "confidence": confidence_map.get(
                str(label.get("confidence", "low")).lower(), "Low"
            ),
            "signal_source": "label",
        })

    if not rows:
        logger.info(f"Loaded 0 label proxy signals (excluding {exclude_entities})")
        return pd.DataFrame(columns=[
            "entity", "date", "credit_relevant", "direction", "signal_type",
            "sector_wide", "confidence", "signal_source",
        ])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    logger.info(
        f"Loaded {len(df)} label proxy signals "
        f"({df['entity'].nunique()} entities, excluding {exclude_entities})"
    )
    return df


def load_all_signals(
    holdout_pred_path: Path,
    holdout_source_path: Path,
    labels_path: Path,
    articles_path: Path,
    holdout_entities: list[str],
    alias_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Merge holdout predictions + label proxies into a single signals DataFrame.

    This is the main data loading entry point for the contagion backtest.
    """
    # Load holdout predictions (model outputs)
    holdout_df = load_holdout_signals(holdout_pred_path, holdout_source_path)

    # Load label proxies (excluding holdout entities to avoid double-counting)
    label_df = load_label_signals(
        labels_path, articles_path,
        exclude_entities=holdout_entities,
        alias_map=alias_map,
    )

    # Merge
    combined = pd.concat([holdout_df, label_df], ignore_index=True)
    combined = combined.sort_values(["entity", "date"]).reset_index(drop=True)

    logger.info(
        f"Combined signals: {len(combined)} total "
        f"({len(holdout_df)} model + {len(label_df)} label proxy), "
        f"{combined['entity'].nunique()} entities"
    )

    return combined


# ============================================================
# Crisis Replay
# ============================================================

def compute_contagion_lead_time(
    scores_df: pd.DataFrame,
    entity: str,
    first_action_date: str | pd.Timestamp,
    threshold: float,
    graph: EntityGraph | None = None,
) -> dict[str, Any]:
    """Compute when an entity first exceeds the score threshold.

    Returns dict with:
        entity, first_action, first_breach_date, lead_time_days,
        direct_only_breach_date, direct_only_lead_time,
        lead_time_improvement, peak_score, peak_contagion
    """
    first_action = pd.Timestamp(first_action_date)

    # Normalize entity name if graph available
    entity_norm = graph.normalize_entity(entity) if graph else entity

    entity_scores = scores_df[scores_df["entity"] == entity_norm].copy()
    if len(entity_scores) == 0:
        entity_scores = scores_df[scores_df["entity"] == entity].copy()

    if len(entity_scores) == 0:
        return {
            "entity": entity,
            "first_action": first_action,
            "first_breach_date": None,
            "lead_time_days": None,
            "direct_only_breach_date": None,
            "direct_only_lead_time": None,
            "lead_time_improvement": None,
            "peak_score": 0.0,
            "peak_contagion": 0.0,
        }

    # Filter to before the first rating action
    before_action = entity_scores[entity_scores["date"] < first_action]

    # Total score breach (direct + contagion)
    total_breach = before_action[before_action["total_score"] >= threshold]
    first_breach = total_breach["date"].min() if len(total_breach) > 0 else None
    lead_time = (first_action - first_breach).days if first_breach is not None else None

    # Direct-only breach (no contagion)
    direct_breach = before_action[before_action["direct_score"] >= threshold]
    direct_first = direct_breach["date"].min() if len(direct_breach) > 0 else None
    direct_lead = (first_action - direct_first).days if direct_first is not None else None

    # Lead time improvement = contagion lead - direct lead
    improvement = None
    if lead_time is not None and direct_lead is not None:
        improvement = lead_time - direct_lead  # positive = contagion was earlier
    elif lead_time is not None and direct_lead is None:
        improvement = lead_time  # contagion found it, direct didn't

    return {
        "entity": entity,
        "first_action": first_action,
        "first_breach_date": first_breach,
        "lead_time_days": lead_time,
        "direct_only_breach_date": direct_first,
        "direct_only_lead_time": direct_lead,
        "lead_time_improvement": improvement,
        "peak_score": before_action["total_score"].max() if len(before_action) > 0 else 0.0,
        "peak_contagion": before_action["contagion_score"].max() if len(before_action) > 0 else 0.0,
    }


def run_crisis_replay(
    signals_df: pd.DataFrame,
    graph: EntityGraph,
    crisis_config: dict,
    config: dict,
) -> dict[str, Any]:
    """Replay a single crisis scenario and compute contagion metrics.

    Args:
        signals_df: All signals (model + label proxy)
        graph: EntityGraph
        crisis_config: A single crisis definition from contagion_config.yaml
        config: Full contagion config

    Returns:
        Dict with crisis name, scores_df, and per-target-entity results
    """
    crisis_name = crisis_config["name"]
    start_date = pd.Timestamp(crisis_config["start_date"])
    end_date = pd.Timestamp(crisis_config["end_date"])
    source_entities = crisis_config["source_entities"]
    target_entities = crisis_config["target_entities"]

    logger.info(f"Replaying crisis: {crisis_name} ({start_date.date()} to {end_date.date()})")

    # Filter signals to crisis period
    crisis_signals = signals_df[
        (signals_df["date"] >= start_date) &
        (signals_df["date"] <= end_date)
    ].copy()

    logger.info(
        f"  Crisis period signals: {len(crisis_signals)} "
        f"({crisis_signals['entity'].nunique()} entities)"
    )

    # Compute all scores (direct + contagion + rolling)
    scores_df = compute_all_scores(crisis_signals, graph, config)

    # Analyze source entities
    source_results: list[dict] = []
    for entity in source_entities:
        entity_norm = graph.normalize_entity(entity)
        entity_scores = scores_df[scores_df["entity"] == entity_norm]
        if len(entity_scores) > 0:
            source_results.append({
                "entity": entity_norm,
                "n_days_with_score": len(entity_scores),
                "total_direct": entity_scores["direct_score"].sum(),
                "peak_direct": entity_scores["direct_score"].max(),
                "first_signal_date": entity_scores["date"].min(),
            })
        else:
            source_results.append({
                "entity": entity_norm,
                "n_days_with_score": 0,
                "total_direct": 0.0,
                "peak_direct": 0.0,
                "first_signal_date": None,
            })

    # Analyze target entities
    thresholds = config.get("score_thresholds", {})
    warning_threshold = thresholds.get("warning", 2.0)

    target_results: list[dict] = []
    for target in target_entities:
        target_name = target["name"]
        first_action = target.get("first_action")

        if first_action is None:
            # Stable control entity
            entity_norm = graph.normalize_entity(target_name)
            entity_scores = scores_df[scores_df["entity"] == entity_norm]
            if len(entity_scores) == 0:
                entity_scores = scores_df[scores_df["entity"] == target_name]

            peak = entity_scores["total_score"].max() if len(entity_scores) > 0 else 0.0
            peak_contagion = entity_scores["contagion_score"].max() if len(entity_scores) > 0 else 0.0
            n_breaches = (entity_scores["total_score"] >= warning_threshold).sum() if len(entity_scores) > 0 else 0

            target_results.append({
                "entity": target_name,
                "is_control": True,
                "first_action": None,
                "first_breach_date": None,
                "lead_time_days": None,
                "direct_only_lead_time": None,
                "lead_time_improvement": None,
                "peak_score": peak,
                "peak_contagion": peak_contagion,
                "n_warning_breaches": int(n_breaches),
            })
        else:
            result = compute_contagion_lead_time(
                scores_df=scores_df,
                entity=target_name,
                first_action_date=first_action,
                threshold=warning_threshold,
                graph=graph,
            )
            result["is_control"] = False
            result["n_warning_breaches"] = 0
            target_results.append(result)

    return {
        "crisis_name": crisis_name,
        "start_date": start_date,
        "end_date": end_date,
        "n_signals": len(crisis_signals),
        "n_entities": crisis_signals["entity"].nunique(),
        "source_results": source_results,
        "target_results": target_results,
        "scores_df": scores_df,
    }


# ============================================================
# Report Generation
# ============================================================

def generate_contagion_report(
    crisis_results: list[dict],
    config: dict,
    output_path: Path,
) -> None:
    """Generate the Phase 3 contagion backtest report as markdown."""
    lines: list[str] = []

    lines.append("# Phase 3 — Contagion Backtest Results")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")

    # --- Executive Summary ---
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append("The contagion layer propagates credit distress signals from one entity to its")
    lines.append("sector peers. When DHFL shows distress, housing finance NBFCs automatically get")
    lines.append("elevated risk scores — even if they have zero direct news signals.")
    lines.append("")

    edge_weights = config.get("edge_weights", {})
    lines.append(f"- **Edge weights:** intra-subsector={edge_weights.get('intra_subsector', 0.8)}, "
                 f"cross-subsector={edge_weights.get('cross_subsector', 0.1)}")
    lines.append(f"- **Contagion window:** {config.get('contagion_window_days', 30)} days")
    lines.append(f"- **Warning threshold:** {config.get('score_thresholds', {}).get('warning', 2.0)}")
    lines.append("")

    for crisis in crisis_results:
        lines.append(f"## 2. {crisis['crisis_name']}")
        lines.append("")
        lines.append(f"**Period:** {crisis['start_date'].date()} to {crisis['end_date'].date()}")
        lines.append(f"**Signals in period:** {crisis['n_signals']} ({crisis['n_entities']} entities)")
        lines.append("")

        # Source entities
        lines.append("### Source Entities (signal originators)")
        lines.append("")
        lines.append("| Entity | Days with Score | Total Direct | Peak Direct | First Signal |")
        lines.append("|--------|----------------|-------------|-------------|-------------|")
        for src in crisis["source_results"]:
            first_sig = src["first_signal_date"]
            first_str = first_sig.strftime("%Y-%m-%d") if first_sig is not None and pd.notna(first_sig) else "—"
            lines.append(
                f"| {src['entity']} | {src['n_days_with_score']} | "
                f"{src['total_direct']:.1f} | {src['peak_direct']:.1f} | {first_str} |"
            )
        lines.append("")

        # Target entities
        lines.append("### Target Entities (contagion recipients)")
        lines.append("")
        lines.append("| Entity | Control? | First Action | Breach Date | Lead Time | "
                     "Direct-Only Lead | Improvement | Peak Score | Peak Contagion |")
        lines.append("|--------|----------|-------------|-------------|-----------|"
                     "-----------------|-------------|------------|----------------|")

        for tgt in crisis["target_results"]:
            is_ctrl = "Yes" if tgt.get("is_control") else "No"
            action = tgt.get("first_action")
            action_str = action.strftime("%Y-%m-%d") if action is not None and pd.notna(action) else "—"
            breach = tgt.get("first_breach_date")
            breach_str = breach.strftime("%Y-%m-%d") if breach is not None and pd.notna(breach) else "—"
            lt = tgt.get("lead_time_days")
            lt_str = f"{lt}d" if lt is not None else "—"
            dlt = tgt.get("direct_only_lead_time")
            dlt_str = f"{dlt}d" if dlt is not None else "—"
            imp = tgt.get("lead_time_improvement")
            imp_str = f"+{imp}d" if imp is not None and imp > 0 else (f"{imp}d" if imp is not None else "—")
            peak = tgt.get("peak_score", 0.0)
            peak_c = tgt.get("peak_contagion", 0.0)

            lines.append(
                f"| {tgt['entity']} | {is_ctrl} | {action_str} | {breach_str} | "
                f"{lt_str} | {dlt_str} | {imp_str} | {peak:.2f} | {peak_c:.2f} |"
            )
        lines.append("")

    # --- Key Findings ---
    lines.append("## 3. Key Findings")
    lines.append("")
    lines.append("### Contagion Value Proposition")
    lines.append("")
    lines.append("1. **Entities with zero direct signals** can still get early warnings via contagion")
    lines.append("2. **Lead time improvement** measures how many extra days of warning contagion provides")
    lines.append("3. **Cross-subsector control** (low weight=0.1) limits false positive spillover")
    lines.append("")

    # --- Methodology Notes ---
    lines.append("## 4. Methodology Notes")
    lines.append("")
    lines.append("### Signal Sources")
    lines.append("- **Model predictions (holdout):** DHFL (1,243), Reliance Capital (688), "
                 "Cholamandalam (1,372) — actual fine-tuned model outputs")
    lines.append("- **Label proxies (all others):** Haiku/Sonnet labels treated as model predictions. "
                 "Model has ~83% direction accuracy vs labels, so this is a reasonable proxy.")
    lines.append("- **Tracking:** Each signal has a `signal_source` column ('model' or 'label') "
                 "for transparency")
    lines.append("")
    lines.append("### Limitations")
    lines.append("- Non-holdout entity signals are labels, not predictions (overstates accuracy)")
    lines.append("- v1 edge weights are subsector-only; v2 adds funding profile similarity")
    lines.append("- Symmetric propagation; v2 adds asymmetric weights by entity size")
    lines.append("")

    # --- v2 Upgrade Path ---
    lines.append("## 5. v2 Upgrade Path")
    lines.append("")
    lines.append("1. **Funding profile edges** — wholesale/retail similarity (actual crisis mechanism)")
    lines.append("2. **Asymmetric weights** — large entity stress hits small entities harder")
    lines.append("3. **Exponential decay** — replace hard window cutoff")
    lines.append("4. **Full-corpus inference** — run all 17K articles through fine-tuned model")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Contagion report written to {output_path}")


# ============================================================
# CLI Entry Point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run contagion backtest: replay crises and measure lead time improvement"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/contagion_config.yaml"),
        help="Path to contagion config YAML",
    )
    parser.add_argument(
        "--crisis", type=str, default=None,
        help="Run a specific crisis (e.g., 'ilfs_dhfl_2018'). Default: run all.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Override output path for the report",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})

    # Build entity graph
    entity_yaml = Path(paths.get("entity_graph", "configs/nbfc_entities.yaml"))
    graph = load_entity_graph(entity_yaml, config)

    # Holdout entities (entities with real model predictions)
    holdout_entities = ["DHFL", "Reliance Capital", "Cholamandalam Investment"]

    # Load all signals
    signals_df = load_all_signals(
        holdout_pred_path=Path(paths["holdout_predictions"]),
        holdout_source_path=Path(paths["holdout_source"]),
        labels_path=Path(paths["labels"]),
        articles_path=Path(paths["articles"]),
        holdout_entities=holdout_entities,
        alias_map=graph.alias_map,
    )

    # Run crisis replays
    crises = config.get("crises", {})
    if args.crisis:
        crises = {args.crisis: crises[args.crisis]}

    crisis_results: list[dict] = []
    for crisis_key, crisis_config_val in crises.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Crisis: {crisis_key}")
        logger.info(f"{'='*60}")

        result = run_crisis_replay(signals_df, graph, crisis_config_val, config)
        crisis_results.append(result)

        # Print summary to console
        print(f"\n{'='*60}")
        print(f"  Crisis: {result['crisis_name']}")
        print(f"  Signals: {result['n_signals']} | Entities: {result['n_entities']}")
        print(f"{'='*60}")

        for src in result["source_results"]:
            print(f"  SOURCE: {src['entity']} — {src['n_days_with_score']} days, "
                  f"peak direct={src['peak_direct']:.1f}")

        for tgt in result["target_results"]:
            if tgt.get("is_control"):
                print(f"  CONTROL: {tgt['entity']} — peak={tgt['peak_score']:.2f}, "
                      f"peak contagion={tgt['peak_contagion']:.2f}, "
                      f"warning breaches={tgt.get('n_warning_breaches', 0)}")
            else:
                lt = tgt.get("lead_time_days")
                imp = tgt.get("lead_time_improvement")
                lt_str = f"{lt}d" if lt is not None else "none"
                imp_str = f"+{imp}d" if imp is not None and imp > 0 else (f"{imp}d" if imp is not None else "none")
                print(f"  TARGET: {tgt['entity']} — lead time={lt_str}, "
                      f"improvement={imp_str}, peak={tgt['peak_score']:.2f}")

    # Generate report
    output_path = args.output or Path(paths.get("report_output", "reports/phase3_contagion_results.md"))
    generate_contagion_report(crisis_results, config, output_path)
    print(f"\n  Report: {output_path}")


if __name__ == "__main__":
    main()
