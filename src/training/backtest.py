# WHY THIS: Backtest module that answers the hardest question in Phase 2 —
# does the model predict actual rating agency downgrades, and how early?
#
# Phase 2.2 eval compared model predictions to LLM-generated labels (Haiku/Sonnet).
# That tells us: "does the model agree with our labeling?" This module compares
# predictions to REAL WORLD OUTCOMES (rating actions from CRISIL/ICRA/CARE).
# That tells us: "does this actually work as an early warning system?"

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger

from src.data.parse_training_output import parse_training_output


# ============================================================
# Data Loading
# ============================================================

def extract_entity_from_input(input_text: str) -> str:
    """Extract entity name from training input text.

    Input format: "Entity: DHFL\\nDate: 2019-06-01\\nTitle: ...\\nArticle: ..."
    """
    for line in input_text.split("\n"):
        if line.startswith("Entity:"):
            return line[len("Entity:"):].strip()
    return "UNKNOWN"


def extract_date_from_input(input_text: str) -> str:
    """Extract date from training input text.

    Input format: "Entity: DHFL\\nDate: 2019-06-01\\nTitle: ...\\nArticle: ..."
    Returns date string in YYYY-MM-DD format, or empty string if not found.
    """
    for line in input_text.split("\n"):
        if line.startswith("Date:"):
            return line[len("Date:"):].strip()
    return ""


def extract_title_from_input(input_text: str) -> str:
    """Extract title from training input text."""
    for line in input_text.split("\n"):
        if line.startswith("Title:"):
            return line[len("Title:"):].strip()
    return ""


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_predictions_with_metadata(
    pred_path: Path,
    source_path: Path,
    match_by: str = "index",
) -> pd.DataFrame:
    """Load model predictions and join with source metadata (entity, date, title).

    # 🎓 WHY two matching strategies:
    # - "index": Holdout predictions were generated in exact order of source JSONL.
    #   Row 0 in predictions = Row 0 in source. Simple, fast, verified.
    # - "expected": Test predictions were from a stratified sample (500 of 2,133).
    #   We match on the 'expected' field (ground truth output text) which is unique
    #   per article due to the REASONING field.

    Returns DataFrame with columns:
        entity, date, title, predicted_cr, predicted_direction, predicted_signal_type,
        predicted_confidence, expected_cr, expected_direction
    """
    preds = load_jsonl(pred_path)
    source = load_jsonl(source_path)

    rows: list[dict] = []

    if match_by == "index":
        # Direct positional matching — same order, same count (or preds <= source)
        for i, pred in enumerate(preds):
            if i >= len(source):
                logger.warning(f"Prediction index {i} exceeds source length {len(source)}")
                break

            src = source[i]
            entity = pred.get("entity") or extract_entity_from_input(src["input"])
            date_str = extract_date_from_input(src["input"])
            title = extract_title_from_input(src["input"])

            # Parse model's generated output
            parsed_pred = parse_training_output(pred["generated"])
            # Parse expected (ground truth label) output
            parsed_expected = parse_training_output(pred["expected"])

            rows.append({
                "entity": entity,
                "date": date_str,
                "title": title,
                "predicted_cr": parsed_pred.credit_relevant if parsed_pred.parse_ok else None,
                "predicted_direction": parsed_pred.direction if parsed_pred.parse_ok else None,
                "predicted_signal_type": parsed_pred.signal_type if parsed_pred.parse_ok else None,
                "predicted_confidence": parsed_pred.confidence if parsed_pred.parse_ok else None,
                "expected_cr": parsed_expected.credit_relevant if parsed_expected.parse_ok else None,
                "expected_direction": parsed_expected.direction if parsed_expected.parse_ok else None,
                "parse_ok": parsed_pred.parse_ok,
            })

    elif match_by == "expected":
        # Match predictions to source by expected output text
        # Build lookup: output text → source record
        source_lookup: dict[str, dict] = {}
        for src in source:
            source_lookup[src["output"]] = src

        for pred in preds:
            src = source_lookup.get(pred["expected"])
            if src is None:
                logger.warning(f"No source match for expected text: {pred['expected'][:80]}...")
                continue

            entity = extract_entity_from_input(src["input"])
            date_str = extract_date_from_input(src["input"])
            title = extract_title_from_input(src["input"])

            parsed_pred = parse_training_output(pred["generated"])
            parsed_expected = parse_training_output(pred["expected"])

            rows.append({
                "entity": entity,
                "date": date_str,
                "title": title,
                "predicted_cr": parsed_pred.credit_relevant if parsed_pred.parse_ok else None,
                "predicted_direction": parsed_pred.direction if parsed_pred.parse_ok else None,
                "predicted_signal_type": parsed_pred.signal_type if parsed_pred.parse_ok else None,
                "predicted_confidence": parsed_pred.confidence if parsed_pred.parse_ok else None,
                "expected_cr": parsed_expected.credit_relevant if parsed_expected.parse_ok else None,
                "expected_direction": parsed_expected.direction if parsed_expected.parse_ok else None,
                "parse_ok": parsed_pred.parse_ok,
            })
    else:
        raise ValueError(f"Unknown match_by: {match_by}. Use 'index' or 'expected'.")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    logger.info(
        f"Loaded {len(df)} predictions ({df['parse_ok'].sum()} parsed OK) "
        f"from {pred_path.name}"
    )
    return df


def load_rating_actions(
    path: Path,
    action_types: list[str] | None = None,
) -> pd.DataFrame:
    """Load rating actions CSV, optionally filtering to specific action types.

    # 🎓 The rating_actions_sourced.csv has columns:
    # entity, entity_full_name, agency, date, action_type, from_rating,
    # to_rating, instrument_type, rationale_url, notes
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if action_types:
        df = df[df["action_type"].isin(action_types)].copy()

    logger.info(
        f"Loaded {len(df)} rating actions"
        + (f" (filtered to {action_types})" if action_types else "")
    )
    return df


def build_entity_alias_map(aliases: dict[str, list[str]]) -> dict[str, str]:
    """Build a reverse map: alias → canonical entity name.

    # 🎓 Why aliases? Rating actions CSV says "DHFL", article predictions
    # also say "DHFL" — but future data might say "Dewan Housing". The alias
    # map normalizes all variants to a single canonical name for matching.
    """
    alias_map: dict[str, str] = {}
    for canonical, variants in aliases.items():
        for variant in variants:
            alias_map[variant.lower()] = canonical
    return alias_map


def normalize_entity(name: str, alias_map: dict[str, str]) -> str:
    """Normalize an entity name using the alias map."""
    return alias_map.get(name.lower(), name)


# ============================================================
# Lead Time Analysis
# ============================================================

def compute_lead_time(
    predictions_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    lookback_days: int = 180,
    alias_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """For each rating action, find the earliest deterioration signal and compute lead time.

    # 🎓 KEY CONCEPT: Lead time = how many days before a downgrade did the model
    # first flag deterioration? This is the #1 metric for an early warning system.
    # A model with 90% recall but 0-day lead time is useless — you need warning
    # BEFORE the rating agency acts, not at the same time.

    Returns DataFrame with columns:
        entity, action_date, action_type, agency, first_signal_date,
        lead_time_days, n_signals_before, n_articles_before
    """
    if alias_map is None:
        alias_map = {}

    # Normalize entity names in predictions
    pred_df = predictions_df.copy()
    pred_df["entity_norm"] = pred_df["entity"].apply(
        lambda x: normalize_entity(x, alias_map)
    )

    # Filter to deterioration signals only
    det_signals = pred_df[
        (pred_df["predicted_cr"] == True) &  # noqa: E712
        (pred_df["predicted_direction"] == "Deterioration") &
        (pred_df["date"].notna())
    ].copy()

    results: list[dict] = []

    for _, action in actions_df.iterrows():
        action_entity = normalize_entity(action["entity"], alias_map)
        action_date = action["date"]

        if pd.isna(action_date):
            continue

        lookback_start = action_date - timedelta(days=lookback_days)

        # All articles about this entity in the lookback window
        entity_articles = pred_df[
            (pred_df["entity_norm"] == action_entity) &
            (pred_df["date"] >= lookback_start) &
            (pred_df["date"] <= action_date)
        ]

        # Deterioration signals in the lookback window
        entity_signals = det_signals[
            (det_signals["entity_norm"] == action_entity) &
            (det_signals["date"] >= lookback_start) &
            (det_signals["date"] <= action_date)
        ]

        first_signal_date = entity_signals["date"].min() if len(entity_signals) > 0 else pd.NaT
        lead_time = (action_date - first_signal_date).days if pd.notna(first_signal_date) else None

        results.append({
            "entity": action_entity,
            "action_date": action_date,
            "action_type": action["action_type"],
            "agency": action.get("agency", ""),
            "first_signal_date": first_signal_date,
            "lead_time_days": lead_time,
            "n_signals_before": len(entity_signals),
            "n_articles_before": len(entity_articles),
        })

    return pd.DataFrame(results)


# ============================================================
# Alert-Based Precision / Recall
# ============================================================

def compute_alert_metrics(
    predictions_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    n_threshold: int,
    window_days: int,
    lookahead_days: int,
    alias_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute precision/recall for an alert threshold.

    # 🎓 CONCEPT: Alert-based evaluation
    #
    # An "alert" fires on day T for entity E if there are ≥ N deterioration
    # signals about E in the window [T - M days, T].
    #
    # True positive: alert fires, and E gets downgraded within K days.
    # False positive: alert fires, but E does NOT get downgraded within K days.
    # False negative: E gets downgraded, but no alert fired in the K days before.
    #
    # This is a more realistic evaluation than per-article accuracy because
    # real users care about: "did the system warn me?" not "was each article right?"
    """
    if alias_map is None:
        alias_map = {}

    pred_df = predictions_df.copy()
    pred_df["entity_norm"] = pred_df["entity"].apply(
        lambda x: normalize_entity(x, alias_map)
    )

    # Build deterioration signal set: (entity, date) pairs
    det_signals = pred_df[
        (pred_df["predicted_cr"] == True) &  # noqa: E712
        (pred_df["predicted_direction"] == "Deterioration") &
        (pred_df["date"].notna())
    ].copy()

    # Build rating action set: (entity, date) pairs
    actions_norm = actions_df.copy()
    actions_norm["entity_norm"] = actions_norm["entity"].apply(
        lambda x: normalize_entity(x, alias_map)
    )

    # Get unique entities in predictions
    entities = pred_df["entity_norm"].unique()

    # For each entity, compute daily signal counts and check for alerts
    alerts: list[dict] = []  # (entity, alert_date, is_true_positive)

    for entity in entities:
        entity_signals = det_signals[det_signals["entity_norm"] == entity].copy()
        if len(entity_signals) == 0:
            continue

        # Get all unique dates for this entity's signals
        signal_dates = entity_signals["date"].dt.normalize().value_counts().sort_index()

        # For each date, count signals in the rolling window
        entity_actions = actions_norm[actions_norm["entity_norm"] == entity]
        action_dates = set(entity_actions["date"].dt.normalize().dropna())

        # Check each day that has at least one signal
        all_dates = pred_df[pred_df["entity_norm"] == entity]["date"].dropna().dt.normalize().unique()

        for check_date in sorted(all_dates):
            window_start = check_date - timedelta(days=window_days)
            signals_in_window = entity_signals[
                (entity_signals["date"].dt.normalize() >= window_start) &
                (entity_signals["date"].dt.normalize() <= check_date)
            ]

            if len(signals_in_window) >= n_threshold:
                # Alert fires! Check if true positive
                lookahead_end = check_date + timedelta(days=lookahead_days)
                has_action = any(
                    check_date <= ad <= lookahead_end
                    for ad in action_dates
                )
                alerts.append({
                    "entity": entity,
                    "alert_date": check_date,
                    "is_tp": has_action,
                })

    # Compute precision/recall
    n_alerts = len(alerts)
    n_tp = sum(1 for a in alerts if a["is_tp"])
    n_fp = n_alerts - n_tp

    # False negatives: rating actions with no alert in the lookback window
    n_fn = 0
    for _, action in actions_norm.iterrows():
        entity = action["entity_norm"]
        action_date = action["date"]
        if pd.isna(action_date):
            continue
        # Check if any alert fired for this entity in [action_date - K, action_date]
        alert_window_start = action_date - timedelta(days=lookahead_days)
        had_alert = any(
            a["entity"] == entity and
            alert_window_start <= a["alert_date"] <= action_date
            for a in alerts
        )
        if not had_alert:
            n_fn += 1

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n_threshold": n_threshold,
        "window_days": window_days,
        "lookahead_days": lookahead_days,
        "n_alerts": n_alerts,
        "n_true_positives": n_tp,
        "n_false_positives": n_fp,
        "n_false_negatives": n_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def sweep_alert_thresholds(
    predictions_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    config: dict,
    alias_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Grid search over alert thresholds and return all results.

    # 🎓 WHY grid search? There's no single "right" threshold. A credit analyst
    # might prefer high recall (catch every downgrade, tolerate false alarms).
    # A portfolio manager might prefer high precision (only act on strong signals).
    # The grid gives you the Pareto frontier — all non-dominated tradeoffs.
    """
    thresholds = config.get("alert_thresholds", {})
    n_signals_list = thresholds.get("n_signals", [1, 2, 3, 5])
    window_list = thresholds.get("window_days", [14, 30, 60, 90])
    lookahead_list = thresholds.get("lookahead_days", [90, 180])

    results: list[dict] = []

    for n in n_signals_list:
        for w in window_list:
            for k in lookahead_list:
                metrics = compute_alert_metrics(
                    predictions_df, actions_df,
                    n_threshold=n, window_days=w, lookahead_days=k,
                    alias_map=alias_map,
                )
                results.append(metrics)

    return pd.DataFrame(results)


# ============================================================
# Entity Timeline
# ============================================================

def compute_entity_timeline(
    predictions_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    entity: str,
    alias_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a daily timeline of signals and rating actions for one entity.

    # 🎓 This powers the "money shot" visualization: signals accumulating
    # over time, with vertical lines at rating actions. The visual proof
    # that signals precede downgrades.

    Returns dict with:
        daily_signals: DataFrame[date, n_det, n_imp, n_neutral, n_not_cr, cumulative_det]
        rating_actions: DataFrame[date, action_type, agency]
    """
    if alias_map is None:
        alias_map = {}

    pred_df = predictions_df.copy()
    pred_df["entity_norm"] = pred_df["entity"].apply(
        lambda x: normalize_entity(x, alias_map)
    )

    entity_norm = normalize_entity(entity, alias_map)

    entity_preds = pred_df[
        (pred_df["entity_norm"] == entity_norm) &
        (pred_df["date"].notna())
    ].copy()
    entity_preds["date_day"] = entity_preds["date"].dt.normalize()

    # Daily signal counts
    daily: list[dict] = []
    if len(entity_preds) > 0:
        for day, group in entity_preds.groupby("date_day"):
            n_det = ((group["predicted_cr"] == True) & (group["predicted_direction"] == "Deterioration")).sum()  # noqa: E712
            n_imp = ((group["predicted_cr"] == True) & (group["predicted_direction"] == "Improvement")).sum()  # noqa: E712
            n_neutral = ((group["predicted_cr"] == True) & (group["predicted_direction"] == "Neutral")).sum()  # noqa: E712
            n_not_cr = (group["predicted_cr"] == False).sum()  # noqa: E712
            daily.append({
                "date": day,
                "n_det": int(n_det),
                "n_imp": int(n_imp),
                "n_neutral": int(n_neutral),
                "n_not_cr": int(n_not_cr),
            })

    daily_df = pd.DataFrame(daily).sort_values("date") if daily else pd.DataFrame(
        columns=["date", "n_det", "n_imp", "n_neutral", "n_not_cr"]
    )
    if len(daily_df) > 0:
        daily_df["cumulative_det"] = daily_df["n_det"].cumsum()
    else:
        daily_df["cumulative_det"] = []

    # Rating actions for this entity
    actions_norm = actions_df.copy()
    actions_norm["entity_norm"] = actions_norm["entity"].apply(
        lambda x: normalize_entity(x, alias_map)
    )
    entity_actions = actions_norm[actions_norm["entity_norm"] == entity_norm][
        ["date", "action_type", "agency"]
    ].sort_values("date")

    return {
        "daily_signals": daily_df,
        "rating_actions": entity_actions,
    }


# ============================================================
# Naive Baselines
# ============================================================

def compute_naive_baselines(
    predictions_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    alias_map: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Compute lead times for naive baseline strategies.

    # 🎓 WHY baselines? Without baselines, "50-day lead time" means nothing.
    # If a baseline that labels everything as deterioration also gets 50-day lead time,
    # then our model isn't doing anything special. Baselines establish the bar.

    Returns dict mapping baseline name → summary dict (same keys as compute_lead_time output).
    """
    if alias_map is None:
        alias_map = {}

    baselines: dict[str, dict] = {}

    # --- Baseline 1: Always-Deterioration ---
    # Predict deterioration for every article. Upper bound on recall.
    always_det = predictions_df.copy()
    always_det["predicted_cr"] = True
    always_det["predicted_direction"] = "Deterioration"
    baselines["always_deterioration"] = {
        "description": "Predict deterioration for every article",
        "predictions": always_det,
    }

    # --- Baseline 2: Ground Truth Labels ---
    # Use the expected (Haiku/Sonnet label) as prediction.
    # Shows: model performance vs labeling agreement.
    label_baseline = predictions_df.copy()
    label_baseline["predicted_cr"] = label_baseline["expected_cr"]
    label_baseline["predicted_direction"] = label_baseline["expected_direction"]
    baselines["ground_truth_labels"] = {
        "description": "Use Haiku/Sonnet labels as predictions",
        "predictions": label_baseline,
    }

    # Compute lead times for each baseline
    results: dict[str, dict] = {}
    for name, bl in baselines.items():
        lt_df = compute_lead_time(
            bl["predictions"], actions_df,
            lookback_days=180, alias_map=alias_map,
        )
        results[name] = {
            "description": bl["description"],
            "lead_times": lt_df,
            "mean_lead_time": lt_df["lead_time_days"].mean() if len(lt_df) > 0 else None,
            "median_lead_time": lt_df["lead_time_days"].median() if len(lt_df) > 0 else None,
            "n_with_signal": (lt_df["lead_time_days"].notna()).sum() if len(lt_df) > 0 else 0,
            "n_total_actions": len(lt_df),
        }

    return results


# ============================================================
# Report Generation
# ============================================================

def generate_backtest_report(
    lead_time_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    entity_timelines: dict[str, dict],
    baseline_results: dict[str, dict],
    predictions_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    output_path: Path,
    alias_map: dict[str, str] | None = None,
) -> None:
    """Generate the Phase 2.4 backtest report as markdown."""
    if alias_map is None:
        alias_map = {}

    lines: list[str] = []

    # --- Header ---
    lines.append("# Phase 2.4 — Backtest Results: Model vs. Actual Rating Actions")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")

    # --- Executive Summary ---
    lines.append("## 1. Executive Summary")
    lines.append("")

    actions_with_signal = lead_time_df[lead_time_df["lead_time_days"].notna()]
    actions_no_signal = lead_time_df[lead_time_df["lead_time_days"].isna()]

    if len(actions_with_signal) > 0:
        mean_lt = actions_with_signal["lead_time_days"].mean()
        median_lt = actions_with_signal["lead_time_days"].median()
        max_lt = actions_with_signal["lead_time_days"].max()
        min_lt = actions_with_signal["lead_time_days"].min()
        lines.append(f"- **Signal coverage:** {len(actions_with_signal)}/{len(lead_time_df)} "
                      f"rating actions had at least one deterioration signal in the prior 180 days")
        lines.append(f"- **Mean lead time:** {mean_lt:.0f} days")
        lines.append(f"- **Median lead time:** {median_lt:.0f} days")
        lines.append(f"- **Lead time range:** {min_lt:.0f} – {max_lt:.0f} days")
    else:
        lines.append("- **No signals detected before any rating actions.**")
    lines.append("")

    # Best threshold (highest F1)
    if len(threshold_df) > 0:
        best = threshold_df.loc[threshold_df["f1"].idxmax()]
        lines.append(f"- **Best alert threshold** (by F1): "
                      f"N≥{best['n_threshold']:.0f} signals in {best['window_days']:.0f}-day window, "
                      f"{best['lookahead_days']:.0f}-day lookahead")
        lines.append(f"  - Precision: {best['precision']:.1%}, Recall: {best['recall']:.1%}, "
                      f"F1: {best['f1']:.3f}")
    lines.append("")

    # --- Lead Time Analysis ---
    lines.append("## 2. Lead Time Analysis")
    lines.append("")
    lines.append("For each rating action (downgrade/default), we looked back 180 days for "
                 "deterioration signals from the model.")
    lines.append("")

    if len(lead_time_df) > 0:
        lines.append("| Entity | Action Date | Type | Agency | First Signal | Lead Time | Signals |")
        lines.append("|--------|------------|------|--------|-------------|-----------|---------|")
        for _, row in lead_time_df.sort_values(["entity", "action_date"]).iterrows():
            first_sig = row["first_signal_date"].strftime("%Y-%m-%d") if pd.notna(row["first_signal_date"]) else "—"
            lt = f"{row['lead_time_days']:.0f}d" if pd.notna(row["lead_time_days"]) else "No signal"
            lines.append(
                f"| {row['entity']} | {row['action_date'].strftime('%Y-%m-%d')} | "
                f"{row['action_type']} | {row['agency']} | {first_sig} | {lt} | "
                f"{row['n_signals_before']} / {row['n_articles_before']} |"
            )
    lines.append("")

    # --- Per-Entity Deep Dives ---
    for entity_name in ["DHFL", "Reliance Capital", "Cholamandalam"]:
        lines.append(f"## 3. {entity_name} Deep Dive")
        lines.append("")

        timeline = entity_timelines.get(entity_name)
        if timeline is None:
            lines.append(f"No timeline data for {entity_name}.")
            lines.append("")
            continue

        daily = timeline["daily_signals"]
        actions = timeline["rating_actions"]

        if len(daily) > 0:
            total_articles = daily[["n_det", "n_imp", "n_neutral", "n_not_cr"]].sum().sum()
            total_det = daily["n_det"].sum()
            date_range = f"{daily['date'].min().strftime('%Y-%m-%d')} to {daily['date'].max().strftime('%Y-%m-%d')}"
            lines.append(f"- **Articles:** {total_articles:.0f} ({date_range})")
            lines.append(f"- **Deterioration signals:** {total_det:.0f} ({total_det/total_articles*100:.1f}%)")

        if len(actions) > 0:
            lines.append(f"- **Rating actions:** {len(actions)}")
            for _, act in actions.iterrows():
                lines.append(f"  - {act['date'].strftime('%Y-%m-%d')}: {act['action_type']} ({act['agency']})")
        else:
            lines.append(f"- **Rating actions:** 0 (clean record)")
        lines.append("")

        # Lead times for this entity
        entity_lt = lead_time_df[lead_time_df["entity"] == entity_name]
        if len(entity_lt) > 0:
            with_signal = entity_lt[entity_lt["lead_time_days"].notna()]
            if len(with_signal) > 0:
                lines.append(f"**Signal-before-action coverage:** {len(with_signal)}/{len(entity_lt)}")
                lines.append(f"**Mean lead time:** {with_signal['lead_time_days'].mean():.0f} days")
                lines.append(f"**Earliest signal:** {with_signal['first_signal_date'].min().strftime('%Y-%m-%d')}")
            else:
                lines.append("**No deterioration signals detected before rating actions.**")
        elif entity_name == "Cholamandalam":
            # False positive analysis
            entity_preds = predictions_df[predictions_df["entity"] == entity_name]
            n_total = len(entity_preds)
            n_det = ((entity_preds["predicted_cr"] == True) &
                     (entity_preds["predicted_direction"] == "Deterioration")).sum()  # noqa: E712
            if n_total > 0:
                lines.append(f"**False positive rate:** {n_det}/{n_total} articles "
                              f"({n_det/n_total*100:.1f}%) predicted as deterioration")
                lines.append(f"{entity_name} has zero downgrades — every deterioration signal is a false positive.")
            else:
                lines.append(f"No predictions for {entity_name} in this dataset.")
        lines.append("")

    # --- Alert Threshold Grid ---
    lines.append("## 4. Alert Threshold Grid")
    lines.append("")
    lines.append("Grid search over N (min signals), M (window days), K (lookahead days).")
    lines.append("")

    if len(threshold_df) > 0:
        lines.append("| N | Window | Lookahead | Alerts | TP | FP | FN | Precision | Recall | F1 |")
        lines.append("|---|--------|-----------|--------|----|----|-----|-----------|--------|----|")
        for _, row in threshold_df.sort_values("f1", ascending=False).iterrows():
            lines.append(
                f"| {row['n_threshold']:.0f} | {row['window_days']:.0f}d | "
                f"{row['lookahead_days']:.0f}d | {row['n_alerts']:.0f} | "
                f"{row['n_true_positives']:.0f} | {row['n_false_positives']:.0f} | "
                f"{row['n_false_negatives']:.0f} | {row['precision']:.1%} | "
                f"{row['recall']:.1%} | {row['f1']:.3f} |"
            )
    lines.append("")

    # --- Naive Baselines ---
    lines.append("## 5. Naive Baselines")
    lines.append("")
    lines.append("| Baseline | Description | Signal Coverage | Mean Lead Time | Median Lead Time |")
    lines.append("|----------|-------------|----------------|----------------|-----------------|")

    # Model results
    if len(lead_time_df) > 0:
        model_coverage = f"{(lead_time_df['lead_time_days'].notna()).sum()}/{len(lead_time_df)}"
        model_mean = f"{lead_time_df['lead_time_days'].mean():.0f}d" if lead_time_df["lead_time_days"].notna().any() else "—"
        model_median = f"{lead_time_df['lead_time_days'].median():.0f}d" if lead_time_df["lead_time_days"].notna().any() else "—"
        lines.append(f"| **Our Model** | LoRA fine-tuned Qwen 2.5-7B | {model_coverage} | {model_mean} | {model_median} |")

    for name, bl in baseline_results.items():
        coverage = f"{bl['n_with_signal']}/{bl['n_total_actions']}"
        mean_lt = f"{bl['mean_lead_time']:.0f}d" if bl['mean_lead_time'] is not None and pd.notna(bl['mean_lead_time']) else "—"
        median_lt = f"{bl['median_lead_time']:.0f}d" if bl['median_lead_time'] is not None and pd.notna(bl['median_lead_time']) else "—"
        lines.append(f"| {name} | {bl['description']} | {coverage} | {mean_lt} | {median_lt} |")
    lines.append("")

    # --- Limitations ---
    lines.append("## 6. Limitations & Next Steps")
    lines.append("")
    lines.append("### Limitations")
    lines.append("- **Small event set:** Only 2-3 entities with downgrades in holdout, limiting statistical power")
    lines.append("- **Label quality:** Training data labels from Haiku (82.3% agreement with Sonnet) — "
                 "model inherits labeling biases")
    lines.append("- **Survivorship bias:** Articles sourced from GDELT may not capture all relevant news")
    lines.append("- **Entity overlap:** Multi-entity articles mentioning holdout entities stayed in training")
    lines.append("")
    lines.append("### Next Steps")
    lines.append("- **Prompted Opus baseline:** Run Claude Opus on holdout articles for apples-to-apples comparison")
    lines.append("- **FinRLlama baseline:** Run FinRLlama 3.2-3B on holdout articles (requires Colab)")
    lines.append("- **Contagion analysis (Phase 3):** Does DHFL crisis signal propagate to other NBFCs?")
    lines.append("- **Dashboard (Phase 4):** Visualize timelines with interactive plots")
    lines.append("")

    # Write the report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report written to {output_path}")


# ============================================================
# CLI Entry Point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest model predictions against actual rating actions"
    )
    parser.add_argument(
        "--predictions", type=Path, required=True,
        help="Path to model predictions JSONL (e.g., finetuned_holdout_outputs.jsonl)"
    )
    parser.add_argument(
        "--source", type=Path, required=True,
        help="Path to source JSONL (e.g., entity_holdout.jsonl)"
    )
    parser.add_argument(
        "--rating-actions", type=Path, required=True,
        help="Path to rating actions CSV"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/backtest_config.yaml"),
        help="Path to backtest config YAML"
    )
    parser.add_argument(
        "--match-by", choices=["index", "expected"], default="index",
        help="How to match predictions to source (index=positional, expected=output text)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/phase2_4_backtest_results.md"),
        help="Path to write the report"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Build entity alias map
    alias_map = build_entity_alias_map(config.get("entity_aliases", {}))

    # Load data
    logger.info("Loading predictions and source data...")
    predictions_df = load_predictions_with_metadata(
        args.predictions, args.source, match_by=args.match_by
    )

    negative_actions = config.get("negative_actions", ["downgrade", "default"])
    actions_df = load_rating_actions(args.rating_actions, action_types=negative_actions)
    lookback_days = config.get("lookback_days", 180)

    # Compute lead times
    logger.info("Computing lead times...")
    lead_time_df = compute_lead_time(
        predictions_df, actions_df,
        lookback_days=lookback_days, alias_map=alias_map,
    )

    # Sweep alert thresholds
    logger.info("Sweeping alert thresholds...")
    threshold_df = sweep_alert_thresholds(
        predictions_df, actions_df, config, alias_map=alias_map,
    )

    # Entity timelines
    logger.info("Building entity timelines...")
    holdout_entities = ["DHFL", "Reliance Capital", "Cholamandalam"]
    entity_timelines = {}
    for entity in holdout_entities:
        entity_timelines[entity] = compute_entity_timeline(
            predictions_df, actions_df, entity, alias_map=alias_map,
        )

    # Naive baselines
    logger.info("Computing naive baselines...")
    baseline_results = compute_naive_baselines(
        predictions_df, actions_df, alias_map=alias_map,
    )

    # Generate report
    logger.info("Generating report...")
    generate_backtest_report(
        lead_time_df=lead_time_df,
        threshold_df=threshold_df,
        entity_timelines=entity_timelines,
        baseline_results=baseline_results,
        predictions_df=predictions_df,
        actions_df=actions_df,
        output_path=args.output,
        alias_map=alias_map,
    )

    # Print summary to console
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    print(f"  Predictions:     {len(predictions_df):>6,d}")
    print(f"  Rating actions:  {len(actions_df):>6,d}")

    if len(lead_time_df) > 0:
        with_signal = lead_time_df[lead_time_df["lead_time_days"].notna()]
        print(f"  Actions with signal: {len(with_signal)}/{len(lead_time_df)}")
        if len(with_signal) > 0:
            print(f"  Mean lead time:  {with_signal['lead_time_days'].mean():>6.0f} days")
            print(f"  Median lead time:{with_signal['lead_time_days'].median():>6.0f} days")

    if len(threshold_df) > 0:
        best = threshold_df.loc[threshold_df["f1"].idxmax()]
        print(f"  Best threshold:  N≥{best['n_threshold']:.0f}, {best['window_days']:.0f}d window, "
              f"{best['lookahead_days']:.0f}d lookahead")
        print(f"  Best F1:         {best['f1']:.3f} "
              f"(P={best['precision']:.1%}, R={best['recall']:.1%})")

    print(f"\n  Report: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
