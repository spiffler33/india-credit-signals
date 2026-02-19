# WHY THIS: Signal propagation engine that turns per-article credit signals into
# per-entity daily scores, then propagates those scores to peer entities via the
# contagion graph. This is the core math that answers: "If DHFL is in distress,
# how much should we worry about Indiabulls?"
#
# Why additive (not multiplicative)? An entity with ZERO direct signals must still
# get contagion from its peers. Indiabulls had no articles flagged before its 2019
# downgrade — contagion from DHFL was its ONLY early warning. Multiplicative would
# give 0 × anything = 0, killing the signal entirely.

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from src.signals.entity_graph import EntityGraph


# ============================================================
# Direct Scoring
# ============================================================

def compute_direct_scores(
    signals_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compute daily direct credit scores per entity from article-level signals.

    # 🎓 Each article contributes a score based on three factors:
    # 1. Direction: Deterioration (+1.0), Improvement (-0.5), Neutral (0.0)
    # 2. Confidence: High (1.0), Medium (0.6), Low (0.3)
    # 3. Sector-wide: 1.5x multiplier if the signal affects the whole sector
    #
    # A single High-confidence Deterioration signal with sector_wide=True
    # scores: 1.0 × 1.0 × 1.5 = 1.5. Three such signals on the same day = 4.5.

    Args:
        signals_df: DataFrame with columns [entity, date, credit_relevant,
                    direction, signal_type, sector_wide, confidence]
        config: contagion config dict

    Returns:
        DataFrame[entity, date, direct_score, n_signals, n_det, n_imp,
                  has_sector_wide]
    """
    direction_mult = config.get("direction_multiplier", {
        "Deterioration": 1.0, "Improvement": -0.5, "Neutral": 0.0,
    })
    confidence_wt = config.get("confidence_weights", {
        "High": 1.0, "Medium": 0.6, "Low": 0.3,
    })
    sector_wide_mult = config.get("sector_wide_multiplier", 1.5)

    # Filter to credit-relevant signals only
    cr_df = signals_df[signals_df["credit_relevant"] == 1].copy()

    if len(cr_df) == 0:
        return pd.DataFrame(columns=[
            "entity", "date", "direct_score", "n_signals", "n_det", "n_imp",
            "has_sector_wide",
        ])

    # Compute per-article score
    cr_df["dir_mult"] = cr_df["direction"].map(direction_mult).fillna(0.0)
    cr_df["conf_wt"] = cr_df["confidence"].map(confidence_wt).fillna(0.3)
    cr_df["sw_mult"] = cr_df["sector_wide"].apply(
        lambda x: sector_wide_mult if x else 1.0
    )
    cr_df["article_score"] = cr_df["dir_mult"] * cr_df["conf_wt"] * cr_df["sw_mult"]

    # Normalize dates to day level
    cr_df["date_day"] = pd.to_datetime(cr_df["date"]).dt.normalize()

    # Aggregate per entity per day
    grouped = cr_df.groupby(["entity", "date_day"]).agg(
        direct_score=("article_score", "sum"),
        n_signals=("article_score", "count"),
        n_det=("direction", lambda x: (x == "Deterioration").sum()),
        n_imp=("direction", lambda x: (x == "Improvement").sum()),
        has_sector_wide=("sector_wide", lambda x: x.any()),
    ).reset_index()

    grouped.rename(columns={"date_day": "date"}, inplace=True)

    logger.debug(
        f"Computed direct scores: {len(grouped)} entity-day rows, "
        f"{len(grouped['entity'].unique())} entities"
    )

    return grouped


# ============================================================
# Contagion Propagation
# ============================================================

def compute_contagion_scores(
    direct_df: pd.DataFrame,
    graph: EntityGraph,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Propagate direct scores to peer entities via the contagion graph.

    # 🎓 KEY CONCEPT: Contagion Propagation
    #
    # For each entity E on each day D:
    #   contagion_score(E, D) = SUM over peers P:
    #       edge_weight(P, E)  ×  rolling_direct(P, D, window)  ×  peer_discount
    #
    # rolling_direct(P, D, window) = sum of P's direct scores in [D - window, D]
    #
    # peer_discount = 0.5 for non-sector-wide signals. Sector-wide signals propagate
    # at full strength because they affect the entire subsector by definition.
    #
    # Example: DHFL has direct_score=3.0 on Nov 15 2018 (accumulated over 30-day window).
    # Indiabulls (same subsector, weight=0.8) gets contagion = 0.8 × 3.0 × 0.5 = 1.2.
    # If DHFL's signals were sector-wide: 0.8 × 3.0 × 1.0 = 2.4.

    Args:
        direct_df: Output of compute_direct_scores()
        graph: EntityGraph for peer lookups
        config: contagion config dict

    Returns:
        DataFrame[entity, date, contagion_score, n_sources, top_source]
    """
    window_days = config.get("contagion_window_days", 30)
    peer_discount = config.get("peer_signal_discount", 0.5)

    if len(direct_df) == 0:
        return pd.DataFrame(columns=[
            "entity", "date", "contagion_score", "n_sources", "top_source",
        ])

    # Build a lookup: entity → sorted list of (date, direct_score, has_sector_wide)
    entity_scores: dict[str, pd.DataFrame] = {}
    for entity in direct_df["entity"].unique():
        entity_df = direct_df[direct_df["entity"] == entity].sort_values("date")
        entity_scores[entity] = entity_df

    # Get all unique dates across all entities
    all_dates = sorted(direct_df["date"].unique())

    # For each entity in the graph (not just those with direct signals),
    # compute contagion on each day
    contagion_rows: list[dict] = []

    # All entities that exist in the graph
    all_entities = set(graph.nodes.keys())
    # Also include entities that have direct signals (may use aliases)
    for entity in direct_df["entity"].unique():
        canonical = graph.normalize_entity(entity)
        all_entities.add(canonical)

    for entity in sorted(all_entities):
        peers = graph.get_peers(entity, min_weight=0.01)
        if not peers:
            continue

        for date in all_dates:
            window_start = date - pd.Timedelta(days=window_days)
            total_contagion = 0.0
            n_sources = 0
            top_source_name = ""
            top_source_contrib = 0.0

            for peer_name, edge_weight in peers:
                # Get peer's direct scores in the window
                peer_df = entity_scores.get(peer_name)
                if peer_df is None:
                    continue

                window_data = peer_df[
                    (peer_df["date"] >= window_start) &
                    (peer_df["date"] <= date)
                ]

                if len(window_data) == 0:
                    continue

                # Sum peer's direct scores in window, applying peer discount
                # If any signal was sector-wide, use full strength (no discount)
                any_sector_wide = window_data["has_sector_wide"].any()
                discount = 1.0 if any_sector_wide else peer_discount

                rolling_direct = window_data["direct_score"].sum()
                contrib = edge_weight * rolling_direct * discount

                if contrib != 0:
                    total_contagion += contrib
                    n_sources += 1
                    if abs(contrib) > abs(top_source_contrib):
                        top_source_contrib = contrib
                        top_source_name = peer_name

            if total_contagion != 0:
                contagion_rows.append({
                    "entity": entity,
                    "date": date,
                    "contagion_score": total_contagion,
                    "n_sources": n_sources,
                    "top_source": top_source_name,
                })

    result = pd.DataFrame(contagion_rows)
    if len(result) > 0:
        logger.debug(
            f"Computed contagion scores: {len(result)} entity-day rows, "
            f"{result['entity'].nunique()} entities affected"
        )
    return result


# ============================================================
# Combined Scores + Rolling Windows
# ============================================================

def compute_rolling_scores(
    total_df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling window averages of total scores.

    # 🎓 Rolling windows smooth out daily noise. A single bad article on one day
    # shouldn't trigger an alert. But 5 bad articles over 7 days = real pattern.
    # We compute 7-day, 30-day, and 90-day rolling averages.

    Args:
        total_df: DataFrame with [entity, date, total_score]
        windows: List of window sizes in days (default: [7, 30, 90])

    Returns:
        total_df with additional columns: rolling_7d, rolling_30d, rolling_90d
    """
    if windows is None:
        windows = [7, 30, 90]

    if len(total_df) == 0:
        for w in windows:
            total_df[f"rolling_{w}d"] = []
        return total_df

    result = total_df.copy()

    for entity in result["entity"].unique():
        entity_mask = result["entity"] == entity
        entity_data = result[entity_mask].sort_values("date")

        if len(entity_data) == 0:
            continue

        # Create a date-indexed series for rolling calculation
        score_series = entity_data.set_index("date")["total_score"]

        # Reindex to daily frequency to handle gaps
        date_range = pd.date_range(
            start=score_series.index.min(),
            end=score_series.index.max(),
            freq="D",
        )
        daily_scores = score_series.reindex(date_range, fill_value=0.0)

        for w in windows:
            col_name = f"rolling_{w}d"
            rolling = daily_scores.rolling(window=w, min_periods=1).mean()

            # Map back to original dates only
            for idx in entity_data.index:
                date = result.loc[idx, "date"]
                if date in rolling.index:
                    result.loc[idx, col_name] = rolling.loc[date]
                else:
                    result.loc[idx, col_name] = 0.0

    return result


def compute_all_scores(
    signals_df: pd.DataFrame,
    graph: EntityGraph,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Main entry point: signals → direct + contagion → rolling windows.

    # 🎓 This is the full pipeline in one call:
    # 1. Per-article signals → daily direct scores per entity
    # 2. Direct scores → contagion propagation to peers
    # 3. Direct + contagion → total score → rolling windows

    Args:
        signals_df: Article-level signals with [entity, date, credit_relevant,
                    direction, signal_type, sector_wide, confidence]
        graph: EntityGraph for peer lookups
        config: contagion config dict

    Returns:
        DataFrame[entity, date, direct_score, contagion_score, total_score,
                  rolling_7d, rolling_30d, rolling_90d, n_signals, n_sources, top_source]
    """
    windows = config.get("rolling_windows", [7, 30, 90])

    # Step 1: Direct scores
    direct_df = compute_direct_scores(signals_df, config)

    # Step 2: Contagion scores
    contagion_df = compute_contagion_scores(direct_df, graph, config)

    # Step 3: Merge direct + contagion into total
    if len(direct_df) == 0 and len(contagion_df) == 0:
        return pd.DataFrame(columns=[
            "entity", "date", "direct_score", "contagion_score", "total_score",
            "n_signals", "n_sources", "top_source",
        ] + [f"rolling_{w}d" for w in windows])

    # Start with direct scores
    if len(direct_df) > 0:
        total_df = direct_df[["entity", "date", "direct_score", "n_signals"]].copy()
    else:
        total_df = pd.DataFrame(columns=["entity", "date", "direct_score", "n_signals"])

    # Merge contagion scores
    if len(contagion_df) > 0:
        total_df = pd.merge(
            total_df,
            contagion_df[["entity", "date", "contagion_score", "n_sources", "top_source"]],
            on=["entity", "date"],
            how="outer",
        )
    else:
        total_df["contagion_score"] = 0.0
        total_df["n_sources"] = 0
        total_df["top_source"] = ""

    # Fill NaN from outer join
    total_df["direct_score"] = total_df["direct_score"].fillna(0.0)
    total_df["contagion_score"] = total_df["contagion_score"].fillna(0.0)
    total_df["n_signals"] = total_df["n_signals"].fillna(0).astype(int)
    total_df["n_sources"] = total_df["n_sources"].fillna(0).astype(int)
    total_df["top_source"] = total_df["top_source"].fillna("")

    # Total = direct + contagion
    total_df["total_score"] = total_df["direct_score"] + total_df["contagion_score"]

    # Step 4: Rolling windows
    total_df = compute_rolling_scores(total_df, windows)

    # Sort for readability
    total_df = total_df.sort_values(["entity", "date"]).reset_index(drop=True)

    logger.info(
        f"Computed all scores: {len(total_df)} entity-day rows, "
        f"{total_df['entity'].nunique()} entities, "
        f"date range {total_df['date'].min()} to {total_df['date'].max()}"
    )

    return total_df
