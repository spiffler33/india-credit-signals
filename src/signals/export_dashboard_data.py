# WHY THIS: Data export pipeline that bridges the scoring engine (Phase 2-3) and
# the Streamlit dashboard (Phase 4). Runs the full pipeline — signal loading,
# direct scoring, contagion propagation, rolling windows — then saves intermediate
# DataFrames as parquet files for fast dashboard consumption.
#
# Why not compute scores live in Streamlit? The scoring pipeline iterates
# 44 entities × ~2,500 dates × 43 peers = ~4.7M iterations. That takes 30-60
# seconds. A dashboard that takes 60s to load is unusable for a demo. Pre-compute
# once, save parquet, load in <1 second.

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
from src.signals.propagation import (
    compute_all_scores,
    compute_direct_scores,
)
from src.signals.contagion_backtest import (
    load_all_signals,
    run_crisis_replay,
)
from src.training.backtest import (
    extract_title_from_input,
    load_jsonl,
    load_rating_actions,
)
from src.data.parse_training_output import parse_training_output


# ============================================================
# 1. Entity Scores — daily direct + contagion + rolling
# ============================================================

def export_entity_scores(
    signals_df: pd.DataFrame,
    graph: EntityGraph,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run full scoring pipeline and return entity-day scores.

    # 🎓 This is the most important export — it powers the Entity Timeline,
    # Sector Heatmap, and Alert Dashboard views. Each row is one entity on
    # one day, with columns for direct score, contagion score, total, and
    # rolling averages (7d, 30d, 90d).

    Returns:
        DataFrame[entity, date, direct_score, contagion_score, total_score,
                  rolling_7d, rolling_30d, rolling_90d, n_signals, n_sources,
                  top_source]
    """
    scores_df = compute_all_scores(signals_df, graph, config)

    if len(scores_df) == 0:
        logger.warning("No scores computed — empty signals?")
        return scores_df

    # Add subsector for easier dashboard filtering
    scores_df["subsector"] = scores_df["entity"].apply(
        lambda e: graph.nodes.get(graph.normalize_entity(e), None)
    ).apply(lambda n: n.subsector if n else "unknown")

    logger.info(
        f"Entity scores: {len(scores_df)} rows, "
        f"{scores_df['entity'].nunique()} entities, "
        f"{scores_df['date'].min().date()} to {scores_df['date'].max().date()}"
    )
    return scores_df


# ============================================================
# 2. Signals — per-article signal details for Signal Feed
# ============================================================

def export_signals(
    signals_df: pd.DataFrame,
    articles_path: Path,
    holdout_source_path: Path,
    graph: EntityGraph,
) -> pd.DataFrame:
    """Enrich signals with article titles/URLs for the Signal Feed view.

    # 🎓 The Signal Feed lets the analyst drill into individual articles.
    # "What did the model extract from this article?" The raw signals_df
    # has entity/date/direction but no title or URL. We join back to the
    # article sources to add those human-readable fields.

    Returns:
        DataFrame[entity, date, credit_relevant, direction, signal_type,
                  confidence, sector_wide, signal_source, title, url, subsector]
    """
    # Build URL→title lookup from articles CSV
    articles_df = pd.read_csv(articles_path)
    url_to_title: dict[str, str] = {}
    for _, row in articles_df.iterrows():
        url_to_title[row["article_url"]] = row.get("article_title", "")

    # Build URL→date lookup for matching
    url_to_date: dict[str, str] = {}
    url_to_entity: dict[str, str] = {}
    for _, row in articles_df.iterrows():
        url_to_date[row["article_url"]] = row["article_date"]
        url_to_entity[row["article_url"]] = row.get("entities", "")

    # For holdout signals, get titles from source JSONL
    holdout_titles: dict[tuple[str, str], str] = {}  # (entity, date) → title
    holdout_source = load_jsonl(holdout_source_path)
    for src in holdout_source:
        input_text = src.get("input", "")
        title = extract_title_from_input(input_text)
        # Extract entity and date from input
        entity = ""
        date_str = ""
        for line in input_text.split("\n"):
            if line.startswith("Entity:"):
                entity = line[len("Entity:"):].strip()
            elif line.startswith("Date:"):
                date_str = line[len("Date:"):].strip()
        if entity and date_str:
            holdout_titles[(entity, date_str)] = title

    # Enrich signals
    enriched = signals_df.copy()

    # Add title — for label-sourced signals, match via articles CSV
    # For holdout signals, match via the input text
    titles: list[str] = []
    urls: list[str] = []

    for _, row in enriched.iterrows():
        entity = row["entity"]
        date_str = str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])[:10]

        if row.get("signal_source") == "model":
            # Holdout — lookup by (entity, date)
            title = holdout_titles.get((entity, date_str), "")
            titles.append(title)
            urls.append("")  # holdout predictions don't have URLs easily
        else:
            # Label-sourced — we'd need a more complex join
            # For now, leave blank — the Signal Feed can show these from labels_final
            titles.append("")
            urls.append("")

    enriched["title"] = titles
    enriched["url"] = urls

    # Add subsector
    enriched["subsector"] = enriched["entity"].apply(
        lambda e: graph.nodes.get(graph.normalize_entity(e), None)
    ).apply(lambda n: n.subsector if n else "unknown")

    logger.info(f"Signals export: {len(enriched)} articles, {enriched['entity'].nunique()} entities")
    return enriched


# ============================================================
# 3. Contagion Edges — for Network Graph view
# ============================================================

def export_contagion_edges(
    graph: EntityGraph,
) -> pd.DataFrame:
    """Export graph edges with weights for the network visualization.

    # 🎓 The Network Graph view needs: source, target, weight, same_subsector.
    # We export the static graph structure. The dashboard overlays dynamic
    # signal strength by multiplying edge_weight × rolling_direct(source)
    # for the selected date.

    Returns:
        DataFrame[source, target, weight, source_subsector, target_subsector,
                  same_subsector]
    """
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for source, targets in graph.edges.items():
        for target, edge in targets.items():
            # Avoid duplicates (graph is symmetric)
            pair = tuple(sorted([source, target]))
            if pair in seen:
                continue
            seen.add(pair)

            source_node = graph.nodes.get(source)
            target_node = graph.nodes.get(target)

            rows.append({
                "source": source,
                "target": target,
                "weight": edge.weight,
                "source_subsector": source_node.subsector if source_node else "unknown",
                "target_subsector": target_node.subsector if target_node else "unknown",
                "same_subsector": (
                    source_node.subsector == target_node.subsector
                    if source_node and target_node else False
                ),
            })

    df = pd.DataFrame(rows)
    logger.info(f"Contagion edges: {len(df)} edges ({df['same_subsector'].sum()} intra-subsector)")
    return df


# ============================================================
# 4. Rating Actions — for Timeline overlay
# ============================================================

def export_rating_actions(
    rating_actions_path: Path,
) -> pd.DataFrame:
    """Load and clean rating actions for dashboard overlay.

    # 🎓 The Entity Timeline shows vertical red lines at each rating action.
    # We filter to downgrades and defaults (the events we're trying to predict)
    # plus upgrades (green lines showing when an entity recovered).

    Returns:
        DataFrame[entity, date, agency, action_type, from_rating, to_rating]
    """
    df = pd.read_csv(rating_actions_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Keep the columns the dashboard needs
    keep_cols = ["entity", "date", "agency", "action_type", "from_rating", "to_rating"]
    available = [c for c in keep_cols if c in df.columns]
    result = df[available].copy()

    # Add a simplified outcome column for coloring
    downgrade_types = {"downgrade", "default", "withdrawn_negative", "suspended"}
    upgrade_types = {"upgrade"}
    result["outcome"] = result["action_type"].apply(
        lambda x: "negative" if str(x).lower() in downgrade_types
        else ("positive" if str(x).lower() in upgrade_types else "neutral")
    )

    logger.info(
        f"Rating actions: {len(result)} total, "
        f"{(result['outcome'] == 'negative').sum()} negative, "
        f"{(result['outcome'] == 'positive').sum()} positive"
    )
    return result


# ============================================================
# 5. Entity Metadata — for filters, labels, dropdowns
# ============================================================

def export_entity_metadata(
    graph: EntityGraph,
) -> dict[str, Any]:
    """Export entity metadata as a JSON-serializable dict.

    # 🎓 The dashboard needs entity info for dropdown labels, subsector
    # grouping, and color coding. This is a small static file — JSON
    # is fine (no need for parquet on 44 rows).

    Returns:
        Dict with entities list and subsector summary.
    """
    entities: list[dict] = []
    for name, node in sorted(graph.nodes.items()):
        peers = graph.get_peers(name, min_weight=0.01)
        entities.append({
            "name": name,
            "full_name": node.full_name,
            "subsector": node.subsector,
            "status": node.status,
            "n_peers": len(peers),
            "n_intra_peers": sum(1 for _, w in peers if w > 0.5),
            "n_cross_peers": sum(1 for _, w in peers if w <= 0.5),
        })

    # Subsector summary
    subsector_counts: dict[str, int] = {}
    for node in graph.nodes.values():
        subsector_counts[node.subsector] = subsector_counts.get(node.subsector, 0) + 1

    return {
        "entities": entities,
        "subsectors": subsector_counts,
        "n_entities": len(entities),
        "generated_at": datetime.now().isoformat(),
    }


# ============================================================
# 6. Crisis Results — for Alert context
# ============================================================

def export_crisis_results(
    signals_df: pd.DataFrame,
    graph: EntityGraph,
    config: dict[str, Any],
) -> list[dict]:
    """Run crisis replays and export results as JSON-serializable dicts.

    # 🎓 The Alert Dashboard shows context like "79% of similar alerts preceded
    # a downgrade within 90 days." This comes from crisis replay results.
    # We serialize lead times, improvements, and control breach rates.

    Returns:
        List of crisis result dicts (JSON-serializable).
    """
    crises = config.get("crises", {})
    results: list[dict] = []

    for crisis_key, crisis_config in crises.items():
        logger.info(f"Running crisis replay: {crisis_key}")
        replay = run_crisis_replay(signals_df, graph, crisis_config, config)

        # Serialize — strip the scores_df (too large for JSON), keep summaries
        serialized = {
            "crisis_key": crisis_key,
            "crisis_name": replay["crisis_name"],
            "start_date": str(replay["start_date"].date()),
            "end_date": str(replay["end_date"].date()),
            "n_signals": replay["n_signals"],
            "n_entities": replay["n_entities"],
            "source_results": [
                {
                    "entity": s["entity"],
                    "n_days_with_score": s["n_days_with_score"],
                    "total_direct": round(s["total_direct"], 2),
                    "peak_direct": round(s["peak_direct"], 2),
                    "first_signal_date": (
                        str(s["first_signal_date"].date())
                        if s["first_signal_date"] is not None and pd.notna(s["first_signal_date"])
                        else None
                    ),
                }
                for s in replay["source_results"]
            ],
            "target_results": [
                _serialize_target_result(t)
                for t in replay["target_results"]
            ],
        }
        results.append(serialized)

    logger.info(f"Crisis results: {len(results)} crises exported")
    return results


def _serialize_target_result(target: dict) -> dict:
    """Convert a target result dict to JSON-serializable form."""
    def _date_str(d: Any) -> str | None:
        if d is None or (hasattr(d, "__class__") and pd.isna(d)):
            return None
        if hasattr(d, "date"):
            return str(d.date())
        return str(d)

    return {
        "entity": target["entity"],
        "is_control": target.get("is_control", False),
        "first_action": _date_str(target.get("first_action")),
        "first_breach_date": _date_str(target.get("first_breach_date")),
        "lead_time_days": target.get("lead_time_days"),
        "direct_only_lead_time": target.get("direct_only_lead_time"),
        "lead_time_improvement": target.get("lead_time_improvement"),
        "peak_score": round(target.get("peak_score", 0.0), 2),
        "peak_contagion": round(target.get("peak_contagion", 0.0), 2),
        "n_warning_breaches": target.get("n_warning_breaches", 0),
    }


# ============================================================
# Main Export Pipeline
# ============================================================

def run_export(config_path: Path, output_dir: Path) -> None:
    """Run the full export pipeline: load data → compute scores → save parquet.

    # 🎓 PIPELINE FLOW:
    # 1. Load config + entity graph
    # 2. Load all signals (holdout predictions + label proxies)
    # 3. Compute full-corpus scores (direct + contagion + rolling)
    # 4. Export 6 output files to data/dashboard/
    #
    # Total runtime: ~60-120 seconds on M1 Mac (dominated by contagion
    # propagation over 44 entities × ~2,500 unique dates × 43 peers).
    """
    logger.info(f"Starting dashboard data export...")
    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})

    # Build entity graph
    entity_yaml = Path(paths.get("entity_graph", "configs/nbfc_entities.yaml"))
    graph = load_entity_graph(entity_yaml, config)

    # Load all signals (holdout predictions + label proxies)
    holdout_entities = ["DHFL", "Reliance Capital", "Cholamandalam Investment"]
    signals_df = load_all_signals(
        holdout_pred_path=Path(paths["holdout_predictions"]),
        holdout_source_path=Path(paths["holdout_source"]),
        labels_path=Path(paths["labels"]),
        articles_path=Path(paths["articles"]),
        holdout_entities=holdout_entities,
        alias_map=graph.alias_map,
    )

    logger.info(f"Loaded {len(signals_df)} signals across {signals_df['entity'].nunique()} entities")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Entity Scores ---
    logger.info("Computing entity scores (direct + contagion + rolling)...")
    scores_df = export_entity_scores(signals_df, graph, config)
    scores_path = output_dir / "entity_scores.parquet"
    scores_df.to_parquet(scores_path, index=False)
    logger.info(f"  Saved: {scores_path} ({len(scores_df)} rows)")

    # --- 2. Signals ---
    logger.info("Exporting article-level signals...")
    signals_export = export_signals(
        signals_df,
        articles_path=Path(paths["articles"]),
        holdout_source_path=Path(paths["holdout_source"]),
        graph=graph,
    )
    signals_path = output_dir / "signals.parquet"
    signals_export.to_parquet(signals_path, index=False)
    logger.info(f"  Saved: {signals_path} ({len(signals_export)} rows)")

    # --- 3. Contagion Edges ---
    logger.info("Exporting contagion graph edges...")
    edges_df = export_contagion_edges(graph)
    edges_path = output_dir / "contagion_edges.parquet"
    edges_df.to_parquet(edges_path, index=False)
    logger.info(f"  Saved: {edges_path} ({len(edges_df)} edges)")

    # --- 4. Rating Actions ---
    logger.info("Exporting rating actions...")
    rating_path = Path(paths.get("rating_actions", "data/raw/rating_actions_sourced.csv"))
    rating_df = export_rating_actions(rating_path)
    rating_out_path = output_dir / "rating_actions.parquet"
    rating_df.to_parquet(rating_out_path, index=False)
    logger.info(f"  Saved: {rating_out_path} ({len(rating_df)} actions)")

    # --- 5. Entity Metadata ---
    logger.info("Exporting entity metadata...")
    metadata = export_entity_metadata(graph)
    metadata_path = output_dir / "entity_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved: {metadata_path} ({metadata['n_entities']} entities)")

    # --- 6. Crisis Results ---
    logger.info("Running crisis replays for alert context...")
    crisis_results = export_crisis_results(signals_df, graph, config)
    crisis_path = output_dir / "crisis_results.json"
    with open(crisis_path, "w", encoding="utf-8") as f:
        json.dump(crisis_results, f, indent=2)
    logger.info(f"  Saved: {crisis_path} ({len(crisis_results)} crises)")

    # --- Summary ---
    total_size = sum(
        p.stat().st_size for p in output_dir.iterdir() if p.is_file()
    )
    logger.info(
        f"\n{'='*60}\n"
        f"  Dashboard export complete!\n"
        f"  Output directory: {output_dir}\n"
        f"  Files: {sum(1 for _ in output_dir.iterdir())}\n"
        f"  Total size: {total_size / 1024 / 1024:.1f} MB\n"
        f"{'='*60}"
    )


# ============================================================
# CLI Entry Point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export dashboard data: signals → scores → parquet files"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/contagion_config.yaml"),
        help="Path to contagion config YAML",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/dashboard"),
        help="Output directory for dashboard data files",
    )
    args = parser.parse_args()

    run_export(args.config, args.output)


if __name__ == "__main__":
    main()
