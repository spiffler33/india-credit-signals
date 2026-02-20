# WHY THIS: The Sector Heatmap answers: "Which subsectors are stressed RIGHT NOW?"
# It's a grid of all 44 entities, colored green→yellow→red by their rolling risk score.
# Grouped by subsector so the boss can instantly see: "housing finance is red,
# vehicle finance is green — the crisis is contained to one sector."
#
# The date slider lets you ANIMATE through time. For the demo, scrub from
# mid-2018 to early-2019 and watch housing finance light up red while others stay calm.
# This is the sector-level view that justifies the contagion layer.

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.utils.data_loader import (
    get_entities_by_subsector,
    load_entity_metadata,
    load_entity_scores,
)
from src.dashboard.utils.styling import (
    PLOTLY_LAYOUT,
    SCORE_COLORSCALE,
    SUBSECTOR_COLORS,
    SUBSECTOR_LABELS,
)


def render() -> None:
    """Render the Sector Heatmap view."""
    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    entity_scores = load_entity_scores()
    metadata = load_entity_metadata()
    entities_by_subsector = get_entities_by_subsector(metadata)

    # ----------------------------------------------------------
    # Controls
    # ----------------------------------------------------------
    ctrl1, ctrl2 = st.columns([3, 2])

    with ctrl1:
        # 🎓 Rolling window: larger = smoother (filters out single-day spikes),
        # smaller = more responsive. 30d is the sweet spot for crisis detection.
        window_col = st.select_slider(
            "Rolling window",
            options=[7, 30, 90],
            value=30,
            format_func=lambda x: f"{x}-day average",
            help=(
                "How many days to average the risk score over. "
                "7 days is noisy but responsive. 90 days is smooth but slow to react."
            ),
        )
        rolling_col = f"rolling_{window_col}d"

    with ctrl2:
        score_type = st.radio(
            "Score type",
            options=["Total (direct + contagion)", "Direct only", "Contagion only"],
            index=0,
            horizontal=True,
            help=(
                "**Total:** Combined signal from direct news + peer contagion. "
                "**Direct:** Only signals from articles mentioning this entity. "
                "**Contagion:** Only signal propagated from distressed peers."
            ),
        )

    # Date selector — the animation driver
    all_dates = entity_scores["date"].dt.date.unique()
    all_dates_sorted = sorted(all_dates)

    selected_date = st.select_slider(
        "Date",
        options=all_dates_sorted,
        value=all_dates_sorted[-1],
        format_func=lambda d: d.strftime("%b %d, %Y"),
        help="Scrub through time to watch crises develop. Try Nov 2018 for the DHFL crisis.",
    )

    # ----------------------------------------------------------
    # Filter to selected date
    # ----------------------------------------------------------
    day_data = entity_scores[
        entity_scores["date"].dt.date == selected_date
    ].copy()

    if day_data.empty:
        st.warning(f"No data for {selected_date}.")
        return

    # Pick the right score column based on user selection
    if "Direct only" in score_type:
        # 🎓 For direct-only, we use direct_score (not a rolling column for direct).
        # The rolling columns are pre-computed on total_score. We approximate by
        # using the direct_score as-is (it's already a daily aggregate).
        day_data["display_score"] = day_data["direct_score"]
    elif "Contagion only" in score_type:
        day_data["display_score"] = day_data["contagion_score"]
    else:
        day_data["display_score"] = day_data[rolling_col]

    # ----------------------------------------------------------
    # Build heatmap: one row per subsector, one column per entity
    # ----------------------------------------------------------
    # 🎓 We build this as a Plotly heatmap (imshow-style) where:
    # - Rows = subsectors (grouped categories)
    # - Columns = entities within each subsector
    # - Cell color = risk score (green→red)
    # This gives the boss a "dashboard of dashboards" — glance and see which sectors hurt.

    # Order subsectors by average score (most stressed at top)
    subsector_avg = (
        day_data.groupby("subsector")["display_score"]
        .mean()
        .sort_values(ascending=False)
    )
    ordered_subsectors = subsector_avg.index.tolist()

    # Determine max score for color scaling
    max_score = max(day_data["display_score"].quantile(0.95), 1.0)

    # Build one horizontal bar per subsector
    fig = go.Figure()

    y_labels = []
    y_positions = []
    current_y = 0

    for subsector in ordered_subsectors:
        entities_in_sub = sorted(entities_by_subsector.get(subsector, []))
        sub_data = day_data[day_data["entity"].isin(entities_in_sub)].set_index("entity")

        if sub_data.empty:
            continue

        sub_label = SUBSECTOR_LABELS.get(subsector, subsector)
        sub_avg = subsector_avg.get(subsector, 0)

        # Sort entities by score within subsector (worst first)
        sub_data = sub_data.sort_values("display_score", ascending=False)
        entities_sorted = sub_data.index.tolist()

        scores = sub_data["display_score"].values
        # Normalize to 0-1 for colorscale
        normalized = [min(max(s / max_score, 0), 1) for s in scores]

        # 🎓 Each entity becomes a bar segment. We use a horizontal bar chart
        # where bar width = fixed, color = score. This gives a treemap-like feel.
        for i, entity in enumerate(entities_sorted):
            score = sub_data.loc[entity, "display_score"]
            norm = min(max(score / max_score, 0), 1)

            # Map normalized score to color using our colorscale
            from src.dashboard.utils.styling import score_to_color
            color = score_to_color(score, max_score)

            fig.add_trace(
                go.Bar(
                    x=[1],
                    y=[f"{sub_label}"],
                    orientation="h",
                    marker=dict(color=color, line=dict(width=0.5, color="white")),
                    name=entity,
                    text=entity,
                    textposition="inside",
                    textfont=dict(size=9, color="white" if norm > 0.4 else "black"),
                    hovertemplate=(
                        f"<b>{entity}</b><br>"
                        f"Subsector: {sub_label}<br>"
                        f"Risk score: {score:.2f}<br>"
                        f"<extra></extra>"
                    ),
                    showlegend=False,
                    width=0.7,
                )
            )

    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("legend", "margin", "hovermode")},
        barmode="stack",
        title=dict(
            text=f"Sector Risk — {selected_date.strftime('%b %d, %Y')}",
            subtitle=dict(
                text=f"Each block = one entity. Color: green (safe) → red (stressed). {window_col}-day rolling average.",
            ),
        ),
        xaxis=dict(visible=False),
        yaxis=dict(
            title="",
            autorange="reversed",
            tickfont=dict(size=12),
        ),
        height=max(350, len(ordered_subsectors) * 80),
        showlegend=False,
        margin=dict(l=160, r=30, t=70, b=30),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------
    # Summary stats below the chart
    # ----------------------------------------------------------
    st.subheader("Subsector Summary")

    summary_rows = []
    for subsector in ordered_subsectors:
        sub_label = SUBSECTOR_LABELS.get(subsector, subsector)
        sub_entities = entities_by_subsector.get(subsector, [])
        sub_data = day_data[day_data["entity"].isin(sub_entities)]

        if sub_data.empty:
            continue

        avg_score = sub_data["display_score"].mean()
        max_entity = sub_data.loc[sub_data["display_score"].idxmax()]
        n_elevated = len(sub_data[sub_data["display_score"] > 4.0])

        summary_rows.append({
            "Subsector": sub_label,
            "Entities": len(sub_data),
            "Avg Score": f"{avg_score:.2f}",
            "Highest": f"{max_entity['entity']} ({max_entity['display_score']:.2f})",
            "Elevated (>4.0)": n_elevated,
        })

    if summary_rows:
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )
