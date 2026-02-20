# WHY THIS: The Alert Dashboard is the "so what?" view. The other views show
# analysis. This one shows ACTION: which entities need attention RIGHT NOW?
#
# It computes alert levels (warning / critical) by comparing rolling scores
# against backtested thresholds. The key metric is precision context:
# "79% of similar alerts preceded an actual downgrade within 90 days."
# That's what makes the boss trust it.
#
# Thresholds from contagion_config.yaml: warning=4.0, critical=10.0
# These were calibrated in Phase 3 (v2 normalization) so that:
# - Housing finance targets breach warning >80% of crisis days
# - Cross-sector controls breach <10% of days

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.utils.data_loader import (
    get_entities_by_subsector,
    load_crisis_results,
    load_entity_metadata,
    load_entity_scores,
    load_rating_actions,
)
from src.dashboard.utils.styling import (
    PLOTLY_LAYOUT,
    SUBSECTOR_COLORS,
    SUBSECTOR_LABELS,
    score_to_color,
)

# 🎓 These thresholds come from Phase 3 backtest calibration.
# warning=4.0: housing finance targets breach this on >80% of crisis days.
# critical=10.0: only during peak crisis periods.
# They were set AFTER normalizing contagion by peer count (v2 fix).
WARNING_THRESHOLD = 4.0
CRITICAL_THRESHOLD = 10.0

ALERT_COLORS = {
    "Critical": "#E63946",
    "Warning": "#F4A261",
    "Normal": "#2A9D8F",
}


def render() -> None:
    """Render the Alert Dashboard view."""
    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    entity_scores = load_entity_scores()
    metadata = load_entity_metadata()
    rating_actions = load_rating_actions()
    crisis_results = load_crisis_results()
    entities_by_subsector = get_entities_by_subsector(metadata)
    subsector_map = {e["name"]: e["subsector"] for e in metadata["entities"]}

    # ----------------------------------------------------------
    # Controls
    # ----------------------------------------------------------
    ctrl1, ctrl2 = st.columns([3, 2])

    with ctrl1:
        all_dates = entity_scores["date"].dt.date.unique()
        all_dates_sorted = sorted(all_dates)
        selected_date = st.select_slider(
            "Date",
            options=all_dates_sorted,
            value=all_dates_sorted[-1],
            format_func=lambda d: d.strftime("%b %d, %Y"),
            help="Check alert status on any historical date. Try Feb 2019 for the DHFL crisis peak.",
        )

    with ctrl2:
        window = st.select_slider(
            "Rolling window",
            options=[7, 30, 90],
            value=30,
            format_func=lambda x: f"{x}-day average",
        )
        rolling_col = f"rolling_{window}d"

    # ----------------------------------------------------------
    # Compute alerts for the selected date
    # ----------------------------------------------------------
    day_data = entity_scores[
        entity_scores["date"].dt.date == selected_date
    ].copy()

    if day_data.empty:
        st.warning(f"No data for {selected_date}.")
        return

    # Classify alert level
    def _alert_level(score: float) -> str:
        if score >= CRITICAL_THRESHOLD:
            return "Critical"
        elif score >= WARNING_THRESHOLD:
            return "Warning"
        return "Normal"

    day_data["alert_level"] = day_data[rolling_col].apply(_alert_level)
    day_data["subsector_label"] = day_data["subsector"].map(SUBSECTOR_LABELS)

    # ----------------------------------------------------------
    # Alert summary metrics
    # ----------------------------------------------------------
    n_critical = len(day_data[day_data["alert_level"] == "Critical"])
    n_warning = len(day_data[day_data["alert_level"] == "Warning"])
    n_normal = len(day_data[day_data["alert_level"] == "Normal"])

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Critical",
            n_critical,
            help=f"Rolling score >= {CRITICAL_THRESHOLD}. Immediate attention required.",
        )
    with m2:
        st.metric(
            "Warning",
            n_warning,
            help=f"Rolling score >= {WARNING_THRESHOLD}. Elevated risk — monitor closely.",
        )
    with m3:
        st.metric("Normal", n_normal)
    with m4:
        # Precision context from backtest
        st.metric(
            "Alert precision",
            "79%",
            help=(
                "From Phase 2.4 backtest: at threshold N>=5 signals in 14-day window, "
                "79% of alerts preceded actual downgrades within 90 days. F1=0.760."
            ),
        )

    # ----------------------------------------------------------
    # Alert table: entities above warning threshold
    # ----------------------------------------------------------
    alerted = day_data[day_data["alert_level"] != "Normal"].sort_values(
        rolling_col, ascending=False
    )

    if alerted.empty:
        st.success(
            f"No alerts on {selected_date.strftime('%b %d, %Y')}. "
            "All entities below warning threshold."
        )
    else:
        st.subheader(f"Active Alerts — {selected_date.strftime('%b %d, %Y')}")

        alert_rows = []
        for _, row in alerted.iterrows():
            entity = row["entity"]
            score = row[rolling_col]
            level = row["alert_level"]

            # Check if this entity had any downgrades
            entity_ratings = rating_actions[rating_actions["entity"] == entity]
            n_downgrades = len(entity_ratings[entity_ratings["outcome"] == "negative"])

            alert_rows.append({
                "Alert": level,
                "Entity": entity,
                "Subsector": SUBSECTOR_LABELS.get(row["subsector"], row["subsector"]),
                "Risk Score": f"{score:.2f}",
                "Direct": f"{row['direct_score']:.2f}",
                "Contagion": f"{row['contagion_score']:.2f}",
                "Historical Downgrades": n_downgrades,
            })

        alert_df = pd.DataFrame(alert_rows)
        st.dataframe(alert_df, use_container_width=True, hide_index=True)

    # ----------------------------------------------------------
    # Subsector-level alert heatmap
    # ----------------------------------------------------------
    st.subheader("Subsector Risk Overview")

    subsector_stats = []
    for subsector in sorted(entities_by_subsector.keys()):
        sub_entities = entities_by_subsector[subsector]
        sub_data = day_data[day_data["entity"].isin(sub_entities)]

        if sub_data.empty:
            continue

        sub_label = SUBSECTOR_LABELS.get(subsector, subsector)
        avg_score = sub_data[rolling_col].mean()
        max_score = sub_data[rolling_col].max()
        n_alert = len(sub_data[sub_data["alert_level"] != "Normal"])
        worst = sub_data.loc[sub_data[rolling_col].idxmax(), "entity"]

        subsector_stats.append({
            "Subsector": sub_label,
            "Entities": len(sub_data),
            "Avg Score": avg_score,
            "Max Score": max_score,
            "Alerts": n_alert,
            "Worst Entity": worst,
        })

    if subsector_stats:
        stats_df = pd.DataFrame(subsector_stats).sort_values("Avg Score", ascending=False)

        # Visual bar chart
        fig = go.Figure()
        for _, row in stats_df.iterrows():
            color = score_to_color(row["Avg Score"], max_score=10.0)
            fig.add_trace(
                go.Bar(
                    x=[row["Avg Score"]],
                    y=[row["Subsector"]],
                    orientation="h",
                    marker=dict(color=color),
                    text=f"{row['Avg Score']:.2f}",
                    textposition="auto",
                    hovertemplate=(
                        f"<b>{row['Subsector']}</b><br>"
                        f"Avg score: {row['Avg Score']:.2f}<br>"
                        f"Max: {row['Max Score']:.2f} ({row['Worst Entity']})<br>"
                        f"Alerts: {row['Alerts']}/{row['Entities']}"
                        f"<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

        # Add threshold lines
        fig.add_vline(
            x=WARNING_THRESHOLD, line=dict(color="#F4A261", width=2, dash="dash"),
            annotation_text="Warning", annotation_position="top",
        )
        fig.add_vline(
            x=CRITICAL_THRESHOLD, line=dict(color="#E63946", width=2, dash="dash"),
            annotation_text="Critical", annotation_position="top",
        )

        fig.update_layout(
            title="Average risk score by subsector",
            height=max(250, len(subsector_stats) * 50),
            margin=dict(l=160, r=30, t=50, b=30),
            xaxis_title="Rolling risk score",
            yaxis=dict(title="", autorange="reversed"),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Table version
        display_stats = stats_df.copy()
        display_stats["Avg Score"] = display_stats["Avg Score"].round(2)
        display_stats["Max Score"] = display_stats["Max Score"].round(2)
        st.dataframe(display_stats, use_container_width=True, hide_index=True)

    # ----------------------------------------------------------
    # Backtest context (collapsible)
    # ----------------------------------------------------------
    with st.expander("Backtest precision context", expanded=False):
        st.markdown(
            """
**How reliable are these alerts?**

From our Phase 2.4 backtest against actual rating actions:

| Metric | Value |
|--------|-------|
| Alert threshold | N >= 5 signals in 14-day window |
| Precision | **79%** (of alerts that fired, 79% preceded a downgrade within 90 days) |
| Recall | **73%** (of actual downgrades, 73% had a prior alert) |
| F1 Score | **0.760** |

**Entity-level backtest:**

| Entity | Coverage | Mean Lead Time | False Positive Rate |
|--------|----------|----------------|---------------------|
| DHFL | 23/23 actions (100%) | 160 days | — |
| Reliance Capital | 15/15 actions (100%) | 156 days | — |
| Cholamandalam | 0 downgrades | — | 13% |

**Threshold calibration (Phase 3 v2):**
- Warning (4.0): Housing finance targets breach >80% of crisis days
- Critical (10.0): Only during peak crisis periods
- Cross-sector controls (Chola, Bajaj): breach <10% of days
"""
        )

    # ----------------------------------------------------------
    # Crisis replay summaries
    # ----------------------------------------------------------
    with st.expander("Crisis replay results", expanded=False):
        for crisis in crisis_results:
            st.markdown(f"### {crisis['crisis_name']}")
            st.caption(f"{crisis['start_date']} to {crisis['end_date']}")

            if "target_results" in crisis:
                targets = crisis["target_results"]
                target_rows = []
                for t in targets:
                    is_control = t.get("is_control", False)
                    target_rows.append({
                        "Entity": t["entity"],
                        "Type": "Control" if is_control else "Target",
                        "Lead Time": f"{t.get('lead_time_days', '—')}d" if not is_control else "—",
                        "Improvement": f"+{t.get('lead_time_improvement', 0)}d",
                        "Peak Score": f"{t.get('peak_score', 0):.1f}",
                        "Peak Contagion": f"{t.get('peak_contagion', 0):.1f}",
                    })

                st.dataframe(
                    pd.DataFrame(target_rows),
                    use_container_width=True,
                    hide_index=True,
                )
