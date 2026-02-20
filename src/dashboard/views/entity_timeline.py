# WHY THIS: Entity Timeline is the "money shot" for the global head demo.
# It answers ONE question: "Did the system alert BEFORE the downgrade?"
#
# VISUAL DESIGN (v3 — rolling score + threshold crossings):
# Y-axis = rolling average credit risk score (7d / 30d / 90d selectable).
# The score is a weighted sum of deterioration articles (+1.0 each) and
# improvement articles (-0.5 each), so it's mostly positive during crises.
#
# The boss sees:
#   1. Stacked area: dark = direct signals, light = contagion from peers
#   2. Horizontal lines at warning (4.0) and critical (10.0) thresholds
#   3. When the score crosses a threshold = "alert issued"
#   4. Red vertical line = actual downgrade by rating agency
#   5. Yellow band between alert and downgrade = lead time
#
# Why rolling score instead of cumulative article count?
# - Cumulative only goes up → you can't tell if risk subsided
# - Rolling score rises AND falls → shows risk intensity right now
# - Threshold crossings give a concrete "alert issued on DATE" moment
# - Stacked area shows direct vs contagion contribution at a glance
# - For Can Fin Homes: direct area is flat, contagion pushes past threshold

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.utils.data_loader import (
    get_entities_by_subsector,
    get_entity_list,
    load_entity_metadata,
    load_entity_scores,
    load_rating_actions,
    load_signals,
)
from src.dashboard.utils.styling import (
    PLOTLY_LAYOUT,
    SUBSECTOR_LABELS,
)

# Alert thresholds — must match configs/contagion_config.yaml
# 🎓 These come from the Phase 3 threshold sweep. warning=4.0 means:
# "the rolling average score is equivalent to ~4 high-confidence deterioration
# articles per day, sustained over the window." In practice, this fires when
# there's a sustained burst of bad news, not a one-off article.
WARNING_THRESHOLD = 4.0
CRITICAL_THRESHOLD = 10.0

# ============================================================
# Chart Color Palette — each element gets its own visual lane
# ============================================================
# 🎓 COLOR DESIGN PRINCIPLE: Red is reserved for ONE thing — actual downgrades.
# Everything else uses blue (model output), amber (thresholds), gold (lead time).
# This prevents the "wall of red" problem where score area, thresholds, and
# downgrades all use the same color and blur together.
#
# The boss's eye path: blue area rises → crosses amber line → red line appears → gold gap = value

SCORE_COLOR = "#4A6FA5"           # Indigo blue — direct evidence from news about THIS entity
CONTAGION_COLOR = "#8B72BE"       # Soft purple — risk spreading through peer network
WARNING_COLOR = "#D4A017"         # Amber/gold — "caution" threshold
CRITICAL_COLOR = "#D35400"        # Deep orange — "serious" threshold
DOWNGRADE_COLOR = "#C1292E"       # Crimson — actual downgrade (THE ONLY red on the chart)
UPGRADE_COLOR = "#2A9D8F"         # Teal — actual upgrade
LEAD_TIME_BAND = "rgba(233, 196, 106, 0.15)"   # Pale gold band
LEAD_TIME_TEXT = "#b8860b"        # Dark gold for annotations

DEMO_PRESETS: list[dict[str, str]] = [
    {"name": "DHFL", "why": "Score crosses warning ~160d before first downgrade"},
    {"name": "Can Fin Homes", "why": "Contagion pushes score past threshold — near-zero direct signal"},
    {"name": "Cholamandalam", "why": "Stable NBFC — score rarely breaches warning, no downgrades"},
]

SIGNAL_TYPE_LABELS: dict[str, str] = {
    "asset_quality": "Asset quality concern",
    "liquidity": "Liquidity stress",
    "funding": "Funding pressure",
    "contagion": "Contagion from peer",
    "governance": "Governance issue",
    "regulatory": "Regulatory action",
    "operational": "Operational risk",
    "other": "Other credit signal",
}


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba() for Plotly fill colors."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ============================================================
# Data Computation
# ============================================================


def _compute_entity_rolling(
    entity_df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Compute rolling averages for direct, contagion, and total scores separately.

    🎓 WHY SEPARATE ROLLING: The pre-computed entity_scores.parquet has rolling
    averages only on total_score. But for the stacked area chart, we need rolling
    direct and rolling contagion independently. Since rolling average is LINEAR
    (rolling(a+b) = rolling(a) + rolling(b)), the two stacked areas will sum
    to the total rolling line.

    For a single entity (~2,000 rows), this computation is instant.
    """
    if entity_df.empty:
        return entity_df

    df = entity_df.set_index("date").sort_index()

    # Reindex to daily frequency so gaps get filled with 0 (no signal = score of 0)
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df["direct_score"] = df["direct_score"].fillna(0.0)
    df["contagion_score"] = df["contagion_score"].fillna(0.0)

    df["rolling_direct"] = (
        df["direct_score"].rolling(window=window, min_periods=1).mean()
    )
    df["rolling_contagion"] = (
        df["contagion_score"].rolling(window=window, min_periods=1).mean()
    )
    df["rolling_total"] = df["rolling_direct"] + df["rolling_contagion"]

    df = df.reset_index().rename(columns={"index": "date"})
    return df


@dataclass
class AlertResult:
    """Result of alert detection — when did the score cross a threshold?"""

    # Warning alert
    warning_date: datetime | None = None
    # Critical alert
    critical_date: datetime | None = None
    # First negative rating action (ground truth)
    first_neg_date: datetime | None = None
    first_neg_agency: str | None = None
    first_neg_action_type: str | None = None
    # Lead time = gap between alert and first downgrade
    warning_lead_days: int | None = None
    critical_lead_days: int | None = None
    # Contagion contribution
    contagion_pct: float | None = None
    # Peak score reached
    peak_score: float | None = None


def _detect_alerts(
    rolling_df: pd.DataFrame,
    entity_ratings: pd.DataFrame,
) -> AlertResult:
    """Find first threshold crossing and compute lead time to first downgrade.

    🎓 WHY THRESHOLD-BASED: Instead of "first article" (which is noisy — one
    random article doesn't mean much), we use the SAME alert thresholds the
    system would actually fire on. This is what would happen in production:
    score crosses 4.0 → warning email goes out → analyst investigates.
    """
    result = AlertResult()

    if rolling_df.empty:
        return result

    # Peak score
    result.peak_score = rolling_df["rolling_total"].max()

    # Contagion contribution (over entire timeline)
    total_direct = rolling_df["direct_score"].sum()
    total_contagion = rolling_df["contagion_score"].sum()
    total_combined = total_direct + total_contagion
    result.contagion_pct = (
        (total_contagion / total_combined) * 100 if total_combined > 0 else 0
    )

    # First negative rating action
    neg_actions = entity_ratings[entity_ratings["outcome"] == "negative"]
    if not neg_actions.empty:
        first_neg = neg_actions.sort_values("date").iloc[0]
        result.first_neg_date = first_neg["date"]
        result.first_neg_agency = first_neg.get("agency", "")
        result.first_neg_action_type = first_neg.get("action_type", "")

    # Find first warning crossing
    warning_rows = rolling_df[rolling_df["rolling_total"] >= WARNING_THRESHOLD]
    if not warning_rows.empty:
        result.warning_date = warning_rows.iloc[0]["date"]
        if result.first_neg_date is not None and result.warning_date < result.first_neg_date:
            result.warning_lead_days = (result.first_neg_date - result.warning_date).days

    # Find first critical crossing
    critical_rows = rolling_df[rolling_df["rolling_total"] >= CRITICAL_THRESHOLD]
    if not critical_rows.empty:
        result.critical_date = critical_rows.iloc[0]["date"]
        if result.first_neg_date is not None and result.critical_date < result.first_neg_date:
            result.critical_lead_days = (result.first_neg_date - result.critical_date).days

    return result


# ============================================================
# Chart Building
# ============================================================


def _build_chart(
    rolling_df: pd.DataFrame,
    ratings_display: pd.DataFrame,
    alert: AlertResult,
    entity_name: str,
    subsector_label: str,
    window: int,
) -> go.Figure:
    """Build the main timeline chart: stacked area + thresholds + rating actions."""
    fig = go.Figure()

    # --- 1. STACKED AREA: direct (bottom, solid) + contagion (top, lighter) ---
    # 🎓 Stacked area lets the boss see at a glance: "how much of this risk
    # came from direct news vs peer contagion?" For DHFL the dark area dominates.
    # For Can Fin Homes, the light area is almost everything.
    #
    # 🎓 TRACE ORDER MATTERS for Plotly's fill="tonexty": it fills between
    # the current trace and the PREVIOUS trace. So we draw:
    #   1. Direct first (fill to zero = dark area)
    #   2. Total second (fill to previous trace = light area between direct and total)
    # The light area = contagion contribution.

    # Direct layer (bottom — darker blue fill from 0 to direct score)
    fig.add_trace(
        go.Scatter(
            x=rolling_df["date"],
            y=rolling_df["rolling_direct"],
            name="Direct signals",
            mode="lines",
            line=dict(color=SCORE_COLOR, width=2),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(SCORE_COLOR, 0.35),
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Direct: <b>%{y:.2f}</b>"
                "<extra></extra>"
            ),
        )
    )

    # Contagion layer (top — lighter blue fill from direct to total)
    fig.add_trace(
        go.Scatter(
            x=rolling_df["date"],
            y=rolling_df["rolling_total"],
            name="+ Contagion from peers",
            mode="lines",
            line=dict(color=CONTAGION_COLOR, width=1),
            fill="tonexty",
            fillcolor=_hex_to_rgba(CONTAGION_COLOR, 0.2),
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Total (direct + contagion): <b>%{y:.2f}</b>"
                "<extra></extra>"
            ),
        )
    )

    # --- 2. THRESHOLD LINES ---
    # 🎓 Amber for warning, deep orange for critical — NOT red.
    # Red is reserved exclusively for actual downgrades (ground truth).
    fig.add_hline(
        y=WARNING_THRESHOLD,
        line=dict(color=WARNING_COLOR, width=1.5, dash="dash"),
        annotation_text="Warning",
        annotation_position="top left",
        annotation_font=dict(size=10, color=WARNING_COLOR),
    )
    fig.add_hline(
        y=CRITICAL_THRESHOLD,
        line=dict(color=CRITICAL_COLOR, width=1.5, dash="dash"),
        annotation_text="Critical",
        annotation_position="top left",
        annotation_font=dict(size=10, color=CRITICAL_COLOR),
    )

    # --- 3. RATING ACTION VERTICAL LINES ---
    y_max = max(rolling_df["rolling_total"].max(), CRITICAL_THRESHOLD * 1.2)
    _add_rating_actions(fig, ratings_display, y_max)

    # --- 4. ALERT-TO-DOWNGRADE BAND (the money visual) ---
    _add_alert_band(fig, alert)
    layout_kwargs = {**PLOTLY_LAYOUT}
    layout_kwargs["legend"] = dict(
        orientation="h",
        yanchor="top",
        y=-0.08,
        xanchor="center",
        x=0.5,
        font=dict(size=11),
    )
    fig.update_layout(
        **layout_kwargs,
        title=dict(
            text=f"{entity_name}",
            subtitle=dict(
                text=f"{subsector_label} — {window}-day rolling risk score vs actual rating actions",
            ),
        ),
        xaxis_title="",
        yaxis_title=f"Risk score ({window}d rolling avg)",
        yaxis_range=[0, y_max],
        height=550,
    )

    return fig


def _add_rating_actions(
    fig: go.Figure,
    entity_ratings: pd.DataFrame,
    y_max: float,
) -> None:
    """Add rating action vertical lines — downgrades (crimson) and upgrades (teal)."""
    marker_y = max(y_max * 0.92, WARNING_THRESHOLD)

    for outcome, color, label_prefix in [
        ("negative", DOWNGRADE_COLOR, "Downgrade"),
        ("positive", UPGRADE_COLOR, "Upgrade"),
    ]:
        actions = entity_ratings[entity_ratings["outcome"] == outcome]
        if actions.empty:
            continue

        for _, action in actions.iterrows():
            fig.add_vline(
                x=action["date"],
                line=dict(color=color, width=2, dash="solid"),
                opacity=0.7,
            )

        hover_text = [
            f"<b>{label_prefix.upper()}</b><br>"
            f"{row['date'].strftime('%b %d, %Y')}<br>"
            f"{row['action_type'].replace('_', ' ').title()} by {row['agency']}<br>"
            f"{row.get('from_rating', '')} → {row.get('to_rating', '')}"
            for _, row in actions.iterrows()
        ]
        fig.add_trace(
            go.Scatter(
                x=actions["date"],
                y=[marker_y] * len(actions),
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color=color,
                    line=dict(width=1, color="white"),
                ),
                name=f"Actual {label_prefix.lower()}",
                hovertemplate="%{text}<extra></extra>",
                text=hover_text,
            )
        )


def _add_alert_band(fig: go.Figure, alert: AlertResult) -> None:
    """Add the yellow band between alert date and first downgrade.

    🎓 This is THE visual for the demo. The yellow band's width = lead time.
    "See this gap? The system issued a warning X days before the agency acted."
    """
    if alert.first_neg_date is None:
        return

    # Use warning alert for the band (more conservative, longer lead time)
    alert_date = alert.warning_date
    lead_days = alert.warning_lead_days

    if alert_date is None or lead_days is None or lead_days <= 0:
        return

    # Gold band from alert to downgrade
    fig.add_shape(
        type="rect",
        x0=alert_date.isoformat() if hasattr(alert_date, "isoformat") else str(alert_date),
        x1=alert.first_neg_date.isoformat() if hasattr(alert.first_neg_date, "isoformat") else str(alert.first_neg_date),
        y0=0,
        y1=1,
        yref="paper",
        fillcolor=LEAD_TIME_BAND,
        line_width=0,
        layer="below",
    )

    # Label in the middle of the band
    midpoint = alert_date + (alert.first_neg_date - alert_date) / 2
    fig.add_annotation(
        x=midpoint.isoformat() if hasattr(midpoint, "isoformat") else str(midpoint),
        y=1.0,
        yref="paper",
        text=f"<b>{lead_days} days early warning</b>",
        showarrow=False,
        font=dict(size=13, color=LEAD_TIME_TEXT),
        bgcolor="rgba(255,255,255,0.85)",
        borderpad=4,
        yshift=10,
    )

    # "Warning issued" annotation — matches the amber threshold line
    fig.add_annotation(
        x=alert_date.isoformat() if hasattr(alert_date, "isoformat") else str(alert_date),
        y=WARNING_THRESHOLD,
        text=f"Warning issued<br>{alert_date.strftime('%b %d, %Y') if hasattr(alert_date, 'strftime') else str(alert_date)}",
        showarrow=True,
        arrowhead=2,
        arrowwidth=1.5,
        arrowcolor=WARNING_COLOR,
        ax=-70,
        ay=-40,
        font=dict(size=10, color=WARNING_COLOR),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=WARNING_COLOR,
        borderwidth=1,
        borderpad=4,
    )

    # "First downgrade" annotation — matches the crimson downgrade lines
    agency_text = f" by {alert.first_neg_agency}" if alert.first_neg_agency else ""
    fig.add_annotation(
        x=alert.first_neg_date.isoformat() if hasattr(alert.first_neg_date, "isoformat") else str(alert.first_neg_date),
        y=WARNING_THRESHOLD,
        text=f"First downgrade{agency_text}<br>{alert.first_neg_date.strftime('%b %d, %Y') if hasattr(alert.first_neg_date, 'strftime') else str(alert.first_neg_date)}",
        showarrow=True,
        arrowhead=2,
        arrowwidth=1.5,
        arrowcolor=DOWNGRADE_COLOR,
        ax=70,
        ay=-40,
        font=dict(size=10, color=DOWNGRADE_COLOR),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=DOWNGRADE_COLOR,
        borderwidth=1,
        borderpad=4,
    )


# ============================================================
# Main Render
# ============================================================


def render() -> None:
    """Render the Entity Timeline view."""
    # Load data
    entity_scores = load_entity_scores()
    rating_actions = load_rating_actions()
    signals = load_signals()
    metadata = load_entity_metadata()
    all_entities = get_entity_list(metadata)

    subsector_map = {e["name"]: e["subsector"] for e in metadata["entities"]}
    entities_by_subsector = get_entities_by_subsector(metadata)

    neg_counts = (
        rating_actions[rating_actions["outcome"] == "negative"]
        .groupby("entity")
        .size()
        .to_dict()
    )

    # ----------------------------------------------------------
    # Entity selection: subsector filter → entity dropdown
    # ----------------------------------------------------------
    if "timeline_entity" not in st.session_state:
        st.session_state.timeline_entity = "DHFL"

    filter_col, entity_col = st.columns([1, 2])

    with filter_col:
        subsector_options = ["All subsectors"] + [
            SUBSECTOR_LABELS.get(s, s) for s in sorted(entities_by_subsector.keys())
        ]
        selected_subsector_label = st.selectbox(
            "Subsector filter",
            options=subsector_options,
            index=0,
        )

    label_to_key = {v: k for k, v in SUBSECTOR_LABELS.items()}
    if selected_subsector_label == "All subsectors":
        filtered_entities = all_entities
    else:
        subsector_key = label_to_key.get(
            selected_subsector_label, selected_subsector_label
        )
        filtered_entities = sorted(entities_by_subsector.get(subsector_key, []))

    with entity_col:
        def _entity_label(name: str) -> str:
            n_neg = neg_counts.get(name, 0)
            sub = SUBSECTOR_LABELS.get(subsector_map.get(name, ""), "")
            parts = [name]
            if sub and selected_subsector_label == "All subsectors":
                parts.append(sub)
            if n_neg > 0:
                parts.append(f"{n_neg} downgrades")
            return " — ".join(parts) if len(parts) > 1 else name

        default_entity = st.session_state.timeline_entity
        if default_entity not in filtered_entities and filtered_entities:
            default_entity = filtered_entities[0]

        selected_entity = st.selectbox(
            "Entity",
            options=filtered_entities,
            index=(
                filtered_entities.index(default_entity)
                if default_entity in filtered_entities
                else 0
            ),
            format_func=_entity_label,
            key="timeline_entity_select",
        )
        st.session_state.timeline_entity = selected_entity

    # Demo presets
    with st.expander("Demo presets", expanded=False):
        demo_cols = st.columns(len(DEMO_PRESETS))
        for i, demo in enumerate(DEMO_PRESETS):
            with demo_cols[i]:
                if st.button(
                    demo["name"],
                    help=demo["why"],
                    use_container_width=True,
                ):
                    st.session_state.timeline_entity = demo["name"]
                    st.rerun()

    # ----------------------------------------------------------
    # Controls: rolling window
    # ----------------------------------------------------------
    window = st.radio(
        "Rolling window",
        options=[7, 30, 90],
        index=1,  # Default 30d — smooths daily noise without losing crisis shape
        horizontal=True,
        format_func=lambda w: f"{w} days",
        help=(
            "**7 days:** Responsive — shows short bursts of bad news. Noisy.\n\n"
            "**30 days:** Balanced — smooths daily noise, shows sustained trends.\n\n"
            "**90 days:** Stable — only fires on prolonged crises. Slow to react."
        ),
    )

    # ----------------------------------------------------------
    # Get entity data and compute rolling scores
    # ----------------------------------------------------------
    entity_df = (
        entity_scores[entity_scores["entity"] == selected_entity]
        .sort_values("date")
        .copy()
    )
    entity_ratings = rating_actions[
        rating_actions["entity"] == selected_entity
    ].sort_values("date")

    if entity_df.empty:
        st.warning(f"No data found for **{selected_entity}**.")
        return

    rolling_df = _compute_entity_rolling(entity_df, window)

    # Detect alerts (from full timeline — not affected by date zoom)
    alert = _detect_alerts(rolling_df, entity_ratings)

    # ----------------------------------------------------------
    # Date range slider
    # ----------------------------------------------------------
    min_date = rolling_df["date"].min().date()
    max_date = rolling_df["date"].max().date()
    date_range = st.slider(
        "Date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM",
    )

    # Filter for display
    rolling_display = rolling_df[
        (rolling_df["date"].dt.date >= date_range[0])
        & (rolling_df["date"].dt.date <= date_range[1])
    ]
    ratings_display = entity_ratings[
        (entity_ratings["date"].dt.date >= date_range[0])
        & (entity_ratings["date"].dt.date <= date_range[1])
    ]

    if rolling_display.empty:
        st.warning("No data in selected date range.")
        return

    # ----------------------------------------------------------
    # Metrics row
    # ----------------------------------------------------------
    subsector = subsector_map.get(selected_entity, "unknown")
    subsector_label = SUBSECTOR_LABELS.get(subsector, subsector)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Subsector", subsector_label)
    with m2:
        if alert.warning_lead_days is not None and alert.warning_lead_days > 0:
            st.metric("Early warning", f"{alert.warning_lead_days}d before downgrade")
        elif alert.first_neg_date is not None:
            neg_count = len(entity_ratings[entity_ratings["outcome"] == "negative"])
            st.metric("Downgrades", neg_count)
        else:
            st.metric("Downgrades", 0)
    with m3:
        if alert.peak_score is not None:
            level = (
                "Critical" if alert.peak_score >= CRITICAL_THRESHOLD
                else "Warning" if alert.peak_score >= WARNING_THRESHOLD
                else "Below threshold"
            )
            st.metric("Peak alert level", level)
        else:
            st.metric("Peak alert level", "No data")
    with m4:
        if alert.contagion_pct is not None and alert.contagion_pct > 10:
            st.metric("From contagion", f"{alert.contagion_pct:.0f}%")
        else:
            st.metric("Peak score", f"{alert.peak_score:.1f}" if alert.peak_score else "0")

    # ----------------------------------------------------------
    # Build and display chart
    # ----------------------------------------------------------
    fig = _build_chart(
        rolling_display,
        ratings_display,
        alert,
        selected_entity,
        subsector_label,
        window,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------
    # Context panel
    # ----------------------------------------------------------
    _render_context_panel(selected_entity, entity_ratings, signals, alert, window)


def _render_context_panel(
    entity: str,
    entity_ratings: pd.DataFrame,
    all_signals: pd.DataFrame,
    alert: AlertResult,
    window: int,
) -> None:
    """Plain-English context below the chart."""
    entity_signals = all_signals[all_signals["entity"] == entity]
    det_count = len(entity_signals[entity_signals["direction"] == "Deterioration"])
    neg_count = len(entity_ratings[entity_ratings["outcome"] == "negative"])

    # --- Main narrative ---
    if alert.first_neg_date is None:
        # No downgrades — false positive story
        if alert.warning_date is not None:
            breach_days = len(
                entity_signals[entity_signals["direction"] == "Deterioration"]
            )
            st.info(
                f"**{entity}** has never been downgraded. "
                f"The warning threshold was breached, but no downgrade followed — "
                f"this is a false alarm. The model flagged {det_count} deterioration "
                f"articles over the full period."
            )
        else:
            st.info(
                f"**{entity}** has never been downgraded and the risk score never "
                f"reached the warning threshold. Clean. "
                + (
                    f"({det_count} articles flagged as risk, but not enough to trigger an alert.)"
                    if det_count > 0
                    else ""
                )
            )
    elif alert.warning_lead_days is not None and alert.warning_lead_days > 0:
        # The good case — alert before downgrade
        st.success(
            f"**{alert.warning_lead_days} days of early warning.**\n\n"
            f"The {window}-day rolling score crossed the warning threshold "
            f"on **{alert.warning_date.strftime('%b %d, %Y')}**. "
            f"The first downgrade came on **{alert.first_neg_date.strftime('%b %d, %Y')}** "
            f"({alert.first_neg_agency}, {(alert.first_neg_action_type or '').replace('_', ' ')}).\n\n"
            f"The yellow band on the chart = {alert.warning_lead_days} days the system "
            f"knew before the rating agency acted."
            + (
                f"\n\n**{alert.contagion_pct:.0f}%** of the signal came from peer contagion."
                if alert.contagion_pct and alert.contagion_pct > 30
                else ""
            )
        )
    else:
        # Downgrade happened but no prior alert
        st.warning(
            f"**{entity}** was downgraded ({neg_count} negative actions) but the "
            f"risk score didn't cross the warning threshold before the first downgrade. "
            f"The alert system would not have caught this one early."
        )

    # Contagion note for contagion-driven entities
    if alert.contagion_pct is not None and alert.contagion_pct > 70:
        st.info(
            f"**{alert.contagion_pct:.0f}% of the risk score came from peer contagion** — "
            f"almost no direct news articles about {entity}. The lighter shaded area "
            f"in the chart is the contagion contribution from same-subsector peers."
        )

    # Rating actions table
    if not entity_ratings.empty:
        with st.expander(
            f"Rating actions detail ({len(entity_ratings)} total)", expanded=False
        ):
            display_df = entity_ratings[
                ["date", "agency", "action_type", "from_rating", "to_rating", "outcome"]
            ].copy()
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
            display_df.columns = ["Date", "Agency", "Action", "From", "To", "Outcome"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Top risk articles
    det_articles = entity_signals[entity_signals["direction"] == "Deterioration"]
    if not det_articles.empty:
        with st.expander(
            f"Risk articles flagged by model ({len(det_articles)} total)", expanded=False
        ):
            sample = det_articles.sort_values("date", ascending=False).head(20)
            display_df = sample[["date", "title", "signal_type", "confidence"]].copy()
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
            display_df["signal_type"] = display_df["signal_type"].map(
                SIGNAL_TYPE_LABELS
            ).fillna(display_df["signal_type"])
            display_df.columns = ["Date", "Headline", "Signal type", "Confidence"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)
