# WHY THIS: Simplified contagion network — a "we built this" showcase, not a
# working analytical tool. The real contagion computation runs at scale in the
# background (propagation.py). This view just demonstrates the concept:
# "when one NBFC shows stress, the system propagates warnings to peers."
#
# Simplified means:
# - Only show the top ~10 most active entities (not all 44)
# - Only intra-subsector edges (the strong ones, weight=0.8)
# - Clean radial layout with readable labels
# - One demo date preset (peak crisis) so the boss doesn't have to hunt

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.utils.data_loader import (
    get_entities_by_subsector,
    load_contagion_edges,
    load_entity_metadata,
    load_entity_scores,
)
from src.dashboard.utils.styling import (
    SUBSECTOR_COLORS,
    SUBSECTOR_LABELS,
    score_to_color,
)

# Entities that tell the contagion story — curated for the demo
DEMO_ENTITIES = {
    "IL&FS/DHFL Crisis (2019)": {
        "date": "2019-06-11",
        "entities": [
            # Sources
            "DHFL", "IL&FS",
            # Housing finance targets (contagion recipients)
            "PNB Housing Finance", "Indiabulls Housing Finance",
            "Can Fin Homes", "Piramal Enterprises",
            # Cross-sector controls (should have weak edges)
            "Cholamandalam Investment", "Bajaj Finance",
            # Other stressed
            "Reliance Home Finance", "YES Bank",
        ],
        "story": (
            "DHFL and IL&FS are the crisis epicenters (large red nodes). "
            "Thick red edges flow to housing finance peers — PNB Housing, "
            "Indiabulls, Can Fin Homes. Thin gray edges to diversified entities "
            "(Bajaj, Chola) show the system correctly weights intra-subsector "
            "contagion 8x higher than cross-subsector."
        ),
    },
    "SREI/RelCap Crisis (2020)": {
        "date": "2020-06-01",
        "entities": [
            "Reliance Capital", "SREI Infrastructure Finance",
            "SREI Equipment Finance", "IL&FS Financial Services",
            "L&T Finance", "Power Finance Corporation",
            "Bajaj Finance", "Cholamandalam Investment",
            "DHFL", "YES Bank",
        ],
        "story": (
            "Reliance Capital and SREI are the stress sources. "
            "Infrastructure finance peers (L&T Finance, PFC, IREDA) receive "
            "contagion through intra-subsector edges. DHFL appears as a "
            "residual node from the earlier housing finance crisis."
        ),
    },
    "Current state": {
        "date": "latest",
        "entities": None,  # top 10 by score
        "story": "Showing the 10 highest-scoring entities on the latest date.",
    },
}


def _radial_positions(
    entities: list[str],
    subsector_map: dict[str, str],
) -> dict[str, tuple[float, float]]:
    """Place nodes in a clean radial layout, grouped by subsector.

    With only 8-12 nodes, this gives a spacious, readable layout.
    Subsectors get their own arc segment. Entities within a subsector
    are spread evenly along that arc.
    """
    # Group entities by subsector
    groups: dict[str, list[str]] = {}
    for e in entities:
        sub = subsector_map.get(e, "unknown")
        groups.setdefault(sub, []).append(e)

    positions: dict[str, tuple[float, float]] = {}
    n_groups = len(groups)
    arc_per_group = 2 * math.pi / max(n_groups, 1)

    for i, (subsector, members) in enumerate(sorted(groups.items())):
        arc_start = i * arc_per_group
        n = len(members)

        for j, entity in enumerate(members):
            if n == 1:
                angle = arc_start + arc_per_group / 2
            else:
                # Spread within the arc, with padding
                padding = arc_per_group * 0.15
                angle = arc_start + padding + (arc_per_group - 2 * padding) * j / (n - 1)

            radius = 2.5
            positions[entity] = (math.cos(angle) * radius, math.sin(angle) * radius)

    return positions


def render() -> None:
    """Render the simplified Contagion Network view."""
    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    edges_df = load_contagion_edges()
    metadata = load_entity_metadata()
    entity_scores = load_entity_scores()
    entities_by_subsector = get_entities_by_subsector(metadata)
    subsector_map = {e["name"]: e["subsector"] for e in metadata["entities"]}

    # ----------------------------------------------------------
    # Preset selector
    # ----------------------------------------------------------
    preset_name = st.radio(
        "Scenario",
        options=list(DEMO_ENTITIES.keys()),
        index=0,
        horizontal=True,
        help="Pre-configured views showing contagion during key crisis periods.",
    )

    preset = DEMO_ENTITIES[preset_name]

    # Resolve date
    all_dates = sorted(entity_scores["date"].dt.date.unique())
    if preset["date"] == "latest":
        selected_date = all_dates[-1]
    else:
        target = pd.Timestamp(preset["date"]).date()
        # Find closest available date
        selected_date = min(all_dates, key=lambda d: abs((d - target).days))

    # Get scores for this date
    day_scores = entity_scores[
        entity_scores["date"].dt.date == selected_date
    ].set_index("entity")

    # Resolve entities
    if preset["entities"] is None:
        # Top 10 by score
        if day_scores.empty:
            st.warning(f"No data for {selected_date}.")
            return
        top = day_scores.nlargest(10, "rolling_30d")
        show_entities = list(top.index)
    else:
        show_entities = [e for e in preset["entities"] if e in day_scores.index]

    if not show_entities:
        st.warning("No entity data available for this scenario.")
        return

    # Show the narrative
    st.info(f"**{preset_name}** — {selected_date.strftime('%b %d, %Y')}\n\n{preset['story']}")

    # ----------------------------------------------------------
    # Filter edges to only connections between visible entities
    # ----------------------------------------------------------
    entity_set = set(show_entities)
    visible_edges = edges_df[
        (edges_df["source"].isin(entity_set))
        & (edges_df["target"].isin(entity_set))
    ].copy()

    # ----------------------------------------------------------
    # Layout
    # ----------------------------------------------------------
    positions = _radial_positions(show_entities, subsector_map)

    # ----------------------------------------------------------
    # Build the graph
    # ----------------------------------------------------------
    fig = go.Figure()

    # --- Edges ---
    for _, edge in visible_edges.iterrows():
        src, tgt = edge["source"], edge["target"]
        if src not in positions or tgt not in positions:
            continue

        x0, y0 = positions[src]
        x1, y1 = positions[tgt]
        weight = edge["weight"]
        is_intra = edge["same_subsector"]

        # Source score drives edge visibility
        src_score = day_scores.loc[src, "rolling_30d"] if src in day_scores.index else 0
        activity = min(max(src_score, 0), 10) / 10

        # Intra-subsector: red when active, thicker. Cross: gray, thinner.
        if is_intra:
            opacity = 0.15 + activity * 0.6
            width = 1.5 + activity * 4
            color = f"rgba(230, 57, 70, {opacity})"
        else:
            opacity = 0.08 + activity * 0.2
            width = 0.5 + activity * 1.5
            color = f"rgba(150, 150, 150, {opacity})"

        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # --- Nodes ---
    # One trace per subsector for clean legend
    seen_subsectors: set[str] = set()
    for entity in show_entities:
        if entity not in positions:
            continue

        subsector = subsector_map.get(entity, "unknown")
        sub_color = SUBSECTOR_COLORS.get(subsector, "#999999")
        sub_label = SUBSECTOR_LABELS.get(subsector, subsector)
        show_in_legend = subsector not in seen_subsectors
        seen_subsectors.add(subsector)

        x, y = positions[entity]
        score = day_scores.loc[entity, "rolling_30d"] if entity in day_scores.index else 0
        direct = day_scores.loc[entity, "direct_score"] if entity in day_scores.index else 0
        contagion = day_scores.loc[entity, "contagion_score"] if entity in day_scores.index else 0

        # Size: bigger = more stressed. Range 25-70 for readability.
        size = max(25, min(70, 25 + score * 4))

        # Color: subsector color but darken/redden with score
        node_color = score_to_color(score, max_score=12.0) if score > 2.0 else sub_color

        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(
                    size=size,
                    color=node_color,
                    line=dict(width=2, color="white"),
                    opacity=0.9,
                ),
                text=[entity],
                textposition="top center",
                textfont=dict(size=11, color="#333"),
                hovertemplate=(
                    f"<b>{entity}</b><br>"
                    f"Subsector: {sub_label}<br>"
                    f"Risk score: {score:.2f}<br>"
                    f"Direct: {direct:.2f}<br>"
                    f"Contagion: {contagion:.2f}"
                    f"<extra></extra>"
                ),
                name=sub_label,
                legendgroup=subsector,
                showlegend=show_in_legend,
            )
        )

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        title=dict(
            text="Contagion Network",
            subtitle=dict(
                text=(
                    "Larger node = higher risk score. "
                    "Red edges = intra-subsector contagion (8x weight). "
                    "Gray edges = cross-subsector (weak)."
                ),
            ),
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        height=600,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        hovermode="closest",
        margin=dict(l=20, r=20, t=70, b=50),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------
    # Edge weight explanation
    # ----------------------------------------------------------
    with st.expander("How contagion works", expanded=False):
        st.markdown(
            """
**Edge weights determine how much stress flows between entities:**

| Connection | Weight | Example |
|-----------|--------|---------|
| Same subsector | **0.8** | DHFL → PNB Housing (both housing finance) |
| Different subsector | **0.1** | DHFL → Bajaj Finance (housing → diversified) |

**This means:** when DHFL's risk score is 10, PNB Housing receives
10 × 0.8 = **8.0 contagion** (before normalization by peer count),
while Bajaj receives only 10 × 0.1 = **1.0 contagion**.

**Real impact from Phase 3 backtest:**
- Can Fin Homes: **+220 days** early warning (zero direct signals — contagion only)
- Piramal: **+210 days** (contagion only)
- PNB Housing: **+483 days** (contagion amplified the weak direct signal)
"""
        )
