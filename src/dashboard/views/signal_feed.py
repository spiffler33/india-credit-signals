# WHY THIS: The Signal Feed is the "show your work" view. When the boss asks
# "What is the model actually reading?", you pull up this table. Every row is
# one news article with the model's verdict: credit-relevant? deterioration or
# improvement? what type of signal? how confident?
#
# This is what makes the system auditable — you can drill into any alert and
# see the exact articles that triggered it. For the demo: find the DHFL
# commercial paper rollover article → "the model tagged this as liquidity
# deterioration, high confidence, 3 months before the first downgrade."

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.utils.data_loader import (
    get_entities_by_subsector,
    get_entity_list,
    load_entity_metadata,
    load_signals,
)
from src.dashboard.utils.styling import (
    DIRECTION_COLORS,
    SUBSECTOR_LABELS,
)

# Plain English signal type labels
SIGNAL_TYPE_LABELS: dict[str, str] = {
    "asset_quality": "Asset Quality",
    "liquidity": "Liquidity",
    "funding": "Funding",
    "contagion": "Contagion",
    "governance": "Governance",
    "regulatory": "Regulatory",
    "operational": "Operational",
    "other": "Other",
}

# How many rows per page (avoid loading 16K rows into the DOM)
PAGE_SIZE = 50


def render() -> None:
    """Render the Signal Feed view."""
    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    signals = load_signals()
    metadata = load_entity_metadata()
    all_entities = get_entity_list(metadata)
    entities_by_subsector = get_entities_by_subsector(metadata)

    # ----------------------------------------------------------
    # Filters
    # ----------------------------------------------------------
    f1, f2, f3 = st.columns(3)

    with f1:
        # Entity filter with "All" option
        entity_options = ["All entities"] + all_entities
        selected_entity = st.selectbox("Entity", options=entity_options, index=0)

    with f2:
        direction_options = ["All directions", "Deterioration", "Improvement", "Neutral"]
        selected_direction = st.selectbox("Direction", options=direction_options, index=0)

    with f3:
        confidence_options = ["All confidence", "High", "Medium", "Low"]
        selected_confidence = st.selectbox("Confidence", options=confidence_options, index=0)

    f4, f5 = st.columns(2)

    with f4:
        signal_type_options = ["All signal types"] + list(SIGNAL_TYPE_LABELS.values())
        selected_signal_type = st.selectbox("Signal type", options=signal_type_options, index=0)

    with f5:
        credit_rel_options = ["All articles", "Credit-relevant only", "Not credit-relevant only"]
        selected_cr = st.selectbox("Credit relevance", options=credit_rel_options, index=1)

    # Date range
    min_date = signals["date"].min().date()
    max_date = signals["date"].max().date()
    date_range = st.slider(
        "Date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM",
    )

    # ----------------------------------------------------------
    # Apply filters
    # ----------------------------------------------------------
    filtered = signals.copy()

    # Date
    filtered = filtered[
        (filtered["date"].dt.date >= date_range[0])
        & (filtered["date"].dt.date <= date_range[1])
    ]

    # Entity
    if selected_entity != "All entities":
        filtered = filtered[filtered["entity"] == selected_entity]

    # Direction
    if selected_direction != "All directions":
        filtered = filtered[filtered["direction"] == selected_direction]

    # Confidence
    if selected_confidence != "All confidence":
        filtered = filtered[filtered["confidence"] == selected_confidence]

    # Signal type (reverse map from display label to data value)
    if selected_signal_type != "All signal types":
        label_to_key = {v: k for k, v in SIGNAL_TYPE_LABELS.items()}
        type_key = label_to_key.get(selected_signal_type, selected_signal_type)
        filtered = filtered[filtered["signal_type"] == type_key]

    # Credit relevance
    if selected_cr == "Credit-relevant only":
        filtered = filtered[filtered["credit_relevant"] == 1]
    elif selected_cr == "Not credit-relevant only":
        filtered = filtered[filtered["credit_relevant"] == 0]

    # Sort by date descending (most recent first)
    filtered = filtered.sort_values("date", ascending=False)

    # ----------------------------------------------------------
    # Summary metrics
    # ----------------------------------------------------------
    total = len(filtered)
    n_det = len(filtered[filtered["direction"] == "Deterioration"])
    n_imp = len(filtered[filtered["direction"] == "Improvement"])
    n_high = len(filtered[filtered["confidence"] == "High"])

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Articles matching", f"{total:,}")
    with m2:
        st.metric("Deterioration", f"{n_det:,}")
    with m3:
        st.metric("Improvement", f"{n_imp:,}")
    with m4:
        st.metric("High confidence", f"{n_high:,}")

    if filtered.empty:
        st.info("No articles match the current filters.")
        return

    # ----------------------------------------------------------
    # Paginated table
    # ----------------------------------------------------------
    # 🎓 Loading 16K rows into a Streamlit dataframe is slow and makes the
    # browser lag. Pagination shows 50 rows at a time — fast and responsive.
    n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

    page_num = st.number_input(
        f"Page (1–{n_pages})",
        min_value=1,
        max_value=n_pages,
        value=1,
        step=1,
    )

    start_idx = (page_num - 1) * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, total)
    page_data = filtered.iloc[start_idx:end_idx]

    st.caption(f"Showing {start_idx + 1}–{end_idx} of {total:,} articles")

    # Build display table
    display_df = page_data[[
        "date", "entity", "title", "direction", "signal_type", "confidence"
    ]].copy()

    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df["signal_type"] = display_df["signal_type"].map(
        SIGNAL_TYPE_LABELS
    ).fillna(display_df["signal_type"])

    display_df.columns = [
        "Date", "Entity", "Headline", "Direction", "Signal Type", "Confidence"
    ]

    # 🎓 column_config lets us style columns — colored pills for Direction,
    # narrower Date column, wider Headline column, etc.
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(35 * (len(display_df) + 1), 700),
    )

    # ----------------------------------------------------------
    # Distribution charts (collapsible)
    # ----------------------------------------------------------
    with st.expander("Signal distribution", expanded=False):
        d1, d2 = st.columns(2)

        with d1:
            # Direction breakdown
            dir_counts = filtered["direction"].value_counts()
            colors = [DIRECTION_COLORS.get(d, "#999") for d in dir_counts.index]

            import plotly.graph_objects as go
            fig_dir = go.Figure(go.Bar(
                x=dir_counts.index,
                y=dir_counts.values,
                marker_color=colors,
                text=dir_counts.values,
                textposition="auto",
            ))
            fig_dir.update_layout(
                title="By direction",
                height=300,
                margin=dict(l=40, r=20, t=40, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_dir, use_container_width=True)

        with d2:
            # Signal type breakdown
            type_counts = (
                filtered[filtered["credit_relevant"] == 1]["signal_type"]
                .map(SIGNAL_TYPE_LABELS)
                .value_counts()
            )

            fig_type = go.Figure(go.Bar(
                x=type_counts.values,
                y=type_counts.index,
                orientation="h",
                marker_color="#2A9D8F",
                text=type_counts.values,
                textposition="auto",
            ))
            fig_type.update_layout(
                title="By signal type (credit-relevant only)",
                height=300,
                margin=dict(l=100, r=20, t=40, b=40),
                showlegend=False,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_type, use_container_width=True)
