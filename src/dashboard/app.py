# WHY THIS: Single-file Streamlit app with sidebar navigation. We use manual
# page routing (radio buttons → if/elif) instead of Streamlit's built-in
# multi-page system (pages/ directory auto-discovery) because:
# - We control the page order (Entity Timeline first — the money shot)
# - We share cached data across pages without re-loading
# - The global head demo follows a specific narrative flow
#
# Run from repo root: streamlit run src/dashboard/app.py

from __future__ import annotations

import sys
from pathlib import Path

# 🎓 Streamlit runs this file directly (`streamlit run src/dashboard/app.py`),
# which means the repo root isn't on sys.path. Without this, Python can't
# resolve `from src.dashboard.utils...` because it doesn't know where `src` is.
# We add the project root (4 levels up from this file) so all `src.*` imports work.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.dashboard.utils.data_loader import (
    load_entity_scores,
    load_signals,
    load_contagion_edges,
    load_rating_actions,
    load_entity_metadata,
    load_crisis_results,
    get_entity_list,
)
from src.dashboard.views.entity_timeline import render as render_entity_timeline
from src.dashboard.views.sector_heatmap import render as render_sector_heatmap
from src.dashboard.views.signal_feed import render as render_signal_feed
from src.dashboard.views.contagion_network import render as render_contagion_network
from src.dashboard.views.alert_dashboard import render as render_alert_dashboard
from src.dashboard.utils.styling import (
    SUBSECTOR_COLORS,
    SUBSECTOR_LABELS,
)

# ============================================================
# Page Config — must be first Streamlit call
# ============================================================

st.set_page_config(
    page_title="NBFC Credit Signal Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Sidebar Navigation
# ============================================================

# 🎓 st.sidebar creates a collapsible left panel. We use st.radio for
# page selection — it's simpler than buttons and shows which page is active.

with st.sidebar:
    st.title("Credit Signal Engine")
    st.caption("India NBFC Early Warning System")
    st.divider()

    page = st.radio(
        "Navigate",
        options=[
            "Entity Timeline",
            "Sector Heatmap",
            "Contagion Network",
            "Signal Feed",
            "Alert Dashboard",
        ],
        index=0,  # Default to Entity Timeline — the demo opener
        label_visibility="collapsed",
    )

    st.divider()

    # Data status indicator
    # 🎓 Loading data in the sidebar so it's cached before any page renders.
    # @st.cache_data means this only runs once per session (or when TTL expires).
    with st.spinner("Loading data..."):
        try:
            metadata = load_entity_metadata()
            entity_scores = load_entity_scores()
            n_entities = metadata["n_entities"]
            n_rows = len(entity_scores)
            date_range = f"{entity_scores['date'].min().date()} to {entity_scores['date'].max().date()}"

            st.success(f"Data loaded")
            st.caption(
                f"{n_entities} entities | {n_rows:,} score rows\n\n"
                f"{date_range}"
            )
        except Exception as e:
            st.error(f"Data load failed: {e}")
            st.stop()

    # Subsector legend
    st.divider()
    st.caption("**Subsectors**")
    for subsector, color in SUBSECTOR_COLORS.items():
        if subsector == "unknown":
            continue
        label = SUBSECTOR_LABELS.get(subsector, subsector)
        count = metadata["subsectors"].get(subsector, 0)
        st.markdown(
            f'<span style="color:{color}">●</span> {label} ({count})',
            unsafe_allow_html=True,
        )


# ============================================================
# Page Routing
# ============================================================

if page == "Entity Timeline":
    st.header("Entity Timeline")
    render_entity_timeline()

elif page == "Sector Heatmap":
    st.header("Sector Heatmap")
    render_sector_heatmap()

elif page == "Contagion Network":
    st.header("Contagion Network")
    render_contagion_network()

elif page == "Signal Feed":
    st.header("Signal Feed")
    render_signal_feed()

elif page == "Alert Dashboard":
    st.header("Alert Dashboard")
    render_alert_dashboard()
