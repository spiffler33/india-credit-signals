# WHY THIS: Centralized color/theme constants so every chart across 5 views
# uses the same visual language. If DHFL is red on the timeline, it's red on
# the heatmap and the network graph. Consistency matters for the demo —
# the global head should never ask "why is this a different color?"
#
# Why not a Streamlit theme TOML? Streamlit themes control page background and
# widget colors, not Plotly chart colors. We need both: Streamlit config for
# page-level theming + this module for chart-level color mapping.

from __future__ import annotations

# ============================================================
# Subsector Colors — 6 distinct, colorblind-safe palette
# ============================================================
# 🎓 Using a categorical palette (not a gradient) because subsectors are
# discrete categories, not a continuous scale. Colorblind-safe means we
# avoid red-green pairs as the only distinguishing feature.

SUBSECTOR_COLORS: dict[str, str] = {
    "housing_finance": "#E63946",       # Red — the crisis epicenter
    "infrastructure_finance": "#F4A261", # Orange — secondary stress sector
    "diversified_nbfc": "#2A9D8F",       # Teal — stable control entities
    "microfinance": "#264653",           # Dark blue — small, distinct group
    "vehicle_finance": "#6A994E",        # Green — stable sector
    "special_situations": "#9B5DE5",     # Purple — unique entities (YES Bank, etc.)
    "unknown": "#999999",               # Gray fallback
}

SUBSECTOR_LABELS: dict[str, str] = {
    "housing_finance": "Housing Finance",
    "infrastructure_finance": "Infrastructure Finance",
    "diversified_nbfc": "Diversified NBFC",
    "microfinance": "Microfinance",
    "vehicle_finance": "Vehicle Finance",
    "special_situations": "Special Situations",
}


# ============================================================
# Signal Direction Colors
# ============================================================

DIRECTION_COLORS: dict[str, str] = {
    "Deterioration": "#E63946",  # Red — bad news
    "Improvement": "#2A9D8F",    # Teal — good news
    "Neutral": "#999999",        # Gray — no signal
}


# ============================================================
# Rating Action Outcome Colors
# ============================================================

OUTCOME_COLORS: dict[str, str] = {
    "negative": "#E63946",   # Downgrades, defaults → red vertical lines
    "positive": "#2A9D8F",   # Upgrades → green vertical lines
    "neutral": "#F4A261",    # Reaffirmations, etc. → orange
}


# ============================================================
# Score-to-Color Gradient (for heatmap)
# ============================================================
# 🎓 Diverging colorscale: green (safe) → yellow (watch) → red (danger).
# We use the score thresholds from contagion_config.yaml (warning=4.0,
# critical=10.0) to anchor the color transitions. This means the heatmap
# colors have the same meaning as the alert thresholds.

SCORE_COLORSCALE: list[list] = [
    [0.0, "#2A9D8F"],    # Green — zero/low score
    [0.3, "#E9C46A"],    # Yellow — moderate
    [0.6, "#F4A261"],    # Orange — elevated
    [1.0, "#E63946"],    # Red — high risk
]


def score_to_color(score: float, max_score: float = 15.0) -> str:
    """Map a numeric score to a hex color using the green→red gradient.

    Args:
        score: The entity's rolling score (higher = worse).
        max_score: Score that maps to full red. Default 15.0 covers most cases.

    Returns:
        Hex color string.
    """
    if max_score <= 0:
        return SCORE_COLORSCALE[0][1]

    ratio = min(max(score / max_score, 0.0), 1.0)

    # Linear interpolation through the colorscale
    for i in range(1, len(SCORE_COLORSCALE)):
        lower_val, lower_color = SCORE_COLORSCALE[i - 1]
        upper_val, upper_color = SCORE_COLORSCALE[i]
        if ratio <= upper_val:
            # Interpolate between lower and upper
            t = (ratio - lower_val) / (upper_val - lower_val) if upper_val > lower_val else 0
            return _interpolate_hex(lower_color, upper_color, t)

    return SCORE_COLORSCALE[-1][1]


def _interpolate_hex(color1: str, color2: str, t: float) -> str:
    """Linearly interpolate between two hex colors."""
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# ============================================================
# Plotly Layout Defaults
# ============================================================
# 🎓 Consistent chart styling across all views. White background,
# minimal gridlines, readable fonts. These get passed to
# fig.update_layout(**PLOTLY_LAYOUT) in each chart builder.

PLOTLY_LAYOUT: dict = {
    "template": "plotly_white",
    # 🎓 Explicitly set white backgrounds so charts look correct regardless
    # of Streamlit theme. "plotly_white" sets the template, but Streamlit can
    # override the paper/plot background via its own theme. These force white.
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#FFFFFF",
    "font": {"family": "Inter, system-ui, sans-serif", "size": 13, "color": "#1A1A2E"},
    "title_font_size": 18,
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": -0.2,
        "xanchor": "center",
        "x": 0.5,
    },
    "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
    "hovermode": "x unified",
}


# ============================================================
# Confidence Badge Styling
# ============================================================

CONFIDENCE_COLORS: dict[str, str] = {
    "High": "#2A9D8F",
    "Medium": "#F4A261",
    "Low": "#E63946",
}
