"""Theme state management and Dark / Light palette definitions."""

from __future__ import annotations

import streamlit as st

THEME_DARK: dict[str, str] = {
    "mode": "dark",
    "bg": "#121212",
    "secondary_bg": "#1E1E1E",
    "card_bg": "#1E1E1E",
    "plot_bg": "#18181B",
    "input_bg": "#262730",
    "hover_bg": "#27272A",
    "border": "#303036",
    "border_strong": "#3F3F46",
    "text": "#F4F4F5",
    "text_secondary": "#E4E4E7",
    "text_muted": "#A1A1AA",
    "tick_text": "#D4D4D8",
    "grid": "#2E2E32",
    "badge_bg": "#27272A",
    "badge_text": "#E4E4E7",
    "obs_color": "#F4F4F5",
    "plotly_template": "plotly_dark",
    "map_style": "carto-darkmatter",
    "test_label_color": "#FCA5A5",
    "annot_bg": "rgba(24, 24, 27, 0.90)",
    "shadow": "0 1px 2px rgba(0, 0, 0, 0.25)",
}

THEME_LIGHT: dict[str, str] = {
    "mode": "light",
    "bg": "#F8FAFC",
    "secondary_bg": "#F1F5F9",
    "card_bg": "#FFFFFF",
    "plot_bg": "#FFFFFF",
    "input_bg": "#FFFFFF",
    "hover_bg": "#E2E8F0",
    "border": "#E2E8F0",
    "border_strong": "#CBD5E1",
    "text": "#0F172A",
    "text_secondary": "#1E293B",
    "text_muted": "#475569",
    "tick_text": "#334155",
    "grid": "#E2E8F0",
    "badge_bg": "#E2E8F0",
    "badge_text": "#1E293B",
    "obs_color": "#0F172A",
    "plotly_template": "plotly_white",
    "map_style": "carto-positron",
    "test_label_color": "#991B1B",
    "annot_bg": "rgba(255, 255, 255, 0.92)",
    "shadow": "0 1px 3px rgba(15, 23, 42, 0.08)",
}


def get_theme_mode() -> str:
    """Return the active UI theme mode ('dark' or 'light')."""
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "dark"
    return st.session_state["theme_mode"]


def is_dark_mode() -> bool:
    """Return True when the active theme is dark mode."""
    return get_theme_mode() == "dark"


def toggle_theme_mode() -> None:
    """Flip the active UI theme between dark and light mode."""
    st.session_state["theme_mode"] = "light" if is_dark_mode() else "dark"


def get_theme_palette() -> dict[str, str]:
    """Return the color and template palette for the active theme."""
    return THEME_DARK if is_dark_mode() else THEME_LIGHT


def get_plotly_template() -> str:
    """Return the Plotly template name for the active theme."""
    return get_theme_palette()["plotly_template"]

