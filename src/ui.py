"""Shared UI styling, theme toggle, headers, download buttons, and footer components."""

from __future__ import annotations

import pandas as pd
import plotly.io as pio
import streamlit as st

from src.config import MODEL_ORDER, MODEL_SPECS, SEASONS, YEARS
from src.theme import get_theme_palette, is_dark_mode, toggle_theme_mode


def _light_mode_css(p: dict[str, str]) -> str:
    """Return CSS overrides to render Streamlit's UI cleanly in Light Mode."""
    return f"""
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stBottom"] {{
            background-color: {p["bg"]} !important;
            color: {p["text"]} !important;
        }}
        [data-testid="stHeader"],
        [data-testid="stToolbar"] {{
            background-color: rgba(248, 250, 252, 0.92) !important;
            color: {p["text"]} !important;
        }}
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child,
        [data-testid="stSidebarContent"],
        [data-testid="stSidebarHeader"] {{
            background-color: {p["secondary_bg"]} !important;
            color: {p["text"]} !important;
            border-color: {p["border"]} !important;
        }}
        [data-testid="stSidebar"] {{
            border-right: 1px solid {p["border"]} !important;
        }}
        [data-testid="stSidebarNav"] a,
        [data-testid="stSidebarNavLink"],
        [data-testid="stSidebarNavLink"] span,
        [data-testid="stSidebarNav"] span {{
            color: {p["text_secondary"]} !important;
        }}
        [data-testid="stSidebarNavLink"][aria-current="page"],
        [data-testid="stSidebarNavLink"]:hover {{
            background-color: {p["hover_bg"]} !important;
            color: {p["text"]} !important;
        }}
        h1, h2, h3, h4, h5, h6,
        .stApp p, .stApp li,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stRadio"] label p,
        [data-testid="stCheckbox"] label p,
        [data-testid="stSlider"] label p {{
            color: {p["text"]} !important;
        }}
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        .stCaption {{
            color: {p["text_muted"]} !important;
        }}
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p {{
            color: {p["text_muted"]} !important;
        }}
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] > div {{
            color: {p["text"]} !important;
        }}
        [data-testid="stHeader"] svg,
        [data-testid="stSidebar"] svg,
        [data-testid="stExpander"] svg,
        [data-baseweb="select"] svg {{
            color: {p["tick_text"]} !important;
            fill: currentColor !important;
        }}
        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button {{
            background-color: {p["card_bg"]} !important;
            color: {p["text"]} !important;
            border: 1px solid {p["border_strong"]} !important;
        }}
        [data-testid="stButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover {{
            background-color: {p["hover_bg"]} !important;
            border-color: {p["text_muted"]} !important;
        }}
        [data-baseweb="tab-list"] {{
            border-bottom: 1px solid {p["border"]} !important;
        }}
        [data-baseweb="tab"] {{
            color: {p["text_muted"]} !important;
        }}
        [data-baseweb="tab"][aria-selected="true"] {{
            color: {p["text"]} !important;
        }}
        [data-baseweb="tab-highlight"] {{
            background-color: #2563EB !important;
        }}
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="base-input"] {{
            background-color: {p["input_bg"]} !important;
            color: {p["text"]} !important;
            border-color: {p["border_strong"]} !important;
        }}
        [data-baseweb="select"] span,
        [data-baseweb="select"] input,
        [data-baseweb="input"] input,
        [data-baseweb="base-input"] input {{
            color: {p["text"]} !important;
            -webkit-text-fill-color: {p["text"]} !important;
        }}
        [data-baseweb="tag"] {{
            background-color: {p["hover_bg"]} !important;
            color: {p["text"]} !important;
            border: 1px solid {p["border_strong"]} !important;
        }}
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="menu"],
        [role="listbox"] {{
            background-color: {p["card_bg"]} !important;
            color: {p["text"]} !important;
            border: 1px solid {p["border_strong"]} !important;
        }}
        [role="option"] {{
            color: {p["text"]} !important;
            background-color: {p["card_bg"]} !important;
        }}
        [role="option"]:hover,
        [role="option"][aria-selected="true"] {{
            background-color: {p["hover_bg"]} !important;
        }}
        [data-testid="stExpander"] {{
            background-color: {p["card_bg"]} !important;
            border: 1px solid {p["border"]} !important;
            border-radius: 10px !important;
        }}
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p {{
            color: {p["text"]} !important;
        }}
        [data-testid="stDataFrame"] {{
            filter: invert(1) hue-rotate(180deg);
            border: 1px solid {p["border_strong"]};
            border-radius: 8px;
            overflow: hidden;
        }}
        [data-testid="stJson"] {{
            filter: invert(0.95) hue-rotate(180deg);
        }}
        [data-testid="stAlert"] {{
            color: {p["text"]} !important;
        }}
        hr {{
            border-color: {p["border"]} !important;
        }}
    """


def inject_custom_css() -> None:
    """Inject shared card, badge, footer, and active Dark/Light theme CSS."""
    p = get_theme_palette()
    pio.templates.default = p["plotly_template"]
    light_overrides = "" if is_dark_mode() else _light_mode_css(p)

    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}
        div[data-testid="stMetric"] {{
            background-color: {p["card_bg"]};
            border: 1px solid {p["border"]};
            border-radius: 10px;
            padding: 12px 16px;
            box-shadow: {p["shadow"]};
        }}
        div[data-testid="stPlotlyChart"] {{
            background-color: {p["card_bg"]};
            border: 1px solid {p["border"]};
            border-radius: 10px;
            overflow: hidden;
            box-shadow: {p["shadow"]};
        }}
        .dk-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: {p["badge_bg"]};
            color: {p["badge_text"]};
            border: 1px solid {p["border_strong"]};
            margin-right: 6px;
        }}
        .dk-footer {{
            text-align: center;
            color: {p["text_muted"]};
            font-size: 0.85rem;
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid {p["border"]};
        }}
        {light_overrides}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str, badges: list[str] | None = None) -> None:
    inject_custom_css()
    col_title, col_toggle = st.columns([6.5, 1.3], vertical_alignment="center")
    with col_title:
        st.title(title)
    with col_toggle:
        btn_label = "☀️ Light Mode" if is_dark_mode() else "🌙 Dark Mode"
        st.button(
            btn_label,
            key="dk_theme_toggle_btn",
            on_click=toggle_theme_mode,
            help="Switch between Dark Mode and Light Mode",
            use_container_width=True,
        )
    st.caption(subtitle)
    if badges:
        html = "".join(f'<span class="dk-badge">{b}</span>' for b in badges)
        st.markdown(html, unsafe_allow_html=True)
        st.write("")


def render_footer() -> None:
    st.markdown(
        """
        <div class="dk-footer">
            <strong>BRIN Research Team</strong> &middot; DeepKriging Ver 5 Java Monthly Rainfall Viewer
            (1985–2014)
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_model_selector(default: list[str] | None = None, key: str = "models") -> list[str]:
    if default is None:
        default = MODEL_ORDER
    return st.sidebar.multiselect(
        "Models",
        options=MODEL_ORDER,
        default=default,
        format_func=lambda k: MODEL_SPECS[k]["label"],
        key=key,
    )


def sidebar_year_range(key_prefix: str = "yr") -> tuple[int, int]:
    c1, c2 = st.sidebar.columns(2)
    with c1:
        y_start = st.selectbox("Start Year", YEARS, index=0, key=f"{key_prefix}_start")
    with c2:
        y_end = st.selectbox("End Year", YEARS, index=len(YEARS) - 1, key=f"{key_prefix}_end")
    if y_start > y_end:
        st.sidebar.error("Start Year must be <= End Year.")
    return int(y_start), int(y_end)


def sidebar_season_filter(key: str = "season") -> list[int]:
    season_name = st.sidebar.selectbox("Season Filter", list(SEASONS.keys()), index=0, key=key)
    return SEASONS[season_name]


def render_chart(fig) -> None:
    """Apply consistent theme-aware card styling and render without Streamlit overrides."""
    p = get_theme_palette()
    fig.update_layout(
        template=p["plotly_template"],
        paper_bgcolor=p["card_bg"],
        plot_bgcolor=p["plot_bg"],
        font=dict(color=p["text_secondary"]),
        title_font_color=p["text"],
        legend_font_color=p["text_secondary"],
    )
    fig.update_xaxes(
        gridcolor=p["grid"],
        zerolinecolor=p["border_strong"],
        tickfont=dict(color=p["tick_text"]),
        title_font=dict(color=p["text_secondary"]),
    )
    fig.update_yaxes(
        gridcolor=p["grid"],
        zerolinecolor=p["border_strong"],
        tickfont=dict(color=p["tick_text"]),
        title_font=dict(color=p["text_secondary"]),
    )
    for annot in fig.layout.annotations or []:
        if not getattr(annot, "bgcolor", None):
            annot.font.color = p["text"]
    for trace in fig.data or []:
        if trace.type in ("heatmap", "choroplethmap", "choropleth", "contour", "surface"):
            trace.colorbar.tickfont = dict(color=p["tick_text"])
            trace.colorbar.title.font = dict(color=p["text_secondary"])

    st.plotly_chart(fig, use_container_width=True, theme=None)


def render_dataframe(data, **kwargs):
    """Render dataframe across Streamlit versions without use_container_width deprecation warnings."""
    if "use_container_width" in kwargs or "width" not in kwargs:
        use_cont = kwargs.pop("use_container_width", True)
        try:
            parts = [int(p) for p in st.__version__.split(".")[:2]]
            is_new = (parts[0], parts[1]) >= (1, 41)
        except Exception:
            is_new = False

        if is_new:
            kwargs["width"] = "stretch" if use_cont else "content"
        else:
            kwargs["use_container_width"] = use_cont

    return st.dataframe(data, **kwargs)


def download_csv_button(df: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    if df.empty:
        return
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )
