"""Shared UI styling, headers, download buttons, and footer components."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import MODEL_ORDER, MODEL_SPECS, MONTHS, MONTH_NAMES, SEASONS, YEARS


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 12px 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .dk-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: #EFF6FF;
            color: #1D4ED8;
            border: 1px solid #BFDBFE;
            margin-right: 6px;
        }
        .dk-footer {
            text-align: center;
            color: #64748B;
            font-size: 0.85rem;
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid #E2E8F0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str, badges: list[str] | None = None) -> None:
    inject_custom_css()
    st.title(title)
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

