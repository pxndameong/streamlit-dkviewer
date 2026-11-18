"""Page 2: Monthly Climatology, Single-Month Interannual & ENSO Anomaly Explorer."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import APP_TITLE, MODEL_ORDER, MODEL_SPECS, MONTHS, MONTH_NAMES, MONTH_SHORT, TEST_POINTS
from src.data_loader import load_paired_with_predictions
from src.metrics import evaluate_models_table
from src.theme import get_plotly_template, get_theme_palette
from src.ui import (
    download_csv_button,
    render_chart,
    render_footer,
    render_header,
    sidebar_model_selector,
    sidebar_year_range,
)

st.set_page_config(
    page_title=f"{APP_TITLE} | Monthly Patterns",
    page_icon="🗓️",
    layout="wide",
)

render_header(
    "🗓️ Monthly & Seasonal Rainfall Patterns",
    "Explore the 12-month rainfall cycle, year-to-year trends for a single month, and wet/dry year differences.",
)

df_all = load_paired_with_predictions()
if df_all.empty:
    st.stop()

st.sidebar.header("Filters")
selected_models = sidebar_model_selector(default=MODEL_ORDER, key="clim_models")
y_start, y_end = sidebar_year_range("clim")

station_opts = ["All 9 Test Stations (Average)", "All Stations (Average)"] + [
    p["short"] for p in TEST_POINTS
]
station_sel = st.sidebar.selectbox("Station Selection", station_opts, key="clim_sta")
focus_month = st.sidebar.selectbox(
    "Month to Inspect (Year-to-Year Tab)",
    MONTHS,
    index=0,
    format_func=lambda m: MONTH_NAMES[m],
    key="clim_month",
)

if not selected_models or y_start > y_end:
    st.warning("Select at least one model and a valid year range.")
    st.stop()

df_sub = df_all[(df_all["year"] >= y_start) & (df_all["year"] <= y_end)].copy()
if station_sel.startswith("All 9"):
    df_sub = df_sub[df_sub["is_test_point"]]
elif station_sel.startswith("S"):
    pid = int(station_sel.split(" ")[0].replace("S", ""))
    df_sub = df_sub[df_sub["point_id"] == pid]

pred_cols = [MODEL_SPECS[k]["pred_col"] for k in selected_models]

tab_cycle, tab_month, tab_anom = st.tabs([
    "🌦️ Monthly Cycle (Jan–Dec)",
    f"📅 Year-to-Year Trend ({MONTH_NAMES[focus_month]})",
    "⚡ Wet & Dry Years",
])

with tab_cycle:
    agg_dict = {"monthly_rainfall": "mean"}
    for pc in pred_cols:
        agg_dict[pc] = "mean"
    df_cyc = df_sub.groupby("month", as_index=False).agg(agg_dict).sort_values("month")
    df_cyc["Month"] = df_cyc["month"].map(MONTH_SHORT)

    fig_cyc = go.Figure()
    fig_cyc.add_trace(go.Scatter(
        x=df_cyc["Month"],
        y=df_cyc["monthly_rainfall"],
        mode="lines+markers",
        name="Observed Rainfall",
        line=dict(color=get_theme_palette()["obs_color"], width=3),
        marker=dict(size=7),
    ))
    for mk in selected_models:
        spec = MODEL_SPECS[mk]
        fig_cyc.add_trace(go.Scatter(
            x=df_cyc["Month"],
            y=df_cyc[spec["pred_col"]],
            mode="lines+markers",
            name=spec["label"],
            line=dict(color=spec["color"], width=2),
        ))
    fig_cyc.update_layout(
        title=f"Average Monthly Rainfall ({y_start}–{y_end}) — {station_sel}",
        template=get_plotly_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        xaxis_title="Month",
        yaxis_title="Average Rainfall (mm/month)",
        height=450,
    )
    render_chart(fig_cyc)

    # Monthly MAE & Bias comparison across 12 months
    monthly_metrics = []
    for m_num in MONTHS:
        dm = df_sub[df_sub["month"] == m_num]
        if dm.empty:
            continue
        for mk in selected_models:
            spec = MODEL_SPECS[mk]
            err = dm[spec["pred_col"]] - dm["monthly_rainfall"]
            monthly_metrics.append({
                "month": m_num,
                "Month": MONTH_SHORT[m_num],
                "Model": spec["label"],
                "MAE": float(err.abs().mean()),
                "Bias": float(err.mean()),
            })
    df_mm = pd.DataFrame(monthly_metrics)
    c1, c2 = st.columns(2)
    with c1:
        fig_mmae = px.line(
            df_mm,
            x="Month",
            y="MAE",
            color="Model",
            markers=True,
            color_discrete_map={MODEL_SPECS[k]["label"]: MODEL_SPECS[k]["color"] for k in selected_models},
            title="Average Error (MAE) by Month",
            template=get_plotly_template(),
        )
        fig_mmae.update_layout(height=380, yaxis_title="MAE (mm)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        render_chart(fig_mmae)
    with c2:
        fig_mbias = px.bar(
            df_mm,
            x="Month",
            y="Bias",
            color="Model",
            barmode="group",
            color_discrete_map={MODEL_SPECS[k]["label"]: MODEL_SPECS[k]["color"] for k in selected_models},
            title="Average Bias (Pred − Obs) by Month",
            template=get_plotly_template(),
        )
        fig_mbias.add_hline(y=0, line_dash="dash", line_color="#A1A1AA")
        fig_mbias.update_layout(height=380, yaxis_title="Mean Bias (mm)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        render_chart(fig_mbias)

with tab_month:
    df_m = df_sub[df_sub["month"] == focus_month].copy()
    df_m_yr = df_m.groupby("year", as_index=False).agg({"monthly_rainfall": "mean", **{pc: "mean" for pc in pred_cols}})

    fig_my = go.Figure()
    fig_my.add_trace(go.Scatter(
        x=df_m_yr["year"],
        y=df_m_yr["monthly_rainfall"],
        mode="lines+markers",
        name="Observed",
        line=dict(color=get_theme_palette()["obs_color"], width=2.6),
    ))
    for mk in selected_models:
        spec = MODEL_SPECS[mk]
        fig_my.add_trace(go.Scatter(
            x=df_m_yr["year"],
            y=df_m_yr[spec["pred_col"]],
            mode="lines+markers",
            name=spec["label"],
            line=dict(color=spec["color"], width=1.8),
        ))
    fig_my.update_layout(
        title=f"{MONTH_NAMES[focus_month]} Rainfall Across Years ({y_start}–{y_end}) — {station_sel}",
        template=get_plotly_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        xaxis=dict(title="Year", dtick=2),
        yaxis_title="Rainfall (mm)",
        height=430,
    )
    render_chart(fig_my)

    tbl_m = evaluate_models_table(df_m, MODEL_SPECS, selected_models)
    st.subheader(f"Accuracy for {MONTH_NAMES[focus_month]} ({y_start}–{y_end})")
    st.dataframe(
        tbl_m.drop(columns=["Model_Key"]).style.format({
            "N": "{:,}", "MAE": "{:.2f}", "RMSE": "{:.2f}",
            "R2": "{:.4f}", "KGE": "{:.4f}", "r": "{:.4f}",
            "MBE": "{:+.2f}", "PBIAS_%": "{:+.2f}%",
            "NRMSE": "{:.3f}", "CV_RMSE_%": "{:.2f}%",
            "Mean_Obs": "{:.2f}", "Mean_Pred": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    download_csv_button(df_m_yr, f"climatology_{MONTH_NAMES[focus_month].lower()}.csv")

with tab_anom:
    st.caption("Difference from the 1985–2014 average rainfall (shows dry years like El Niño 1997 and wet years like La Niña 2010).")
    df_yr = df_sub.groupby("year", as_index=False).agg({"monthly_rainfall": "mean", **{pc: "mean" for pc in pred_cols}})
    clim_obs = df_yr["monthly_rainfall"].mean()
    df_yr["Obs_Anomaly"] = df_yr["monthly_rainfall"] - clim_obs

    fig_anom = go.Figure()
    colors = ["#2563EB" if v >= 0 else "#DC2626" for v in df_yr["Obs_Anomaly"]]
    fig_anom.add_trace(go.Bar(
        x=df_yr["year"],
        y=df_yr["Obs_Anomaly"],
        name="Observed Anomaly",
        marker_color=colors,
        opacity=0.6,
    ))
    for mk in selected_models:
        spec = MODEL_SPECS[mk]
        pred_anom = df_yr[spec["pred_col"]] - df_yr[spec["pred_col"]].mean()
        fig_anom.add_trace(go.Scatter(
            x=df_yr["year"],
            y=pred_anom,
            mode="lines+markers",
            name=f"{spec['short']} Anomaly",
            line=dict(color=spec["color"], width=1.8),
        ))
    fig_anom.add_hline(y=0, line_color="#A1A1AA", line_width=1)
    fig_anom.update_layout(
        title=f"Yearly Rainfall Difference vs Long-Term Average ({clim_obs:.1f} mm/mo) — {station_sel}",
        template=get_plotly_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        xaxis=dict(title="Year", dtick=2),
        yaxis_title="Difference from Average (mm/month)",
        height=440,
    )
    render_chart(fig_anom)

render_footer()

