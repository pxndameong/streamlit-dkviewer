"""Page 3: Spatio-Temporal Hovmoller Diagnostics (Year x Station, Month x Year, Longitude x Time)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import APP_TITLE, MODEL_ORDER, MODEL_SPECS, MONTHS, MONTH_SHORT, TEST_POINTS
from src.data_loader import load_paired_with_predictions, load_predictions
from src.theme import get_plotly_template
from src.ui import (
    download_csv_button,
    render_chart,
    render_footer,
    render_header,
    sidebar_model_selector,
    sidebar_year_range,
)

st.set_page_config(
    page_title=f"{APP_TITLE} | Time & Station Heatmaps",
    page_icon="🌡️",
    layout="wide",
)

render_header(
    "🌡️ Time & Station Heatmaps",
    "See how rainfall and prediction errors change across years, months, and stations from West to East Java.",
)

df_eval = load_paired_with_predictions()
if df_eval.empty:
    st.stop()

st.sidebar.header("Filters")
selected_models = sidebar_model_selector(default=MODEL_ORDER, key="hov_models")
y_start, y_end = sidebar_year_range("hov")

hov_mode = st.sidebar.radio(
    "Heatmap View",
    [
        "Year × Test Stations (West → East)",
        "Month × Year Calendar Grid",
        "West-to-East Rainfall Over Time (Full Grid)",
    ],
    key="hov_mode",
)

metric_options = (
    ["MAE", "RMSE", "Mean Bias (Pred - Obs)", "Predicted Rainfall", "Observed Rainfall"]
    if hov_mode != "West-to-East Rainfall Over Time (Full Grid)"
    else ["Predicted Rainfall"]
)
metric_choice = st.sidebar.selectbox("Value to Show (Color)", metric_options, key="hov_metric")

scale_mode = st.sidebar.radio("Color Range", ["Auto", "Custom"], key="hov_scale")
zmin_val, zmax_val = None, None
if scale_mode == "Custom":
    c1, c2 = st.sidebar.columns(2)
    with c1:
        zmin_val = st.number_input("Z min", value=-100.0 if "Bias" in metric_choice else 0.0, step=10.0)
    with c2:
        zmax_val = st.number_input("Z max", value=100.0 if "Bias" in metric_choice else 400.0, step=10.0)

if not selected_models or y_start > y_end:
    st.warning("Select at least one model and a valid year range.")
    st.stop()

# Colorscale selection
if "Bias" in metric_choice:
    cscale = "RdBu_r"
elif "Rainfall" in metric_choice:
    cscale = "YlGnBu"
else:
    cscale = "YlOrRd"


def _compute_agg_metric(df_g: pd.DataFrame, pred_col: str, metric: str) -> float:
    if df_g.empty:
        return np.nan
    yt = df_g["monthly_rainfall"].to_numpy()
    yp = df_g[pred_col].to_numpy()
    if metric == "MAE":
        return float(np.mean(np.abs(yp - yt)))
    if metric == "RMSE":
        return float(np.sqrt(np.mean((yp - yt) ** 2)))
    if metric == "Mean Bias (Pred - Obs)":
        return float(np.mean(yp - yt))
    if metric == "Predicted Rainfall":
        return float(np.mean(yp))
    return float(np.mean(yt))


# ---------------------------------------------------------------------------
# Mode 1: Year x Test Stations (sorted West to East)
# ---------------------------------------------------------------------------
if hov_mode == "Year × Test Stations (West → East)":
    tp_sorted = sorted(TEST_POINTS, key=lambda p: p["lon"])
    tp_labels = [p["short"] for p in tp_sorted]
    selected_pts = st.sidebar.multiselect(
        "Test Stations (West → East)",
        options=tp_labels,
        default=tp_labels,
        key="hov_tp",
    )
    if not selected_pts:
        st.warning("Select at least one station.")
        st.stop()

    df_sub = df_eval[
        (df_eval["is_test_point"])
        & (df_eval["year"] >= y_start)
        & (df_eval["year"] <= y_end)
    ].copy()
    pid_to_short = {p["point_id"]: p["short"] for p in tp_sorted}
    df_sub["station_short"] = df_sub["point_id"].map(pid_to_short)
    df_sub = df_sub[df_sub["station_short"].isin(selected_pts)]

    years_list = list(range(y_start, y_end + 1))
    ordered_pts = [s for s in tp_labels if s in selected_pts]

    # Build tables per model
    export_rows = []
    fig = make_subplots(
        rows=1,
        cols=len(selected_models),
        subplot_titles=[MODEL_SPECS[k]["label"] for k in selected_models],
        shared_yaxes=True,
        horizontal_spacing=0.04,
    )

    all_vals = []
    pivots = {}
    for mk in selected_models:
        pcol = MODEL_SPECS[mk]["pred_col"]
        mat = np.full((len(years_list), len(ordered_pts)), np.nan)
        for yi, yr in enumerate(years_list):
            for xi, st_name in enumerate(ordered_pts):
                cell = df_sub[(df_sub["year"] == yr) & (df_sub["station_short"] == st_name)]
                val = _compute_agg_metric(cell, pcol, metric_choice)
                mat[yi, xi] = val
                if np.isfinite(val):
                    all_vals.append(val)
                export_rows.append({
                    "model": MODEL_SPECS[mk]["short"],
                    "year": yr,
                    "station": st_name,
                    "metric": metric_choice,
                    "value": val,
                })
        pivots[mk] = mat

    if scale_mode == "Auto" and all_vals:
        if "Bias" in metric_choice:
            vmax = max(abs(min(all_vals)), abs(max(all_vals)))
            zmin_f, zmax_f = -vmax, vmax
        else:
            zmin_f, zmax_f = min(all_vals), max(all_vals)
    else:
        zmin_f, zmax_f = zmin_val, zmax_val

    for idx, mk in enumerate(selected_models):
        fig.add_trace(
            go.Heatmap(
                z=pivots[mk],
                x=ordered_pts,
                y=years_list,
                colorscale=cscale,
                zmin=zmin_f,
                zmax=zmax_f,
                showscale=(idx == len(selected_models) - 1),
                colorbar=dict(title=metric_choice) if idx == len(selected_models) - 1 else None,
                hovertemplate="Station: %{x}<br>Year: %{y}<br>" + metric_choice + ": %{z:.2f}<extra></extra>",
            ),
            row=1,
            col=idx + 1,
        )
        fig.update_xaxes(tickangle=35, row=1, col=idx + 1)

    fig.update_yaxes(title_text="Year", dtick=2, row=1, col=1)
    fig.update_layout(
        title=f"{metric_choice} by Year & Test Station (West → East Java)",
        template=get_plotly_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=650,
    )
    render_chart(fig)
    download_csv_button(pd.DataFrame(export_rows), "heatmap_year_station.csv")

# ---------------------------------------------------------------------------
# Mode 2: Month x Year Calendar Grid
# ---------------------------------------------------------------------------
elif hov_mode == "Month × Year Calendar Grid":
    scope_m2 = st.sidebar.radio("Stations", ["9 Test Stations", "All Stations"], key="hov_m2_scope")
    df_sub = df_eval[(df_eval["year"] >= y_start) & (df_eval["year"] <= y_end)].copy()
    if scope_m2.startswith("9"):
        df_sub = df_sub[df_sub["is_test_point"]]

    years_list = list(range(y_start, y_end + 1))
    month_labels = [MONTH_SHORT[m] for m in MONTHS]

    fig = make_subplots(
        rows=1,
        cols=len(selected_models),
        subplot_titles=[MODEL_SPECS[k]["label"] for k in selected_models],
        shared_yaxes=True,
        horizontal_spacing=0.04,
    )
    all_vals = []
    pivots = {}
    for mk in selected_models:
        pcol = MODEL_SPECS[mk]["pred_col"]
        mat = np.full((len(years_list), 12), np.nan)
        for yi, yr in enumerate(years_list):
            for mi, mo in enumerate(MONTHS):
                cell = df_sub[(df_sub["year"] == yr) & (df_sub["month"] == mo)]
                val = _compute_agg_metric(cell, pcol, metric_choice)
                mat[yi, mi] = val
                if np.isfinite(val):
                    all_vals.append(val)
        pivots[mk] = mat

    if scale_mode == "Auto" and all_vals:
        if "Bias" in metric_choice:
            vmax = max(abs(min(all_vals)), abs(max(all_vals)))
            zmin_f, zmax_f = -vmax, vmax
        else:
            zmin_f, zmax_f = min(all_vals), max(all_vals)
    else:
        zmin_f, zmax_f = zmin_val, zmax_val

    for idx, mk in enumerate(selected_models):
        fig.add_trace(
            go.Heatmap(
                z=pivots[mk],
                x=month_labels,
                y=years_list,
                colorscale=cscale,
                zmin=zmin_f,
                zmax=zmax_f,
                showscale=(idx == len(selected_models) - 1),
                colorbar=dict(title=metric_choice) if idx == len(selected_models) - 1 else None,
                hovertemplate="Month: %{x}<br>Year: %{y}<br>" + metric_choice + ": %{z:.2f}<extra></extra>",
            ),
            row=1,
            col=idx + 1,
        )

    fig.update_yaxes(title_text="Year", dtick=2, row=1, col=1)
    fig.update_layout(
        title=f"{metric_choice} by Month & Year ({scope_m2})",
        template=get_plotly_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=650,
    )
    render_chart(fig)

# ---------------------------------------------------------------------------
# Mode 3: West-to-East Rainfall Over Time across Full 0.25 deg Grid
# ---------------------------------------------------------------------------
else:
    df_pred = load_predictions()
    df_pred = df_pred[(df_pred["year"] >= y_start) & (df_pred["year"] <= y_end)].copy()
    df_pred["date"] = pd.to_datetime(df_pred[["year", "month"]].assign(day=1))
    st.caption("Average predicted rainfall (mm/month) from West to East Java across the full map grid.")

    fig = make_subplots(
        rows=1,
        cols=len(selected_models),
        subplot_titles=[MODEL_SPECS[k]["label"] for k in selected_models],
        shared_yaxes=True,
        horizontal_spacing=0.04,
    )

    all_vals = []
    pivots = {}
    for mk in selected_models:
        pcol = MODEL_SPECS[mk]["pred_col"]
        piv = df_pred.groupby(["date", "lon"], as_index=False)[pcol].mean().pivot(
            index="date", columns="lon", values=pcol
        )
        pivots[mk] = piv
        all_vals.append(np.nanmin(piv.values))
        all_vals.append(np.nanmax(piv.values))

    zmin_f = min(all_vals) if scale_mode == "Auto" else zmin_val
    zmax_f = max(all_vals) if scale_mode == "Auto" else zmax_val

    for idx, mk in enumerate(selected_models):
        piv = pivots[mk]
        fig.add_trace(
            go.Heatmap(
                z=piv.values,
                x=piv.columns.astype(float),
                y=piv.index,
                colorscale="YlGnBu",
                zmin=zmin_f,
                zmax=zmax_f,
                showscale=(idx == len(selected_models) - 1),
                colorbar=dict(title="Rainfall (mm)") if idx == len(selected_models) - 1 else None,
            ),
            row=1,
            col=idx + 1,
        )
        fig.update_xaxes(title_text="Longitude (°E)", row=1, col=idx + 1)

    fig.update_yaxes(title_text="Time", row=1, col=1)
    fig.update_layout(
        title=f"West-to-East Rainfall Over Time ({y_start}–{y_end})",
        template=get_plotly_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=680,
    )
    render_chart(fig)

render_footer()

