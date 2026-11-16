"""Page 1: Multi-Model Comparison, Time Series, Scatter 1:1 & Error Diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.charts import make_scatter_comparison, make_timeseries_figure
from src.config import APP_TITLE, MODEL_ORDER, MODEL_SPECS, TEST_POINTS
from src.data_loader import load_paired_with_predictions
from src.metrics import compute_metrics, evaluate_models_table
from src.ui import (
    download_csv_button,
    render_footer,
    render_header,
    sidebar_model_selector,
    sidebar_season_filter,
    sidebar_year_range,
)

st.set_page_config(
    page_title=f"{APP_TITLE} | Model Comparison",
    page_icon="📊",
    layout="wide",
)

render_header(
    "📊 Model Comparison & Station Explorer",
    "Quantitative multi-model evaluation across independent hold-out points, training grids, and Java regions.",
)

df_all = load_paired_with_predictions()
if df_all.empty:
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
st.sidebar.header("Configuration")
selected_models = sidebar_model_selector(default=MODEL_ORDER, key="cmp_models")
y_start, y_end = sidebar_year_range("cmp")
season_months = sidebar_season_filter("cmp_season")

scope_mode = st.sidebar.radio(
    "Spatial Evaluation Scope",
    [
        "All 9 Hold-Out Test Points (Mean)",
        "Single Hold-Out Test Point",
        "All Training Grid Points (Mean)",
        "Single Training Grid Cell",
        "Regional Average (West/Central/East Java)",
    ],
    key="cmp_scope",
)

selected_point_label = None
if scope_mode == "Single Hold-Out Test Point":
    tp_options = [f"P{p['point_id']}: {p['location_name']} ({p['lat']:.2f}, {p['lon']:.2f})" for p in TEST_POINTS]
    selected_point_label = st.sidebar.selectbox("Select Hold-Out Point", tp_options, key="cmp_tp")
elif scope_mode == "Single Training Grid Cell":
    train_pts = (
        df_all[~df_all["is_test_point"]][["idx_new", "lat", "lon", "region"]]
        .drop_duplicates()
        .sort_values("idx_new")
    )
    tr_options = [
        f"Grid #{int(r['idx_new'])} ({r['lat']:.2f}, {r['lon']:.2f}) — {r['region']}"
        for _, r in train_pts.iterrows()
    ]
    selected_point_label = st.sidebar.selectbox("Select Training Grid", tr_options, key="cmp_tr")
elif scope_mode == "Regional Average (West/Central/East Java)":
    reg_choice = st.sidebar.selectbox(
        "Select Region",
        ["West Java & Banten", "Central Java & DIY", "East Java"],
        key="cmp_reg",
    )
    reg_include = st.sidebar.radio("Points in Region", ["Hold-Out Test Only", "All Paired Points"], key="cmp_reg_inc")

if not selected_models or y_start > y_end:
    st.warning("Please select at least one model and a valid year range.")
    st.stop()

# Filter data
df_sub = df_all[
    (df_all["year"] >= y_start)
    & (df_all["year"] <= y_end)
    & (df_all["month"].isin(season_months))
].copy()

if scope_mode == "All 9 Hold-Out Test Points (Mean)":
    df_sub = df_sub[df_sub["is_test_point"]].copy()
    scope_title = "Mean Across 9 Hold-Out Test Points"
elif scope_mode == "Single Hold-Out Test Point":
    pid = int(selected_point_label.split(":")[0].replace("P", ""))
    df_sub = df_sub[df_sub["point_id"] == pid].copy()
    scope_title = selected_point_label
elif scope_mode == "All Training Grid Points (Mean)":
    df_sub = df_sub[~df_sub["is_test_point"]].copy()
    scope_title = "Mean Across All Training Grid Cells"
elif scope_mode == "Single Training Grid Cell":
    gid = int(selected_point_label.split("#")[1].split(" ")[0])
    df_sub = df_sub[df_sub["idx_new"] == gid].copy()
    scope_title = selected_point_label
else:
    df_sub = df_sub[df_sub["region"] == reg_choice].copy()
    if reg_include == "Hold-Out Test Only":
        df_sub = df_sub[df_sub["is_test_point"]].copy()
    scope_title = f"{reg_choice} ({reg_include})"

if df_sub.empty:
    st.warning("No matching data found for the selected filters.")
    st.stop()

# Aggregate by month-year for time series view
pred_cols = [MODEL_SPECS[k]["pred_col"] for k in selected_models]
agg_dict = {"monthly_rainfall": "mean"}
for pc in pred_cols:
    agg_dict[pc] = "mean"
df_ts = df_sub.groupby(["year", "month", "date"], as_index=False).agg(agg_dict).sort_values("date")

# ---------------------------------------------------------------------------
# Metrics Table (Point-Month Level vs Spatial-Mean Level)
# ---------------------------------------------------------------------------
st.subheader(f"Evaluation Metrics — {scope_title} ({y_start}–{y_end})")
metric_basis = st.radio(
    "Metric Aggregation Basis",
    ["All Point-Month Pairs (Unaggregated)", "Spatially Averaged Monthly Series"],
    horizontal=True,
)
eval_df = df_sub if metric_basis.startswith("All Point") else df_ts
metrics_tbl = evaluate_models_table(eval_df, MODEL_SPECS, selected_models)

st.dataframe(
    metrics_tbl.drop(columns=["Model_Key"]).style.format({
        "N": "{:,}",
        "MAE": "{:.2f}",
        "RMSE": "{:.2f}",
        "R2": "{:.4f}",
        "KGE": "{:.4f}",
        "r": "{:.4f}",
        "MBE": "{:+.2f}",
        "PBIAS_%": "{:+.2f}%",
        "NRMSE": "{:.3f}",
        "CV_RMSE_%": "{:.2f}%",
        "Mean_Obs": "{:.2f}",
        "Mean_Pred": "{:.2f}",
    }).highlight_min(subset=["MAE", "RMSE"], color="#DCFCE7").highlight_max(subset=["R2", "KGE"], color="#DBEAFE"),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Tabs for Deep Visual Analytics
# ---------------------------------------------------------------------------
tab_ts, tab_scatter, tab_res, tab_pts, tab_tbl = st.tabs([
    "📈 Time Series & Error Trace",
    "🎯 Scatter 1:1 & Fit",
    "📦 Residual Distribution",
    "📍 Per-Test-Point Breakdown",
    "📋 Raw Paired Table",
])

with tab_ts:
    ts_mode = st.radio(
        "Plot Variable",
        ["Rainfall (Observed vs Predicted)", "Error Bias (Pred − Obs)", "Absolute Error |Pred − Obs|"],
        horizontal=True,
    )
    if ts_mode.startswith("Rainfall"):
        fig_ts = make_timeseries_figure(
            df_ts,
            selected_models,
            show_obs=True,
            title=f"Monthly Rainfall Comparison — {scope_title}",
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        fig_err_ts = go.Figure()
        for mk in selected_models:
            spec = MODEL_SPECS[mk]
            err = df_ts[spec["pred_col"]] - df_ts["monthly_rainfall"]
            if ts_mode.startswith("Absolute"):
                err = err.abs()
            fig_err_ts.add_trace(go.Scatter(
                x=df_ts["date"],
                y=err,
                mode="lines",
                name=spec["label"],
                line=dict(color=spec["color"], width=1.6),
            ))
        if ts_mode.startswith("Error Bias"):
            fig_err_ts.add_hline(y=0, line_dash="dash", line_color="#0F172A", line_width=1)
        fig_err_ts.update_layout(
            title=f"{ts_mode} — {scope_title}",
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Error (mm)",
            height=440,
        )
        st.plotly_chart(fig_err_ts, use_container_width=True)

with tab_scatter:
    fig_sc = make_scatter_comparison(
        eval_df,
        selected_models,
        title=f"Observed vs Predicted Monthly Rainfall — {scope_title}",
    )
    st.plotly_chart(fig_sc, use_container_width=True)

with tab_res:
    res_rows = []
    for mk in selected_models:
        spec = MODEL_SPECS[mk]
        res_vals = eval_df[spec["pred_col"]] - eval_df["monthly_rainfall"]
        res_rows.append(pd.DataFrame({
            "Model": spec["label"],
            "Residual (Pred - Obs)": res_vals,
            "Abs Error": res_vals.abs(),
        }))
    df_res_long = pd.concat(res_rows, ignore_index=True)

    c_r1, c_r2 = st.columns(2)
    with c_r1:
        fig_box = px.box(
            df_res_long,
            x="Model",
            y="Residual (Pred - Obs)",
            color="Model",
            color_discrete_map={MODEL_SPECS[k]["label"]: MODEL_SPECS[k]["color"] for k in selected_models},
            title="Residual Distribution (Pred − Obs)",
            template="plotly_white",
        )
        fig_box.add_hline(y=0, line_dash="dash", line_color="#0F172A")
        fig_box.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig_box, use_container_width=True)

    with c_r2:
        fig_ecdf = px.ecdf(
            df_res_long,
            x="Abs Error",
            color="Model",
            color_discrete_map={MODEL_SPECS[k]["label"]: MODEL_SPECS[k]["color"] for k in selected_models},
            title="Cumulative Distribution (CDF) of Absolute Error",
            template="plotly_white",
        )
        fig_ecdf.update_layout(xaxis_title="Absolute Error |Pred − Obs| (mm)", yaxis_title="Cumulative Probability", height=420)
        st.plotly_chart(fig_ecdf, use_container_width=True)

with tab_pts:
    st.caption("Cross-station comparison across all 9 independent hold-out points for the selected year range and season.")
    df_tp_all = df_all[
        (df_all["is_test_point"])
        & (df_all["year"] >= y_start)
        & (df_all["year"] <= y_end)
        & (df_all["month"].isin(season_months))
    ]
    pt_rows = []
    for tp in TEST_POINTS:
        dpt = df_tp_all[df_tp_all["point_id"] == tp["point_id"]]
        if dpt.empty:
            continue
        for mk in selected_models:
            spec = MODEL_SPECS[mk]
            m = compute_metrics(dpt["monthly_rainfall"], dpt[spec["pred_col"]])
            pt_rows.append({
                "Point": f"P{tp['point_id']}",
                "Location": tp["location_name"],
                "Lon": tp["lon"],
                "Lat": tp["lat"],
                "Model": spec["short"],
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "R2": m["R2"],
                "KGE": m["KGE"],
                "PBIAS_%": m["PBIAS_%"],
            })
    df_pt = pd.DataFrame(pt_rows)
    if not df_pt.empty:
        metric_choice = st.selectbox("Heatmap / Pivot Metric", ["MAE", "RMSE", "R2", "KGE", "PBIAS_%"])
        piv = df_pt.pivot(index="Location", columns="Model", values=metric_choice)
        ordered_cols = [MODEL_SPECS[k]["short"] for k in selected_models if MODEL_SPECS[k]["short"] in piv.columns]
        piv = piv[ordered_cols]

        fig_hm = px.imshow(
            piv,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdYlGn" if metric_choice in ("R2", "KGE") else "YlOrRd",
            title=f"Per-Station {metric_choice} Across 9 Hold-Out Points ({y_start}–{y_end})",
        )
        fig_hm.update_layout(height=420)
        st.plotly_chart(fig_hm, use_container_width=True)
        download_csv_button(df_pt, "per_test_point_metrics.csv", "⬇️ Download Per-Point Metrics CSV")

with tab_tbl:
    show_cols = ["year", "month", "idx_new", "lat", "lon", "location_name", "is_test_point", "monthly_rainfall"] + pred_cols
    df_export = df_sub[show_cols].sort_values(["year", "month", "idx_new"])
    st.dataframe(df_export.head(1000), use_container_width=True, hide_index=True)
    st.caption(f"Showing first 1,000 of {len(df_export):,} rows.")
    download_csv_button(df_export, "filtered_paired_predictions.csv", "⬇️ Download Full Filtered Table CSV")

render_footer()

