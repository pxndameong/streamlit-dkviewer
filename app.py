"""DK Viewer v2 — Main Entrypoint (Overview & Experiment Hub)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    APP_SUBTITLE,
    APP_TITLE,
    MODEL_ORDER,
    MODEL_SPECS,
    TEST_POINTS,
)
from src.data_loader import (
    load_history_summary,
    load_metadata,
    load_paired_with_predictions,
    load_shap_ranking,
    load_station_catalog,
)
from src.metrics import evaluate_models_table
from src.ui import download_csv_button, render_footer, render_header

st.set_page_config(
    page_title=f"{APP_TITLE} | Overview",
    page_icon="🌧️",
    layout="wide",
)

render_header(
    f"🌧️ {APP_TITLE} — DeepKriging Rainfall Analytics Hub",
    APP_SUBTITLE,
    badges=[
        "Ver 5 Basis [16, 64, 256]",
        "5 SHAP Feature Variants (1–20 Var)",
        "1985–2014 (360 Months)",
        "9 Hold-Out Validation Points",
    ],
)

meta = load_metadata()
df_eval = load_paired_with_predictions()
catalog = load_station_catalog()
hist_sum = load_history_summary()
shap_df = load_shap_ranking()

if df_eval.empty:
    st.stop()

# ---------------------------------------------------------------------------
# Top KPI Cards
# ---------------------------------------------------------------------------
df_test = df_eval[df_eval["is_test_point"]].copy()
df_train = df_eval[~df_eval["is_test_point"]].copy()

leaderboard_test = evaluate_models_table(df_test, MODEL_SPECS, MODEL_ORDER)
best_mae_row = leaderboard_test.loc[leaderboard_test["MAE"].idxmin()]
best_kge_row = leaderboard_test.loc[leaderboard_test["KGE"].idxmax()]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Prediction Grid", "675 cells", "0.25° × 0.25° Java")
c2.metric("Validation Samples", f"{len(df_test):,}", "9 Hold-out Points × 360 mo")
c3.metric("Raw BMKG Stations", f"{catalog['stanam'].nunique():,}" if not catalog.empty else "—", "KDTree Paired to ERA5")
c4.metric("Best Test MAE", f"{best_mae_row['MAE']:.2f} mm", best_mae_row["Model"])
c5.metric("Best Test KGE", f"{best_kge_row['KGE']:.3f}", best_kge_row["Model"])

st.markdown("---")

# ---------------------------------------------------------------------------
# Model Architecture & Hold-Out Leaderboard
# ---------------------------------------------------------------------------
tab_leader, tab_map, tab_arch = st.tabs([
    "🏆 Hold-Out vs Training Leaderboard",
    "🗺️ Observation & Validation Network Map",
    "⚙️ Experiment & Model Specifications",
])

with tab_leader:
    scope_col1, scope_col2 = st.columns([1, 2])
    with scope_col1:
        eval_scope = st.radio(
            "Evaluation Subset",
            ["Independent Hold-Out (9 Test Points)", "Training Grid Points", "All Paired Grid Points"],
            horizontal=False,
        )
    if eval_scope.startswith("Independent"):
        sub_df = df_test
    elif eval_scope.startswith("Training"):
        sub_df = df_train
    else:
        sub_df = df_eval

    tbl = evaluate_models_table(sub_df, MODEL_SPECS, MODEL_ORDER)

    st.subheader(f"Performance Summary — {eval_scope}")
    styled_tbl = tbl.drop(columns=["Model_Key"]).style.format({
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
    }).highlight_min(subset=["MAE", "RMSE", "NRMSE"], color="#DCFCE7").highlight_max(
        subset=["R2", "KGE", "r"], color="#DBEAFE"
    )
    st.dataframe(styled_tbl, use_container_width=True, hide_index=True)
    download_csv_button(tbl, "dk_v2_model_leaderboard.csv", "⬇️ Download Leaderboard CSV")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            name="MAE (mm)",
            x=tbl["Model"],
            y=tbl["MAE"],
            marker_color="#2563EB",
            text=tbl["MAE"].round(2),
            textposition="outside",
        ))
        fig_err.add_trace(go.Bar(
            name="RMSE (mm)",
            x=tbl["Model"],
            y=tbl["RMSE"],
            marker_color="#DC2626",
            text=tbl["RMSE"].round(2),
            textposition="outside",
        ))
        fig_err.update_layout(
            title="Error Magnitude Across Models (Lower is Better)",
            barmode="group",
            template="plotly_white",
            yaxis_title="mm / month",
            height=380,
        )
        st.plotly_chart(fig_err, use_container_width=True)

    with col_b:
        fig_skill = go.Figure()
        fig_skill.add_trace(go.Bar(
            name="R²",
            x=tbl["Model"],
            y=tbl["R2"],
            marker_color="#10B981",
            text=tbl["R2"].round(3),
            textposition="outside",
        ))
        fig_skill.add_trace(go.Bar(
            name="KGE",
            x=tbl["Model"],
            y=tbl["KGE"],
            marker_color="#F59E0B",
            text=tbl["KGE"].round(3),
            textposition="outside",
        ))
        fig_skill.update_layout(
            title="Skill Scores Across Models (Higher is Better)",
            barmode="group",
            template="plotly_white",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1.05]),
            height=380,
        )
        st.plotly_chart(fig_skill, use_container_width=True)

with tab_map:
    st.subheader("Java Station & Paired Grid Network")
    st.caption(
        "Comparison between raw BMKG rainfall stations (small dots), ERA5 paired training grid cells (blue), "
        "and the 9 independent hold-out validation grid cells excluded from training (red diamonds)."
    )

    show_raw = st.checkbox("Overlay Raw BMKG Stations (486 unique station names)", value=True)
    grid_summary = (
        df_eval.groupby(["idx_new", "lat", "lon", "is_test_point", "point_id", "location_name", "region"], as_index=False)
        .agg(mean_rain=("monthly_rainfall", "mean"), n_months=("monthly_rainfall", "count"))
    )

    fig_net = go.Figure()
    if show_raw and not catalog.empty:
        fig_net.add_trace(go.Scattermapbox(
            lat=catalog["lat"],
            lon=catalog["long"],
            mode="markers",
            marker=dict(size=5, color="#64748B", opacity=0.55),
            name="Raw BMKG Stations",
            text=[
                f"{r['stanam']} (Elev: {r['elev']:.0f} m)<br>Paired Grid: ({r['grid_lat']:.2f}, {r['grid_lon']:.2f}) | Dist: {r['pair_dist_km']:.1f} km"
                for _, r in catalog.iterrows()
            ],
            hoverinfo="text",
        ))

    train_pts = grid_summary[~grid_summary["is_test_point"]]
    fig_net.add_trace(go.Scattermapbox(
        lat=train_pts["lat"],
        lon=train_pts["lon"],
        mode="markers",
        marker=dict(size=9, color="#2563EB", opacity=0.8),
        name="Training Grid Cells",
        text=[
            f"Grid #{r['idx_new']} ({r['lat']:.2f}, {r['lon']:.2f})<br>Region: {r['region']}<br>Mean Obs: {r['mean_rain']:.1f} mm/mo"
            for _, r in train_pts.iterrows()
        ],
        hoverinfo="text",
    ))

    test_pts = grid_summary[grid_summary["is_test_point"]].sort_values("point_id")
    fig_net.add_trace(go.Scattermapbox(
        lat=test_pts["lat"],
        lon=test_pts["lon"],
        mode="markers+text",
        marker=dict(size=14, color="#DC2626"),
        text=[f"P{r['point_id']}" for _, r in test_pts.iterrows()],
        textposition="top right",
        textfont=dict(size=11, color="#991B1B"),
        name="9 Hold-Out Test Points",
        hovertext=[
            f"P{r['point_id']}: {r['location_name']}<br>Coord: ({r['lat']:.2f}, {r['lon']:.2f})<br>Mean Obs: {r['mean_rain']:.1f} mm/mo ({r['n_months']} mo)"
            for _, r in test_pts.iterrows()
        ],
        hoverinfo="text",
    ))

    fig_net.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=-7.25, lon=109.8), zoom=6.2),
        margin=dict(l=0, r=0, t=10, b=0),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    st.plotly_chart(fig_net, use_container_width=True)

    st.markdown("#### 9 Independent Hold-Out Validation Locations")
    st.dataframe(pd.DataFrame(TEST_POINTS), use_container_width=True, hide_index=True)

with tab_arch:
    st.subheader("DeepKriging Ver 5 Model Configurations")
    spec_rows = []
    for k in MODEL_ORDER:
        s = MODEL_SPECS[k]
        spec_rows.append({
            "Model": s["label"],
            "Exogenous Features (#)": len(s["features"]),
            "Selected Covariates": ", ".join(s["features"]),
            "Spatial Basis (Wendland C2)": "16 + 64 + 256 (336 knots)",
            "Training Horizon": "1985–2014 (360 monthly models)",
        })
    st.dataframe(pd.DataFrame(spec_rows), use_container_width=True, hide_index=True)

    if not hist_sum.empty:
        st.subheader("Training Convergence Snapshot (Mean Across 360 Monthly Runs)")
        conv_tbl = (
            hist_sum.groupby("model", as_index=False)
            .agg(
                Runs=("year", "count"),
                Avg_Epochs=("total_epochs", "mean"),
                Avg_Final_Train_MAE=("final_mae_train", "mean"),
                Avg_Final_Test_MAE=("final_mae_test", "mean"),
                Avg_Best_Test_MAE=("best_mae_test", "mean"),
                Avg_Best_Test_Epoch=("best_mae_test_epoch", "mean"),
                Avg_Overfit_Gap=("overfit_gap", "mean"),
            )
        )
        conv_tbl["order"] = conv_tbl["model"].map({k: i for i, k in enumerate(MODEL_ORDER)})
        conv_tbl = conv_tbl.sort_values("order").drop(columns=["order"])
        conv_tbl["model"] = conv_tbl["model"].map(lambda x: MODEL_SPECS[x]["label"])
        st.dataframe(
            conv_tbl.style.format({
                "Avg_Epochs": "{:.0f}",
                "Avg_Final_Train_MAE": "{:.2f}",
                "Avg_Final_Test_MAE": "{:.2f}",
                "Avg_Best_Test_MAE": "{:.2f}",
                "Avg_Best_Test_Epoch": "{:.0f}",
                "Avg_Overfit_Gap": "{:+.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    if meta:
        with st.expander("📦 Packaged Parquet Dataset Metadata"):
            st.json(meta)

render_footer()

