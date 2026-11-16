"""Page 5: Training Convergence, Generalization Gap & Epoch Diagnostics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import APP_TITLE, MODEL_ORDER, MODEL_SPECS, MONTHS, MONTH_NAMES, YEARS
from src.data_loader import load_history_curves, load_history_summary
from src.ui import (
    download_csv_button,
    render_footer,
    render_header,
    sidebar_model_selector,
    sidebar_year_range,
)

st.set_page_config(
    page_title=f"{APP_TITLE} | Training Progress",
    page_icon="📈",
    layout="wide",
)

render_header(
    "📈 Model Training Progress",
    "Check how training and test errors change up to 10,000 epochs, and see when the models reach their best accuracy.",
)

hist_sum = load_history_summary()
hist_curves = load_history_curves()

if hist_sum.empty or hist_curves.empty:
    st.stop()

st.sidebar.header("Filters")
selected_models = sidebar_model_selector(default=MODEL_ORDER, key="tr_models")
y_start, y_end = sidebar_year_range("tr")

c1, c2 = st.sidebar.columns(2)
with c1:
    inspect_year = st.selectbox("Year", YEARS, index=YEARS.index(2010), key="tr_iy")
with c2:
    inspect_month = st.selectbox("Month", MONTHS, index=0, format_func=lambda m: MONTH_NAMES[m], key="tr_im")

if not selected_models or y_start > y_end:
    st.warning("Select at least one model and a valid year range.")
    st.stop()

df_sum_sub = hist_sum[
    (hist_sum["model"].isin(selected_models))
    & (hist_sum["year"] >= y_start)
    & (hist_sum["year"] <= y_end)
].copy()
df_sum_sub["Model"] = df_sum_sub["model"].map(lambda k: MODEL_SPECS[k]["label"])

tab_curve, tab_global, tab_overfit = st.tabs([
    f"🔬 Single Month Curve ({MONTH_NAMES[inspect_month]} {inspect_year})",
    "📊 Overall Summary (1985–2014)",
    "⚖️ Best Epoch vs Epoch 10,000",
])

with tab_curve:
    run_curves = hist_curves[
        (hist_curves["model"].isin(selected_models))
        & (hist_curves["year"] == inspect_year)
        & (hist_curves["month"] == inspect_month)
    ].sort_values("epoch")

    run_sum = hist_sum[
        (hist_sum["model"].isin(selected_models))
        & (hist_sum["year"] == inspect_year)
        & (hist_sum["month"] == inspect_month)
    ].copy()
    run_sum["Model"] = run_sum["model"].map(lambda k: MODEL_SPECS[k]["label"])

    log_x = st.checkbox("Use Log Scale for Epoch Axis (easier to see early steps)", value=True)
    metric_pair = st.radio("Error Metric", ["MAE (mm)", "MSE Loss (mm²)"], horizontal=True)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Training Stations Error", "Test Stations Error (9 Stations)"],
        horizontal_spacing=0.08,
    )

    tr_col = "mae_train" if metric_pair.startswith("MAE") else "loss_train"
    te_col = "mae_test" if metric_pair.startswith("MAE") else "loss_test"

    for mk in selected_models:
        spec = MODEL_SPECS[mk]
        sub = run_curves[run_curves["model"] == mk]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["epoch"],
                y=sub[tr_col],
                mode="lines",
                name=spec["label"],
                line=dict(color=spec["color"], width=2),
                legendgroup=mk,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["epoch"],
                y=sub[te_col],
                mode="lines",
                name=spec["label"],
                line=dict(color=spec["color"], width=2),
                legendgroup=mk,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    if log_x:
        fig.update_xaxes(type="log", title_text="Epoch (log scale)")
    else:
        fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text=metric_pair)
    fig.update_layout(
        title=f"Training Error Curves — {MONTH_NAMES[inspect_month]} {inspect_year}",
        template="plotly_white",
        hovermode="x unified",
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Summary — {MONTH_NAMES[inspect_month]} {inspect_year}")
    st.dataframe(
        run_sum[[
            "Model", "total_epochs", "final_mae_train", "final_mae_test",
            "best_mae_test", "best_mae_test_epoch", "overfit_gap",
        ]].style.format({
            "final_mae_train": "{:.2f}",
            "final_mae_test": "{:.2f}",
            "best_mae_test": "{:.2f}",
            "overfit_gap": "{:+.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

with tab_global:
    agg_sum = (
        df_sum_sub.groupby(["model", "Model"], as_index=False)
        .agg(
            Runs=("year", "count"),
            Mean_Final_Train_MAE=("final_mae_train", "mean"),
            Mean_Final_Test_MAE=("final_mae_test", "mean"),
            Mean_Best_Test_MAE=("best_mae_test", "mean"),
            Median_Best_Epoch=("best_mae_test_epoch", "median"),
            Mean_Overfit_Gap=("overfit_gap", "mean"),
        )
    )
    agg_sum["order"] = agg_sum["model"].map({k: i for i, k in enumerate(MODEL_ORDER)})
    agg_sum = agg_sum.sort_values("order").drop(columns=["order", "model"])

    st.dataframe(
        agg_sum.style.format({
            "Mean_Final_Train_MAE": "{:.2f}",
            "Mean_Final_Test_MAE": "{:.2f}",
            "Mean_Best_Test_MAE": "{:.2f}",
            "Median_Best_Epoch": "{:.0f}",
            "Mean_Overfit_Gap": "{:+.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    c_g1, c_g2 = st.columns(2)
    with c_g1:
        fig_gap = px.box(
            df_sum_sub,
            x="Model",
            y="overfit_gap",
            color="Model",
            color_discrete_map={MODEL_SPECS[k]["label"]: MODEL_SPECS[k]["color"] for k in selected_models},
            title="Train vs Test Error Gap (Test MAE − Train MAE)",
            template="plotly_white",
        )
        fig_gap.update_layout(showlegend=False, height=400, yaxis_title="Error Gap (mm)")
        st.plotly_chart(fig_gap, use_container_width=True)

    with c_g2:
        fig_ep = px.histogram(
            df_sum_sub,
            x="best_mae_test_epoch",
            color="Model",
            nbins=40,
            barmode="overlay",
            opacity=0.65,
            color_discrete_map={MODEL_SPECS[k]["label"]: MODEL_SPECS[k]["color"] for k in selected_models},
            title="When Models Reach Best Test Accuracy (Epoch)",
            template="plotly_white",
        )
        fig_ep.update_layout(height=400, xaxis_title="Best Epoch", yaxis_title="Number of Monthly Runs")
        st.plotly_chart(fig_ep, use_container_width=True)

with tab_overfit:
    st.caption(
        "Compares the lowest Test MAE during training vs Test MAE at the final epoch (10,000). "
        "Points above the dashed line mean stopping earlier gives lower error."
    )
    fig_es = px.scatter(
        df_sum_sub,
        x="best_mae_test",
        y="final_mae_test",
        color="Model",
        hover_data=["year", "month", "best_mae_test_epoch", "final_mae_train"],
        color_discrete_map={MODEL_SPECS[k]["label"]: MODEL_SPECS[k]["color"] for k in selected_models},
        title=f"Best Test MAE vs Final Epoch Test MAE ({y_start}–{y_end})",
        template="plotly_white",
        opacity=0.65,
    )
    lim = max(df_sum_sub["final_mae_test"].max(), df_sum_sub["best_mae_test"].max()) * 1.05
    fig_es.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(dash="dash", color="#0F172A"))
    fig_es.update_layout(
        xaxis_title="Best Test MAE During Training (mm)",
        yaxis_title="Final Test MAE at Epoch 10,000 (mm)",
        height=470,
    )
    st.plotly_chart(fig_es, use_container_width=True)
    download_csv_button(df_sum_sub, "training_convergence_summary.csv")

render_footer()

