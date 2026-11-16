"""Reusable Plotly chart builders with consistent scientific styling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import MODEL_SPECS, OBS_COLOR
from src.metrics import compute_metrics

PLOT_TEMPLATE = "plotly_dark"


def make_scatter_comparison(
    df: pd.DataFrame,
    model_keys: list[str],
    obs_col: str = "monthly_rainfall",
    title: str = "Observed vs Predicted Monthly Rainfall",
) -> go.Figure:
    n = len(model_keys)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[MODEL_SPECS[k]["label"] for k in model_keys],
        horizontal_spacing=0.07,
        vertical_spacing=0.14,
    )

    max_val = float(df[obs_col].max()) if not df.empty else 500.0
    for k in model_keys:
        pcol = MODEL_SPECS[k]["pred_col"]
        if pcol in df.columns and not df[pcol].isna().all():
            max_val = max(max_val, float(df[pcol].max()))
    limit = max(max_val * 1.05, 100.0)

    for idx, mk in enumerate(model_keys):
        r = idx // cols + 1
        c = idx % cols + 1
        spec = MODEL_SPECS[mk]
        sub = df[[obs_col, spec["pred_col"]]].dropna()
        m = compute_metrics(sub[obs_col], sub[spec["pred_col"]])

        # Subsample points if huge so browser stays snappy
        if len(sub) > 5000:
            sub_plot = sub.sample(5000, random_state=42)
        else:
            sub_plot = sub

        fig.add_trace(
            go.Scattergl(
                x=sub_plot[obs_col],
                y=sub_plot[spec["pred_col"]],
                mode="markers",
                marker=dict(color=spec["color"], size=5, opacity=0.5),
                name=spec["short"],
                showlegend=False,
                hovertemplate="Obs: %{x:.1f} mm<br>Pred: %{y:.1f} mm<extra></extra>",
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=[0, limit],
                y=[0, limit],
                mode="lines",
                line=dict(color="#CBD5E1", dash="dash", width=1.2),
                showlegend=False,
            ),
            row=r,
            col=c,
        )

        # Linear regression fit line
        if len(sub) > 2:
            slope, intercept = np.polyfit(sub[obs_col], sub[spec["pred_col"]], 1)
            fig.add_trace(
                go.Scatter(
                    x=[0, limit],
                    y=[intercept, slope * limit + intercept],
                    mode="lines",
                    line=dict(color=spec["color"], width=1.5),
                    showlegend=False,
                ),
                row=r,
                col=c,
            )

        annot_text = (
            f"MAE = {m['MAE']:.1f} mm<br>"
            f"RMSE = {m['RMSE']:.1f} mm<br>"
            f"R² = {m['R2']:.3f} | KGE = {m['KGE']:.3f}"
        )
        xref = "x domain" if idx == 0 else f"x{idx + 1} domain"
        yref = "y domain" if idx == 0 else f"y{idx + 1} domain"
        fig.add_annotation(
            x=0.97,
            y=0.04,
            xref=xref,
            yref=yref,
            text=annot_text,
            showarrow=False,
            align="right",
            bgcolor="rgba(15, 23, 42, 0.88)",
            bordercolor="#475569",
            borderwidth=1,
            font=dict(size=10, color="#F8FAFC"),
        )
        fig.update_xaxes(title_text="Observed (mm)", range=[0, limit], row=r, col=c)
        fig.update_yaxes(title_text="Predicted (mm)", range=[0, limit], row=r, col=c)

    fig.update_layout(
        title=title,
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.35)",
        height=390 * rows,
        margin=dict(t=60, b=40, l=40, r=20),
    )
    return fig


def make_timeseries_figure(
    df_agg: pd.DataFrame,
    model_keys: list[str],
    obs_col: str = "monthly_rainfall",
    show_obs: bool = True,
    title: str = "Monthly Rainfall Time Series",
) -> go.Figure:
    fig = go.Figure()
    if show_obs and obs_col in df_agg.columns:
        fig.add_trace(
            go.Scatter(
                x=df_agg["date"],
                y=df_agg[obs_col],
                mode="lines+markers",
                name="Observed Rainfall",
                line=dict(color=OBS_COLOR, width=2.4),
                marker=dict(size=4),
            )
        )
    for mk in model_keys:
        spec = MODEL_SPECS[mk]
        fig.add_trace(
            go.Scatter(
                x=df_agg["date"],
                y=df_agg[spec["pred_col"]],
                mode="lines",
                name=spec["label"],
                line=dict(color=spec["color"], width=1.8),
            )
        )
    fig.update_layout(
        title=title,
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.35)",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Monthly Rainfall (mm)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=450,
    )
    return fig

