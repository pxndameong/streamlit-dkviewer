"""Page 6: Raw BMKG Stations, KDTree Pairing, Exogenous Covariates & SHAP Importance."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import APP_TITLE, EKSOGEN_20_COLS, EXOGENOUS_INFO, TEST_POINTS
from src.data_loader import (
    load_paired_with_predictions,
    load_raw_stations,
    load_shap_ranking,
    load_station_catalog,
)
from src.theme import get_plotly_template
from src.ui import download_csv_button, render_chart, render_footer, render_header

st.set_page_config(
    page_title=f"{APP_TITLE} | Stations & Variables",
    page_icon="🔬",
    layout="wide",
)

render_header(
    "🔬 BMKG Stations, Weather Variables & Feature Importance",
    "Explore raw BMKG station records, how each station matches the nearest grid cell, ERA5 weather variables, and top predictors (SHAP).",
)

catalog = load_station_catalog()
df_raw = load_raw_stations()
df_paired = load_paired_with_predictions()
shap_df = load_shap_ranking()

tab_raw, tab_cov, tab_shap = st.tabs([
    "🛰️ BMKG Stations",
    "🌦️ Weather Variables vs Rainfall",
    "🧠 Most Important Variables (SHAP)",
])

# ---------------------------------------------------------------------------
# Tab 1: BMKG Stations & Nearest Grid
# ---------------------------------------------------------------------------
with tab_raw:
    if catalog.empty or df_raw.empty:
        st.warning("Raw station dataset not available.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Station Records", f"{len(catalog):,}")
        c2.metric("Unique Stations", f"{catalog['stanam'].nunique():,}")
        c3.metric("Matched Grid Cells", f"{catalog['grid_idx'].nunique():,}")
        c4.metric("Typical Distance to Grid", f"{catalog['pair_dist_km'].median():.1f} km")

        filter_col1, filter_col2 = st.columns([1, 2])
        with filter_col1:
            test_grid_only = st.checkbox("Only Show Stations Inside the 9 Test Locations", value=False)
            reg_filter = st.multiselect(
                "Region",
                ["West Java & Banten", "Central Java & DIY", "East Java"],
                default=["West Java & Banten", "Central Java & DIY", "East Java"],
            )
        with filter_col2:
            search_txt = st.text_input("Search Station Name (`stanam`) or City", "")

        cat_filt = catalog[catalog["region"].isin(reg_filter)].copy()
        if test_grid_only:
            cat_filt = cat_filt[cat_filt["is_test_grid"]]
        if search_txt.strip():
            q = search_txt.strip().lower()
            cat_filt = cat_filt[
                cat_filt["stanam"].str.lower().str.contains(q)
                | cat_filt["test_location_name"].str.lower().str.contains(q)
            ]

        st.subheader(f"Station List ({len(cat_filt):,} stations)")
        st.dataframe(
            cat_filt.sort_values(["is_test_grid", "grid_idx", "stanam"], ascending=[False, True, True]),
            use_container_width=True,
            hide_index=True,
        )
        download_csv_button(cat_filt, "bmkg_station_catalog_filtered.csv")

        # Interactive drilldown into a specific station or test grid
        st.markdown("---")
        st.subheader("View Station Rainfall Over Time")
        drill_mode = st.radio(
            "View By",
            ["Test Station Area (All BMKG Stations Inside)", "Single BMKG Station"],
            horizontal=True,
        )
        if drill_mode.startswith("Test Station"):
            tp_sel = st.selectbox(
                "Select Test Station",
                [p["short"] for p in TEST_POINTS],
            )
            pid = int(tp_sel.split(" ")[0].replace("S", ""))
            sub_raw = df_raw[df_raw["point_id"] == pid].copy()
            st.caption(
                f"BMKG stations inside Station {pid}: **{', '.join(sorted(sub_raw['stanam'].unique()))}**"
            )
            sub_raw["date"] = pd.to_datetime(sub_raw[["year", "month"]].assign(day=1))
            fig_raw = px.line(
                sub_raw.sort_values("date"),
                x="date",
                y="monthly_rainfall",
                color="stanam",
                title=f"Raw BMKG Station Rainfall — {tp_sel}",
                template=get_plotly_template(),
            )
            fig_raw.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            render_chart(fig_raw)
        else:
            st_opts = sorted(cat_filt["stanam"].unique())
            if st_opts:
                st_choice = st.selectbox("Select Station Name", st_opts)
                sub_raw = df_raw[df_raw["stanam"] == st_choice].copy()
                sub_raw["date"] = pd.to_datetime(sub_raw[["year", "month"]].assign(day=1))
                fig_raw = px.line(
                    sub_raw.sort_values("date"),
                    x="date",
                    y="monthly_rainfall",
                    markers=True,
                    title=f"Raw Monthly Rainfall — {st_choice} (Elev: {sub_raw['elev'].iloc[0]:.0f} m)",
                    template=get_plotly_template(),
                )
                fig_raw.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                render_chart(fig_raw)

# ---------------------------------------------------------------------------
# Tab 2: Weather Variables Explorer
# ---------------------------------------------------------------------------
with tab_cov:
    if df_paired.empty:
        st.warning("Dataset unavailable.")
    else:
        st.subheader("How Weather Variables Relate to Rainfall & Error")
        c1, c2 = st.columns(2)
        with c1:
            cov_x = st.selectbox(
                "Select Weather Variable (X-Axis)",
                EKSOGEN_20_COLS,
                format_func=lambda c: f"{c} — {EXOGENOUS_INFO[c]['name']}",
            )
        with c2:
            y_target = st.selectbox(
                "Select Target (Y-Axis)",
                ["monthly_rainfall", "pred_20var", "pred_1var", "Error_20var (pred - obs)", "Abs_Error_20var"],
            )

        sample_df = df_paired[df_paired["is_test_point"]].copy()
        sample_df["Error_20var (pred - obs)"] = sample_df["pred_20var"] - sample_df["monthly_rainfall"]
        sample_df["Abs_Error_20var"] = sample_df["Error_20var (pred - obs)"].abs()

        # Correlation table across all 20 weather variables
        corr_rows = []
        for col in EKSOGEN_20_COLS:
            corr_obs = sample_df[col].corr(sample_df["monthly_rainfall"])
            corr_err = sample_df[col].corr(sample_df["Abs_Error_20var"])
            corr_rows.append({
                "Variable": col,
                "Description": EXOGENOUS_INFO[col]["name"],
                "Unit": EXOGENOUS_INFO[col]["unit"],
                "Corr w/ Rainfall (r)": corr_obs,
                "Corr w/ 20-Var |Error| (r)": corr_err,
            })
        df_corr = pd.DataFrame(corr_rows)

        col_l, col_r = st.columns([1.3, 1])
        with col_l:
            fig_cov = px.scatter(
                sample_df,
                x=cov_x,
                y=y_target,
                color="location_name",
                opacity=0.55,
                title=f"{cov_x} vs {y_target} (9 Test Stations, 1985–2014)",
                template=get_plotly_template(),
            )
            fig_cov.update_layout(
                xaxis_title=f"{cov_x} [{EXOGENOUS_INFO[cov_x]['unit']}]",
                height=440,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            render_chart(fig_cov)
        with col_r:
            st.markdown("#### Variable Correlation Summary (Test Stations)")
            st.dataframe(
                df_corr.style.format({
                    "Corr w/ Rainfall (r)": "{:+.3f}",
                    "Corr w/ 20-Var |Error| (r)": "{:+.3f}",
                }),
                use_container_width=True,
                hide_index=True,
                height=400,
            )

# ---------------------------------------------------------------------------
# Tab 3: SHAP Importance Ranking
# ---------------------------------------------------------------------------
with tab_shap:
    if shap_df.empty:
        st.warning("SHAP ranking table not found.")
    else:
        st.caption(
            "Importance ranking of all 51 ERA5 weather variables, showing which variables are used in the 1-Var, 2-Var, 5-Var, 9-Var, and 20-Var models."
        )
        top_n = st.slider("Number of Top Variables to Show", 10, len(shap_df), 25, step=5)
        sub_shap = shap_df.head(top_n).copy()
        sub_shap["Tier"] = "Not in Top-20"
        sub_shap.loc[sub_shap["in_20var"], "Tier"] = "20-Var"
        sub_shap.loc[sub_shap["in_9var"], "Tier"] = "9-Var"
        sub_shap.loc[sub_shap["in_5var"], "Tier"] = "5-Var"
        sub_shap.loc[sub_shap["in_2var"], "Tier"] = "2-Var"
        sub_shap.loc[sub_shap["in_1var"], "Tier"] = "1-Var (OLR)"

        fig_shap = px.bar(
            sub_shap.iloc[::-1],
            x="Mean_Abs_SHAP",
            y="Feature",
            color="Tier",
            orientation="h",
            title=f"Top {top_n} Most Important Weather Variables (SHAP)",
            template=get_plotly_template(),
            color_discrete_map={
                "1-Var (OLR)": "#64748B",
                "2-Var": "#2563EB",
                "5-Var": "#F59E0B",
                "9-Var": "#10B981",
                "20-Var": "#DC2626",
                "Not in Top-20": "#52525B",
            },
        )
        fig_shap.update_layout(
            height=max(420, top_n * 22),
            xaxis_title="Mean |SHAP Value|",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        render_chart(fig_shap)
        download_csv_button(shap_df, "global_shap_ranking_all51.csv")

render_footer()
