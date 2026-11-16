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
from src.ui import download_csv_button, render_footer, render_header

st.set_page_config(
    page_title=f"{APP_TITLE} | Raw Data & SHAP",
    page_icon="🔬",
    layout="wide",
)

render_header(
    "🔬 Raw Stations, Exogenous Covariates & SHAP Explorer",
    "Inspect raw BMKG observation stations, KDTree grid pairing geometry, ERA5 covariate correlations, and SHAP feature importance rankings.",
)

catalog = load_station_catalog()
df_raw = load_raw_stations()
df_paired = load_paired_with_predictions()
shap_df = load_shap_ranking()

tab_raw, tab_cov, tab_shap = st.tabs([
    "🛰️ Raw BMKG Stations & KDTree Pairing",
    "🌦️ Exogenous Covariates vs Rainfall & Error",
    "🧠 SHAP Global Feature Importance (51 Var)",
])

# ---------------------------------------------------------------------------
# Tab 1: Raw Stations & KDTree Pairing
# ---------------------------------------------------------------------------
with tab_raw:
    if catalog.empty or df_raw.empty:
        st.warning("Raw station dataset not available.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unique Station Records", f"{len(catalog):,}")
        c2.metric("Unique Station Names", f"{catalog['stanam'].nunique():,}")
        c3.metric("Paired ERA5 Grid Cells", f"{catalog['grid_idx'].nunique():,}")
        c4.metric("Median Pairing Distance", f"{catalog['pair_dist_km'].median():.1f} km")

        filter_col1, filter_col2 = st.columns([1, 2])
        with filter_col1:
            test_grid_only = st.checkbox("Filter to Stations Paired to the 9 Hold-Out Test Grids", value=False)
            reg_filter = st.multiselect(
                "Region",
                ["West Java & Banten", "Central Java & DIY", "East Java"],
                default=["West Java & Banten", "Central Java & DIY", "East Java"],
            )
        with filter_col2:
            search_txt = st.text_input("Search Station Name (`stanam`) or Test Location", "")

        cat_filt = catalog[catalog["region"].isin(reg_filter)].copy()
        if test_grid_only:
            cat_filt = cat_filt[cat_filt["is_test_grid"]]
        if search_txt.strip():
            q = search_txt.strip().lower()
            cat_filt = cat_filt[
                cat_filt["stanam"].str.lower().str.contains(q)
                | cat_filt["test_location_name"].str.lower().str.contains(q)
            ]

        st.subheader(f"Station Catalog ({len(cat_filt):,} entries)")
        st.dataframe(
            cat_filt.sort_values(["is_test_grid", "grid_idx", "stanam"], ascending=[False, True, True]),
            use_container_width=True,
            hide_index=True,
        )
        download_csv_button(cat_filt, "bmkg_station_catalog_filtered.csv")

        # Interactive drilldown into a specific station or test grid
        st.markdown("---")
        st.subheader("Raw Station Drilldown (Time Series Before Grid Aggregation)")
        drill_mode = st.radio(
            "Select By",
            ["Hold-Out Test Grid (Shows All Raw Stations Inside Grid)", "Individual BMKG Station"],
            horizontal=True,
        )
        if drill_mode.startswith("Hold-Out"):
            tp_sel = st.selectbox(
                "Select Hold-Out Grid",
                [f"P{p['point_id']}: {p['location_name']}" for p in TEST_POINTS],
            )
            pid = int(tp_sel.split(":")[0].replace("P", ""))
            sub_raw = df_raw[df_raw["point_id"] == pid].copy()
            st.caption(
                f"Raw stations mapped into Grid P{pid}: **{', '.join(sorted(sub_raw['stanam'].unique()))}**"
            )
            sub_raw["date"] = pd.to_datetime(sub_raw[["year", "month"]].assign(day=1))
            fig_raw = px.line(
                sub_raw.sort_values("date"),
                x="date",
                y="monthly_rainfall",
                color="stanam",
                title=f"Individual Raw Station Series Mapped to {tp_sel}",
                template="plotly_white",
            )
            st.plotly_chart(fig_raw, use_container_width=True)
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
                    template="plotly_white",
                )
                st.plotly_chart(fig_raw, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Exogenous Covariate Explorer
# ---------------------------------------------------------------------------
with tab_cov:
    if df_paired.empty:
        st.warning("Paired dataset unavailable.")
    else:
        st.subheader("ERA5 Exogenous Relationship with Observed Rainfall & 20-Var Error")
        c1, c2 = st.columns(2)
        with c1:
            cov_x = st.selectbox(
                "Select Exogenous Feature",
                EKSOGEN_20_COLS,
                format_func=lambda c: f"{c} — {EXOGENOUS_INFO[c]['name']}",
            )
        with c2:
            y_target = st.selectbox(
                "Target Y Variable",
                ["monthly_rainfall", "pred_20var", "pred_1var", "Error_20var (pred - obs)", "Abs_Error_20var"],
            )

        sample_df = df_paired[df_paired["is_test_point"]].copy()
        sample_df["Error_20var (pred - obs)"] = sample_df["pred_20var"] - sample_df["monthly_rainfall"]
        sample_df["Abs_Error_20var"] = sample_df["Error_20var (pred - obs)"].abs()

        # Pearson correlation table across all 20 covariates
        corr_rows = []
        for col in EKSOGEN_20_COLS:
            corr_obs = sample_df[col].corr(sample_df["monthly_rainfall"])
            corr_err = sample_df[col].corr(sample_df["Abs_Error_20var"])
            corr_rows.append({
                "Feature": col,
                "Description": EXOGENOUS_INFO[col]["name"],
                "Unit": EXOGENOUS_INFO[col]["unit"],
                "Corr w/ Observed Rain (r)": corr_obs,
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
                title=f"{cov_x} vs {y_target} (9 Hold-Out Points, 1985–2014)",
                template="plotly_white",
            )
            fig_cov.update_layout(
                xaxis_title=f"{cov_x} [{EXOGENOUS_INFO[cov_x]['unit']}]",
                height=440,
            )
            st.plotly_chart(fig_cov, use_container_width=True)
        with col_r:
            st.markdown("#### Feature Correlation Summary (Test Points)")
            st.dataframe(
                df_corr.style.format({
                    "Corr w/ Observed Rain (r)": "{:+.3f}",
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
            "Global Mean |SHAP| ranking across all 51 ERA5 atmospheric features, showing how the 1-Var, 2-Var, 5-Var, 9-Var, and 20-Var models select top predictors."
        )
        top_n = st.slider("Show Top-N Features", 10, len(shap_df), 25, step=5)
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
            title=f"Top {top_n} Global SHAP Feature Importance & Model Inclusion Tiers",
            template="plotly_white",
            color_discrete_map={
                "1-Var (OLR)": "#64748B",
                "2-Var": "#2563EB",
                "5-Var": "#F59E0B",
                "9-Var": "#10B981",
                "20-Var": "#DC2626",
                "Not in Top-20": "#CBD5E1",
            },
        )
        fig_shap.update_layout(height=max(420, top_n * 22), xaxis_title="Mean |SHAP Value|")
        st.plotly_chart(fig_shap, use_container_width=True)
        download_csv_button(shap_df, "global_shap_ranking_all51.csv")

render_footer()
