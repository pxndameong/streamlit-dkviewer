"""Page 4: Spatial Map, Model Difference & Interpolated Surface Viewer."""

from __future__ import annotations

import base64
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata
import streamlit as st

from src.config import APP_TITLE, DATA_DIR, MODEL_ORDER, MODEL_SPECS, MONTHS, MONTH_NAMES, YEARS
from src.data_loader import load_era5_grid, load_paired_obs, load_predictions
from src.theme import get_plotly_template, get_theme_palette
from src.ui import download_csv_button, render_chart, render_footer, render_header

st.set_page_config(
    page_title=f"{APP_TITLE} | Rainfall Maps",
    page_icon="🗺️",
    layout="wide",
)

render_header(
    "🗺️ Java Rainfall Maps",
    "View predicted rainfall across Java, compare models side-by-side, check differences between models, and inspect ERA5 weather variables.",
)

df_pred = load_predictions()
df_obs = load_paired_obs()
if df_pred.empty:
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
st.sidebar.header("Map Settings")
time_agg = st.sidebar.radio(
    "Time Period",
    ["Single Month & Year", "Monthly Average (1985–2014)", "30-Year Average (All Months)"],
    key="map_tmode",
)

sel_year = 2010
sel_month = 1
if time_agg == "Single Month & Year":
    c1, c2 = st.sidebar.columns(2)
    with c1:
        sel_year = st.selectbox("Year", YEARS, index=YEARS.index(2010), key="map_yr")
    with c2:
        sel_month = st.selectbox("Month", MONTHS, index=0, format_func=lambda m: MONTH_NAMES[m], key="map_mo")
elif time_agg == "Monthly Average (1985–2014)":
    sel_month = st.sidebar.selectbox("Month", MONTHS, index=0, format_func=lambda m: MONTH_NAMES[m], key="map_mo_clim")

view_mode = st.sidebar.radio(
    "Map View",
    [
        "Single Model Map",
        "Compare Models Side-by-Side",
        "Difference Between Models (A − B)",
        "Weather Variable Map (ERA5)",
    ],
    key="map_vmode",
)

render_style = st.sidebar.radio(
    "Map Style",
    ["Smooth Heatmap on Map", "Grid Boxes on Map"],
    key="map_render",
)
basemap_choice = st.sidebar.radio(
    "Base Map",
    ["Auto (Follow Theme)", "Dark Map (Carto)", "Light Map (Carto)", "Street Map (OSM)"],
    key="map_basemap",
)
if basemap_choice.startswith("Auto"):
    basemap_style = get_theme_palette()["map_style"]
elif basemap_choice.startswith("Dark"):
    basemap_style = "carto-darkmatter"
elif basemap_choice.startswith("Light"):
    basemap_style = "carto-positron"
else:
    basemap_style = "open-street-map"
overlay_obs = st.sidebar.checkbox("Show 9 Test Stations on Map", value=True, key="map_overlay")
mask_land = st.sidebar.checkbox("Mask Ocean (Daratan Saja)", value=True, key="map_mask_land")


@st.cache_data(show_spinner=False)
def _load_land_mask() -> np.ndarray | None:
    mask_file = DATA_DIR / "java_land_mask_240x100.npy"
    if mask_file.exists():
        return np.load(mask_file)
    return None


@st.cache_data(show_spinner=False)
def _load_land_coords() -> set[tuple[float, float]]:
    coords_file = DATA_DIR / "java_land_coords.json"
    if coords_file.exists():
        with open(coords_file, "r") as f:
            return {(round(float(lat), 2), round(float(lon), 2)) for lat, lon in json.load(f)}
    return set()

# Filter temporal slice
if time_agg == "Single Month & Year":
    pred_slice = df_pred[(df_pred["year"] == sel_year) & (df_pred["month"] == sel_month)].copy()
    obs_slice = df_obs[(df_obs["year"] == sel_year) & (df_obs["month"] == sel_month)].copy()
    time_label = f"{MONTH_NAMES[sel_month]} {sel_year}"
elif time_agg == "Monthly Average (1985–2014)":
    pred_slice = df_pred[df_pred["month"] == sel_month].groupby(["lat", "lon"], as_index=False)[
        [MODEL_SPECS[k]["pred_col"] for k in MODEL_ORDER]
    ].mean()
    obs_slice = df_obs[df_obs["month"] == sel_month].groupby(
        ["lat", "lon", "is_test_point", "point_id", "idx_new", "location_name"], as_index=False
    )["monthly_rainfall"].mean()
    time_label = f"{MONTH_NAMES[sel_month]} Average (1985–2014)"
else:
    pred_slice = df_pred.groupby(["lat", "lon"], as_index=False)[
        [MODEL_SPECS[k]["pred_col"] for k in MODEL_ORDER]
    ].mean()
    obs_slice = df_obs.groupby(
        ["lat", "lon", "is_test_point", "point_id", "idx_new", "location_name"], as_index=False
    )["monthly_rainfall"].mean()
    time_label = "30-Year Average (1985–2014)"


def _add_test_station_trace(fig: go.Figure) -> None:
    if overlay_obs and not obs_slice.empty:
        tp = obs_slice[obs_slice["is_test_point"]]
        fig.add_trace(go.Scattermap(
            lat=tp["lat"],
            lon=tp["lon"],
            mode="markers+text",
            marker=dict(size=12, color="#F59E0B"),
            text=[f"S{int(r['point_id'])} {int(r['idx_new'])}" for _, r in tp.iterrows()],
            textposition="top right",
            textfont=dict(size=11, color="#FDE68A" if basemap_style == "carto-darkmatter" else "#111827"),
            name="9 Test Stations",
            hovertext=[f"{r['location_name']}<br>Obs: {r['monthly_rainfall']:.1f} mm" for _, r in tp.iterrows()],
            hoverinfo="text",
        ))


def _plot_smooth_map(df_pts: pd.DataFrame, val_col: str, title: str, ztitle: str, cscale: str = "YlGnBu", symmetric: bool = False) -> go.Figure:
    lon_u = np.sort(df_pts["lon"].unique())
    lat_u = np.sort(df_pts["lat"].unique())
    lon_min, lon_max = float(lon_u.min()), float(lon_u.max())
    lat_min, lat_max = float(lat_u.min()), float(lat_u.max())

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(lon_min, lon_max, 240),
        np.linspace(lat_min, lat_max, 100),
    )
    z_interp = griddata(
        df_pts[["lon", "lat"]].values,
        df_pts[val_col].values,
        (grid_lon, grid_lat),
        method="cubic",
    )

    if mask_land:
        land_mask = _load_land_mask()
        if land_mask is not None and land_mask.shape == z_interp.shape:
            z_interp[~land_mask] = np.nan

    if symmetric:
        vmax = float(np.nanmax(np.abs(z_interp))) if np.isfinite(z_interp).any() else float(df_pts[val_col].abs().max())
        zmin_f, zmax_f = -vmax, vmax
    else:
        zmin_f = float(np.nanmin(z_interp)) if np.isfinite(z_interp).any() else float(df_pts[val_col].min())
        zmax_f = float(np.nanmax(z_interp)) if np.isfinite(z_interp).any() else float(df_pts[val_col].max())

    # Render interpolated surface to a transparent RGBA PNG placed over the geographic basemap
    z_flipped = z_interp[::-1, :]
    cmap_name = "viridis" if cscale == "Viridis" else cscale
    cmap = matplotlib.colormaps[cmap_name]
    norm = mcolors.Normalize(vmin=zmin_f, vmax=zmax_f if zmax_f > zmin_f else zmin_f + 1.0, clip=True)
    rgba = cmap(norm(np.nan_to_num(z_flipped, nan=zmin_f)))
    rgba[..., 3] = np.where(np.isnan(z_flipped), 0.0, 0.72)

    buf = io.BytesIO()
    mpimg.imsave(buf, rgba, format="png")
    img_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    # Filter hover markers to land only when mask_land is on
    df_markers = df_pts
    if mask_land:
        land_set = _load_land_coords()
        if land_set:
            df_markers = df_pts[df_pts.apply(lambda r: (round(float(r["lat"]), 2), round(float(r["lon"]), 2)) in land_set, axis=1)]

    fig = go.Figure()
    # Nearly invisible markers supply exact hover tooltips and the color bar
    fig.add_trace(go.Scattermap(
        lat=df_markers["lat"],
        lon=df_markers["lon"],
        mode="markers",
        marker=dict(
            size=10,
            color=df_markers[val_col],
            colorscale=cscale,
            cmin=zmin_f,
            cmax=zmax_f,
            showscale=True,
            colorbar=dict(title=ztitle),
            opacity=0.02,
        ),
        text=[f"({r['lat']:.2f}, {r['lon']:.2f})<br>{ztitle}: {r[val_col]:.2f}" for _, r in df_markers.iterrows()],
        hoverinfo="text",
        name=ztitle,
        showlegend=False,
    ))
    _add_test_station_trace(fig)

    fig.update_layout(
        title=title,
        template=get_plotly_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        map=dict(
            style=basemap_style,
            center=dict(lat=-7.25, lon=109.8),
            zoom=6.1,
            layers=[
                {
                    "sourcetype": "image",
                    "source": img_uri,
                    "coordinates": [
                        [lon_min, lat_max],
                        [lon_max, lat_max],
                        [lon_max, lat_min],
                        [lon_min, lat_min],
                    ],
                    "below": "traces",
                }
            ],
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=490,
    )
    return fig


@st.cache_data(show_spinner=False)
def _build_grid_geojson(lats: tuple[float, ...], lons: tuple[float, ...]) -> dict:
    """Build GeoJSON rectangular cell polygons centered at each grid coordinate."""
    lat_u = np.sort(np.unique(lats))
    lon_u = np.sort(np.unique(lons))
    d_lat = float(np.median(np.diff(lat_u))) if len(lat_u) > 1 else 0.25
    d_lon = float(np.median(np.diff(lon_u))) if len(lon_u) > 1 else 0.25
    half_lat = d_lat / 2.0
    half_lon = d_lon / 2.0

    features = []
    for idx, (lat, lon) in enumerate(zip(lats, lons)):
        lon0, lon1 = float(lon - half_lon), float(lon + half_lon)
        lat0, lat1 = float(lat - half_lat), float(lat + half_lat)
        features.append(
            {
                "type": "Feature",
                "id": str(idx),
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon0, lat0],
                        [lon1, lat0],
                        [lon1, lat1],
                        [lon0, lat1],
                        [lon0, lat0],
                    ]],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _plot_grid_box_map(df_pts: pd.DataFrame, val_col: str, title: str, ztitle: str, cscale: str = "YlGnBu", symmetric: bool = False) -> go.Figure:
    if mask_land:
        land_set = _load_land_coords()
        if land_set:
            df_pts = df_pts[df_pts.apply(lambda r: (round(float(r["lat"]), 2), round(float(r["lon"]), 2)) in land_set, axis=1)]

    df_valid = df_pts.dropna(subset=["lat", "lon", val_col]).reset_index(drop=True)
    zmin, zmax = None, None
    if symmetric and not df_valid.empty:
        vmax = float(df_valid[val_col].abs().max())
        zmin, zmax = -vmax, vmax

    geojson = _build_grid_geojson(
        tuple(df_valid["lat"].astype(float)),
        tuple(df_valid["lon"].astype(float)),
    )
    locations = [str(i) for i in range(len(df_valid))]
    border_color = "rgba(255, 255, 255, 0.25)" if basemap_style == "carto-darkmatter" else "rgba(15, 23, 42, 0.25)"

    fig = go.Figure()
    fig.add_trace(go.Choroplethmap(
        geojson=geojson,
        locations=locations,
        z=df_valid[val_col],
        colorscale=cscale,
        zmin=zmin,
        zmax=zmax,
        marker_opacity=0.72,
        marker_line_width=0.6,
        marker_line_color=border_color,
        colorbar=dict(title=ztitle),
        text=[f"({r['lat']:.2f}, {r['lon']:.2f})<br>{ztitle}: {r[val_col]:.2f}" for _, r in df_valid.iterrows()],
        hoverinfo="text",
        name=ztitle,
        showlegend=False,
    ))
    _add_test_station_trace(fig)
    fig.update_layout(
        title=title,
        template=get_plotly_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        map=dict(style=basemap_style, center=dict(lat=-7.25, lon=109.8), zoom=6.1),
        margin=dict(l=0, r=0, t=40, b=0),
        height=490,
    )
    return fig


# ---------------------------------------------------------------------------
# Render Chosen Map View
# ---------------------------------------------------------------------------
if view_mode == "Single Model Map":
    mk = st.sidebar.selectbox("Select Model", MODEL_ORDER, index=4, format_func=lambda k: MODEL_SPECS[k]["label"])
    pcol = MODEL_SPECS[mk]["pred_col"]
    title = f"{MODEL_SPECS[mk]['label']} — {time_label}"
    fig = _plot_smooth_map(pred_slice, pcol, title, "Rainfall (mm)") if render_style.startswith("Smooth") else _plot_grid_box_map(pred_slice, pcol, title, "Rainfall (mm)")
    render_chart(fig)

    # Spatial summary KPIs
    kpi_slice = pred_slice
    if mask_land:
        land_set = _load_land_coords()
        if land_set:
            kpi_slice = pred_slice[pred_slice.apply(lambda r: (round(float(r["lat"]), 2), round(float(r["lon"]), 2)) in land_set, axis=1)]

    lbl_suffix = " (Land)" if mask_land else ""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Map Average{lbl_suffix}", f"{kpi_slice[pcol].mean():.1f} mm")
    c2.metric(f"Map Max{lbl_suffix}", f"{kpi_slice[pcol].max():.1f} mm")
    c3.metric(f"Map Min{lbl_suffix}", f"{kpi_slice[pcol].min():.1f} mm")
    c4.metric("Spread (Std Dev)", f"{kpi_slice[pcol].std():.1f} mm")

    if not obs_slice.empty:
        merged_check = obs_slice.merge(pred_slice[["lat", "lon", pcol]], on=["lat", "lon"], how="left")
        merged_check["error"] = merged_check[pcol] - merged_check["monthly_rainfall"]
        st.subheader("Test Station Check (Predicted vs Observed)")
        st.dataframe(
            merged_check[merged_check["is_test_point"]][["point_id", "idx_new", "location_name", "lat", "lon", "monthly_rainfall", pcol, "error"]].sort_values("point_id"),
            use_container_width=True,
            hide_index=True,
        )
    download_csv_button(pred_slice, f"spatial_grid_{mk}_{time_label.replace(' ', '_')}.csv")

elif view_mode == "Compare Models Side-by-Side":
    models_cmp = st.sidebar.multiselect(
        "Models to Compare",
        MODEL_ORDER,
        default=["1var", "5var", "20var"],
        format_func=lambda k: MODEL_SPECS[k]["label"],
    )
    if not models_cmp:
        st.warning("Select at least one model.")
        st.stop()
    cols = st.columns(min(len(models_cmp), 2))
    for i, mk in enumerate(models_cmp):
        pcol = MODEL_SPECS[mk]["pred_col"]
        title = f"{MODEL_SPECS[mk]['short']} — {time_label}"
        with cols[i % 2]:
            fig = _plot_smooth_map(pred_slice, pcol, title, "Rainfall (mm)") if render_style.startswith("Smooth") else _plot_grid_box_map(pred_slice, pcol, title, "Rainfall (mm)")
            render_chart(fig)

elif view_mode == "Difference Between Models (A − B)":
    c1, c2 = st.sidebar.columns(2)
    with c1:
        m_a = st.selectbox("Model A", MODEL_ORDER, index=4, format_func=lambda k: MODEL_SPECS[k]["short"])
    with c2:
        m_b = st.selectbox("Model B", MODEL_ORDER, index=0, format_func=lambda k: MODEL_SPECS[k]["short"])

    diff_df = pred_slice[["lat", "lon"]].copy()
    diff_df["diff"] = pred_slice[MODEL_SPECS[m_a]["pred_col"]] - pred_slice[MODEL_SPECS[m_b]["pred_col"]]
    title = f"Difference: {MODEL_SPECS[m_a]['short']} − {MODEL_SPECS[m_b]['short']} ({time_label})"
    fig = (
        _plot_smooth_map(diff_df, "diff", title, "Diff (mm)", cscale="RdBu_r", symmetric=True)
        if render_style.startswith("Smooth")
        else _plot_grid_box_map(diff_df, "diff", title, "Diff (mm)", cscale="RdBu_r", symmetric=True)
    )
    render_chart(fig)
    st.caption(
        f"Average difference across map: **{diff_df['diff'].mean():+.2f} mm** | "
        f"Range: **[{diff_df['diff'].min():+.2f}, {diff_df['diff'].max():+.2f}] mm**"
    )

else:
    df_era = load_era5_grid()
    if df_era.empty:
        st.stop()
    from src.config import EKSOGEN_20_COLS, EXOGENOUS_INFO
    cov_choice = st.sidebar.selectbox(
        "Weather Variable (ERA5)",
        EKSOGEN_20_COLS,
        format_func=lambda c: f"{c} — {EXOGENOUS_INFO[c]['name']}",
    )
    if time_agg == "Single Month & Year":
        era_slice = df_era[(df_era["year"] == sel_year) & (df_era["month"] == sel_month)].copy()
    elif time_agg == "Monthly Average (1985–2014)":
        era_slice = df_era[df_era["month"] == sel_month].groupby(["lat", "lon"], as_index=False)[cov_choice].mean()
    else:
        era_slice = df_era.groupby(["lat", "lon"], as_index=False)[cov_choice].mean()

    unit = EXOGENOUS_INFO[cov_choice]["unit"]
    title = f"ERA5 {cov_choice} ({EXOGENOUS_INFO[cov_choice]['name']}) — {time_label}"
    fig = (
        _plot_smooth_map(era_slice, cov_choice, title, unit, cscale="Viridis")
        if render_style.startswith("Smooth")
        else _plot_grid_box_map(era_slice, cov_choice, title, unit, cscale="Viridis")
    )
    render_chart(fig)

render_footer()

