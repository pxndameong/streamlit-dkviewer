"""Cached Parquet data loaders for DK Viewer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DATA_DIR


def _read_parquet(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        st.error(f"Dataset file not found: `{path}`. Run `python tools/build_dataset.py` first.")
        return pd.DataFrame()
    return pd.read_parquet(path, engine="pyarrow")


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    path = DATA_DIR / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Loading model predictions...")
def load_predictions() -> pd.DataFrame:
    df = _read_parquet("predictions.parquet")
    if not df.empty:
        df["lat"] = df["lat"].round(4)
        df["lon"] = df["lon"].round(4)
        for col in df.columns:
            if col.startswith("pred_"):
                df[col] = df[col].clip(lower=0.0)
    return df


@st.cache_data(show_spinner="Loading paired station observations...")
def load_paired_obs() -> pd.DataFrame:
    df = _read_parquet("paired_obs.parquet")
    if not df.empty:
        df["lat"] = df["lat"].round(4)
        df["lon"] = df["lon"].round(4)
    return df


@st.cache_data(show_spinner="Joining observations with 5 DeepKriging models...")
def load_paired_with_predictions() -> pd.DataFrame:
    obs = load_paired_obs()
    pred = load_predictions()
    if obs.empty or pred.empty:
        return pd.DataFrame()
    merged = obs.merge(
        pred,
        on=["year", "month", "lat", "lon"],
        how="inner",
    )
    merged["date"] = pd.to_datetime(merged[["year", "month"]].assign(day=1))
    return merged


@st.cache_data(show_spinner="Loading observation station records...")
def load_raw_stations() -> pd.DataFrame:
    return _read_parquet("raw_stations.parquet")


@st.cache_data(show_spinner=False)
def load_station_catalog() -> pd.DataFrame:
    return _read_parquet("station_catalog.parquet")


@st.cache_data(show_spinner="Loading ERA5 weather variables...")
def load_era5_grid() -> pd.DataFrame:
    df = _read_parquet("era5_grid_20var.parquet")
    if not df.empty:
        df["lat"] = df["lat"].round(4)
        df["lon"] = df["lon"].round(4)
    return df


@st.cache_data(show_spinner=False)
def load_history_summary() -> pd.DataFrame:
    return _read_parquet("history_summary.parquet")


@st.cache_data(show_spinner="Loading training convergence curves...")
def load_history_curves() -> pd.DataFrame:
    return _read_parquet("history_curves.parquet")


@st.cache_data(show_spinner=False)
def load_shap_ranking() -> pd.DataFrame:
    return _read_parquet("shap_ranking.parquet")

