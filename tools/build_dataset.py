#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETL Pipeline for DeepKriging Java Rainfall Viewer (DK Viewer).

Converts raw training outputs from HASIL VER 5 (1var, 2var, 5var, 9var, 20var),
raw station rainfall observations, raw ERA5 0.25-deg grid covariates, and SHAP
importance rankings into compact, zstd-compressed Parquet datasets ready for
GitHub and Streamlit deployment.

Usage:
    python tools/build_dataset.py
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import spatial

# ---------------------------------------------------------------------------
# Source paths (raw workspace on local server)
# ---------------------------------------------------------------------------
MODEL_DIRS = {
    "1var": Path("/home/deepkriging/tsaqib/1_2026_DKNEW/1var_olr/HASIL VER 5"),
    "2var": Path("/home/deepkriging/tsaqib/2_2026_DKNEWEST/2var_newest/HASIL VER 5"),
    "5var": Path("/home/deepkriging/tsaqib/2_2026_DKNEWEST/5var_newest/HASIL VER 5"),
    "9var": Path("/home/deepkriging/tsaqib/2_2026_DKNEWEST/9var_newest/HASIL VER 5"),
    "20var": Path("/home/deepkriging/tsaqib/2_2026_DKNEWEST/20var_newest/HASIL VER 5"),
}

RAW_OBS_FILE = Path("/home/deepkriging/data/sum_bulanan_rainfall_1.txt")
ERA5_DIR = Path("/home/deepkriging/data/DATA BULANAN ERA5 51 VAR 1 TP 1985_2024")
SHAP_CSV = Path(
    "/home/deepkriging/tsaqib/2_2026_DKNEWEST/SHAP_RESULT_ALLVAR/GLOBAL_SHAP_Importance_Ranking_ALL51.csv"
)

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(1985, 2015))
MONTHS = list(range(1, 13))

EKSOGEN_20_COLS = [
    "olr", "q_200", "mcc", "v_1000", "t_200",
    "tcrw", "q_925", "r_925", "u_1000", "u_850",
    "u_600", "w_925", "slhf", "t2m", "u_200",
    "msl", "v_850", "w_850", "v_200", "hcc",
]

MODEL_FEATURE_COUNTS = {
    "1var": 1,
    "2var": 2,
    "5var": 5,
    "9var": 9,
    "20var": 20,
}

# 9 Independent Hold-out Test Points (points_to_remove in training scripts)
TEST_POINTS = [
    {"point_id": 1, "lon": 106.50, "lat": -6.25, "location_name": "Tangerang / Bogor (West Java)"},
    {"point_id": 2, "lon": 107.00, "lat": -6.75, "location_name": "Cianjur / Bandung (West Java)"},
    {"point_id": 3, "lon": 107.50, "lat": -7.25, "location_name": "Garut / Tasikmalaya (West Java)"},
    {"point_id": 4, "lon": 110.00, "lat": -7.00, "location_name": "Semarang (Central Java)"},
    {"point_id": 5, "lon": 110.00, "lat": -7.50, "location_name": "Sleman / Yogyakarta (DIY)"},
    {"point_id": 6, "lon": 111.75, "lat": -7.75, "location_name": "Madiun / Ponorogo (East Java)"},
    {"point_id": 7, "lon": 112.25, "lat": -7.00, "location_name": "Gresik / Surabaya (East Java)"},
    {"point_id": 8, "lon": 112.25, "lat": -8.25, "location_name": "South Malang (East Java)"},
    {"point_id": 9, "lon": 113.50, "lat": -8.00, "location_name": "Jember / Lumajang (East Java)"},
]
TEST_POINT_MAP = {(round(p["lon"], 4), round(p["lat"], 4)): p for p in TEST_POINTS}


def assign_region(lon: float) -> str:
    if lon < 108.75:
        return "West Java & Banten"
    if lon < 111.00:
        return "Central Java & DIY"
    return "East Java"


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_predictions() -> dict:
    print("[1/6] Building predictions.parquet (wide table for 5 models, 1985-2014)...")
    t0 = time.time()
    yearly_frames = []

    for year in YEARS:
        for month in MONTHS:
            merged = None
            for model_key, base_dir in MODEL_DIRS.items():
                fp = base_dir / str(year) / f"HASIL_ch_pred_{month}-{year}.csv"
                if not fp.exists():
                    raise FileNotFoundError(f"Missing prediction file: {fp}")
                df = pd.read_csv(fp)
                df = df.rename(columns={
                    "longitude": "lon",
                    "latitude": "lat",
                    "ch_pred": f"pred_{model_key}",
                })
                df["lat"] = df["lat"].round(4).astype(np.float32)
                df["lon"] = df["lon"].round(4).astype(np.float32)
                df[f"pred_{model_key}"] = df[f"pred_{model_key}"].astype(np.float32)

                if merged is None:
                    merged = df[["lat", "lon", f"pred_{model_key}"]]
                else:
                    merged = merged.merge(
                        df[["lat", "lon", f"pred_{model_key}"]],
                        on=["lat", "lon"],
                        how="inner",
                    )
            merged.insert(0, "month", np.int8(month))
            merged.insert(0, "year", np.int16(year))
            yearly_frames.append(merged)

    df_all = pd.concat(yearly_frames, ignore_index=True)
    out_path = OUT_DIR / "predictions.parquet"
    df_all.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    mb = out_path.stat().st_size / (1024 * 1024)
    print(f"    -> Saved {out_path.name}: {df_all.shape} ({mb:.2f} MB) in {time.time() - t0:.1f}s")
    return {"rows": len(df_all), "size_mb": round(mb, 3)}


def build_paired_obs() -> dict:
    print("[2/6] Building paired_obs.parquet (DF_PADAN + test/train flags + 20 covariates)...")
    t0 = time.time()
    frames = []
    base_dir = MODEL_DIRS["20var"]

    for year in YEARS:
        for month in MONTHS:
            fp = base_dir / str(year) / f"DF_PADAN_{month}_{year}.csv"
            if not fp.exists():
                raise FileNotFoundError(f"Missing paired file: {fp}")
            df = pd.read_csv(fp)
            df.insert(0, "month", np.int8(month))
            df.insert(0, "year", np.int16(year))
            frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    df_all["idx_new"] = df_all["idx_new"].astype(np.int16)
    df_all["lat"] = df_all["lat"].round(4).astype(np.float32)
    df_all["lon"] = df_all["lon"].round(4).astype(np.float32)
    df_all["monthly_rainfall"] = df_all["monthly_rainfall"].astype(np.float32)
    for col in EKSOGEN_20_COLS:
        df_all[col] = df_all[col].astype(np.float32)

    keys = list(zip(df_all["lon"].round(4), df_all["lat"].round(4)))
    df_all["is_test_point"] = [k in TEST_POINT_MAP for k in keys]
    df_all["point_id"] = np.array(
        [TEST_POINT_MAP[k]["point_id"] if k in TEST_POINT_MAP else 0 for k in keys],
        dtype=np.int16,
    )
    df_all["location_name"] = [
        TEST_POINT_MAP[k]["location_name"]
        if k in TEST_POINT_MAP
        else f"Train Grid #{idx} ({lat:.2f}, {lon:.2f})"
        for k, idx, lat, lon in zip(keys, df_all["idx_new"], df_all["lat"], df_all["lon"])
    ]
    df_all["region"] = [assign_region(float(x)) for x in df_all["lon"]]

    out_path = OUT_DIR / "paired_obs.parquet"
    df_all.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    mb = out_path.stat().st_size / (1024 * 1024)
    print(f"    -> Saved {out_path.name}: {df_all.shape} ({mb:.2f} MB) in {time.time() - t0:.1f}s")
    return {
        "rows": len(df_all),
        "test_rows": int(df_all["is_test_point"].sum()),
        "train_rows": int((~df_all["is_test_point"]).sum()),
        "size_mb": round(mb, 3),
    }


def build_raw_stations() -> dict:
    print("[3/6] Building raw_stations.parquet & station_catalog.parquet...")
    t0 = time.time()

    df_obs = pd.read_csv(RAW_OBS_FILE, sep="\t")
    df_obs = df_obs[~df_obs["monthly_rainfall"].isna()].copy()
    df_obs = df_obs[(df_obs["year"] >= 1985) & (df_obs["year"] <= 2014)].reset_index(drop=True)

    ref_era5 = pd.read_excel(ERA5_DIR / "processed_era5jawa_1985_01.xlsx")[["lat", "lon"]]
    tree = spatial.KDTree(ref_era5[["lon", "lat"]].values)
    dists_deg, idx_map = tree.query(df_obs[["long", "lat"]].values)

    df_obs["grid_idx"] = idx_map.astype(np.int16)
    df_obs["grid_lat"] = ref_era5.iloc[idx_map]["lat"].values.round(4).astype(np.float32)
    df_obs["grid_lon"] = ref_era5.iloc[idx_map]["lon"].values.round(4).astype(np.float32)
    df_obs["pair_dist_km"] = haversine_km(
        df_obs["lat"].values,
        df_obs["long"].values,
        df_obs["grid_lat"].values,
        df_obs["grid_lon"].values,
    ).astype(np.float32)

    grid_keys = list(zip(df_obs["grid_lon"].round(4), df_obs["grid_lat"].round(4)))
    df_obs["is_test_grid"] = [k in TEST_POINT_MAP for k in grid_keys]
    df_obs["point_id"] = np.array(
        [TEST_POINT_MAP[k]["point_id"] if k in TEST_POINT_MAP else 0 for k in grid_keys],
        dtype=np.int16,
    )
    df_obs["test_location_name"] = [
        TEST_POINT_MAP[k]["location_name"] if k in TEST_POINT_MAP else "Training Grid"
        for k in grid_keys
    ]
    df_obs["region"] = [assign_region(float(x)) for x in df_obs["grid_lon"]]

    # Compact dtypes
    df_obs["year"] = df_obs["year"].astype(np.int16)
    df_obs["month"] = df_obs["month"].astype(np.int8)
    df_obs["monthly_rainfall"] = df_obs["monthly_rainfall"].astype(np.float32)
    df_obs["lat"] = df_obs["lat"].astype(np.float32)
    df_obs["long"] = df_obs["long"].astype(np.float32)
    df_obs["elev"] = df_obs["elev"].astype(np.float32)
    df_obs["no"] = df_obs["no"].astype(str)
    df_obs["stanam"] = df_obs["stanam"].astype(str).str.strip()

    keep_cols = [
        "year", "month", "no", "stanam", "lat", "long", "elev",
        "monthly_rainfall", "grid_idx", "grid_lat", "grid_lon",
        "pair_dist_km", "is_test_grid", "point_id", "test_location_name", "region",
    ]
    df_raw_out = df_obs[keep_cols].copy()

    raw_path = OUT_DIR / "raw_stations.parquet"
    df_raw_out.to_parquet(raw_path, engine="pyarrow", compression="zstd", index=False)

    # Station catalog (1 row per unique physical station)
    catalog = (
        df_raw_out.groupby(["stanam", "lat", "long", "elev", "grid_idx", "grid_lat", "grid_lon", "pair_dist_km", "is_test_grid", "point_id", "test_location_name", "region"], as_index=False)
        .agg(
            n_months=("monthly_rainfall", "count"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            mean_rainfall=("monthly_rainfall", "mean"),
            max_rainfall=("monthly_rainfall", "max"),
        )
    )
    catalog_path = OUT_DIR / "station_catalog.parquet"
    catalog.to_parquet(catalog_path, engine="pyarrow", compression="zstd", index=False)

    mb_raw = raw_path.stat().st_size / (1024 * 1024)
    mb_cat = catalog_path.stat().st_size / (1024 * 1024)
    print(f"    -> Saved {raw_path.name}: {df_raw_out.shape} ({mb_raw:.2f} MB), {catalog_path.name}: {catalog.shape} ({mb_cat:.2f} MB) in {time.time() - t0:.1f}s")
    return {
        "raw_rows": len(df_raw_out),
        "unique_stations": int(catalog["stanam"].nunique()),
        "raw_size_mb": round(mb_raw, 3),
        "catalog_size_mb": round(mb_cat, 3),
    }


def build_era5_grid() -> dict:
    print("[4/6] Building era5_grid_20var.parquet (full 0.25° grid for 20 SHAP covariates, 1985-2014)...")
    t0 = time.time()
    cols = ["lat", "lon"] + EKSOGEN_20_COLS
    frames = []

    for year in YEARS:
        for month in MONTHS:
            fp = ERA5_DIR / f"processed_era5jawa_{year}_{str(month).zfill(2)}.xlsx"
            if not fp.exists():
                raise FileNotFoundError(f"Missing ERA5 file: {fp}")
            df = pd.read_excel(fp, usecols=cols)
            df.insert(0, "month", np.int8(month))
            df.insert(0, "year", np.int16(year))
            df["lat"] = df["lat"].round(4).astype(np.float32)
            df["lon"] = df["lon"].round(4).astype(np.float32)
            for c in EKSOGEN_20_COLS:
                df[c] = df[c].astype(np.float32)
            frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    out_path = OUT_DIR / "era5_grid_20var.parquet"
    df_all.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    mb = out_path.stat().st_size / (1024 * 1024)
    print(f"    -> Saved {out_path.name}: {df_all.shape} ({mb:.2f} MB) in {time.time() - t0:.1f}s")
    return {"rows": len(df_all), "size_mb": round(mb, 3)}


def _sample_epochs(max_epoch: int, best_epoch: int) -> set[int]:
    epochs = set(range(1, min(11, max_epoch + 1)))
    epochs.update(range(12, min(51, max_epoch + 1), 2))
    epochs.update(range(60, min(201, max_epoch + 1), 10))
    epochs.update(range(250, min(1001, max_epoch + 1), 50))
    epochs.update(range(1100, max_epoch + 1, 100))
    epochs.add(best_epoch)
    epochs.add(max_epoch)
    return epochs


def build_history() -> dict:
    print("[5/6] Building history_summary.parquet & history_curves.parquet (1,800 training runs)...")
    t0 = time.time()
    summary_rows = []
    curve_frames = []

    for model_key, base_dir in MODEL_DIRS.items():
        for year in YEARS:
            for month in MONTHS:
                fp = base_dir / str(year) / f"HISTORY_METRICS_{month}_{year}.csv"
                if not fp.exists():
                    raise FileNotFoundError(f"Missing history file: {fp}")
                df = pd.read_csv(fp)
                max_ep = int(df["epoch"].max())
                best_idx = int(df["mae_test"].idxmin())
                best_loss_idx = int(df["loss_test"].idxmin())
                last_row = df.iloc[-1]
                best_row = df.iloc[best_idx]
                best_loss_row = df.iloc[best_loss_idx]

                summary_rows.append({
                    "model": model_key,
                    "year": year,
                    "month": month,
                    "total_epochs": max_ep,
                    "initial_mae_train": float(df.iloc[0]["mae_train"]),
                    "initial_mae_test": float(df.iloc[0]["mae_test"]),
                    "final_mae_train": float(last_row["mae_train"]),
                    "final_mae_test": float(last_row["mae_test"]),
                    "final_loss_train": float(last_row["loss_train"]),
                    "final_loss_test": float(last_row["loss_test"]),
                    "best_mae_test": float(best_row["mae_test"]),
                    "best_mae_test_epoch": int(best_row["epoch"]),
                    "train_mae_at_best_test": float(best_row["mae_train"]),
                    "min_loss_test": float(best_loss_row["loss_test"]),
                    "min_loss_test_epoch": int(best_loss_row["epoch"]),
                    "overfit_gap": float(last_row["mae_test"] - last_row["mae_train"]),
                })

                keep_eps = _sample_epochs(max_ep, int(best_row["epoch"]))
                sub = df[df["epoch"].isin(keep_eps)].copy()
                sub.insert(0, "month", np.int8(month))
                sub.insert(0, "year", np.int16(year))
                sub.insert(0, "model", model_key)
                sub["epoch"] = sub["epoch"].astype(np.int16)
                for col in ["mae_train", "mae_test", "loss_train", "loss_test"]:
                    sub[col] = sub[col].astype(np.float32)
                curve_frames.append(sub)

    df_sum = pd.DataFrame(summary_rows)
    df_sum["year"] = df_sum["year"].astype(np.int16)
    df_sum["month"] = df_sum["month"].astype(np.int8)
    df_sum["total_epochs"] = df_sum["total_epochs"].astype(np.int16)
    df_sum["best_mae_test_epoch"] = df_sum["best_mae_test_epoch"].astype(np.int16)
    df_sum["min_loss_test_epoch"] = df_sum["min_loss_test_epoch"].astype(np.int16)

    sum_path = OUT_DIR / "history_summary.parquet"
    df_sum.to_parquet(sum_path, engine="pyarrow", compression="zstd", index=False)

    df_curves = pd.concat(curve_frames, ignore_index=True)
    curves_path = OUT_DIR / "history_curves.parquet"
    df_curves.to_parquet(curves_path, engine="pyarrow", compression="zstd", index=False)

    mb_sum = sum_path.stat().st_size / (1024 * 1024)
    mb_cur = curves_path.stat().st_size / (1024 * 1024)
    print(f"    -> Saved {sum_path.name}: {df_sum.shape} ({mb_sum:.2f} MB), {curves_path.name}: {df_curves.shape} ({mb_cur:.2f} MB) in {time.time() - t0:.1f}s")
    return {
        "summary_rows": len(df_sum),
        "curve_rows": len(df_curves),
        "summary_size_mb": round(mb_sum, 3),
        "curves_size_mb": round(mb_cur, 3),
    }


def build_shap() -> dict:
    print("[6/6] Building shap_ranking.parquet...")
    df = pd.read_csv(SHAP_CSV)
    for model_key, k in MODEL_FEATURE_COUNTS.items():
        selected = set(EKSOGEN_20_COLS[:k])
        df[f"in_{model_key}"] = df["Feature"].isin(selected)
    out_path = OUT_DIR / "shap_ranking.parquet"
    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    mb = out_path.stat().st_size / (1024 * 1024)
    print(f"    -> Saved {out_path.name}: {df.shape} ({mb:.3f} MB)")
    return {"rows": len(df), "size_mb": round(mb, 3)}


def main() -> None:
    t_start = time.time()
    stats = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "years": [min(YEARS), max(YEARS)],
        "models": list(MODEL_DIRS.keys()),
        "test_points": TEST_POINTS,
        "datasets": {},
    }
    stats["datasets"]["predictions"] = build_predictions()
    stats["datasets"]["paired_obs"] = build_paired_obs()
    stats["datasets"]["raw_stations"] = build_raw_stations()
    stats["datasets"]["era5_grid_20var"] = build_era5_grid()
    stats["datasets"]["history"] = build_history()
    stats["datasets"]["shap_ranking"] = build_shap()

    total_mb = sum(
        p.stat().st_size for p in OUT_DIR.glob("*.parquet")
    ) / (1024 * 1024)
    stats["total_parquet_size_mb"] = round(total_mb, 2)

    meta_path = OUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nALL DONE in {time.time() - t_start:.1f}s | Total Parquet size: {total_mb:.2f} MB")


if __name__ == "__main__":
    main()
