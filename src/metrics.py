"""Hydrological and statistical evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    """Compute MAE, RMSE, NRMSE, CV-RMSE, R2, Pearson r, MBE, PBIAS, and KGE."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]

    n = len(yt)
    if n == 0:
        return {
            "N": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "NRMSE": np.nan,
            "CV_RMSE_%": np.nan,
            "R2": np.nan,
            "r": np.nan,
            "MBE": np.nan,
            "PBIAS_%": np.nan,
            "KGE": np.nan,
            "Mean_Obs": np.nan,
            "Mean_Pred": np.nan,
        }

    res = yp - yt
    mae = float(np.mean(np.abs(res)))
    mse = float(np.mean(res ** 2))
    rmse = float(np.sqrt(mse))
    mbe = float(np.mean(res))

    mean_obs = float(np.mean(yt))
    std_obs = float(np.std(yt))
    mean_pred = float(np.mean(yp))
    std_pred = float(np.std(yp))

    nrmse = rmse / std_obs if std_obs > 0 else np.nan
    cv_rmse = (rmse / mean_obs * 100.0) if mean_obs > 0 else np.nan

    ss_tot = float(np.sum((yt - mean_obs) ** 2))
    ss_res = float(np.sum(res ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else (1.0 if ss_res == 0 else np.nan)

    if n > 1 and std_obs > 0 and std_pred > 0:
        r_corr = float(np.corrcoef(yt, yp)[0, 1])
    else:
        r_corr = np.nan

    sum_obs = float(np.sum(yt))
    pbias = (float(np.sum(res)) / sum_obs * 100.0) if sum_obs != 0 else np.nan

    if np.isfinite(r_corr) and mean_obs != 0 and mean_pred != 0 and std_obs > 0:
        beta = mean_pred / mean_obs
        gamma = (std_pred / mean_pred) / (std_obs / mean_obs)
        kge = float(1.0 - np.sqrt((r_corr - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2))
    else:
        kge = np.nan

    return {
        "N": int(n),
        "MAE": mae,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "CV_RMSE_%": cv_rmse,
        "R2": r2,
        "r": r_corr,
        "MBE": mbe,
        "PBIAS_%": pbias,
        "KGE": kge,
        "Mean_Obs": mean_obs,
        "Mean_Pred": mean_pred,
    }


def evaluate_models_table(
    df_merged: pd.DataFrame,
    model_specs: dict,
    model_keys: list[str],
    obs_col: str = "monthly_rainfall",
) -> pd.DataFrame:
    """Return a comparison DataFrame with one row per selected model."""
    rows = []
    for mk in model_keys:
        spec = model_specs[mk]
        m = compute_metrics(df_merged[obs_col], df_merged[spec["pred_col"]])
        m["Model_Key"] = mk
        m["Model"] = spec["label"]
        rows.append(m)
    cols = [
        "Model_Key", "Model", "N", "MAE", "RMSE", "R2", "KGE",
        "r", "MBE", "PBIAS_%", "NRMSE", "CV_RMSE_%", "Mean_Obs", "Mean_Pred",
    ]
    return pd.DataFrame(rows)[cols]

