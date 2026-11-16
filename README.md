# DK Viewer v2 — DeepKriging Spatio-Temporal Rainfall Analytics (Java, 1985–2014)

An interactive, multi-page **Streamlit** analytics dashboard for evaluating **DeepKriging Ver 5** monthly rainfall models across Java, Indonesia (`1985–2014`, `0.25° × 0.25°` grid, `675` cells).

Built as a clean, self-contained, GitHub-ready upgrade over `10k_dkviewer`, featuring an English scientific UI, zstd-compressed Parquet datasets, and full integration of model predictions, independent hold-out test points, raw BMKG station observations, ERA5 atmospheric covariates, training convergence histories, and SHAP rankings.

---

## Key Features

1. **Overview & Experiment Hub (`app.py`)**
   - Global hold-out vs training leaderboard (`MAE`, `RMSE`, `R²`, `KGE`, `r`, `MBE`, `PBIAS%`, `NRMSE`, `CV-RMSE%`).
   - Interactive Java network map overlaying the **486 raw BMKG stations**, **~102 paired training grid cells**, and **9 independent hold-out test points** (`points_to_remove`).
   - Model architecture & convergence summary across 1,800 monthly training runs.

2. **Model Comparison & Station Explorer (`pages/1_Model_Comparison.py`)**
   - Evaluate across **All 9 Hold-Out Points**, **Single Test Point**, **All Training Grids**, **Single Training Grid**, or **Java Sub-Regions** (West, Central/DIY, East).
   - Season filtering (`DJF`, `MAM`, `JJA`, `SON`), interactive time series, residual & error traces, multi-model 1:1 scatter plots with regression lines, error boxplots, empirical CDFs, and per-station metric heatmaps.

3. **Monthly & Seasonal Climatology (`pages/2_Monthly_Climatology.py`)**
   - 12-month annual rainfall cycle, seasonal MAE/Bias profiles, single-month 30-year interannual trends, and wet/dry anomaly analysis (e.g., El Niño / La Niña years).

4. **Spatio-Temporal Hovmöller Diagnostics (`pages/3_Hovmoller_Diagnostics.py`)**
   - **Year × Station** (ordered West → East Java) and **Month × Year** matrices for `MAE`, `RMSE`, `Bias`, `Predicted`, and `Observed` rainfall.
   - **Longitude × Time** rainfall propagation across the full `0.25°` Java grid.

5. **Spatial Map & Difference Viewer (`pages/4_Spatial_Map_Viewer.py`)**
   - Cubic-interpolated spatial contours and OpenStreetMap scatter views for single months, monthly climatologies, or 30-year means.
   - **Side-by-Side Model Comparison**, **Spatial Difference Maps (`Model A − Model B`)**, and **ERA5 Exogenous Field Inspector**.

6. **Training Convergence & Epoch Diagnostics (`pages/5_Training_Dynamics.py`)**
   - Inspect Train vs Test `MAE` and `MSE Loss` learning curves up to `10,000` epochs.
   - Compare optimal early-stopping epochs vs final epoch performance and analyze generalization gaps across all 1,800 runs.

7. **Raw BMKG Stations, Exogenous Covariates & SHAP (`pages/6_Raw_Data_and_SHAP.py`)**
   - Inspect the exact `KDTree` spatial pairing between the 486 raw BMKG stations (`stanam`, `elev`) and ERA5 `0.25°` grid cells.
   - Drill down into raw station time series prior to grid aggregation, inspect covariate correlations, and view the global **51-variable SHAP importance ranking**.

---

## Models Included (DeepKriging Ver 5)

All 5 variants use Wendland $C^2$ radial basis functions (`NUM_BASIS = [16, 64, 256]`) combined with top-$k$ SHAP-selected ERA5 atmospheric covariates:

| Variant | Exogenous Features |
| :--- | :--- |
| **1-Var (`1var_olr`)** | `olr` |
| **2-Var (`2var_newest`)** | `olr`, `q_200` |
| **5-Var (`5var_newest`)** | `olr`, `q_200`, `mcc`, `v_1000`, `t_200` |
| **9-Var (`9var_newest`)** | Top 5 + `tcrw`, `q_925`, `r_925`, `u_1000` |
| **20-Var (`20var_newest`)** | Top 9 + `u_850`, `u_600`, `w_925`, `slhf`, `t2m`, `u_200`, `msl`, `v_850`, `w_850`, `v_200`, `hcc` |

---

## Repository Structure

```text
├── app.py                              # Main Overview & Experiment Hub
├── pages/
│   ├── 1_Model_Comparison.py           # Time series, scatter 1:1, residuals, per-point metrics
│   ├── 2_Monthly_Climatology.py        # 12-month cycle, interannual & anomaly analysis
│   ├── 3_Hovmoller_Diagnostics.py      # Year×Station, Month×Year, Lon×Time Hovmöller plots
│   ├── 4_Spatial_Map_Viewer.py         # Spatial contours, OSM maps, model diff, ERA5 fields
│   ├── 5_Training_Dynamics.py          # Epoch convergence & early-stopping diagnostics
│   └── 6_Raw_Data_and_SHAP.py          # Raw BMKG stations, KDTree pairing & SHAP rankings
├── src/
│   ├── config.py                       # Model specs, 9 test points, ERA5 metadata
│   ├── data_loader.py                  # Cached Parquet loaders
│   ├── metrics.py                      # MAE, RMSE, R², KGE, PBIAS, MBE, NRMSE, CV-RMSE
│   ├── charts.py                       # Reusable Plotly visualizers
│   └── ui.py                           # Shared styling, sidebar controls, CSV exports
├── tools/
│   └── build_dataset.py                # Reproducible ETL script (raw CSV/XLSX -> Parquet)
├── data/                               # Compact zstd Parquet datasets (~36 MB total)
├── .streamlit/config.toml              # Light scientific UI theme
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 2. Rebuild Parquet Datasets from Raw Outputs (Optional)
The `data/` directory already contains the pre-packaged Parquet datasets (~36 MB total), so the repository runs out-of-the-box on GitHub and Streamlit Community Cloud without needing the raw training folders. To regenerate from raw `HASIL VER 5`, ERA5 `.xlsx`, and station `.txt` files:
```bash
python tools/build_dataset.py
```

### 3. Deploy to Streamlit Community Cloud
1. Push this directory to a fresh GitHub repository (`git init`, `git add .`, `git commit -m "Initial commit"`, `git push`).
2. In Streamlit Community Cloud, create a new app pointing to `app.py`.

