# DK Viewer v2 — Java Monthly Rainfall Viewer (1985–2014)

An interactive, multi-page **Streamlit** dashboard for comparing **DeepKriging Ver 5** monthly rainfall models across Java, Indonesia (`1985–2014`, `0.25° × 0.25°` grid, `675` cells).

Built as a clean, GitHub-ready upgrade over `10k_dkviewer`, featuring simple labels, compressed Parquet datasets, and combined views for model predictions, 9 test stations, raw BMKG station records, ERA5 weather variables, training curves, and SHAP variable rankings.

---

## Key Features

1. **Overview (`app.py`)**
   - Model accuracy ranking (`MAE`, `RMSE`, `R²`, `KGE`, `r`, `MBE`, `PBIAS%`, `NRMSE`, `CV-RMSE%`).
   - Interactive Java map showing the **486 BMKG stations**, **training stations**, and **9 test stations**.
   - Summary of model settings and training results across 1,800 monthly runs.

2. **Model Comparison (`pages/1_Model_Comparison.py`)**
   - Compare models across **All 9 Test Stations**, **Single Test Station**, **All Training Stations**, **Single Training Station**, or **Java Regions** (West, Central/DIY, East).
   - Season filter (`DJF`, `MAM`, `JJA`, `SON`), time series charts, error charts, scatter plots, error spread, and per-station heatmaps.

3. **Monthly & Seasonal Patterns (`pages/2_Monthly_Climatology.py`)**
   - 12-month rainfall cycle, monthly MAE/Bias charts, single-month trends over 30 years, and wet/dry year comparisons (e.g., El Niño / La Niña years).

4. **Time & Station Heatmaps (`pages/3_Hovmoller_Diagnostics.py`)**
   - **Year × Test Station** (West → East Java) and **Month × Year** heatmaps for `MAE`, `RMSE`, `Bias`, `Predicted`, and `Observed` rainfall.
   - **West-to-East Rainfall Over Time** across the full `0.25°` Java grid.

5. **Rainfall Maps (`pages/4_Spatial_Map_Viewer.py`)**
   - Smooth contour maps and OpenStreetMap point views for single months, monthly averages, or 30-year averages.
   - **Side-by-Side Comparison**, **Difference Maps (`Model A − Model B`)**, and **ERA5 Weather Variable Maps**.

6. **Training Progress (`pages/5_Training_Dynamics.py`)**
   - Inspect training vs test `MAE` and `MSE Loss` curves up to `10,000` epochs.
   - Compare the best epoch during training against the final epoch (`10,000`).

7. **BMKG Stations, Weather Variables & SHAP (`pages/6_Raw_Data_and_SHAP.py`)**
   - View the 486 raw BMKG stations (`stanam`, `elev`) and how each matches the nearest ERA5 `0.25°` grid cell.
   - Inspect raw station rainfall over time, correlations with ERA5 weather variables, and the **51-variable SHAP importance ranking**.

---

## Models Included (DeepKriging Ver 5)

All 5 variants combine spatial basis functions (`[16, 64, 256]`) with the top-$k$ ERA5 weather variables selected by SHAP:

| Variant | Weather Variables |
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

