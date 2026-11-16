"""Central configuration, model registry, station metadata, and variable dictionary."""

from __future__ import annotations

from pathlib import Path

APP_TITLE = "DK Viewer"
APP_SUBTITLE = "Java Monthly Rainfall Model Viewer (DeepKriging Ver 5, 1985–2014)"

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

YEARS = list(range(1985, 2015))
MONTHS = list(range(1, 13))

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

MONTH_SHORT = {k: v[:3] for k, v in MONTH_NAMES.items()}

SEASONS = {
    "All Months": list(range(1, 13)),
    "DJF (Peak Wet / NW Monsoon)": [12, 1, 2],
    "MAM (Transition I)": [3, 4, 5],
    "JJA (Dry / SE Monsoon)": [6, 7, 8],
    "SON (Transition II)": [9, 10, 11],
}

MODEL_ORDER = ["1var", "2var", "5var", "9var", "20var"]

MODEL_SPECS = {
    "1var": {
        "label": "1-Var (OLR)",
        "short": "1-Var",
        "pred_col": "pred_1var",
        "color": "#64748B",
        "features": ["olr"],
        "description": "Uses 1 weather variable (OLR) + spatial basis.",
    },
    "2var": {
        "label": "2-Var (Top 2)",
        "short": "2-Var",
        "pred_col": "pred_2var",
        "color": "#2563EB",
        "features": ["olr", "q_200"],
        "description": "Uses top 2 weather variables (olr, q_200) + spatial basis.",
    },
    "5var": {
        "label": "5-Var (Top 5)",
        "short": "5-Var",
        "pred_col": "pred_5var",
        "color": "#F59E0B",
        "features": ["olr", "q_200", "mcc", "v_1000", "t_200"],
        "description": "Uses top 5 weather variables + spatial basis.",
    },
    "9var": {
        "label": "9-Var (Top 9)",
        "short": "9-Var",
        "pred_col": "pred_9var",
        "color": "#10B981",
        "features": ["olr", "q_200", "mcc", "v_1000", "t_200", "tcrw", "q_925", "r_925", "u_1000"],
        "description": "Uses top 9 weather variables + spatial basis.",
    },
    "20var": {
        "label": "20-Var (Top 20)",
        "short": "20-Var",
        "pred_col": "pred_20var",
        "color": "#DC2626",
        "features": [
            "olr", "q_200", "mcc", "v_1000", "t_200",
            "tcrw", "q_925", "r_925", "u_1000", "u_850",
            "u_600", "w_925", "slhf", "t2m", "u_200",
            "msl", "v_850", "w_850", "v_200", "hcc",
        ],
        "description": "Uses top 20 weather variables + spatial basis.",
    },
}

OBS_COLOR = "#F4F4F5"

TEST_POINTS = [
    {"point_id": 1, "lon": 106.50, "lat": -6.25, "location_name": "Tangerang / Bogor (West Java)", "short": "S1 Tangerang/Bogor"},
    {"point_id": 2, "lon": 107.00, "lat": -6.75, "location_name": "Cianjur / Bandung (West Java)", "short": "S2 Cianjur/Bandung"},
    {"point_id": 3, "lon": 107.50, "lat": -7.25, "location_name": "Garut / Tasikmalaya (West Java)", "short": "S3 Garut/Tasik"},
    {"point_id": 4, "lon": 110.00, "lat": -7.00, "location_name": "Semarang (Central Java)", "short": "S4 Semarang"},
    {"point_id": 5, "lon": 110.00, "lat": -7.50, "location_name": "Sleman / Yogyakarta (DIY)", "short": "S5 Sleman/DIY"},
    {"point_id": 6, "lon": 111.75, "lat": -7.75, "location_name": "Madiun / Ponorogo (East Java)", "short": "S6 Madiun/Ponorogo"},
    {"point_id": 7, "lon": 112.25, "lat": -7.00, "location_name": "Gresik / Surabaya (East Java)", "short": "S7 Gresik/Surabaya"},
    {"point_id": 8, "lon": 112.25, "lat": -8.25, "location_name": "South Malang (East Java)", "short": "S8 South Malang"},
    {"point_id": 9, "lon": 113.50, "lat": -8.00, "location_name": "Jember / Lumajang (East Java)", "short": "S9 Jember/Lumajang"},
]

EXOGENOUS_INFO = {
    "olr": {"name": "Outgoing Longwave Radiation (Top Thermal Rad.)", "unit": "J m⁻²"},
    "q_200": {"name": "Specific Humidity at 200 hPa", "unit": "kg kg⁻¹"},
    "mcc": {"name": "Medium Cloud Cover", "unit": "fraction (0–1)"},
    "v_1000": {"name": "Meridional Wind at 1000 hPa", "unit": "m s⁻¹"},
    "t_200": {"name": "Air Temperature at 200 hPa", "unit": "K"},
    "tcrw": {"name": "Total Column Rain Water", "unit": "kg m⁻²"},
    "q_925": {"name": "Specific Humidity at 925 hPa", "unit": "kg kg⁻¹"},
    "r_925": {"name": "Relative Humidity at 925 hPa", "unit": "%"},
    "u_1000": {"name": "Zonal Wind at 1000 hPa", "unit": "m s⁻¹"},
    "u_850": {"name": "Zonal Wind at 850 hPa", "unit": "m s⁻¹"},
    "u_600": {"name": "Zonal Wind at 600 hPa", "unit": "m s⁻¹"},
    "w_925": {"name": "Vertical Velocity (Omega) at 925 hPa", "unit": "Pa s⁻¹"},
    "slhf": {"name": "Surface Latent Heat Flux", "unit": "J m⁻²"},
    "t2m": {"name": "2-Meter Air Temperature", "unit": "K"},
    "u_200": {"name": "Zonal Wind at 200 hPa", "unit": "m s⁻¹"},
    "msl": {"name": "Mean Sea Level Pressure", "unit": "Pa"},
    "v_850": {"name": "Meridional Wind at 850 hPa", "unit": "m s⁻¹"},
    "w_850": {"name": "Vertical Velocity (Omega) at 850 hPa", "unit": "Pa s⁻¹"},
    "v_200": {"name": "Meridional Wind at 200 hPa", "unit": "m s⁻¹"},
    "hcc": {"name": "High Cloud Cover", "unit": "fraction (0–1)"},
}

EKSOGEN_20_COLS = list(EXOGENOUS_INFO.keys())

