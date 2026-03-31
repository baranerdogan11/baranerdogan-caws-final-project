"""
Forging Line — Piece Travel Time Dashboard

Displays processed pieces with predicted bath time and per-stage
timing detail.

Usage:
    uv run streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# TODO: import Predictor from vaultech_analysis.inference

GOLD_FILE = PROJECT_ROOT / "data" / "gold" / "pieces.parquet"

# Column definitions — process order
PARTIAL_COLS = [
    "partial_furnace_to_2nd_strike_s",
    "partial_2nd_to_3rd_strike_s",
    "partial_3rd_to_4th_strike_s",
    "partial_4th_strike_to_auxiliary_press_s",
    "partial_auxiliary_press_to_bath_s",
]
PARTIAL_LABELS = [
    "Furnace → 2nd strike",
    "2nd strike → 3rd strike",
    "3rd strike → 4th strike",
    "4th strike → Aux. press",
    "Aux. press → Bath",
]
CUMULATIVE_COLS = [
    "lifetime_2nd_strike_s",
    "lifetime_3rd_strike_s",
    "lifetime_4th_strike_s",
    "lifetime_auxiliary_press_s",
    "lifetime_bath_s",
]
CUMULATIVE_LABELS = [
    "2nd strike (1st op)",
    "3rd strike (2nd op)",
    "4th strike (drill)",
    "Auxiliary press",
    "Bath",
]


# TODO: implement load_predictor() with @st.cache_resource
# TODO: implement load_data() with @st.cache_data
#   - read gold parquet
#   - run predict_batch() to add predicted_bath_s and prediction_error_s
# TODO: implement get_reference() with @st.cache_data
#   - group by die_matrix, compute median of partial + cumulative cols

st.set_page_config(page_title="Forging Line Dashboard", layout="wide")
st.title("Forging Line — Piece Travel Time Dashboard")

# TODO: load data and reference

# TODO: sidebar filters (die matrix, date range, slow pieces only)

# TODO: summary metrics (total pieces, median bath, median predicted, MAE)

# TODO: pieces table with row selection

# TODO: piece detail panel (cumulative times, partial times, synoptic chart)

st.info("Select a piece from the table above to see its per-stage timing detail.")
