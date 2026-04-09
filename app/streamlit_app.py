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

from vaultech_analysis.inference import Predictor

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


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor(
        model_dir=PROJECT_ROOT / "models",
        gold_file=GOLD_FILE,
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(GOLD_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date

    predictor = load_predictor()
    df["predicted_bath_s"] = predictor.predict_batch(df)
    df["prediction_error_s"] = df["lifetime_bath_s"] - df["predicted_bath_s"]

    return df


@st.cache_data
def get_reference(df: pd.DataFrame) -> pd.DataFrame:
    ref_cols = PARTIAL_COLS + CUMULATIVE_COLS
    available = [c for c in ref_cols if c in df.columns]
    return df.groupby("die_matrix")[available].median()


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Forging Line Dashboard", layout="wide")
st.title("Forging Line — Piece Travel Time Dashboard")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data and running predictions…"):
    df_all = load_data()
    ref = get_reference(df_all)

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")

matrices = sorted(df_all["die_matrix"].dropna().unique().tolist())
selected_matrices = st.sidebar.multiselect(
    "Die Matrix", matrices, default=matrices
)

min_date = df_all["date"].min()
max_date = df_all["date"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

slow_only = st.sidebar.checkbox("Show slow pieces only (bath > P90 per matrix)")

# Apply filters
df = df_all[df_all["die_matrix"].isin(selected_matrices)].copy()

if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

if slow_only:
    p90 = df.groupby("die_matrix")["lifetime_bath_s"].transform(lambda x: x.quantile(0.90))
    df = df[df["lifetime_bath_s"] > p90]

# ── Summary metrics ───────────────────────────────────────────────────────────
st.subheader("Summary")

if len(df) == 0:
    st.warning("No pieces match the current filters.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total pieces", f"{len(df):,}")
col2.metric("Median bath time", f"{df['lifetime_bath_s'].median():.2f}s")
col3.metric("Median predicted", f"{df['predicted_bath_s'].median():.2f}s")
col4.metric("MAE", f"{df['prediction_error_s'].abs().mean():.3f}s")

# ── Pieces table ──────────────────────────────────────────────────────────────
st.subheader("Pieces")

table_cols = [
    "timestamp", "piece_id", "die_matrix",
    "lifetime_bath_s", "predicted_bath_s", "prediction_error_s",
    "oee_cycle_time_s",
]
available_table_cols = [c for c in table_cols if c in df.columns]
df_display = df[available_table_cols].copy()
df_display["timestamp"] = df_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

df_display = df_display.rename(columns={
    "timestamp": "Timestamp",
    "piece_id": "Piece ID",
    "die_matrix": "Die Matrix",
    "lifetime_bath_s": "Bath (s)",
    "predicted_bath_s": "Predicted (s)",
    "prediction_error_s": "Error (s)",
    "oee_cycle_time_s": "OEE (s)",
})

# Row selection
st.caption("Select a row to see piece detail below.")
selected_rows = st.dataframe(
    df_display.reset_index(drop=True),
    use_container_width=True,
    height=300,
    on_select="rerun",
    selection_mode="single-row",
)

# ── Piece detail panel ────────────────────────────────────────────────────────
selected_indices = selected_rows.selection.get("rows", [])

if not selected_indices:
    st.info("Select a piece from the table above to see its per-stage timing detail.")
else:
    selected_idx = selected_indices[0]
    piece = df.iloc[selected_idx]
    matrix = int(piece["die_matrix"])
    matrix_ref = ref.loc[matrix] if matrix in ref.index else None

    st.subheader(f"Piece Detail — {piece.get('piece_id', 'N/A')} | Die Matrix {matrix}")
    st.caption(f"Timestamp: {piece['timestamp']} | Bath: {piece['lifetime_bath_s']:.2f}s | Predicted: {piece['predicted_bath_s']:.2f}s")

    col_cum, col_partial = st.columns(2)

    # Cumulative times
    with col_cum:
        st.markdown("**Cumulative times vs reference**")
        cum_rows = []
        for col, label in zip(CUMULATIVE_COLS, CUMULATIVE_LABELS):
            if col not in df.columns:
                continue
            actual = piece.get(col)
            ref_val = matrix_ref[col] if matrix_ref is not None and col in matrix_ref else None
            deviation = (actual - ref_val) if (actual is not None and ref_val is not None) else None
            cum_rows.append({
                "Stage": label,
                "Actual (s)": f"{actual:.2f}" if actual is not None and not pd.isna(actual) else "N/A",
                "Reference (s)": f"{ref_val:.2f}" if ref_val is not None else "N/A",
                "Deviation (s)": f"{deviation:+.2f}" if deviation is not None else "N/A",
            })
        st.dataframe(pd.DataFrame(cum_rows), use_container_width=True, hide_index=True)

    # Partial times
    with col_partial:
        st.markdown("**Partial times vs reference**")
        partial_rows = []
        for col, label in zip(PARTIAL_COLS, PARTIAL_LABELS):
            if col not in df.columns:
                continue
            actual = piece.get(col)
            ref_val = matrix_ref[col] if matrix_ref is not None and col in matrix_ref else None
            deviation = (actual - ref_val) if (actual is not None and ref_val is not None and not pd.isna(actual)) else None
            if deviation is not None:
                status = "🔴 Slow" if deviation > 1.0 else "✅ OK"
            else:
                status = "N/A"
            partial_rows.append({
                "Segment": label,
                "Actual (s)": f"{actual:.2f}" if actual is not None and not pd.isna(actual) else "N/A",
                "Reference (s)": f"{ref_val:.2f}" if ref_val is not None else "N/A",
                "Deviation (s)": f"{deviation:+.2f}" if deviation is not None else "N/A",
                "Status": status,
            })
        st.dataframe(pd.DataFrame(partial_rows), use_container_width=True, hide_index=True)

    # Synoptic bar chart
    st.markdown("**Process synoptic — actual vs reference partial times**")
    chart_data = []
    for col, label in zip(PARTIAL_COLS, PARTIAL_LABELS):
        if col not in df.columns:
            continue
        actual = piece.get(col)
        ref_val = matrix_ref[col] if matrix_ref is not None and col in matrix_ref else None
        if actual is not None and not pd.isna(actual):
            chart_data.append({"Segment": label, "Time (s)": actual, "Type": "Actual"})
        if ref_val is not None:
            chart_data.append({"Segment": label, "Time (s)": ref_val, "Type": "Reference"})

    if chart_data:
        import altair as alt
        chart_df = pd.DataFrame(chart_data)
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Segment:N", sort=None, axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("Time (s):Q"),
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=["Actual", "Reference"],
                    range=["#1f77b4", "#ff7f0e"]
                )),
                xOffset="Type:N",
                tooltip=["Segment", "Type", "Time (s)"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
