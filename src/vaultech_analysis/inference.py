"""
Inference service for predicting total piece travel time.

Loads the trained XGBoost model and provides predictions.

Usage as CLI:
    uv run python -m vaultech_analysis.inference --die-matrix 5052 --strike2 18.3 --oee 13.5

Usage as module (for Streamlit):
    from vaultech_analysis.inference import Predictor
    predictor = Predictor()
    result = predictor.predict(die_matrix=5052, lifetime_2nd_strike_s=18.3, oee_cycle_time_s=13.5)
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor


MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
GOLD_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "gold" / "pieces.parquet"


class Predictor:
    """Loads the trained model and provides predictions."""

    def __init__(self, model_dir: Path = MODEL_DIR, gold_file: Path = GOLD_FILE):
        # TODO: load the XGBoost model from model_dir
        # TODO: load model metadata (features, metrics, die_matrices)
        # TODO: load reference medians per die matrix from gold_file
        pass

    def predict(
        self,
        die_matrix: int,
        lifetime_2nd_strike_s: float,
        oee_cycle_time_s: float | None = None,
    ) -> dict:
        """Predict total bath time from early-stage features.

        Returns a dict with predicted_bath_time_s, input values, and model_metrics.
        Returns {"error": "..."} for unknown die_matrix values.
        Missing oee_cycle_time_s should default to the median (~13.8s).
        """
        # TODO: implement
        raise NotImplementedError

    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        """Predict bath time for a DataFrame of pieces.

        Handle missing oee_cycle_time_s by filling with the median.
        """
        # TODO: implement
        raise NotImplementedError


def main():
    # TODO: implement CLI entry point
    # Parse args: --die-matrix, --strike2, --oee (optional)
    # Create Predictor, call predict(), print JSON result
    raise NotImplementedError


if __name__ == "__main__":
    main()
