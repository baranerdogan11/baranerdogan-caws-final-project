"""
Inference service for predicting total piece travel time.

Loads the trained model and provides predictions.

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

import joblib
import pandas as pd

# Note: xgboost is imported lazily inside __init__ only if needed

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
GOLD_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "gold" / "pieces.parquet"


class Predictor:
    """Loads the trained model and provides predictions."""

    def __init__(self, model_dir: Path = MODEL_DIR, gold_file: Path = GOLD_FILE):
        model_dir = Path(model_dir)
        gold_file = Path(gold_file)

        # Load model — support both joblib and XGBoost JSON formats
        joblib_path = model_dir / "xgboost_bath_predictor.joblib"
        json_path = model_dir / "xgboost_bath_predictor.json"

        if joblib_path.exists():
            self.model = joblib.load(joblib_path)
        elif json_path.exists():
            from xgboost import XGBRegressor
            self.model = XGBRegressor()
            self.model.load_model(json_path)
        else:
            raise FileNotFoundError(
                f"No model file found in {model_dir}. "
                "Run notebooks/05_feature_selection_and_model.ipynb first."
            )

        # Load model metadata
        metadata_path = model_dir / "model_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.features = self.metadata["features"]
        self.metrics = self.metadata["metrics"]
        self.valid_matrices = {
            int(m["die_matrix"]) for m in self.metadata["per_matrix_metrics"]
        }

        # Load reference medians from gold file for OEE fallback
        if gold_file.exists():
            df_gold = pd.read_parquet(gold_file, columns=["oee_cycle_time_s"])
            self._oee_median = float(df_gold["oee_cycle_time_s"].median())
        else:
            self._oee_median = 13.8  # fallback default

    def predict(
        self,
        die_matrix: int,
        lifetime_2nd_strike_s: float,
        oee_cycle_time_s: float | None = None,
    ) -> dict:
        """Predict total bath time from early-stage features.

        Returns a dict with predicted_bath_time_s, input values, and model_metrics.
        Returns {"error": "..."} for unknown die_matrix values.
        Missing oee_cycle_time_s defaults to the median OEE from training data.
        """
        if int(die_matrix) not in self.valid_matrices:
            return {
                "error": f"Unknown die_matrix {die_matrix}. "
                         f"Valid values: {sorted(self.valid_matrices)}"
            }

        oee_for_model = oee_cycle_time_s if oee_cycle_time_s is not None else self._oee_median

        X = pd.DataFrame([{
            "die_matrix": int(die_matrix),
            "lifetime_2nd_strike_s": float(lifetime_2nd_strike_s),
            "oee_cycle_time_s": float(oee_for_model),
        }])[self.features]

        predicted = float(self.model.predict(X)[0])

        return {
            "predicted_bath_time_s": round(predicted, 3),
            "die_matrix": int(die_matrix),
            "lifetime_2nd_strike_s": float(lifetime_2nd_strike_s),
            "oee_cycle_time_s": oee_cycle_time_s,  # None if not provided
            "oee_used_for_prediction": round(oee_for_model, 3),
            "model_metrics": self.metrics,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        """Predict bath time for a DataFrame of pieces.

        Handles missing oee_cycle_time_s by filling with the median.
        """
        df = df.copy()

        # Fill missing OEE with median
        if "oee_cycle_time_s" in df.columns:
            df["oee_cycle_time_s"] = df["oee_cycle_time_s"].fillna(self._oee_median)
        else:
            df["oee_cycle_time_s"] = self._oee_median

        df["die_matrix"] = df["die_matrix"].astype(int)

        X = df[self.features]
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=df.index)


def main():
    parser = argparse.ArgumentParser(
        description="Predict total bath time from early-stage forging features."
    )
    parser.add_argument(
        "--die-matrix", type=int, required=True,
        help="Die matrix number (4974, 5052, 5090, or 5091)"
    )
    parser.add_argument(
        "--strike2", type=float, required=True,
        help="Cumulative time at 2nd strike in seconds (~17-19s)"
    )
    parser.add_argument(
        "--oee", type=float, default=None,
        help="OEE cycle time in seconds (optional, defaults to training median)"
    )
    args = parser.parse_args()

    predictor = Predictor()
    result = predictor.predict(
        die_matrix=args.die_matrix,
        lifetime_2nd_strike_s=args.strike2,
        oee_cycle_time_s=args.oee,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
