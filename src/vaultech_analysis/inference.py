"""
Inference service for predicting total piece travel time.

Supports two backends:
- Local: loads the trained model from disk (default for local dev)
- SageMaker: calls the deployed endpoint via boto3 (used in production)

Usage as CLI:
    uv run python -m vaultech_analysis.inference --die-matrix 5052 --strike2 18.3 --oee 13.5

Usage as module:
    from vaultech_analysis.inference import Predictor
    predictor = Predictor()                          # local model
    predictor = Predictor(endpoint_name="vaultech-bath-endpoint")  # SageMaker
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
GOLD_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "gold" / "pieces.parquet"

FEATURES = ["die_matrix", "lifetime_2nd_strike_s", "oee_cycle_time_s"]
VALID_MATRICES = {4974, 5052, 5090, 5091}


class Predictor:
    """Loads the trained model and provides predictions.

    If endpoint_name is provided, predictions are served via the
    SageMaker endpoint. Otherwise, the local joblib model is used.
    """

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        gold_file: Path = GOLD_FILE,
        endpoint_name: str | None = None,
        region: str = "eu-west-1",
    ):
        self.endpoint_name = endpoint_name
        self.region = region
        self.features = FEATURES
        self.valid_matrices = VALID_MATRICES

        # Load OEE median from gold file for fallback
        gold_file = Path(gold_file)
        if gold_file.exists():
            df_gold = pd.read_parquet(gold_file, columns=["oee_cycle_time_s"])
            self._oee_median = float(df_gold["oee_cycle_time_s"].median())
        else:
            self._oee_median = 13.8

        if endpoint_name:
            # SageMaker mode — lazy import boto3
            import boto3
            self._runtime = boto3.client(
                "sagemaker-runtime", region_name=region
            )
            # Load metadata for metrics (from local file if available)
            metadata_path = Path(model_dir) / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.metadata = json.load(f)
                self.metrics = self.metadata.get("metrics", {})
            else:
                self.metrics = {}
            self.model = None
        else:
            # Local mode — load joblib model
            model_dir = Path(model_dir)
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

            metadata_path = model_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.metadata = json.load(f)
                self.metrics = self.metadata.get("metrics", {})
            else:
                self.metrics = {}

    def _invoke_endpoint(
        self,
        die_matrix: int,
        lifetime_2nd_strike_s: float,
        oee_cycle_time_s: float,
    ) -> tuple[float, float, str]:
        """Call the SageMaker endpoint and return (prediction, latency_ms, payload)."""
        payload = f"{die_matrix},{lifetime_2nd_strike_s},{oee_cycle_time_s}"
        t0 = time.time()
        response = self._runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="text/csv",
            Body=payload,
        )
        latency_ms = (time.time() - t0) * 1000
        prediction = float(response["Body"].read().decode("utf-8").strip())
        return prediction, latency_ms, payload

    def predict(
        self,
        die_matrix: int,
        lifetime_2nd_strike_s: float,
        oee_cycle_time_s: float | None = None,
    ) -> dict:
        """Predict total bath time from early-stage features.

        Returns a dict with predicted_bath_time_s, input values, model_metrics,
        and (if using SageMaker) inference debug info.
        Returns {"error": "..."} for unknown die_matrix values.
        """
        if int(die_matrix) not in self.valid_matrices:
            return {
                "error": f"Unknown die_matrix {die_matrix}. "
                         f"Valid values: {sorted(self.valid_matrices)}"
            }

        oee_for_model = (
            oee_cycle_time_s if oee_cycle_time_s is not None
            else self._oee_median
        )

        if self.endpoint_name:
            predicted, latency_ms, payload = self._invoke_endpoint(
                int(die_matrix), float(lifetime_2nd_strike_s), float(oee_for_model)
            )
            return {
                "predicted_bath_time_s": round(predicted, 3),
                "die_matrix": int(die_matrix),
                "lifetime_2nd_strike_s": float(lifetime_2nd_strike_s),
                "oee_cycle_time_s": oee_cycle_time_s,
                "oee_used_for_prediction": round(oee_for_model, 3),
                "model_metrics": self.metrics,
                "inference_debug": {
                    "backend": "sagemaker",
                    "endpoint": self.endpoint_name,
                    "payload": payload,
                    "raw_response": str(round(predicted, 4)),
                    "latency_ms": round(latency_ms, 1),
                },
            }
        else:
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
                "oee_cycle_time_s": oee_cycle_time_s,
                "oee_used_for_prediction": round(oee_for_model, 3),
                "model_metrics": self.metrics,
                "inference_debug": {
                    "backend": "local",
                    "endpoint": None,
                    "payload": f"{int(die_matrix)},{lifetime_2nd_strike_s},{oee_for_model}",
                    "raw_response": str(round(predicted, 4)),
                    "latency_ms": None,
                },
            }

    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        """Predict bath time for a DataFrame of pieces."""
        df = df.copy()

        if "oee_cycle_time_s" in df.columns:
            df["oee_cycle_time_s"] = df["oee_cycle_time_s"].fillna(self._oee_median)
        else:
            df["oee_cycle_time_s"] = self._oee_median

        df["die_matrix"] = df["die_matrix"].astype(int)

        for col in self.features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        if self.endpoint_name:
            predictions = []
            for _, row in df[self.features].iterrows():
                payload = f"{int(row['die_matrix'])},{row['lifetime_2nd_strike_s']},{row['oee_cycle_time_s']}"
                response = self._runtime.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType="text/csv",
                    Body=payload,
                )
                pred = float(response["Body"].read().decode("utf-8").strip())
                predictions.append(pred)
            return pd.Series(predictions, index=df.index)
        else:
            X = df[self.features]
            predictions = self.model.predict(X)
            return pd.Series(predictions, index=df.index)


def main():
    parser = argparse.ArgumentParser(
        description="Predict total bath time from early-stage forging features."
    )
    parser.add_argument("--die-matrix", type=int, required=True)
    parser.add_argument("--strike2", type=float, required=True)
    parser.add_argument("--oee", type=float, default=None)
    parser.add_argument("--endpoint", type=str, default=None,
                        help="SageMaker endpoint name (omit for local model)")
    parser.add_argument("--region", type=str, default="eu-west-1")
    args = parser.parse_args()

    predictor = Predictor(endpoint_name=args.endpoint, region=args.region)
    result = predictor.predict(
        die_matrix=args.die_matrix,
        lifetime_2nd_strike_s=args.strike2,
        oee_cycle_time_s=args.oee,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
