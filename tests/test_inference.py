import pytest
import pandas as pd
from pathlib import Path

from vaultech_analysis.inference import Predictor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
GOLD_FILE = PROJECT_ROOT / "data" / "gold" / "pieces.parquet"


@pytest.fixture(scope="module")
def predictor():
    return Predictor(model_dir=MODEL_DIR, gold_file=GOLD_FILE)


def test_predict_returns_prediction(predictor):
    result = predictor.predict(die_matrix=5052, lifetime_2nd_strike_s=18.3, oee_cycle_time_s=13.5)
    assert "predicted_bath_time_s" in result
    assert 40 < result["predicted_bath_time_s"] < 80


def test_predict_without_oee(predictor):
    result = predictor.predict(die_matrix=5052, lifetime_2nd_strike_s=18.3)
    assert "predicted_bath_time_s" in result
    assert result["oee_cycle_time_s"] is None


def test_predict_all_matrices(predictor):
    for matrix in [4974, 5052, 5090, 5091]:
        result = predictor.predict(die_matrix=matrix, lifetime_2nd_strike_s=18.0)
        assert "predicted_bath_time_s" in result
        assert result["predicted_bath_time_s"] > 0


def test_predict_invalid_matrix(predictor):
    result = predictor.predict(die_matrix=9999, lifetime_2nd_strike_s=18.0)
    assert "error" in result


def test_predict_includes_metrics(predictor):
    result = predictor.predict(die_matrix=5052, lifetime_2nd_strike_s=18.3)
    assert "model_metrics" in result
    assert "rmse" in result["model_metrics"]
    assert "mae" in result["model_metrics"]
    assert "r2" in result["model_metrics"]


def test_predict_slow_piece_higher_prediction(predictor):
    normal = predictor.predict(die_matrix=5052, lifetime_2nd_strike_s=18.0)
    slow = predictor.predict(die_matrix=5052, lifetime_2nd_strike_s=30.0)
    assert slow["predicted_bath_time_s"] > normal["predicted_bath_time_s"]


def test_predict_batch(predictor):
    df = pd.DataFrame([
        {"die_matrix": 5052, "lifetime_2nd_strike_s": 18.3, "oee_cycle_time_s": 13.5},
        {"die_matrix": 5090, "lifetime_2nd_strike_s": 17.8, "oee_cycle_time_s": 14.0},
        {"die_matrix": 5091, "lifetime_2nd_strike_s": 18.6, "oee_cycle_time_s": None},
    ])
    predictions = predictor.predict_batch(df)
    assert len(predictions) == 3
    assert all(predictions > 0)
