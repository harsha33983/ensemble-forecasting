"""
Unit Tests for Forecasting Engine.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

from src.forecasting import ForecastingEngine
from src.config import TARGET_COLUMN


@pytest.fixture
def mock_history_context() -> pd.DataFrame:
    """Creates a 200-hour historical sequence with datetime and count."""
    dates = pd.date_range("2012-01-01 00:00:00", periods=200, freq="h")
    df = pd.DataFrame({
        "datetime": dates,
        "temp": np.linspace(0.3, 0.7, 200),
        "atemp": np.linspace(0.35, 0.75, 200),
        "hum": np.linspace(0.4, 0.8, 200),
        "windspeed": [0.2] * 200,
        "weathersit": [1] * 200,
        TARGET_COLUMN: np.random.randint(50, 400, 200),
    })
    return df


def test_forecasting_input_validation(mock_history_context):
    """Test validation of required columns and minimum historical sequence depth."""
    engine = ForecastingEngine()

    # Empty / small dataframe should raise ValueError
    too_short = mock_history_context.iloc[:50]
    with pytest.raises(ValueError, match="at least 168 hourly rows"):
        engine.validate_historical_context(too_short)

    # Missing column should raise ValueError
    missing_col = mock_history_context.drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError, match="missing mandatory columns"):
        engine.validate_historical_context(missing_col)


def test_recursive_forecasting_execution(tmp_path, mock_history_context):
    """Test multi-step recursive forecasting outputs, intervals, and CSV export."""
    engine = ForecastingEngine()

    # Assign a simple mock model to test execution without pre-saved artifacts
    class SimpleModel:
        def predict(self, X):
            return np.full(len(X), 150.0)

    engine.model = SimpleModel()
    engine.feature_names = ["temp", "atemp", "hum", "windspeed", "lag_1", "rolling_mean_24h"]

    out_csv = tmp_path / "test_forecast.csv"
    forecast_df = engine.forecast(
        history_df=mock_history_context,
        horizon_hours=12,
        output_csv=out_csv,
    )

    assert len(forecast_df) == 12
    assert "datetime" in forecast_df.columns
    assert "forecast_cnt" in forecast_df.columns
    assert "lower_95" in forecast_df.columns
    assert "upper_95" in forecast_df.columns

    # Verify intervals are ordered correctly
    assert (forecast_df["lower_95"] <= forecast_df["forecast_cnt"]).all()
    assert (forecast_df["forecast_cnt"] <= forecast_df["upper_95"]).all()

    # Verify CSV was created
    assert out_csv.exists()

