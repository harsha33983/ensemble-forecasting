"""
Unit Tests for Feature Engineering and Leakage Prevention.
"""

import pytest
import numpy as np
import pandas as pd

from src.feature_engineering import FeatureEngineer
from src.config import TARGET_COLUMN


@pytest.fixture
def sample_feature_data() -> pd.DataFrame:
    """Creates a sequential hourly dataset of 200 hours for feature testing."""
    dates = pd.date_range("2011-01-01 00:00:00", periods=200, freq="h")
    df = pd.DataFrame({
        "datetime": dates,
        "season": [1] * 200,
        "yr": [0] * 200,
        "mnth": dates.month,
        "hr": dates.hour,
        "holiday": [0] * 200,
        "weekday": dates.dayofweek,
        "workingday": [1 if d < 5 else 0 for d in dates.dayofweek],
        "weathersit": [1] * 200,
        "temp": np.linspace(0.2, 0.6, 200),
        "atemp": np.linspace(0.25, 0.65, 200),
        "hum": np.linspace(0.4, 0.8, 200),
        "windspeed": np.linspace(0.1, 0.3, 200),
        TARGET_COLUMN: np.arange(100, 300),  # strictly increasing sequence 100, 101, ...
    })
    return df


def test_calendar_and_cyclical_features(sample_feature_data):
    """Test calendar extraction and cyclical bounds [-1, 1]."""
    fe = FeatureEngineer()
    df_cal = fe.create_calendar_features(sample_feature_data)

    assert "hour" in df_cal.columns
    assert "sin_hour" in df_cal.columns
    assert "cos_hour" in df_cal.columns
    assert "is_weekend" in df_cal.columns
    assert "is_rush_hour" in df_cal.columns

    # Verify cyclical bounds
    assert df_cal["sin_hour"].between(-1.0001, 1.0001).all()
    assert df_cal["cos_hour"].between(-1.0001, 1.0001).all()


def test_meteorological_interaction_features(sample_feature_data):
    """Test meteorological interaction terms."""
    fe = FeatureEngineer()
    df_met = fe.create_meteorological_features(sample_feature_data)

    assert "temp_hum_ratio" in df_met.columns
    assert "atemp_diff" in df_met.columns
    assert "wind_temp_mult" in df_met.columns
    assert not df_met["temp_hum_ratio"].isnull().any()


def test_strict_anti_leakage_in_lags(sample_feature_data):
    """
    CRITICAL TEST: Verify that lag_1 at row t equals target value at row t-1,
    and row t target is NEVER visible at row t.
    """
    fe = FeatureEngineer(lag_hours=[1, 2, 24])
    df_lag = fe.create_lag_features(sample_feature_data, target_col=TARGET_COLUMN)

    assert "lag_1" in df_lag.columns
    assert "lag_2" in df_lag.columns
    assert "lag_24" in df_lag.columns

    # Check row 10: target is 110. lag_1 must be 109, lag_2 must be 108.
    idx = 10
    assert df_lag.loc[idx, TARGET_COLUMN] == 110
    assert df_lag.loc[idx, "lag_1"] == 109
    assert df_lag.loc[idx, "lag_2"] == 108


def test_strict_anti_leakage_in_rolling_statistics(sample_feature_data):
    """
    CRITICAL TEST: Verify that rolling aggregations at time t are calculated
    strictly using target observations up to time t-1.
    """
    fe = FeatureEngineer(rolling_windows=[6, 24])
    df_roll = fe.create_rolling_features(sample_feature_data, target_col=TARGET_COLUMN)

    assert "rolling_mean_6h" in df_roll.columns
    assert "rolling_mean_24h" in df_roll.columns

    # At row 10 (target=110), past target values are 104, 105, 106, 107, 108, 109.
    # Mean of past 6 values = 106.5.
    # If unshifted target was leaked, mean would be 107.0.
    expected_mean_6h = np.mean([104, 105, 106, 107, 108, 109])
    actual_mean_6h = df_roll.loc[10, "rolling_mean_6h"]
    assert np.isclose(actual_mean_6h, expected_mean_6h)

