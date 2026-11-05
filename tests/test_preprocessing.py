"""
Unit Tests for Data Loading, Validation, and Preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_validation import DataValidator
from src.preprocessing import Preprocessor
from src.config import TARGET_COLUMN, LEAKAGE_COLUMNS


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Creates a synthetic hourly dataframe mimicking UCI Bike Sharing schema."""
    dates = pd.date_range("2011-01-01 00:00:00", periods=50, freq="h")
    df = pd.DataFrame({
        "instant": range(1, 51),
        "dteday": dates.strftime("%Y-%m-%d"),
        "hr": dates.hour,
        "season": [1] * 50,
        "yr": [0] * 50,
        "mnth": [1] * 50,
        "holiday": [0] * 50,
        "weekday": dates.dayofweek,
        "workingday": [1 if d < 5 else 0 for d in dates.dayofweek],
        "weathersit": [1] * 50,
        "temp": np.linspace(0.2, 0.6, 50),
        "atemp": np.linspace(0.25, 0.65, 50),
        "hum": np.linspace(0.4, 0.8, 50),
        "windspeed": np.linspace(0.1, 0.3, 50),
        "casual": np.random.randint(1, 20, 50),
        "registered": np.random.randint(10, 100, 50),
    })
    df[TARGET_COLUMN] = df["casual"] + df["registered"]
    return df


def test_data_validator_schema(sample_raw_data):
    """Test schema verification detects all valid raw columns."""
    valid, errors = DataValidator.validate_raw_schema(sample_raw_data)
    assert valid is True
    assert len(errors) == 0


def test_data_validator_missing_values(sample_raw_data):
    """Test missing values counting."""
    nulls = DataValidator.validate_missing_values(sample_raw_data)
    assert sum(nulls.values()) == 0


def test_data_validator_leakage_detection():
    """Test that target leakage check flags forbidden columns."""
    clean_cols = ["temp", "hum", "windspeed", "hour"]
    ok, leaks = DataValidator.check_for_target_leakage(clean_cols)
    assert ok is True
    assert len(leaks) == 0

    leaky_cols = ["temp", "casual", "registered"]
    ok_leak, leaks_found = DataValidator.check_for_target_leakage(leaky_cols)
    assert ok_leak is False
    assert "casual" in leaks_found
    assert "registered" in leaks_found


def test_preprocessor_datetime_and_chronology(sample_raw_data):
    """Test datetime creation and strict chronological sorting."""
    # Shuffle intentionally
    shuffled = sample_raw_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
    prep = Preprocessor()
    clean = prep.clean_raw_data(shuffled)

    assert "datetime" in clean.columns
    assert clean["datetime"].is_monotonic_increasing is True
    assert len(clean) == len(sample_raw_data)


def test_preprocessor_leakage_removal(sample_raw_data):
    """Test removal of casual, registered, and instant columns."""
    prep = Preprocessor()
    no_leak = prep.remove_leakage_columns(sample_raw_data, keep_target=True)

    assert "casual" not in no_leak.columns
    assert "registered" not in no_leak.columns
    assert "instant" not in no_leak.columns
    assert TARGET_COLUMN in no_leak.columns


def test_chronological_splitting(sample_raw_data):
    """Test chronological split ensures train < val < test in time."""
    prep = Preprocessor()
    clean = prep.clean_raw_data(sample_raw_data)
    train_df, val_df, test_df = prep.split_chronologically(clean, train_ratio=0.6, val_ratio=0.2)

    assert len(train_df) == int(len(clean) * 0.6)
    assert len(val_df) == int(len(clean) * 0.2)
    assert len(test_df) == len(clean) - len(train_df) - len(val_df)

    # Verify temporal boundary
    assert train_df["datetime"].max() < val_df["datetime"].min()
    assert val_df["datetime"].max() < test_df["datetime"].min()

