"""
Unit Tests for Base and Ensemble Models.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.baseline_models import HistoricalMeanBaseline, NaiveBaseline, SeasonalNaiveBaseline
from src.ensemble_model import WeightedAveragingEnsemble, StackingEnsemble
from src.utils import save_pickle, load_pickle


@pytest.fixture
def mock_training_data():
    """Generates synthetic features and continuous target."""
    np.random.seed(42)
    X = pd.DataFrame({
        "lag_1": np.linspace(10, 100, 80),
        "lag_24": np.linspace(10, 100, 80) + np.random.normal(0, 5, 80),
        "temp": np.random.uniform(0.1, 0.9, 80),
        "hour": np.random.randint(0, 24, 80),
    })
    y = 0.7 * X["lag_1"] + 0.3 * X["lag_24"] + 10 * X["temp"] + np.random.normal(0, 2, 80)
    return X, y.to_numpy()


def test_baseline_models(mock_training_data):
    """Test fitting and prediction logic of all baseline models."""
    X, y = mock_training_data

    mean_base = HistoricalMeanBaseline().fit(X, y)
    mean_preds = mean_base.predict(X)
    assert len(mean_preds) == len(X)
    assert np.allclose(mean_preds, np.mean(y))

    naive_base = NaiveBaseline().fit(X, y)
    naive_preds = naive_base.predict(X)
    assert len(naive_preds) == len(X)
    assert np.allclose(naive_preds, X["lag_1"].values)

    seasonal_base = SeasonalNaiveBaseline().fit(X, y)
    seasonal_preds = seasonal_base.predict(X)
    assert len(seasonal_preds) == len(X)
    assert np.allclose(seasonal_preds, X["lag_24"].values)


def test_weighted_averaging_ensemble(mock_training_data):
    """Test weight optimization and prediction summing to 1."""
    X, y = mock_training_data

    # Train mini base models
    rf = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42).fit(X, y)
    gb = GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=42).fit(X, y)

    weighted = WeightedAveragingEnsemble(rf_model=rf, gb_model=gb)
    w_rf, w_gb, val_rmse = weighted.optimize_weights(X, y)

    assert np.isclose(w_rf + w_gb, 1.0)
    preds = weighted.predict(X)
    assert len(preds) == len(X)
    assert (preds >= 0).all()


def test_stacking_ensemble(mock_training_data):
    """Test chronological stacking ensemble with out-of-fold training."""
    X, y = mock_training_data

    rf = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=42)

    stack = StackingEnsemble(rf_model=rf, gb_model=gb, n_splits=3)
    stack.fit_with_oof(X, y)

    preds = stack.predict(X)
    assert len(preds) == len(X)
    assert (preds >= 0).all()


def test_model_serialization(tmp_path, mock_training_data):
    """Test that models can be pickled and unpickled without state corruption."""
    X, y = mock_training_data
    rf = RandomForestRegressor(n_estimators=5, random_state=42).fit(X, y)

    save_path = tmp_path / "test_rf.pkl"
    save_pickle(rf, save_path)
    loaded_rf = load_pickle(save_path)

    orig_preds = rf.predict(X)
    loaded_preds = loaded_rf.predict(X)
    assert np.allclose(orig_preds, loaded_preds)

