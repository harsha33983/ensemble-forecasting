"""
Baseline Models Module.

Implements reference benchmark models:
1. Historical Mean Baseline
2. Naive Forecasting Baseline (t-1)
3. Seasonal Naive Baseline (t-24 / same hour previous day)
4. Linear Regression Baseline
5. Decision Tree Regressor Baseline
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from src.config import RANDOM_STATE
from src.utils import calculate_metrics, setup_logger

logger = setup_logger("baseline_models")


class HistoricalMeanBaseline(BaseEstimator, RegressorMixin):
    """Predicts constant historical mean observed during training."""

    def __init__(self):
        self.mean_value_: float = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "HistoricalMeanBaseline":
        self.mean_value_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n_samples = len(X)
        return np.full(shape=(n_samples,), fill_value=self.mean_value_, dtype=np.float64)


class NaiveBaseline(BaseEstimator, RegressorMixin):
    """Predicts previous hour value (lag_1). Falls back to mean if lag_1 is unavailable."""

    def __init__(self, lag_col: str = "lag_1"):
        self.lag_col = lag_col
        self.fallback_value_: float = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "NaiveBaseline":
        self.fallback_value_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and self.lag_col in X.columns:
            preds = X[self.lag_col].fillna(self.fallback_value_).to_numpy()
        else:
            preds = np.full(len(X), self.fallback_value_)
        return np.clip(preds, 0, None)


class SeasonalNaiveBaseline(BaseEstimator, RegressorMixin):
    """Predicts value from the same hour yesterday (lag_24). Falls back to mean."""

    def __init__(self, seasonal_col: str = "lag_24"):
        self.seasonal_col = seasonal_col
        self.fallback_value_: float = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SeasonalNaiveBaseline":
        self.fallback_value_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and self.seasonal_col in X.columns:
            preds = X[self.seasonal_col].fillna(self.fallback_value_).to_numpy()
        else:
            preds = np.full(len(X), self.fallback_value_)
        return np.clip(preds, 0, None)


def train_and_evaluate_baselines(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate all 5 baseline models on test data.
    Returns dictionary with fitted models, predictions, and evaluation metrics.
    """
    logger.info("Training and evaluating baseline models...")

    models = {
        "Historical Mean": HistoricalMeanBaseline(),
        "Naive (Lag 1)": NaiveBaseline(lag_col="lag_1"),
        "Seasonal Naive (Lag 24)": SeasonalNaiveBaseline(seasonal_col="lag_24"),
        "Linear Regression (Ridge)": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_clipped = np.clip(preds, 0, None)
        metrics = calculate_metrics(y_test, preds_clipped)
        logger.info("Baseline '%s': RMSE=%.2f, MAE=%.2f, R2=%.4f", name, metrics["RMSE"], metrics["MAE"], metrics["R2"])
        results[name] = {
            "model": model,
            "predictions": preds_clipped,
            "metrics": metrics,
        }

    return results

