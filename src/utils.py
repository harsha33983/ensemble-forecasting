"""
Utility Functions for Logging, Serialization, and Metrics.
"""

import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def setup_logger(name: str = "ensemble_forecasting", log_file: Union[str, Path, None] = None) -> logging.Logger:
    """Set up and return a structured console and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


import joblib


def save_pickle(obj: Any, file_path: Union[str, Path]) -> None:
    """Save an object using joblib with compression and directory creation."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)


def load_pickle(file_path: Union[str, Path]) -> Any:
    """Load an object using joblib (supports compressed and uncompressed files)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    return joblib.load(path)


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save a dictionary as a formatted JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a dictionary from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression and forecasting evaluation metrics.
    Handles potential zeros in y_true for MAPE gracefully.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Avoid negative counts for count forecasts
    y_pred_clipped = np.clip(y_pred, 0, None)

    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    mse = float(mean_squared_error(y_true, y_pred_clipped))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred_clipped))

    # Safe MAPE: exclude exact zeros to avoid division by zero
    nonzero_mask = y_true > 0
    if np.any(nonzero_mask):
        mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred_clipped[nonzero_mask]) / y_true[nonzero_mask])) * 100.0)
    else:
        mape = float("nan")

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
    }


def calculate_improvement(baseline_rmse: float, candidate_rmse: float) -> float:
    """
    Calculate relative percentage improvement in RMSE:
    Improvement (%) = ((baseline_rmse - candidate_rmse) / baseline_rmse) * 100
    """
    if baseline_rmse <= 0:
        return 0.0
    return round(((baseline_rmse - candidate_rmse) / baseline_rmse) * 100.0, 2)

