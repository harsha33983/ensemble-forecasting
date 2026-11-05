"""
Configuration Module for Ensemble Forecasting Project.

Defines all file paths, random seeds, split ratios, feature definitions,
and hyperparameters for reproducible execution.
"""

from pathlib import Path
from typing import Dict, List, Any

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
TESTS_DIR = BASE_DIR / "tests"
DATABASE_DIR = BASE_DIR / "database"
DATABASE_FILE = DATABASE_DIR / "ensemble_forecasting.db"
DATABASE_PATH = DATABASE_FILE
DATABASE_SCHEMA_FILE = DATABASE_DIR / "schema.sql"
DATABASE_BACKUPS_DIR = DATABASE_DIR / "backups"
DATABASE_URL = f"sqlite:///{DATABASE_FILE.as_posix()}"

# Dataset Settings
DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
FALLBACK_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
RAW_ZIP_FILE = RAW_DATA_DIR / "bike_sharing_dataset.zip"
RAW_HOUR_FILE = RAW_DATA_DIR / "hour.csv"
RAW_DAY_FILE = RAW_DATA_DIR / "day.csv"
RAW_FILE = RAW_HOUR_FILE
PROCESSED_FILE = PROCESSED_DATA_DIR / "bike_sharing_featured.csv"

# Global Random Seed
RANDOM_STATE = 42

# Target and Dropped Columns
TARGET_COLUMN = "cnt"
LEAKAGE_COLUMNS = ["casual", "registered", "instant"]

# Chronological Splitting (Time-series split: Train 70%, Validation 15%, Test 15%)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Time Series Cross-Validation
N_CV_SPLITS = 5

# Lag and Rolling Configuration
LAG_HOURS: List[int] = [1, 2, 3, 24, 48, 168]
ROLLING_WINDOWS: List[int] = [6, 24, 168]

# Hyperparameter Search Spaces
RF_PARAM_GRID: Dict[str, List[Any]] = {
    "n_estimators": [100, 200, 300],
    "max_depth": [15, 20, 25, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.8, 1.0],
}

GB_PARAM_GRID: Dict[str, List[Any]] = {
    "n_estimators": [150, 250, 350],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [4, 6, 8],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
    "subsample": [0.8, 0.9, 1.0],
}

# Plotting Settings
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

