"""
Preprocessing Pipeline Module.

Handles datetime parsing, chronological sorting, leakage removal,
and scikit-learn compatible transformation pipelines strictly fitted on training data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.config import (
    LEAKAGE_COLUMNS,
    PROCESSED_DATA_DIR,
    PROCESSED_FILE,
    TARGET_COLUMN,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)
from src.utils import setup_logger

logger = setup_logger("preprocessing")


class Preprocessor:
    """Preprocesses raw bike sharing data chronologically without future leakage."""

    def __init__(self):
        self.fitted = False
        self.numeric_features = ["temp", "atemp", "hum", "windspeed"]
        self.categorical_features = ["season", "weathersit"]
        self.pipeline: Optional[ColumnTransformer] = None

    def create_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine 'dteday' and 'hr' into a single precise datetime column
        and sort strictly in chronological order.
        """
        df_copy = df.copy()
        if "datetime" not in df_copy.columns:
            df_copy["datetime"] = pd.to_datetime(df_copy["dteday"]) + pd.to_timedelta(df_copy["hr"], unit="h")

        df_copy = df_copy.sort_values("datetime").reset_index(drop=True)
        return df_copy

    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by converting datetime, dropping duplicates,
        and checking target validity.
        """
        df_clean = self.create_datetime_index(df)

        # Remove duplicate timestamps if any exist, keeping the first occurrence
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=["datetime"], keep="first").reset_index(drop=True)
        dropped_dups = initial_count - len(df_clean)
        if dropped_dups > 0:
            logger.info("Dropped %d duplicate timestamp rows.", dropped_dups)

        # Handle missing values if any
        if df_clean.isnull().any().any():
            logger.info("Imputing missing values with forward fill followed by backward fill.")
            df_clean = df_clean.ffill().bfill()

        return df_clean

    def remove_leakage_columns(self, df: pd.DataFrame, keep_target: bool = True) -> pd.DataFrame:
        """
        Remove casual, registered, and instant identifiers to prevent target leakage.
        """
        drop_cols = [c for c in LEAKAGE_COLUMNS if c in df.columns]
        if not keep_target and TARGET_COLUMN in df.columns:
            drop_cols.append(TARGET_COLUMN)

        cleaned_df = df.drop(columns=drop_cols, errors="ignore")
        logger.info("Removed leakage columns: %s", drop_cols)
        return cleaned_df

    def split_chronologically(
        self,
        df: pd.DataFrame,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test chronologically.
        Guarantees that train < val < test in time, eliminating future data leakage.
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy().reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_df = df.iloc[val_end:].copy().reset_index(drop=True)

        logger.info(
            "Chronological split sizes: Train=%d (%0.1f%%), Val=%d (%0.1f%%), Test=%d (%0.1f%%)",
            len(train_df), (len(train_df)/n)*100,
            len(val_df), (len(val_df)/n)*100,
            len(test_df), (len(test_df)/n)*100,
        )
        if "datetime" in df.columns:
            logger.info("Train timeframe: %s to %s", train_df["datetime"].min(), train_df["datetime"].max())
            logger.info("Val timeframe:   %s to %s", val_df["datetime"].min(), val_df["datetime"].max())
            logger.info("Test timeframe:  %s to %s", test_df["datetime"].min(), test_df["datetime"].max())

        return train_df, val_df, test_df

    def fit_transformer(self, train_df: pd.DataFrame) -> ColumnTransformer:
        """
        Fit sklearn ColumnTransformer on TRAINING data only.
        Scales continuous environmental features and one-hot encodes weather/season categories.
        """
        num_cols = [c for c in self.numeric_features if c in train_df.columns]
        cat_cols = [c for c in self.categorical_features if c in train_df.columns]

        transformers = []
        if num_cols:
            transformers.append(("num_scaler", StandardScaler(), num_cols))
        if cat_cols:
            transformers.append(("cat_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))

        self.pipeline = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        self.pipeline.fit(train_df)
        self.fitted = True
        logger.info("ColumnTransformer fitted successfully on %d training rows.", len(train_df))
        return self.pipeline

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using previously fitted ColumnTransformer."""
        if not self.fitted or self.pipeline is None:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform()!")

        transformed_array = self.pipeline.transform(df)
        feature_names = self.pipeline.get_feature_names_out()
        return pd.DataFrame(transformed_array, columns=feature_names, index=df.index)

