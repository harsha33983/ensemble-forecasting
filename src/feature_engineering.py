"""
Feature Engineering Module for Time-Series Forecasting.

Constructs cyclical calendar features, lagged demand observations,
rolling statistical aggregations, and meteorological interaction indices
with strict anti-leakage guarantees.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

from src.config import LAG_HOURS, ROLLING_WINDOWS, TARGET_COLUMN
from src.utils import setup_logger

logger = setup_logger("feature_engineering")


class FeatureEngineer:
    """Generates temporal, cyclical, meteorological, lag, and rolling features."""

    def __init__(
        self,
        lag_hours: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
    ):
        self.lag_hours = lag_hours or LAG_HOURS
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS
        self.feature_columns: List[str] = []

    def create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract date-time calendar attributes and compute cyclical sine/cosine transformations.
        """
        df_out = df.copy()

        if "datetime" not in df_out.columns:
            if "dteday" in df_out.columns and "hr" in df_out.columns:
                df_out["datetime"] = pd.to_datetime(df_out["dteday"]) + pd.to_timedelta(df_out["hr"], unit="h")
            else:
                raise ValueError("DataFrame must contain 'datetime' or ('dteday' and 'hr').")

        dt = pd.to_datetime(df_out["datetime"])

        # Temporal components
        df_out["year"] = dt.dt.year
        df_out["month"] = dt.dt.month
        df_out["day"] = dt.dt.day
        df_out["hour"] = dt.dt.hour
        df_out["dayofweek"] = dt.dt.dayofweek
        df_out["quarter"] = dt.dt.quarter
        df_out["dayofyear"] = dt.dt.dayofyear
        df_out["is_weekend"] = (df_out["dayofweek"] >= 5).astype(int)

        # Rush hour indicators (commuter peaks at 8-9 AM and 5-6 PM on working days)
        is_rush_morning = (df_out["hour"].isin([8, 9])) & (df_out["is_weekend"] == 0)
        is_rush_evening = (df_out["hour"].isin([17, 18])) & (df_out["is_weekend"] == 0)
        df_out["is_rush_hour"] = (is_rush_morning | is_rush_evening).astype(int)

        # Cyclical transformations (smooth boundary periodicity)
        df_out["sin_hour"] = np.sin(2 * np.pi * df_out["hour"] / 24.0)
        df_out["cos_hour"] = np.cos(2 * np.pi * df_out["hour"] / 24.0)

        df_out["sin_month"] = np.sin(2 * np.pi * df_out["month"] / 12.0)
        df_out["cos_month"] = np.cos(2 * np.pi * df_out["month"] / 12.0)

        df_out["sin_dayofweek"] = np.sin(2 * np.pi * df_out["dayofweek"] / 7.0)
        df_out["cos_dayofweek"] = np.cos(2 * np.pi * df_out["dayofweek"] / 7.0)

        return df_out

    def create_meteorological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive interaction terms and physical comfort indices from weather observations.
        """
        df_out = df.copy()

        if "temp" in df_out.columns and "hum" in df_out.columns:
            # Heat/Humidity combined stress indicator
            df_out["temp_hum_ratio"] = df_out["temp"] / (df_out["hum"] + 0.05)

        if "atemp" in df_out.columns and "temp" in df_out.columns:
            # Perceived temperature difference (wind chill or solar radiation effect)
            df_out["atemp_diff"] = df_out["atemp"] - df_out["temp"]

        if "windspeed" in df_out.columns and "temp" in df_out.columns:
            # Wind cooling effect
            df_out["wind_temp_mult"] = df_out["windspeed"] * df_out["temp"]

        # Weather severity penalty index (weathersit: 1=nice to 4=severe)
        if "weathersit" in df_out.columns:
            df_out["is_bad_weather"] = (df_out["weathersit"] >= 3).astype(int)

        return df_out

    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET_COLUMN,
    ) -> pd.DataFrame:
        """
        Create chronological historical lag features.
        CRITICAL: All lags are shifted by >= 1 step to prevent using future target values.
        """
        df_out = df.copy()

        if target_col not in df_out.columns:
            logger.warning("Target column '%s' not found for lag feature generation.", target_col)
            return df_out

        # Ensure sorted chronologically
        df_out = df_out.sort_values("datetime").reset_index(drop=True)

        for lag in self.lag_hours:
            col_name = f"lag_{lag}"
            df_out[col_name] = df_out[target_col].shift(lag)

        return df_out

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET_COLUMN,
    ) -> pd.DataFrame:
        """
        Create rolling window statistics and expanding historical demand statistics.
        CRITICAL: Target is strictly shifted by 1 before any rolling aggregation is computed.
        """
        df_out = df.copy()

        if target_col not in df_out.columns:
            logger.warning("Target column '%s' not found for rolling feature generation.", target_col)
            return df_out

        df_out = df_out.sort_values("datetime").reset_index(drop=True)

        # Strictly shifted base: only past data up to t-1 is visible at time t
        past_target = df_out[target_col].shift(1)

        for window in self.rolling_windows:
            df_out[f"rolling_mean_{window}h"] = past_target.rolling(window=window, min_periods=1).mean()
            if window in [24, 168]:
                df_out[f"rolling_std_{window}h"] = past_target.rolling(window=window, min_periods=2).std().fillna(0.0)

        # Expanding historical average up to t-1
        df_out["expanding_mean"] = past_target.expanding(min_periods=1).mean()

        return df_out

    def build_full_feature_pipeline(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET_COLUMN,
        drop_na: bool = True,
    ) -> pd.DataFrame:
        """
        Execute full end-to-end feature creation sequence:
        1. Calendar & cyclical features
        2. Meteorological interactions
        3. Shifted lag features
        4. Shifted rolling statistics
        """
        logger.info("Executing comprehensive feature engineering...")
        df_feat = self.create_calendar_features(df)
        df_feat = self.create_meteorological_features(df_feat)

        if target_col in df_feat.columns:
            df_feat = self.create_lag_features(df_feat, target_col=target_col)
            df_feat = self.create_rolling_features(df_feat, target_col=target_col)

        if drop_na:
            initial_len = len(df_feat)
            df_feat = df_feat.dropna().reset_index(drop=True)
            dropped = initial_len - len(df_feat)
            logger.info("Dropped %d warm-up rows containing NaN from largest lag window.", dropped)

        # Identify all engineered feature columns
        metadata_cols = ["datetime", "dteday", "instant", "casual", "registered", target_col]
        self.feature_columns = [c for c in df_feat.columns if c not in metadata_cols]
        logger.info("Engineered %d total features. Ready for model ingestion.", len(self.feature_columns))

        return df_feat

