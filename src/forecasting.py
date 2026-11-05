"""
Forecasting Engine Module.

Implements reusable multi-step recursive forecasting:
- Dynamically updates autoregressive lags and rolling statistical windows
- Generates future timestamps and cyclical calendars
- Accepts customizable meteorological scenarios
- Validates input schemas
- Produces point forecasts with predictive intervals and exports to CSV
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PREDICTIONS_DIR, RANDOM_STATE
from src.utils import load_json, load_pickle, save_pickle, setup_logger

logger = setup_logger("forecasting")


class ForecastingEngine:
    """Multi-step recursive time-series forecasting engine using trained ensemble."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ):
        self.models_dir = Path(MODELS_DIR)
        self.model_path = Path(model_path) if model_path else self.models_dir / "ensemble_model.pkl"
        self.metadata_path = Path(metadata_path) if metadata_path else self.models_dir / "model_metadata.json"

        self.model = None
        self.metadata: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self._load_resources()

    def _load_resources(self) -> None:
        """Load model weights and feature schema metadata."""
        if self.model_path.exists():
            self.model = load_pickle(self.model_path)
            logger.info("ForecastingEngine loaded ensemble model from %s", self.model_path)
        else:
            logger.warning("Ensemble model not found at %s. ForecastingEngine requires trained model.", self.model_path)

        if self.metadata_path.exists():
            self.metadata = load_json(self.metadata_path)
            self.feature_names = self.metadata.get("feature_names", [])
            logger.info("Loaded metadata with %d expected features.", len(self.feature_names))

    def validate_historical_context(self, history_df: pd.DataFrame) -> None:
        """Verify that historical context contains sufficient sequence depth and required columns."""
        if not isinstance(history_df, pd.DataFrame):
            raise TypeError("history_df must be a pandas DataFrame.")

        required = ["datetime", "cnt"]
        missing = [c for c in required if c not in history_df.columns]
        if missing:
            raise ValueError(f"Historical context is missing mandatory columns: {missing}")

        if len(history_df) < 168:
            raise ValueError(
                f"Historical context must contain at least 168 hourly rows to seed weekly lags and rolling statistics. "
                f"Provided only {len(history_df)} rows."
            )

    def forecast(
        self,
        history_df: pd.DataFrame,
        horizon_hours: int = 24,
        future_weather: Optional[pd.DataFrame] = None,
        output_csv: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Execute recursive multi-step forecasting for the next `horizon_hours`.

        Parameters:
        -----------
        history_df : pd.DataFrame
            Historical context containing at least 168 rows with 'datetime' and 'cnt'.
        horizon_hours : int
            Number of future hours to forecast (e.g. 24, 48, 72).
        future_weather : Optional[pd.DataFrame]
            Optional future exogenous meteorological variables (temp, atemp, hum, windspeed, weathersit).
            If omitted, uses seasonal historical mean values.
        output_csv : Optional[Union[str, Path]]
            Path to export forecast results as CSV.

        Returns:
        --------
        pd.DataFrame
            Forecasting results with columns: datetime, forecast_cnt, lower_95, upper_95.
        """
        if self.model is None:
            raise RuntimeError("Cannot forecast: No trained ensemble model is loaded!")

        self.validate_historical_context(history_df)

        if horizon_hours < 1 or horizon_hours > 168:
            raise ValueError(f"Horizon hours must be between 1 and 168. Given: {horizon_hours}")

        logger.info("Initiating recursive forecast for %d hours...", horizon_hours)

        # Sort history chronologically
        history_clean = history_df.sort_values("datetime").reset_index(drop=True).copy()
        last_dt = pd.to_datetime(history_clean["datetime"].iloc[-1])

        # Buffer of historical demand counts for lag & rolling computation
        demand_buffer = history_clean["cnt"].tolist()
        timestamps_buffer = pd.to_datetime(history_clean["datetime"]).tolist()

        # Typical default weather values if future weather is not explicitly provided
        recent_temp = float(history_clean["temp"].iloc[-24:].mean()) if "temp" in history_clean else 0.5
        recent_atemp = float(history_clean["atemp"].iloc[-24:].mean()) if "atemp" in history_clean else 0.5
        recent_hum = float(history_clean["hum"].iloc[-24:].mean()) if "hum" in history_clean else 0.6
        recent_wind = float(history_clean["windspeed"].iloc[-24:].mean()) if "windspeed" in history_clean else 0.2
        recent_weather = int(history_clean["weathersit"].mode()[0]) if "weathersit" in history_clean else 1

        forecast_records = []

        for step in range(1, horizon_hours + 1):
            curr_dt = last_dt + pd.Timedelta(hours=step)

            # 1. Weather features for this future hour
            if future_weather is not None and len(future_weather) >= step:
                w_row = future_weather.iloc[step - 1]
                t_val = float(w_row.get("temp", recent_temp))
                at_val = float(w_row.get("atemp", recent_atemp))
                h_val = float(w_row.get("hum", recent_hum))
                w_spd = float(w_row.get("windspeed", recent_wind))
                w_sit = int(w_row.get("weathersit", recent_weather))
            else:
                t_val = recent_temp
                at_val = recent_atemp
                h_val = recent_hum
                w_spd = recent_wind
                w_sit = recent_weather

            # 2. Calendar and cyclical features
            hr = curr_dt.hour
            month = curr_dt.month
            dow = curr_dt.dayofweek
            is_weekend = int(dow >= 5)
            is_rush_morning = (hr in [8, 9]) and (is_weekend == 0)
            is_rush_evening = (hr in [17, 18]) and (is_weekend == 0)
            is_rush = int(is_rush_morning or is_rush_evening)

            # Approximate season from month
            if month in [12, 1, 2]:
                season = 1
            elif month in [3, 4, 5]:
                season = 2
            elif month in [6, 7, 8]:
                season = 3
            else:
                season = 4

            yr = 1 if curr_dt.year >= 2012 else 0

            # 3. Autoregressive lags from dynamic demand buffer
            # demand_buffer[-1] is the value at t+step-1
            lag_1 = demand_buffer[-1]
            lag_2 = demand_buffer[-2]
            lag_3 = demand_buffer[-3]
            lag_24 = demand_buffer[-24]
            lag_48 = demand_buffer[-48]
            lag_168 = demand_buffer[-168]

            # 4. Rolling statistics strictly up to t+step-1
            rolling_6h = float(np.mean(demand_buffer[-6:]))
            rolling_24h = float(np.mean(demand_buffer[-24:]))
            rolling_std_24h = float(np.std(demand_buffer[-24:]))
            rolling_168h = float(np.mean(demand_buffer[-168:]))
            expanding_mean = float(np.mean(demand_buffer))

            # 5. Assemble candidate feature dictionary
            feat_dict = {
                "season": season,
                "yr": yr,
                "mnth": month,
                "hr": hr,
                "holiday": 0,
                "weekday": dow,
                "workingday": 1 if is_weekend == 0 else 0,
                "weathersit": w_sit,
                "temp": t_val,
                "atemp": at_val,
                "hum": h_val,
                "windspeed": w_spd,
                "year": curr_dt.year,
                "month": month,
                "day": curr_dt.day,
                "hour": hr,
                "dayofweek": dow,
                "quarter": curr_dt.quarter,
                "dayofyear": curr_dt.dayofyear,
                "is_weekend": is_weekend,
                "is_rush_hour": is_rush,
                "sin_hour": np.sin(2 * np.pi * hr / 24.0),
                "cos_hour": np.cos(2 * np.pi * hr / 24.0),
                "sin_month": np.sin(2 * np.pi * month / 12.0),
                "cos_month": np.cos(2 * np.pi * month / 12.0),
                "sin_dayofweek": np.sin(2 * np.pi * dow / 7.0),
                "cos_dayofweek": np.cos(2 * np.pi * dow / 7.0),
                "temp_hum_ratio": t_val / (h_val + 0.05),
                "atemp_diff": at_val - t_val,
                "wind_temp_mult": w_spd * t_val,
                "is_bad_weather": int(w_sit >= 3),
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "lag_24": lag_24,
                "lag_48": lag_48,
                "lag_168": lag_168,
                "rolling_mean_6h": rolling_6h,
                "rolling_mean_24h": rolling_24h,
                "rolling_std_24h": rolling_std_24h,
                "rolling_mean_168h": rolling_168h,
                "expanding_mean": expanding_mean,
            }

            # Align with model's expected feature order
            if self.feature_names:
                row_df = pd.DataFrame([{col: feat_dict.get(col, 0.0) for col in self.feature_names}])
            else:
                row_df = pd.DataFrame([feat_dict])

            # Predict next step
            pred_step = float(self.model.predict(row_df)[0])
            pred_step = max(0.0, pred_step)

            # Approximate 95% prediction interval (using empirical residual error ~ 1.96 * RMSE)
            est_rmse = 40.0
            val_metrics = self.metadata.get("validation_metrics", {}).get(self.metadata.get("selected_ensemble", "Weighted Ensemble"), {})
            if "RMSE" in val_metrics:
                est_rmse = val_metrics["RMSE"]

            lower_95 = max(0.0, pred_step - 1.96 * est_rmse * np.sqrt(1 + 0.02 * step))
            upper_95 = pred_step + 1.96 * est_rmse * np.sqrt(1 + 0.02 * step)

            # Record prediction
            forecast_records.append({
                "datetime": curr_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "hour": hr,
                "forecast_cnt": round(pred_step, 1),
                "lower_95": round(lower_95, 1),
                "upper_95": round(upper_95, 1),
            })

            # Append prediction to recursive historical buffer
            demand_buffer.append(pred_step)
            timestamps_buffer.append(curr_dt)

        forecast_df = pd.DataFrame(forecast_records)

        # Export CSV if requested or default
        csv_dest = Path(output_csv) if output_csv else Path(PREDICTIONS_DIR) / "future_forecast.csv"
        csv_dest.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(csv_dest, index=False)
        logger.info("Successfully generated %d-hour recursive forecast. Saved to %s", horizon_hours, csv_dest)

        return forecast_df

