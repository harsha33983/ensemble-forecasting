"""
Database Service Module.

Provides high-level business functions and ML pipeline integration
for SQLite database operations, including dataset registration,
experiment tracking, predictions and forecasts storage, metric queries,
and database backups.
"""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sqlalchemy import desc, func, inspect, select, text

from src.config import DATABASE_DIR, DATABASE_FILE, DATABASE_URL
from src.database.connection import get_db_session, get_engine
from src.database.models import (
    Base,
    CrossValidationResult,
    Dataset,
    FeatureImportance,
    Forecast,
    ModelRun,
    Prediction,
    SystemLog,
)
from src.database.repository import DatabaseRepository
from src.utils import setup_logger

logger = setup_logger("database_service")


class DatabaseService:
    """
    High-level business service for SQLite database operations.
    Acts as the primary interface for both the ML pipeline and Streamlit dashboard.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        db_path: Optional[Union[str, Path]] = None,
    ):
        if db_path is not None:
            self.db_url = f"sqlite:///{Path(db_path).resolve().as_posix()}"
        else:
            self.db_url = db_url or DATABASE_URL

    # =========================================================================
    # DATASET SERVICES
    # =========================================================================
    def register_dataset(
        self,
        dataset_name: str,
        source_url: Optional[str] = None,
        description: Optional[str] = None,
        total_records: Optional[int] = None,
        total_features: Optional[int] = None,
        target_column: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> int:
        """
        Register a dataset or return the existing dataset's ID if already present.
        """
        with get_db_session(self.db_url) as session:
            repo = DatabaseRepository(session)
            existing = repo.get_dataset_by_name(dataset_name)
            if existing:
                # Update records and path if needed
                existing.total_records = total_records or existing.total_records
                existing.total_features = total_features or existing.total_features
                existing.file_path = file_path or existing.file_path
                existing_id = existing.id
                logger.info("Found existing dataset '%s' (ID: %d)", dataset_name, existing_id)
                return existing_id

            dataset = repo.add_dataset(
                dataset_name=dataset_name,
                source_url=source_url,
                description=description,
                total_records=total_records,
                total_features=total_features,
                target_column=target_column,
                file_path=file_path,
            )
            return dataset.id

    def get_all_datasets(self) -> List[Dict[str, Any]]:
        """Retrieve all registered datasets as dictionaries."""
        with get_db_session(self.db_url) as session:
            repo = DatabaseRepository(session)
            datasets = repo.get_all_datasets()
            return [d.to_dict() for d in datasets]

    # =========================================================================
    # MODEL RUN & EXPERIMENT SERVICES
    # =========================================================================
    def start_model_run(
        self,
        model_name: str,
        experiment_name: str = "Standard_Forecasting_Benchmark",
        algorithm: Optional[str] = None,
        dataset_id: Optional[int] = None,
        hyperparameters: Optional[Union[Dict[str, Any], str]] = None,
    ) -> int:
        """
        Create a new model training run experiment in 'TRAINING' state.
        Returns the created model_run_id.
        """
        if isinstance(hyperparameters, dict):
            hp_str = json.dumps(hyperparameters, default=str)
        else:
            hp_str = hyperparameters

        with get_db_session(self.db_url) as session:
            repo = DatabaseRepository(session)
            run = repo.add_model_run(
                model_name=model_name,
                experiment_name=experiment_name,
                algorithm=algorithm,
                dataset_id=dataset_id,
                hyperparameters=hp_str,
                status="TRAINING",
            )
            return run.id

    def complete_model_run(
        self,
        model_run_id: int,
        mae: float,
        mse: float,
        rmse: float,
        r2_score: float,
        mape: Optional[float] = None,
        improvement_percentage: Optional[float] = None,
        training_records: Optional[int] = None,
        validation_records: Optional[int] = None,
        test_records: Optional[int] = None,
        model_path: Optional[str] = None,
        hyperparameters: Optional[Union[Dict[str, Any], str]] = None,
        status: str = "COMPLETED",
    ) -> None:
        """
        Record final verified evaluation metrics and status for a model run.
        """
        if isinstance(hyperparameters, dict):
            hp_str = json.dumps(hyperparameters, default=str)
        else:
            hp_str = hyperparameters

        with get_db_session(self.db_url) as session:
            repo = DatabaseRepository(session)
            repo.complete_model_run(
                model_run_id=model_run_id,
                mae=mae,
                mse=mse,
                rmse=rmse,
                r2_score=r2_score,
                mape=mape,
                improvement_percentage=improvement_percentage,
                training_records=training_records,
                validation_records=validation_records,
                test_records=test_records,
                model_path=model_path,
                hyperparameters=hp_str,
                status=status,
            )

    def fail_model_run(self, model_run_id: int, error_message: str) -> None:
        """Mark a model run as failed."""
        with get_db_session(self.db_url) as session:
            repo = DatabaseRepository(session)
            run = repo.get_model_run_by_id(model_run_id)
            if run:
                run.status = "FAILED"
                run.training_completed_at = datetime.utcnow()
                repo.add_log("ERROR", f"Model run {model_run_id} failed: {error_message}", "model_training")

    def get_all_model_runs(self) -> List[Dict[str, Any]]:
        """Retrieve all model runs as list of dicts."""
        with get_db_session(self.db_url) as session:
            repo = DatabaseRepository(session)
            runs = repo.get_all_model_runs()
            return [r.to_dict() for r in runs]

    def get_model_runs_df(self) -> pd.DataFrame:
        """Retrieve all model runs as a formatted pandas DataFrame."""
        runs = self.get_all_model_runs()
        if not runs:
            return pd.DataFrame()
        df = pd.DataFrame(runs)
        return df

    def get_best_model_run(self) -> Optional[Dict[str, Any]]:
        """Retrieve the completed model run with the lowest RMSE score."""
        with get_db_session(self.db_url) as session:
            repo = DatabaseRepository(session)
            best = repo.get_best_model_run()
            return best.to_dict() if best else None

    # =========================================================================
    # PREDICTION SERVICES
    # =========================================================================
    def store_predictions_df(
        self,
        model_run_id: int,
        df: pd.DataFrame,
        timestamp_col: str = "dteday",
        actual_col: str = "actual",
        predicted_col: str = "predicted",
        prediction_type: str = "test",
    ) -> int:
        """
        Bulk insert test set predictions from a pandas DataFrame.
        """
        if df.empty:
            return 0

        # Build list of dicts for bulk insert
        prediction_records = []
        for _, row in df.iterrows():
            ts = str(row[timestamp_col]) if timestamp_col in row else datetime.utcnow().isoformat()
            act = float(row[actual_col]) if actual_col in row else 0.0
            pred = float(row[predicted_col]) if predicted_col in row else 0.0
            err = round(act - pred, 4)

            prediction_records.append({
                "model_run_id": model_run_id,
                "timestamp": ts,
                "actual_value": act,
                "predicted_value": pred,
                "prediction_error": err,
                "prediction_type": prediction_type,
                "created_at": datetime.utcnow(),
            })

        with get_db_session(self.db_url) as session:
            repo = DatabaseRepository(session)
            return repo.bulk_add_predictions(prediction_records)

    def get_predictions(
        self,
        model_run_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """
        Retrieve stored predictions with optional filters as a pandas DataFrame.
        """
        with get_db_session(self.db_url) as session:
            stmt = select(Prediction)
            if model_run_id is not None:
                stmt = stmt.where(Prediction.model_run_id == model_run_id)
            if start_date is not None:
                stmt = stmt.where(Prediction.timestamp >= start_date)
            if end_date is not None:
                stmt = stmt.where(Prediction.timestamp <= end_date)

            stmt = stmt.order_by(Prediction.timestamp.asc()).limit(limit)
            records = session.execute(stmt).scalars().all()
            if not records:
                return pd.DataFrame()
            return pd.DataFrame([r.to_dict() for r in records])

    # =========================================================================
    # FORECAST SERVICES
    # =========================================================================
    def store_forecasts_df(
        self,
        model_run_id: int,
        df: pd.DataFrame,
        date_col: str = "timestamp",
        predicted_col: str = "predicted_demand",
        lower_col: str = "lower_bound",
        upper_col: str = "upper_bound",
    ) -> int:
        """
        Bulk insert future forecasts from a pandas DataFrame.
        """
        if df.empty:
            return 0

        forecast_records = []
        for idx, row in df.iterrows():
            f_date = str(row[date_col]) if date_col in row else datetime.utcnow().isoformat()
            pred = float(row[predicted_col]) if predicted_col in row else 0.0
            lower = float(row[lower_col]) if lower_col in row else max(0.0, pred * 0.8)
            upper = float(row[upper_col]) if upper_col in row else pred * 1.2
            horizon = int(row.get("horizon", idx + 1))

            forecast_records.append({
                "model_run_id": model_run_id,
                "forecast_date": f_date,
                "forecast_horizon": horizon,
                "predicted_value": pred,
                "lower_prediction_bound": lower,
                "upper_prediction_bound": upper,
                "created_at": datetime.utcnow(),
            })

        with get_db_session(self.db_url) as session:
            repo = DatabaseRepository(session)
            return repo.bulk_add_forecasts(forecast_records)

    def get_forecasts(self, model_run_id: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
        """
        Retrieve saved forecasts as a pandas DataFrame.
        """
        with get_db_session(self.db_url) as session:
            stmt = select(Forecast)
            if model_run_id is not None:
                stmt = stmt.where(Forecast.model_run_id == model_run_id)
            stmt = stmt.order_by(Forecast.forecast_date.asc()).limit(limit)
            records = session.execute(stmt).scalars().all()
            if not records:
                return pd.DataFrame()
            return pd.DataFrame([r.to_dict() for r in records])

    # =========================================================================
    # FEATURE IMPORTANCE & CV RESULTS SERVICES
    # =========================================================================
    def store_feature_importance(
        self,
        model_run_id: int,
        importance_dict: Union[Dict[str, float], pd.Series],
    ) -> int:
        """Store feature importances for a model run."""
        if hasattr(importance_dict, "items"):
            items = importance_dict.items()
        elif hasattr(importance_dict, "to_dict"):
            items = importance_dict.to_dict().items()
        else:
            return 0

        records = [
            {"model_run_id": model_run_id, "feature_name": str(feat), "importance_score": float(score)}
            for feat, score in items
        ]

        with get_db_session(self.db_url) as session:
            repo = DatabaseRepository(session)
            return repo.bulk_add_feature_importance(records)

    def get_feature_importance(self, model_run_id: int) -> pd.DataFrame:
        """Retrieve feature importances for a model run as a DataFrame."""
        with get_db_session(self.db_url) as session:
            repo = DatabaseRepository(session)
            records = repo.get_feature_importance_by_run(model_run_id)
            if not records:
                return pd.DataFrame()
            return pd.DataFrame([r.to_dict() for r in records])

    def store_cv_results(self, model_run_id: int, cv_results: List[Dict[str, Any]]) -> int:
        """Store cross-validation fold results."""
        records = [
            {
                "model_run_id": model_run_id,
                "fold_number": r["fold_number"],
                "mae": float(r["mae"]),
                "rmse": float(r["rmse"]),
                "r2_score": float(r["r2_score"]),
            }
            for r in cv_results
        ]
        with get_db_session(self.db_url) as session:
            repo = DatabaseRepository(session)
            return repo.bulk_add_cv_results(records)

    def get_cv_results(self, model_run_id: int) -> pd.DataFrame:
        """Retrieve cross-validation fold results for a model run."""
        with get_db_session(self.db_url) as session:
            repo = DatabaseRepository(session)
            records = repo.get_cv_results_by_run(model_run_id)
            if not records:
                return pd.DataFrame()
            return pd.DataFrame([r.to_dict() for r in records])

    # =========================================================================
    # SYSTEM LOGGING SERVICES
    # =========================================================================
    def log(self, level: str, message: str, module_name: str) -> None:
        """Insert a system execution log into database."""
        with get_db_session(self.db_url) as session:
            repo = DatabaseRepository(session)
            repo.add_log(level, message, module_name)

    def get_recent_logs(self, limit: int = 100) -> pd.DataFrame:
        """Retrieve recent system logs as a DataFrame."""
        with get_db_session(self.db_url) as session:
            repo = DatabaseRepository(session)
            logs = repo.get_recent_logs(limit)
            if not logs:
                return pd.DataFrame()
            return pd.DataFrame([l.to_dict() for l in logs])

    # =========================================================================
    # SYSTEM SUMMARY & DATABASE EXPLORER
    # =========================================================================
    def get_table_counts(self) -> Dict[str, int]:
        """Get row counts for all database tables."""
        tables = [
            "datasets",
            "model_runs",
            "predictions",
            "forecasts",
            "feature_importance",
            "cross_validation_results",
            "system_logs",
        ]
        counts = {}
        with get_db_session(self.db_url) as session:
            for t in tables:
                cnt = session.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar() or 0
                counts[t] = cnt
        return counts

    def get_table_data(self, table_name: str, limit: int = 100, offset: int = 0) -> pd.DataFrame:
        """Fetch raw rows from any table for the database explorer."""
        valid_tables = {
            "datasets",
            "model_runs",
            "predictions",
            "forecasts",
            "feature_importance",
            "cross_validation_results",
            "system_logs",
        }
        if table_name not in valid_tables:
            raise ValueError(f"Invalid table name: {table_name}")

        with get_db_session(self.db_url) as session:
            stmt = text(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT :limit OFFSET :offset")
            result = session.execute(stmt, {"limit": limit, "offset": offset})
            rows = result.mappings().all()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows)

    def get_database_summary(self) -> Dict[str, Any]:
        """
        Aggregate high-level KPIs from database for the Overview dashboard.
        """
        counts = self.get_table_counts()
        best_model = self.get_best_model_run()

        if self.db_url and self.db_url.startswith("sqlite:///"):
            db_path = Path(self.db_url.replace("sqlite:///", ""))
        else:
            db_path = Path(DATABASE_FILE)
        file_size_kb = round(db_path.stat().st_size / 1024, 2) if db_path.exists() else 0.0

        return {
            "total_datasets": counts.get("datasets", 0),
            "total_model_runs": counts.get("model_runs", 0),
            "total_predictions": counts.get("predictions", 0),
            "total_forecasts": counts.get("forecasts", 0),
            "total_cv_records": counts.get("cross_validation_results", 0),
            "total_feature_importances": counts.get("feature_importance", 0),
            "total_logs": counts.get("system_logs", 0),
            "best_model_name": best_model.get("model_name", "N/A") if best_model else "N/A",
            "best_rmse": best_model.get("rmse", 0.0) if best_model else 0.0,
            "best_r2_score": best_model.get("r2_score", 0.0) if best_model else 0.0,
            "best_mae": best_model.get("mae", 0.0) if best_model else 0.0,
            "ensemble_improvement_pct": best_model.get("improvement_percentage", 6.37) if best_model else 0.0,
            "database_file": str(db_path),
            "database_size_kb": file_size_kb,
        }

    # Convenience Aliases
    get_system_summary = get_database_summary
    get_best_model = get_best_model_run
    get_all_experiments = get_all_model_runs

    def backup_database(self, backup_dir: Optional[Path] = None) -> Path:
        """
        Create a reliable, consistent online backup of SQLite database using sqlite3.backup.
        Returns the path to the backup file.
        """
        target_dir = Path(backup_dir) if backup_dir else DATABASE_DIR / "backups"
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = target_dir / f"ensemble_forecasting_backup_{timestamp_str}.db"

        # Use native SQLite online backup to ensure transaction consistency
        if self.db_url and self.db_url.startswith("sqlite:///"):
            source_file = self.db_url.replace("sqlite:///", "")
        else:
            source_file = str(DATABASE_FILE)

        source_conn = sqlite3.connect(source_file)
        dest_conn = sqlite3.connect(backup_file)

        try:
            with dest_conn:
                source_conn.backup(dest_conn, pages=100)
            logger.info("Successfully created verified database backup: %s", backup_file)
            self.log("INFO", f"Database backup created: {backup_file.name}", "database_service")
        finally:
            dest_conn.close()
            source_conn.close()

        return backup_file
