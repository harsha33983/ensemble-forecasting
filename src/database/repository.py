"""
Database Repository Module.

Provides comprehensive CRUD operations and bulk data access methods
for datasets, model runs, predictions, forecasts, feature importances,
cross-validation results, and system logs.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from src.database.models import (
    CrossValidationResult,
    Dataset,
    FeatureImportance,
    Forecast,
    ModelRun,
    Prediction,
    SystemLog,
)
from src.utils import setup_logger

logger = setup_logger("database_repository")


class DatabaseRepository:
    """Encapsulates data persistence and retrieval operations."""

    def __init__(self, session: Session):
        self.session = session

    # =========================================================================
    # DATASET OPERATIONS
    # =========================================================================
    def add_dataset(
        self,
        dataset_name: str,
        source_url: Optional[str] = None,
        description: Optional[str] = None,
        total_records: Optional[int] = None,
        total_features: Optional[int] = None,
        target_column: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> Dataset:
        """Create and persist a new dataset metadata entry."""
        dataset = Dataset(
            dataset_name=dataset_name,
            source_url=source_url,
            description=description,
            total_records=total_records,
            total_features=total_features,
            target_column=target_column,
            file_path=file_path,
            created_at=datetime.utcnow(),
        )
        self.session.add(dataset)
        self.session.flush()
        logger.info("Persisted dataset metadata: %s (ID: %d)", dataset_name, dataset.id)
        return dataset

    def get_dataset_by_id(self, dataset_id: int) -> Optional[Dataset]:
        """Retrieve dataset by primary key."""
        return self.session.get(Dataset, dataset_id)

    def get_dataset_by_name(self, dataset_name: str) -> Optional[Dataset]:
        """Retrieve dataset by unique name."""
        stmt = select(Dataset).where(Dataset.dataset_name == dataset_name).order_by(desc(Dataset.id))
        return self.session.execute(stmt).scalars().first()

    def get_all_datasets(self) -> List[Dataset]:
        """Retrieve all registered datasets."""
        stmt = select(Dataset).order_by(desc(Dataset.created_at))
        return list(self.session.execute(stmt).scalars().all())

    def update_dataset(self, dataset_id: int, **kwargs) -> Optional[Dataset]:
        """Update dataset attributes."""
        dataset = self.get_dataset_by_id(dataset_id)
        if dataset:
            for key, val in kwargs.items():
                if hasattr(dataset, key):
                    setattr(dataset, key, val)
            self.session.flush()
        return dataset

    def delete_dataset(self, dataset_id: int) -> bool:
        """Delete dataset by ID."""
        dataset = self.get_dataset_by_id(dataset_id)
        if dataset:
            self.session.delete(dataset)
            self.session.flush()
            logger.info("Deleted dataset ID: %d", dataset_id)
            return True
        return False

    # =========================================================================
    # MODEL RUN OPERATIONS
    # =========================================================================
    def add_model_run(
        self,
        model_name: str,
        experiment_name: Optional[str] = "Standard_Forecasting_Benchmark",
        algorithm: Optional[str] = None,
        dataset_id: Optional[int] = None,
        hyperparameters: Optional[str] = None,
        status: str = "PENDING",
    ) -> ModelRun:
        """Create a new model training run experiment record."""
        model_run = ModelRun(
            experiment_name=experiment_name,
            model_name=model_name,
            algorithm=algorithm,
            dataset_id=dataset_id,
            hyperparameters=hyperparameters,
            training_started_at=datetime.utcnow(),
            status=status,
        )
        self.session.add(model_run)
        self.session.flush()
        logger.info("Created model run experiment: %s (ID: %d)", model_name, model_run.id)
        return model_run

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
        hyperparameters: Optional[str] = None,
        status: str = "COMPLETED",
    ) -> Optional[ModelRun]:
        """Update model run with final evaluation metrics and completion status."""
        run = self.session.get(ModelRun, model_run_id)
        if run:
            run.mae = mae
            run.mse = mse
            run.rmse = rmse
            run.r2_score = r2_score
            run.mape = mape
            run.improvement_percentage = improvement_percentage
            run.training_records = training_records
            run.validation_records = validation_records
            run.test_records = test_records
            if model_path:
                run.model_path = model_path
            if hyperparameters:
                run.hyperparameters = hyperparameters
            run.training_completed_at = datetime.utcnow()
            run.status = status
            self.session.flush()
            logger.info("Completed model run ID %d (%s) -> RMSE: %.2f, R2: %.4f", run.id, run.model_name, rmse, r2_score)
        return run

    def get_model_run_by_id(self, run_id: int) -> Optional[ModelRun]:
        """Retrieve model run by ID."""
        return self.session.get(ModelRun, run_id)

    def get_all_model_runs(self) -> List[ModelRun]:
        """Retrieve all model runs ordered by completion time."""
        stmt = select(ModelRun).order_by(desc(ModelRun.id))
        return list(self.session.execute(stmt).scalars().all())

    def get_best_model_run(self) -> Optional[ModelRun]:
        """Retrieve the completed model run with the lowest RMSE score."""
        stmt = (
            select(ModelRun)
            .where(ModelRun.status == "COMPLETED", ModelRun.rmse.isnot(None))
            .order_by(ModelRun.rmse.asc())
        )
        return self.session.execute(stmt).scalars().first()

    # =========================================================================
    # PREDICTION OPERATIONS
    # =========================================================================
    def add_prediction(
        self,
        model_run_id: int,
        timestamp: str,
        actual_value: float,
        predicted_value: float,
        prediction_type: str = "test",
    ) -> Prediction:
        """Add a single prediction record."""
        pred = Prediction(
            model_run_id=model_run_id,
            timestamp=timestamp,
            actual_value=actual_value,
            predicted_value=predicted_value,
            prediction_error=round(actual_value - predicted_value, 4),
            prediction_type=prediction_type,
            created_at=datetime.utcnow(),
        )
        self.session.add(pred)
        self.session.flush()
        return pred

    def bulk_add_predictions(self, prediction_dicts: List[Dict[str, Any]]) -> int:
        """Efficiently bulk insert predictions using mapping."""
        if not prediction_dicts:
            return 0
        now = datetime.utcnow()
        for p in prediction_dicts:
            if "created_at" not in p:
                p["created_at"] = now
            if "prediction_error" not in p and "actual_value" in p and "predicted_value" in p:
                p["prediction_error"] = round(p["actual_value"] - p["predicted_value"], 4)

        self.session.bulk_insert_mappings(Prediction, prediction_dicts)
        self.session.flush()
        logger.info("Bulk inserted %d predictions into database.", len(prediction_dicts))
        return len(prediction_dicts)

    def get_predictions_by_model_run(self, model_run_id: int, limit: int = 1000) -> List[Prediction]:
        """Retrieve predictions for a given model run."""
        stmt = (
            select(Prediction)
            .where(Prediction.model_run_id == model_run_id)
            .order_by(Prediction.timestamp.asc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    # =========================================================================
    # FORECAST OPERATIONS
    # =========================================================================
    def add_forecast(
        self,
        forecast_date: str,
        forecast_horizon: int,
        predicted_value: float,
        lower_prediction_bound: float,
        upper_prediction_bound: float,
        model_run_id: Optional[int] = None,
    ) -> Forecast:
        """Add a single forecast record."""
        fc = Forecast(
            model_run_id=model_run_id,
            forecast_date=forecast_date,
            forecast_horizon=forecast_horizon,
            predicted_value=predicted_value,
            lower_prediction_bound=lower_prediction_bound,
            upper_prediction_bound=upper_prediction_bound,
            created_at=datetime.utcnow(),
        )
        self.session.add(fc)
        self.session.flush()
        return fc

    def bulk_add_forecasts(self, forecast_dicts: List[Dict[str, Any]]) -> int:
        """Bulk insert forecast records."""
        if not forecast_dicts:
            return 0
        now = datetime.utcnow()
        for fc in forecast_dicts:
            if "created_at" not in fc:
                fc["created_at"] = now

        self.session.bulk_insert_mappings(Forecast, forecast_dicts)
        self.session.flush()
        logger.info("Bulk inserted %d forecast entries.", len(forecast_dicts))
        return len(forecast_dicts)

    def get_latest_forecasts(self, limit: int = 100) -> List[Forecast]:
        """Retrieve latest forecasts ordered by date."""
        stmt = select(Forecast).order_by(desc(Forecast.id)).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    # =========================================================================
    # FEATURE IMPORTANCE & CV RESULTS
    # =========================================================================
    def bulk_add_feature_importance(self, importance_dicts: List[Dict[str, Any]]) -> int:
        """Bulk insert feature importances for a model run."""
        if not importance_dicts:
            return 0
        self.session.bulk_insert_mappings(FeatureImportance, importance_dicts)
        self.session.flush()
        return len(importance_dicts)

    def get_feature_importance_by_run(self, model_run_id: int) -> List[FeatureImportance]:
        """Retrieve feature importances for a run ordered by score descending."""
        stmt = (
            select(FeatureImportance)
            .where(FeatureImportance.model_run_id == model_run_id)
            .order_by(desc(FeatureImportance.importance_score))
        )
        return list(self.session.execute(stmt).scalars().all())

    def bulk_add_cv_results(self, cv_dicts: List[Dict[str, Any]]) -> int:
        """Bulk insert cross-validation fold results."""
        if not cv_dicts:
            return 0
        self.session.bulk_insert_mappings(CrossValidationResult, cv_dicts)
        self.session.flush()
        return len(cv_dicts)

    def get_cv_results_by_run(self, model_run_id: int) -> List[CrossValidationResult]:
        """Retrieve cross-validation fold results for a model run."""
        stmt = (
            select(CrossValidationResult)
            .where(CrossValidationResult.model_run_id == model_run_id)
            .order_by(CrossValidationResult.fold_number.asc())
        )
        return list(self.session.execute(stmt).scalars().all())

    # =========================================================================
    # SYSTEM LOGS
    # =========================================================================
    def add_log(self, log_level: str, message: str, module_name: str) -> SystemLog:
        """Record an execution event log in database."""
        log_entry = SystemLog(
            log_level=log_level.upper(),
            message=message,
            module_name=module_name,
            created_at=datetime.utcnow(),
        )
        self.session.add(log_entry)
        self.session.flush()
        return log_entry

    def get_recent_logs(self, limit: int = 100) -> List[SystemLog]:
        """Retrieve recent system logs."""
        stmt = select(SystemLog).order_by(desc(SystemLog.id)).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

