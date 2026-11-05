"""
SQLAlchemy ORM Models for Ensemble Forecasting Database.

Defines schemas for:
1. datasets
2. model_runs
3. predictions
4. forecasts
5. feature_importance
6. cross_validation_results
7. system_logs
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class BaseModelMixin:
    """Provides dictionary serialization for all SQLAlchemy models."""

    def to_dict(self):
        result = {}
        for column in self.__table__.columns:
            val = getattr(self, column.name)
            if isinstance(val, datetime):
                val = val.isoformat()
            result[column.name] = val
        return result


class Dataset(BaseModelMixin, Base):
    """Stores metadata and source details of datasets used for training."""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String(100), nullable=False, index=True)
    source_url = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    total_records = Column(Integer, nullable=True)
    total_features = Column(Integer, nullable=True)
    target_column = Column(String(50), nullable=True)
    file_path = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    model_runs = relationship("ModelRun", back_populates="dataset", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, name='{self.dataset_name}', records={self.total_records})>"


class ModelRun(BaseModelMixin, Base):
    """Tracks training runs, evaluation metrics, and hyperparameter configurations."""

    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_name = Column(String(100), nullable=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    algorithm = Column(String(100), nullable=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=True, index=True)
    training_records = Column(Integer, nullable=True)
    validation_records = Column(Integer, nullable=True)
    test_records = Column(Integer, nullable=True)
    mae = Column(Float, nullable=True)
    mse = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    r2_score = Column(Float, nullable=True)
    mape = Column(Float, nullable=True)
    improvement_percentage = Column(Float, nullable=True)
    hyperparameters = Column(Text, nullable=True)
    model_path = Column(Text, nullable=True)
    training_started_at = Column(DateTime, nullable=True)
    training_completed_at = Column(DateTime, nullable=True)
    status = Column(String(50), default="PENDING", nullable=False, index=True)

    # Relationships
    dataset = relationship("Dataset", back_populates="model_runs")
    predictions = relationship("Prediction", back_populates="model_run", cascade="all, delete-orphan")
    forecasts = relationship("Forecast", back_populates="model_run", cascade="all, delete-orphan")
    feature_importances = relationship("FeatureImportance", back_populates="model_run", cascade="all, delete-orphan")
    cv_results = relationship("CrossValidationResult", back_populates="model_run", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ModelRun(id={self.id}, model='{self.model_name}', rmse={self.rmse}, status='{self.status}')>"


class Prediction(BaseModelMixin, Base):
    """Stores individual test/validation predictions and residual errors."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(String(50), nullable=False, index=True)
    actual_value = Column(Float, nullable=False)
    predicted_value = Column(Float, nullable=False)
    prediction_error = Column(Float, nullable=False)
    prediction_type = Column(String(50), default="test", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    model_run = relationship("ModelRun", back_populates="predictions")

    def __repr__(self) -> str:
        return f"<Prediction(id={self.id}, run_id={self.model_run_id}, actual={self.actual_value}, pred={self.predicted_value})>"


class Forecast(BaseModelMixin, Base):
    """Stores future recursive predictions with confidence intervals."""

    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=True, index=True)
    forecast_date = Column(String(50), nullable=False, index=True)
    forecast_horizon = Column(Integer, nullable=False)
    predicted_value = Column(Float, nullable=False)
    lower_prediction_bound = Column(Float, nullable=False)
    upper_prediction_bound = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    model_run = relationship("ModelRun", back_populates="forecasts")

    def __repr__(self) -> str:
        return f"<Forecast(id={self.id}, horizon={self.forecast_horizon}, pred={self.predicted_value})>"


class FeatureImportance(BaseModelMixin, Base):
    """Stores relative feature importance scores from decision tree and gradient boosted models."""

    __tablename__ = "feature_importance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    feature_name = Column(String(100), nullable=False)
    importance_score = Column(Float, nullable=False)

    # Relationships
    model_run = relationship("ModelRun", back_populates="feature_importances")

    def __repr__(self) -> str:
        return f"<FeatureImportance(feature='{self.feature_name}', score={self.importance_score})>"


class CrossValidationResult(BaseModelMixin, Base):
    """Stores cross-validation metrics across chronological folds."""

    __tablename__ = "cross_validation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    fold_number = Column(Integer, nullable=False)
    mae = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    r2_score = Column(Float, nullable=False)

    # Relationships
    model_run = relationship("ModelRun", back_populates="cv_results")

    def __repr__(self) -> str:
        return f"<CrossValidationResult(run_id={self.model_run_id}, fold={self.fold_number}, rmse={self.rmse})>"


class SystemLog(BaseModelMixin, Base):
    """Logs runtime events, model training activities, and system alerts."""

    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    log_level = Column(String(20), nullable=False, index=True)
    message = Column(Text, nullable=False)
    module_name = Column(String(100), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def __repr__(self) -> str:
        return f"<SystemLog(level='{self.log_level}', module='{self.module_name}')>"

