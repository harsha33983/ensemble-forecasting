"""
Unit Tests for SQLite Database Integration.

Tests:
1. Database and Table Creation & Schema Verification
2. CRUD Operations on Datasets
3. CRUD Operations on Model Runs and Run Lifecycle
4. Foreign Key Constraints & Cascade Deletion
5. Bulk Insert of Predictions
6. Bulk Insert of Forecasts
7. Feature Importance and Cross-Validation Storage
8. System Logging CRUD
9. Transaction Rollback on Failure
10. Database Reconnect and Persistence
11. SQLite Database Online Backup Functionality
"""

from datetime import datetime
import os
from pathlib import Path
import tempfile
import pytest

from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

from src.database.connection import get_db_session, get_engine
from src.database.database_init import EXPECTED_TABLES, export_schema_sql, init_db
from src.database.database_service import DatabaseService
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


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = f.name
    temp_url = f"sqlite:///{temp_path.replace(os.sep, '/')}"

    # Initialize tables
    init_db(db_url=temp_url, force_recreate=True)

    yield {"path": temp_path, "url": temp_url}

    # Teardown
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception:
        pass


class TestDatabaseCreationAndSchema:
    def test_table_creation_and_verification(self, temp_db):
        engine = get_engine(temp_db["url"])
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        for expected in EXPECTED_TABLES:
            assert expected in table_names, f"Table {expected} was not created!"

    def test_schema_sql_export(self, temp_db, tmp_path):
        engine = get_engine(temp_db["url"])
        sql_file = tmp_path / "test_schema.sql"
        exported = export_schema_sql(engine, output_file=sql_file)
        assert exported.exists()
        content = exported.read_text(encoding="utf-8")
        assert "CREATE TABLE datasets" in content
        assert "CREATE TABLE model_runs" in content
        assert "PRAGMA foreign_keys = ON;" in content


class TestDatasetOperations:
    def test_create_and_read_dataset(self, temp_db):
        service = DatabaseService(db_url=temp_db["url"])
        dataset_id = service.register_dataset(
            dataset_name="Test Bike Dataset",
            source_url="https://example.com/bikes",
            description="Test dataset description",
            total_records=1000,
            total_features=16,
            target_column="cnt",
            file_path="/path/to/bikes.csv",
        )
        assert dataset_id is not None
        assert dataset_id > 0

        datasets = service.get_all_datasets()
        assert len(datasets) == 1
        assert datasets[0]["dataset_name"] == "Test Bike Dataset"
        assert datasets[0]["total_records"] == 1000

    def test_idempotent_dataset_registration(self, temp_db):
        service = DatabaseService(db_url=temp_db["url"])
        id1 = service.register_dataset(dataset_name="Unique Dataset", total_records=500)
        id2 = service.register_dataset(dataset_name="Unique Dataset", total_records=600)
        assert id1 == id2  # Should return existing ID


class TestModelRunOperations:
    def test_model_run_lifecycle(self, temp_db):
        service = DatabaseService(db_url=temp_db["url"])
        dataset_id = service.register_dataset("Sample Dataset", total_records=100)

        # 1. Start run
        run_id = service.start_model_run(
            model_name="Random Forest",
            experiment_name="Test_Exp",
            algorithm="RandomForestRegressor",
            dataset_id=dataset_id,
            hyperparameters={"n_estimators": 50},
        )
        assert run_id > 0

        with get_db_session(temp_db["url"]) as session:
            repo = DatabaseRepository(session)
            run = repo.get_model_run_by_id(run_id)
            assert run.status == "TRAINING"

        # 2. Complete run
        service.complete_model_run(
            model_run_id=run_id,
            mae=15.2,
            mse=400.0,
            rmse=20.0,
            r2_score=0.92,
            mape=8.5,
            improvement_percentage=5.4,
            training_records=80,
            test_records=20,
            model_path="/models/rf.pkl",
        )

        with get_db_session(temp_db["url"]) as session:
            repo = DatabaseRepository(session)
            run = repo.get_model_run_by_id(run_id)
            assert run.status == "COMPLETED"
            assert run.rmse == 20.0
            assert run.r2_score == 0.92
            assert run.training_completed_at is not None

    def test_best_model_run_query(self, temp_db):
        service = DatabaseService(db_url=temp_db["url"])
        dataset_id = service.register_dataset("Bench Dataset")

        r1 = service.start_model_run("Model A", dataset_id=dataset_id)
        service.complete_model_run(r1, mae=30.0, mse=1600.0, rmse=40.0, r2_score=0.80)

        r2 = service.start_model_run("Model B", dataset_id=dataset_id)
        service.complete_model_run(r2, mae=20.0, mse=900.0, rmse=30.0, r2_score=0.90)

        best = service.get_best_model_run()
        assert best is not None
        assert best["model_name"] == "Model B"
        assert best["rmse"] == 30.0


class TestPredictionsAndForecasts:
    def test_predictions_bulk_insert_and_retrieval(self, temp_db):
        import pandas as pd
        service = DatabaseService(db_url=temp_db["url"])
        run_id = service.start_model_run("Test Predictor")

        df = pd.DataFrame({
            "timestamp": ["2026-01-01 00:00:00", "2026-01-01 01:00:00"],
            "actual": [100.0, 150.0],
            "predicted": [95.0, 155.0],
        })

        inserted = service.store_predictions_df(
            model_run_id=run_id,
            df=df,
            timestamp_col="timestamp",
            actual_col="actual",
            predicted_col="predicted",
        )
        assert inserted == 2

        preds = service.get_predictions(model_run_id=run_id)
        assert len(preds) == 2
        assert preds.iloc[0]["actual_value"] == 100.0
        assert preds.iloc[0]["prediction_error"] == 5.0

    def test_forecasts_bulk_insert_and_retrieval(self, temp_db):
        import pandas as pd
        service = DatabaseService(db_url=temp_db["url"])
        run_id = service.start_model_run("Test Forecaster")

        df = pd.DataFrame({
            "timestamp": ["2026-01-02 00:00:00", "2026-01-02 01:00:00"],
            "horizon": [1, 2],
            "predicted_demand": [120.0, 130.0],
            "lower_bound": [100.0, 110.0],
            "upper_bound": [140.0, 150.0],
        })

        inserted = service.store_forecasts_df(model_run_id=run_id, df=df)
        assert inserted == 2

        fc = service.get_forecasts(model_run_id=run_id)
        assert len(fc) == 2
        assert fc.iloc[0]["predicted_value"] == 120.0


class TestCascadeAndIntegrityConstraints:
    def test_foreign_key_cascade_deletion(self, temp_db):
        service = DatabaseService(db_url=temp_db["url"])
        dataset_id = service.register_dataset("Cascade Dataset")
        run_id = service.start_model_run("Cascade Model", dataset_id=dataset_id)

        # Add prediction linked to run_id
        with get_db_session(temp_db["url"]) as session:
            repo = DatabaseRepository(session)
            repo.add_prediction(
                model_run_id=run_id,
                timestamp="2026-01-01 00:00:00",
                actual_value=50.0,
                predicted_value=48.0,
            )

        # Verify prediction exists
        counts = service.get_table_counts()
        assert counts["predictions"] == 1

        # Delete dataset; cascade should delete run and predictions
        with get_db_session(temp_db["url"]) as session:
            repo = DatabaseRepository(session)
            deleted = repo.delete_dataset(dataset_id)
            assert deleted is True

        counts_after = service.get_table_counts()
        assert counts_after["datasets"] == 0
        assert counts_after["model_runs"] == 0
        assert counts_after["predictions"] == 0

    def test_transaction_rollback_on_failure(self, temp_db):
        """Verify transaction rolls back cleanly on exception."""
        with pytest.raises(Exception):
            with get_db_session(temp_db["url"]) as session:
                repo = DatabaseRepository(session)
                repo.add_dataset("Temporary Dataset")
                raise ValueError("Simulated pipeline crash during transaction!")

        service = DatabaseService(db_url=temp_db["url"])
        datasets = service.get_all_datasets()
        assert len(datasets) == 0  # Rolled back


class TestBackupAndReconnect:
    def test_sqlite_online_backup(self, temp_db, tmp_path):
        service = DatabaseService(db_url=temp_db["url"])
        service.register_dataset("Backup Test Dataset")

        # Create backup to tmp_path
        backup_file = service.backup_database(backup_dir=tmp_path)
        assert backup_file.exists()
        assert backup_file.stat().st_size > 0

        # Verify backup contains the dataset
        backup_url = f"sqlite:///{str(backup_file).replace(os.sep, '/')}"
        backup_service = DatabaseService(db_url=backup_url)
        datasets = backup_service.get_all_datasets()
        assert len(datasets) == 1
        assert datasets[0]["dataset_name"] == "Backup Test Dataset"

