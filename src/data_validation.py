"""
Data Validation Module.

Performs schema verification, missing value detection, range checks,
chronological order validation, and data leakage checks.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from src.config import TARGET_COLUMN, LEAKAGE_COLUMNS
from src.utils import setup_logger

logger = setup_logger("data_validation")


class DataValidator:
    """Validates data integrity, ranges, ordering, and absence of target leakage."""

    EXPECTED_RAW_COLUMNS = [
        "instant", "dteday", "season", "yr", "mnth", "hr",
        "holiday", "weekday", "workingday", "weathersit",
        "temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"
    ]

    NUMERICAL_RANGES = {
        "season": (1, 4),
        "yr": (0, 1),
        "mnth": (1, 12),
        "hr": (0, 23),
        "holiday": (0, 1),
        "weekday": (0, 6),
        "workingday": (0, 1),
        "weathersit": (1, 4),
        "temp": (0.0, 1.0),
        "atemp": (0.0, 1.0),
        "hum": (0.0, 1.0),
        "windspeed": (0.0, 1.0),
        "cnt": (0, 2000),
    }

    @classmethod
    def validate_raw_schema(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Verify that all expected raw columns exist."""
        errors = []
        missing = [col for col in cls.EXPECTED_RAW_COLUMNS if col not in df.columns]
        if missing:
            errors.append(f"Missing expected raw columns: {missing}")
        return len(errors) == 0, errors

    @classmethod
    def validate_missing_values(cls, df: pd.DataFrame) -> Dict[str, int]:
        """Count missing values per column and verify total."""
        null_counts = df.isnull().sum().to_dict()
        total_nulls = sum(null_counts.values())
        if total_nulls > 0:
            logger.warning("Found %d total missing values across dataset: %s", total_nulls, null_counts)
        else:
            logger.info("Zero missing values detected in dataset.")
        return null_counts

    @classmethod
    def validate_duplicates(cls, df: pd.DataFrame) -> int:
        """Check for duplicate records based on dteday and hr or full rows."""
        if "dteday" in df.columns and "hr" in df.columns:
            dups = int(df.duplicated(subset=["dteday", "hr"]).sum())
        else:
            dups = int(df.duplicated().sum())

        if dups > 0:
            logger.warning("Found %d duplicate timestamp entries.", dups)
        else:
            logger.info("Zero duplicate timestamp entries found.")
        return dups

    @classmethod
    def validate_numerical_ranges(cls, df: pd.DataFrame) -> Dict[str, bool]:
        """Validate that numerical attributes fall within physically/logically expected bounds."""
        range_results = {}
        for col, (min_val, max_val) in cls.NUMERICAL_RANGES.items():
            if col in df.columns:
                actual_min = df[col].min()
                actual_max = df[col].max()
                is_valid = (actual_min >= min_val) and (actual_max <= max_val)
                range_results[col] = bool(is_valid)
                if not is_valid:
                    logger.warning(
                        "Column '%s' out of range [%s, %s]: observed [%s, %s]",
                        col, min_val, max_val, actual_min, actual_max
                    )
        return range_results

    @classmethod
    def validate_chronological_order(cls, df: pd.DataFrame, datetime_col: str = "datetime") -> bool:
        """Check whether the dataframe is strictly chronologically ordered (monotonically non-decreasing)."""
        if datetime_col not in df.columns:
            if "dteday" in df.columns and "hr" in df.columns:
                dt_series = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
            else:
                logger.warning("No datetime column available for chronological ordering check.")
                return False
        else:
            dt_series = pd.to_datetime(df[datetime_col])

        is_monotonic = dt_series.is_monotonic_increasing
        if not is_monotonic:
            logger.error("Dataset is NOT sorted chronologically!")
        else:
            logger.info("Verified dataset is strictly monotonically increasing in time.")
        return bool(is_monotonic)

    @classmethod
    def check_for_target_leakage(cls, feature_columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Verify that no direct target leakage variables (casual, registered, cnt)
        exist in the feature matrix.
        """
        forbidden = set(LEAKAGE_COLUMNS + [TARGET_COLUMN])
        leaks = [c for c in feature_columns if c in forbidden]
        if leaks:
            logger.error("DATA LEAKAGE DETECTED! Feature matrix includes: %s", leaks)
            return False, leaks
        logger.info("Leakage check PASSED. No component or target variables in feature set.")
        return True, []

    @classmethod
    def run_full_validation(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute comprehensive validation suite and return summary dictionary."""
        schema_valid, schema_errors = cls.validate_raw_schema(df)
        null_counts = cls.validate_missing_values(df)
        duplicates = cls.validate_duplicates(df)
        ranges = cls.validate_numerical_ranges(df)
        chrono_valid = cls.validate_chronological_order(df)

        report = {
            "n_records": int(len(df)),
            "n_columns": int(len(df.columns)),
            "schema_valid": schema_valid,
            "schema_errors": schema_errors,
            "total_missing_values": int(sum(null_counts.values())),
            "duplicate_records": duplicates,
            "ranges_valid": all(ranges.values()),
            "range_details": ranges,
            "chronologically_sorted": chrono_valid,
        }
        return report

