"""
Cross-Validation Module.

Implements rigorous chronological TimeSeriesSplit cross-validation (5 folds)
comparing Random Forest, Gradient Boosting, Weighted Ensemble, and Stacking Ensemble.
Saves detailed fold-by-fold and aggregate metrics to CSV.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from src.config import METRICS_DIR, N_CV_SPLITS, RANDOM_STATE
from src.ensemble_model import WeightedAveragingEnsemble, StackingEnsemble
from src.utils import calculate_metrics, setup_logger

logger = setup_logger("cross_validation")


class TimeSeriesCrossValidator:
    """Evaluates time-series models across multiple chronological expanding folds."""

    def __init__(self, n_splits: int = N_CV_SPLITS, metrics_dir: Path = METRICS_DIR):
        self.n_splits = n_splits
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def run_cv(
        self,
        models_dict: Dict[str, Any],
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> pd.DataFrame:
        """
        Execute chronological TimeSeriesSplit evaluation across all provided models.
        """
        logger.info("Executing %d-fold TimeSeriesSplit cross-validation...", self.n_splits)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        results_records: List[Dict[str, Any]] = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_te, y_te = X.iloc[test_idx], y[test_idx]

            logger.info("Fold %d/%d: Train size=%d, Test size=%d", fold, self.n_splits, len(X_tr), len(X_te))

            # Train base models for this fold
            rf_fold = clone(models_dict.get("Random Forest", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)))
            gb_fold = clone(models_dict.get("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)))

            rf_fold.fit(X_tr, y_tr)
            gb_fold.fit(X_tr, y_tr)

            rf_preds = np.clip(rf_fold.predict(X_te), 0, None)
            gb_preds = np.clip(gb_fold.predict(X_te), 0, None)

            # Measure RF
            rf_m = calculate_metrics(y_te, rf_preds)
            results_records.append({"Fold": fold, "Model": "Random Forest", **rf_m})

            # Measure GB
            gb_m = calculate_metrics(y_te, gb_preds)
            results_records.append({"Fold": fold, "Model": "Gradient Boosting", **gb_m})

            # Measure Weighted Ensemble
            # Optimize weights on the last 20% of fold training set to simulate validation
            split_point = int(len(X_tr) * 0.8)
            fold_sub_tr_X, fold_sub_tr_y = X_tr.iloc[:split_point], y_tr[:split_point]
            fold_sub_va_X, fold_sub_val_y = X_tr.iloc[split_point:], y_tr[split_point:]

            rf_sub = clone(rf_fold).fit(fold_sub_tr_X, fold_sub_tr_y)
            gb_sub = clone(gb_fold).fit(fold_sub_tr_X, fold_sub_tr_y)

            weighted_ens = WeightedAveragingEnsemble(rf_model=rf_sub, gb_model=gb_sub)
            weighted_ens.optimize_weights(fold_sub_va_X, fold_sub_val_y)
            # Re-assign full fitted models with discovered weights
            weighted_ens.rf_model = rf_fold
            weighted_ens.gb_model = gb_fold
            weighted_preds = weighted_ens.predict(X_te)
            weighted_m = calculate_metrics(y_te, weighted_preds)
            results_records.append({"Fold": fold, "Model": "Weighted Ensemble", **weighted_m})

            # Measure Stacking Ensemble
            stacking_ens = StackingEnsemble(rf_model=clone(rf_fold), gb_model=clone(gb_fold), n_splits=3)
            stacking_ens.fit_with_oof(X_tr, y_tr)
            stacking_preds = stacking_ens.predict(X_te)
            stacking_m = calculate_metrics(y_te, stacking_preds)
            results_records.append({"Fold": fold, "Model": "Stacking Ensemble", **stacking_m})

        cv_df = pd.DataFrame(results_records)

        # Compute summary mean and std per model
        summary_rows = []
        for model_name, group in cv_df.groupby("Model"):
            summary_rows.append({
                "Fold": "Mean",
                "Model": model_name,
                "MAE": round(group["MAE"].mean(), 4),
                "MSE": round(group["MSE"].mean(), 4),
                "RMSE": round(group["RMSE"].mean(), 4),
                "R2": round(group["R2"].mean(), 4),
                "MAPE": round(group["MAPE"].mean(), 2),
            })
            summary_rows.append({
                "Fold": "Std",
                "Model": model_name,
                "MAE": round(group["MAE"].std(), 4),
                "MSE": round(group["MSE"].std(), 4),
                "RMSE": round(group["RMSE"].std(), 4),
                "R2": round(group["R2"].std(), 4),
                "MAPE": round(group["MAPE"].std(), 2),
            })

        final_cv_df = pd.concat([cv_df, pd.DataFrame(summary_rows)], ignore_index=True)
        csv_path = self.metrics_dir / "cross_validation_results.csv"
        final_cv_df.to_csv(csv_path, index=False)
        logger.info("Saved complete cross-validation results to %s", csv_path)

        return final_cv_df

