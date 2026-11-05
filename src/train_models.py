"""
Model Training and Hyperparameter Tuning Module.

Trains and tunes:
1. RandomForestRegressor
2. GradientBoostingRegressor
Using strict chronological TimeSeriesSplit cross-validation.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from src.config import MODELS_DIR, PLOTS_DIR, RANDOM_STATE
from src.utils import calculate_metrics, save_pickle, setup_logger
from src.visualization import plot_feature_importance, plot_residuals

logger = setup_logger("train_models")


class ModelTrainer:
    """Manages training, chronological hyperparameter search, and diagnostics."""

    def __init__(self, models_dir: Path = MODELS_DIR, plots_dir: Path = PLOTS_DIR):
        self.models_dir = Path(models_dir)
        self.plots_dir = Path(plots_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def tune_and_train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
        """
        Tune RandomForestRegressor using chronological validation search,
        train optimal model, and compute feature importances and residuals.
        """
        logger.info("Starting Random Forest optimization with chronological validation...")

        candidate_configs = [
            {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 2, "max_features": "sqrt", "random_state": RANDOM_STATE},
            {"n_estimators": 120, "max_depth": 20, "min_samples_leaf": 2, "max_features": 0.8, "random_state": RANDOM_STATE},
            {"n_estimators": 150, "max_depth": 20, "min_samples_leaf": 2, "max_features": 0.8, "random_state": RANDOM_STATE},
        ]

        tscv = TimeSeriesSplit(n_splits=3)
        best_score = float("inf")
        best_params = candidate_configs[0]

        for idx, config in enumerate(candidate_configs):
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train):
                cv_X_tr, cv_y_tr = X_train.iloc[train_idx], y_train[train_idx]
                cv_X_va, cv_y_va = X_train.iloc[val_idx], y_train[val_idx]

                rf = RandomForestRegressor(**config)
                rf.fit(cv_X_tr, cv_y_tr)
                preds = np.clip(rf.predict(cv_X_va), 0, None)
                rmse = float(np.sqrt(np.mean((cv_y_va - preds) ** 2)))
                cv_scores.append(rmse)

            mean_rmse = float(np.mean(cv_scores))
            logger.info("RF Config %d: %s -> Chronological CV RMSE: %.2f", idx + 1, config, mean_rmse)

            if mean_rmse < best_score:
                best_score = mean_rmse
                best_params = config

        logger.info("Best Random Forest parameters found: %s (CV RMSE: %.2f)", best_params, best_score)

        # Train final RF on full training set
        best_rf = RandomForestRegressor(**best_params)
        best_rf.fit(X_train, y_train)

        # Feature importances
        feature_names = list(X_train.columns)
        importances = best_rf.feature_importances_
        importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
        plot_feature_importance(importance_df, "Random Forest", "rf_feature_importance.png")

        # Save model
        rf_path = self.models_dir / "random_forest.pkl"
        save_pickle(best_rf, rf_path)
        logger.info("Saved trained Random Forest model to %s", rf_path)

        # Residuals plot on validation set if provided
        if X_val is not None and y_val is not None:
            val_preds = np.clip(best_rf.predict(X_val), 0, None)
            plot_residuals(y_val, val_preds, "Random Forest", "rf_residuals.png")

        return best_rf, {"best_params": best_params, "cv_rmse": best_score, "feature_importances": importance_df.to_dict(orient="records")}

    def tune_and_train_gradient_boosting(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Tuple[GradientBoostingRegressor, Dict[str, Any]]:
        """
        Tune GradientBoostingRegressor using chronological validation search,
        train optimal model, and compute feature importances and residuals.
        """
        logger.info("Starting Gradient Boosting optimization with chronological validation...")

        candidate_configs = [
            {"n_estimators": 100, "learning_rate": 0.08, "max_depth": 5, "min_samples_leaf": 4, "subsample": 0.85, "random_state": RANDOM_STATE},
            {"n_estimators": 120, "learning_rate": 0.06, "max_depth": 6, "min_samples_leaf": 3, "subsample": 0.85, "random_state": RANDOM_STATE},
            {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 6, "min_samples_leaf": 2, "subsample": 0.90, "random_state": RANDOM_STATE},
        ]

        tscv = TimeSeriesSplit(n_splits=3)
        best_score = float("inf")
        best_params = candidate_configs[0]

        for idx, config in enumerate(candidate_configs):
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train):
                cv_X_tr, cv_y_tr = X_train.iloc[train_idx], y_train[train_idx]
                cv_X_va, cv_y_va = X_train.iloc[val_idx], y_train[val_idx]

                gb = GradientBoostingRegressor(**config)
                gb.fit(cv_X_tr, cv_y_tr)
                preds = np.clip(gb.predict(cv_X_va), 0, None)
                rmse = float(np.sqrt(np.mean((cv_y_va - preds) ** 2)))
                cv_scores.append(rmse)

            mean_rmse = float(np.mean(cv_scores))
            logger.info("GB Config %d: %s -> Chronological CV RMSE: %.2f", idx + 1, config, mean_rmse)

            if mean_rmse < best_score:
                best_score = mean_rmse
                best_params = config

        logger.info("Best Gradient Boosting parameters found: %s (CV RMSE: %.2f)", best_params, best_score)

        # Train final GB on full training set
        best_gb = GradientBoostingRegressor(**best_params)
        best_gb.fit(X_train, y_train)

        # Feature importances
        feature_names = list(X_train.columns)
        importances = best_gb.feature_importances_
        importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
        plot_feature_importance(importance_df, "Gradient Boosting", "gb_feature_importance.png")

        # Save model
        gb_path = self.models_dir / "gradient_boosting.pkl"
        save_pickle(best_gb, gb_path)
        logger.info("Saved trained Gradient Boosting model to %s", gb_path)

        # Residuals plot on validation set if provided
        if X_val is not None and y_val is not None:
            val_preds = np.clip(best_gb.predict(X_val), 0, None)
            plot_residuals(y_val, val_preds, "Gradient Boosting", "gb_residuals.png")

        return best_gb, {"best_params": best_params, "cv_rmse": best_score, "feature_importances": importance_df.to_dict(orient="records")}
