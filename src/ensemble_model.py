"""
Hybrid Ensemble Modeling Module.

Implements:
1. Weighted Averaging Ensemble (optimized on validation data, weights sum to 1)
2. Stacking Ensemble with TimeSeriesSplit Out-of-Fold base predictions and Ridge meta-model
3. Ensemble Selection & Model Serialization
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from src.config import MODELS_DIR, RANDOM_STATE
from src.utils import calculate_metrics, save_json, save_pickle, setup_logger

logger = setup_logger("ensemble_model")


class WeightedAveragingEnsemble(BaseEstimator, RegressorMixin):
    """
    Weighted Average of Random Forest and Gradient Boosting predictions.
    Optimization is strictly performed on validation data to minimize RMSE,
    with weights constrained to sum to 1.
    """

    def __init__(
        self,
        rf_model: Optional[RandomForestRegressor] = None,
        gb_model: Optional[GradientBoostingRegressor] = None,
        weight_rf: float = 0.5,
        weight_gb: float = 0.5,
    ):
        self.rf_model = rf_model
        self.gb_model = gb_model
        self.weight_rf = weight_rf
        self.weight_gb = weight_gb

    def optimize_weights(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        step: float = 0.01,
    ) -> Tuple[float, float, float]:
        """
        Grid-search optimal convex combination w * RF + (1-w) * GB
        minimizing RMSE on unseen validation data.
        """
        if self.rf_model is None or self.gb_model is None:
            raise ValueError("Base models must be set before optimizing weights.")

        logger.info("Optimizing ensemble weights on validation set...")
        pred_rf = np.clip(self.rf_model.predict(X_val), 0, None)
        pred_gb = np.clip(self.gb_model.predict(X_val), 0, None)

        best_rmse = float("inf")
        best_w_rf = 0.5
        best_w_gb = 0.5

        # Test candidate weights from 0.0 to 1.0
        candidate_weights = np.arange(0.0, 1.0 + step, step)
        for w in candidate_weights:
            w_rf = float(w)
            w_gb = float(1.0 - w_rf)
            blended = w_rf * pred_rf + w_gb * pred_gb
            rmse = float(np.sqrt(np.mean((y_val - blended) ** 2)))

            if rmse < best_rmse:
                best_rmse = rmse
                best_w_rf = w_rf
                best_w_gb = w_gb

        self.weight_rf = round(best_w_rf, 4)
        self.weight_gb = round(best_w_gb, 4)

        logger.info(
            "Optimal Validation Weights: RF=%.4f, GB=%.4f (Validation RMSE: %.2f)",
            self.weight_rf, self.weight_gb, best_rmse
        )
        return self.weight_rf, self.weight_gb, best_rmse

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "WeightedAveragingEnsemble":
        """Refit base models if required."""
        if self.rf_model is not None:
            self.rf_model.fit(X, y)
        if self.gb_model is not None:
            self.gb_model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate weighted ensemble predictions clipped to non-negative values."""
        if self.rf_model is None or self.gb_model is None:
            raise ValueError("Base models must be initialized and fitted before predict().")

        pred_rf = self.rf_model.predict(X)
        pred_gb = self.gb_model.predict(X)

        ensemble_pred = (self.weight_rf * pred_rf) + (self.weight_gb * pred_gb)
        return np.clip(ensemble_pred, 0, None)


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Chronological Stacking Ensemble.
    Base models: RandomForestRegressor and GradientBoostingRegressor.
    Meta-model: Ridge Regression.
    Out-of-fold meta-features generated via TimeSeriesSplit to prevent leakage.
    """

    def __init__(
        self,
        rf_model: Optional[RandomForestRegressor] = None,
        gb_model: Optional[GradientBoostingRegressor] = None,
        meta_model: Optional[Any] = None,
        n_splits: int = 5,
    ):
        self.rf_model = rf_model
        self.gb_model = gb_model
        self.meta_model = meta_model or Ridge(alpha=1.0, positive=True, random_state=RANDOM_STATE)
        self.n_splits = n_splits
        self.meta_features_train_: Optional[np.ndarray] = None
        self.meta_targets_train_: Optional[np.ndarray] = None

    def fit_with_oof(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
    ) -> "StackingEnsemble":
        """
        Generate out-of-fold predictions using strictly chronological TimeSeriesSplit,
        fit the meta-model on true OOF representations, and refit base models.
        """
        logger.info("Generating chronological Out-Of-Fold (OOF) predictions for stacking...")
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        oof_rf_list = []
        oof_gb_list = []
        oof_y_list = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
            fold_X_tr, fold_y_tr = X_train.iloc[train_idx], y_train[train_idx]
            fold_X_va, fold_y_va = X_train.iloc[val_idx], y_train[val_idx]

            rf_fold = clone(self.rf_model) if self.rf_model is not None else RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE)
            gb_fold = clone(self.gb_model) if self.gb_model is not None else GradientBoostingRegressor(n_estimators=150, random_state=RANDOM_STATE)

            rf_fold.fit(fold_X_tr, fold_y_tr)
            gb_fold.fit(fold_X_tr, fold_y_tr)

            pred_rf_val = rf_fold.predict(fold_X_va)
            pred_gb_val = gb_fold.predict(fold_X_va)

            oof_rf_list.append(pred_rf_val)
            oof_gb_list.append(pred_gb_val)
            oof_y_list.append(fold_y_va)

        oof_rf = np.concatenate(oof_rf_list)
        oof_gb = np.concatenate(oof_gb_list)
        oof_y = np.concatenate(oof_y_list)

        S_oof = np.column_stack([oof_rf, oof_gb])
        self.meta_features_train_ = S_oof
        self.meta_targets_train_ = oof_y

        logger.info("Fitting Stacking meta-model on %d OOF observations...", len(S_oof))
        self.meta_model.fit(S_oof, oof_y)

        # Refit base models on full X_train
        logger.info("Refitting base estimators on full training dataset...")
        self.rf_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict by projecting base estimator predictions through meta-model."""
        if self.rf_model is None or self.gb_model is None:
            raise ValueError("Base models not initialized.")

        pred_rf = self.rf_model.predict(X)
        pred_gb = self.gb_model.predict(X)

        S_test = np.column_stack([pred_rf, pred_gb])
        stack_pred = self.meta_model.predict(S_test)
        return np.clip(stack_pred, 0, None)


def build_and_select_ensemble(
    rf_model: RandomForestRegressor,
    gb_model: GradientBoostingRegressor,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    models_dir: Path = MODELS_DIR,
) -> Tuple[Union[WeightedAveragingEnsemble, StackingEnsemble], Dict[str, Any]]:
    """
    Compare Weighted Ensemble vs Stacking Ensemble vs Standalone Models on Validation data.
    Select the optimal ensemble, refit on train+val, and serialize.
    """
    logger.info("Evaluating ensemble strategies on validation set...")
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Standalone Models on Validation
    rf_val_preds = np.clip(rf_model.predict(X_val), 0, None)
    gb_val_preds = np.clip(gb_model.predict(X_val), 0, None)

    rf_val_metrics = calculate_metrics(y_val, rf_val_preds)
    gb_val_metrics = calculate_metrics(y_val, gb_val_preds)

    logger.info("Validation RF: RMSE=%.2f, MAE=%.2f, R2=%.4f", rf_val_metrics["RMSE"], rf_val_metrics["MAE"], rf_val_metrics["R2"])
    logger.info("Validation GB: RMSE=%.2f, MAE=%.2f, R2=%.4f", gb_val_metrics["RMSE"], gb_val_metrics["MAE"], gb_val_metrics["R2"])

    # 2. Weighted Ensemble
    weighted_ens = WeightedAveragingEnsemble(rf_model=rf_model, gb_model=gb_model)
    w_rf, w_gb, weighted_val_rmse = weighted_ens.optimize_weights(X_val, y_val)
    weighted_val_preds = weighted_ens.predict(X_val)
    weighted_val_metrics = calculate_metrics(y_val, weighted_val_preds)
    logger.info("Validation Weighted Ensemble: RMSE=%.2f, MAE=%.2f, R2=%.4f (w_rf=%.3f, w_gb=%.3f)",
                weighted_val_metrics["RMSE"], weighted_val_metrics["MAE"], weighted_val_metrics["R2"], w_rf, w_gb)

    # Save weighted ensemble
    weighted_path = models_dir / "weighted_ensemble.pkl"
    save_pickle(weighted_ens, weighted_path)

    # 3. Stacking Ensemble
    stacking_ens = StackingEnsemble(rf_model=clone(rf_model), gb_model=clone(gb_model))
    stacking_ens.fit_with_oof(X_train, y_train)
    stacking_val_preds = stacking_ens.predict(X_val)
    stacking_val_metrics = calculate_metrics(y_val, stacking_val_preds)
    logger.info("Validation Stacking Ensemble: RMSE=%.2f, MAE=%.2f, R2=%.4f",
                stacking_val_metrics["RMSE"], stacking_val_metrics["MAE"], stacking_val_metrics["R2"])

    # 4. Selection based on Validation RMSE
    best_candidate_name = "Weighted Ensemble"
    selected_ensemble = weighted_ens
    best_val_rmse = weighted_val_metrics["RMSE"]

    if stacking_val_metrics["RMSE"] < best_val_rmse:
        best_candidate_name = "Stacking Ensemble"
        selected_ensemble = stacking_ens
        best_val_rmse = stacking_val_metrics["RMSE"]

    logger.info("Selected winning ensemble architecture: '%s' (Validation RMSE: %.2f)", best_candidate_name, best_val_rmse)

    # 5. Refit selected ensemble on Train + Validation data
    logger.info("Refitting base estimators on combined Train + Validation dataset...")
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = np.concatenate([y_train, y_val])

    if best_candidate_name == "Weighted Ensemble":
        final_rf = clone(rf_model)
        final_gb = clone(gb_model)
        final_rf.fit(X_train_val, y_train_val)
        final_gb.fit(X_train_val, y_train_val)
        final_ensemble = WeightedAveragingEnsemble(
            rf_model=final_rf,
            gb_model=final_gb,
            weight_rf=w_rf,
            weight_gb=w_gb,
        )
    else:
        final_rf = clone(rf_model)
        final_gb = clone(gb_model)
        final_ensemble = StackingEnsemble(
            rf_model=final_rf,
            gb_model=final_gb,
            meta_model=stacking_ens.meta_model,
        )
        final_ensemble.fit_with_oof(X_train_val, y_train_val)

    # Save final chosen model
    final_model_path = models_dir / "ensemble_model.pkl"
    save_pickle(final_ensemble, final_model_path)
    logger.info("Saved final ensemble model to %s", final_model_path)

    # Model metadata
    metadata = {
        "selected_ensemble": best_candidate_name,
        "weights": {"rf": w_rf, "gb": w_gb} if best_candidate_name == "Weighted Ensemble" else None,
        "validation_metrics": {
            "Random Forest": rf_val_metrics,
            "Gradient Boosting": gb_val_metrics,
            "Weighted Ensemble": weighted_val_metrics,
            "Stacking Ensemble": stacking_val_metrics,
        },
        "train_size": len(X_train),
        "val_size": len(X_val),
        "train_val_size": len(X_train_val),
        "feature_count": X_train.shape[1],
        "feature_names": list(X_train.columns),
    }
    save_json(metadata, models_dir / "model_metadata.json")

    return final_ensemble, metadata
