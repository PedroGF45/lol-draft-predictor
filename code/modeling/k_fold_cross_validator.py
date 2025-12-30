"""
K-Fold Cross Validator for League of Legends Draft Prediction.

Implements k-fold cross-validation with comprehensive metrics for both classification and regression.
"""

import time
import logging
import numpy as np
import warnings
from typing import Dict, Tuple, Optional, Any

from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore")


class KFoldCrossValidator:
    """
    Performs k-fold cross-validation with comprehensive metrics.

    Supports both classification and regression tasks with detailed metric calculation,
    including mean and standard deviation across folds.
    """

    def __init__(
        self,
        k: int = 5,
        is_classification: bool = True,
        random_state: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
    ):
        """
        Initialize K-Fold Cross Validator.

        Args:
            k: Number of folds (must be >= 2)
            is_classification: True for classification, False for regression
            random_state: Random seed for reproducibility
            logger: Logger instance
            verbose: Print detailed progress
        """
        if k < 2:
            raise ValueError("Number of folds k must be at least 2.")

        self.k = k
        self.kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        self.is_classification = is_classification
        self.logger = logger or self._setup_logger()
        self.verbose = verbose

        # Metric storage
        self._reset_scores()

    def _setup_logger(self) -> logging.Logger:
        """Setup logger if not provided."""
        logger = logging.getLogger("KFoldCrossValidator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _reset_scores(self):
        """Reset score lists for a new validation run."""
        if self.is_classification:
            self.accuracy_scores = []
            self.precision_scores = []
            self.recall_scores = []
            self.f1_scores = []
            self.roc_auc_scores = []
        else:
            self.mse_scores = []
            self.rmse_scores = []
            self.mae_scores = []
            self.r2_scores = []

        self.training_times = []

    def cross_validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[Any, Dict[str, Tuple[float, float]]]:
        """
        Perform k-fold cross-validation.

        Args:
            model: Scikit-learn compatible model with fit() and predict()
            X: Features (n_samples, n_features)
            y: Labels

        Returns:
            Tuple of (fitted_model, summary_metrics_dict)
            summary_metrics_dict: {metric_name: (mean, std), ...}
        """
        if self.verbose:
            self.logger.info(f"Starting {self.k}-fold cross-validation...")

        self._reset_scores()

        for fold_idx, (train_idx, val_idx) in enumerate(self.kf.split(X)):
            if self.verbose:
                self.logger.info(f"\n--- Fold {fold_idx + 1}/{self.k} ---")

            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if self.verbose:
                self.logger.info(f"  Train: {X_train.shape[0]} samples | " f"Val: {X_val.shape[0]} samples")

            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            self.training_times.append(training_time)

            if self.verbose:
                self.logger.info(f"  Training time: {training_time:.4f}s")

            # Predict
            y_pred = model.predict(X_val)

            # Calculate metrics
            if self.is_classification:
                self._calculate_classification_metrics(y_val, y_pred, fold_idx)
            else:
                self._calculate_regression_metrics(y_val, y_pred, fold_idx)

        # Compute summary statistics
        summary_metrics = self._compute_summary_metrics()

        if self.verbose:
            self.logger.info("\n" + "=" * 60)
            self.logger.info("Cross-Validation Summary")
            self.logger.info("=" * 60)
            for metric, (mean, std) in summary_metrics.items():
                self.logger.info(f"  {metric}: {mean:.4f} ± {std:.4f}")

        return model, summary_metrics

    def _calculate_classification_metrics(self, y_val: np.ndarray, y_pred: np.ndarray, fold_idx: int):
        """Calculate classification metrics for a fold."""
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        self.accuracy_scores.append(accuracy)
        self.precision_scores.append(precision)
        self.recall_scores.append(recall)
        self.f1_scores.append(f1)

        if self.verbose:
            self.logger.info(f"  Accuracy: {accuracy:.4f}")
            self.logger.info(f"  Precision: {precision:.4f}")
            self.logger.info(f"  Recall: {recall:.4f}")
            self.logger.info(f"  F1-Score: {f1:.4f}")

        # Calculate ROC-AUC for binary classification
        try:
            if len(np.unique(y_val)) == 2:
                roc_auc = roc_auc_score(y_val, y_pred)
                self.roc_auc_scores.append(roc_auc)
                if self.verbose:
                    self.logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        except Exception as e:
            if self.verbose:
                self.logger.debug(f"Could not calculate ROC-AUC: {e}")

    def _calculate_regression_metrics(self, y_val: np.ndarray, y_pred: np.ndarray, fold_idx: int):
        """Calculate regression metrics for a fold."""
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        self.mse_scores.append(mse)
        self.rmse_scores.append(rmse)
        self.mae_scores.append(mae)
        self.r2_scores.append(r2)

        if self.verbose:
            self.logger.info(f"  MSE: {mse:.4f}")
            self.logger.info(f"  RMSE: {rmse:.4f}")
            self.logger.info(f"  MAE: {mae:.4f}")
            self.logger.info(f"  R²: {r2:.4f}")

    def _compute_summary_metrics(self) -> Dict[str, Tuple[float, float]]:
        """Compute mean and std for all metrics."""
        summary = {}

        if self.is_classification:
            summary["accuracy"] = (
                np.mean(self.accuracy_scores),
                np.std(self.accuracy_scores),
            )
            summary["precision"] = (
                np.mean(self.precision_scores),
                np.std(self.precision_scores),
            )
            summary["recall"] = (
                np.mean(self.recall_scores),
                np.std(self.recall_scores),
            )
            summary["f1"] = (
                np.mean(self.f1_scores),
                np.std(self.f1_scores),
            )
            if self.roc_auc_scores:
                summary["roc_auc"] = (
                    np.mean(self.roc_auc_scores),
                    np.std(self.roc_auc_scores),
                )
        else:
            summary["mse"] = (
                np.mean(self.mse_scores),
                np.std(self.mse_scores),
            )
            summary["rmse"] = (
                np.mean(self.rmse_scores),
                np.std(self.rmse_scores),
            )
            summary["mae"] = (
                np.mean(self.mae_scores),
                np.std(self.mae_scores),
            )
            summary["r2"] = (
                np.mean(self.r2_scores),
                np.std(self.r2_scores),
            )

        summary["training_time"] = (
            np.mean(self.training_times),
            np.std(self.training_times),
        )

        return summary

    def evaluate_on_test_set(
        self,
        y_pred: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate predictions on test set.

        Args:
            y_pred: Predicted labels
            y_test: Ground truth labels

        Returns:
            Dictionary of test metrics
        """
        if self.verbose:
            self.logger.info("\nEvaluating on test set...")

        metrics = {}

        if self.is_classification:
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            metrics["recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            metrics["f1"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            try:
                if len(np.unique(y_test)) == 2:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_pred)
            except Exception:
                pass

            if self.verbose:
                self.logger.info("Classification Metrics:")
                for metric, value in metrics.items():
                    self.logger.info(f"  {metric}: {value:.4f}")
        else:
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_test, y_pred)
            metrics["r2"] = r2_score(y_test, y_pred)

            if self.verbose:
                self.logger.info("Regression Metrics:")
                for metric, value in metrics.items():
                    self.logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed cross-validation report."""
        report = {
            "num_folds": self.k,
            "is_classification": self.is_classification,
        }

        if self.is_classification:
            report["accuracy_scores"] = self.accuracy_scores
            report["precision_scores"] = self.precision_scores
            report["recall_scores"] = self.recall_scores
            report["f1_scores"] = self.f1_scores
            report["roc_auc_scores"] = self.roc_auc_scores
        else:
            report["mse_scores"] = self.mse_scores
            report["rmse_scores"] = self.rmse_scores
            report["mae_scores"] = self.mae_scores
            report["r2_scores"] = self.r2_scores

        report["training_times"] = self.training_times

        return report
