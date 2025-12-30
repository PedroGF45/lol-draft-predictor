"""
Model Builder for League of Legends Draft Prediction.

Orchestrates the building, optimization, and evaluation of multiple ML models
using genetic algorithm-based hyperparameter optimization and K-fold cross-validation.
Supports classification and regression tasks with sklearn-based models.
"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from modeling.deep_learner import DeepLearningClassifier
from modeling.k_fold_cross_validator import KFoldCrossValidator
from modeling.model_optimizer import ModelOptimizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge, SGDClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR


class ModelBuilder:
    """
    Builds and optimizes multiple ML models for draft prediction.

    Orchestrates the pipeline: model selection → hyperparameter optimization via GA →
    cross-validation → test set evaluation → model persistence.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        is_classification: bool = True,
        k_folds: int = 5,
        save_models: bool = False,
        output_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        verbose: bool = True,
        auto_preprocess: bool = True,
    ):
        """
        Initialize ModelBuilder.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            is_classification: True for classification, False for regression
            k_folds: Number of cross-validation folds
            save_models: Whether to save trained models
            output_path: Directory to save models and plots
            logger: Logger instance
            verbose: Print detailed progress messages
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.is_classification = is_classification
        self.k_folds = k_folds if k_folds >= 2 else 5
        self.save_models = save_models
        self.output_path = output_path or "./models"
        self.logger = logger or self._setup_logger()
        self.verbose = verbose
        self.auto_preprocess = auto_preprocess

        # Preprocessing strategy
        self.preprocessor = None
        self.scaler = None  # kept for backward compatibility

        try:
            if self.auto_preprocess:
                # Auto handle numeric + categorical
                if isinstance(X_train, pd.DataFrame):
                    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

                    transformers = []
                    if numeric_cols:
                        transformers.append(("num", StandardScaler(), numeric_cols))
                    if categorical_cols:
                        transformers.append(
                            (
                                "cat",
                                OneHotEncoder(handle_unknown="ignore", sparse=False),
                                categorical_cols,
                            )
                        )

                    if transformers:
                        self.preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
                        if self.verbose:
                            self.logger.info(
                                f"Preprocessing: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical columns"
                            )
                    else:
                        # Fallback to simple scaling if no columns detected
                        self.preprocessor = StandardScaler()
                else:
                    # Numpy arrays or other types: assume numeric only
                    self.preprocessor = StandardScaler()

                self.X_train_scaled = self.preprocessor.fit_transform(X_train)
                self.X_test_scaled = self.preprocessor.transform(X_test)
                # mirror attribute name used previously
                self.scaler = self.preprocessor
            else:
                # Assume upstream pipeline has already produced numeric, model-ready features
                if isinstance(X_train, pd.DataFrame):
                    # Verify no non-numeric columns sneak in
                    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
                    if non_numeric:
                        raise ValueError(
                            "auto_preprocess=False but non-numeric columns found: " + ", ".join(non_numeric)
                        )
                    self.X_train_scaled = X_train.values
                    self.X_test_scaled = X_test.values
                else:
                    self.X_train_scaled = X_train
                    self.X_test_scaled = X_test
                if self.verbose:
                    self.logger.info("Skipping internal preprocessing (auto_preprocess=False)")

            # Ensure labels are always numpy arrays (not Series or DataFrames)
            if isinstance(self.y_train, (pd.DataFrame, pd.Series)):
                self.y_train = self.y_train.values.ravel()
            if isinstance(self.y_test, (pd.DataFrame, pd.Series)):
                self.y_test = self.y_test.values.ravel()
        except Exception as e:
            self.logger.error(
                "Failed during feature preprocessing. Ensure inputs are numeric or categoricals are handled. Error: %s",
                str(e),
            )
            raise

        # Best model tracking
        self.best_model = None
        self.best_model_name = None
        self.best_model_params = None
        self.best_score = float("-inf") if is_classification else float("inf")

        # History tracking
        self.history = {}  # Stores results for each model
        self.models_built = []  # List of (model_name, model_instance, scores)

        # Create output directory
        if self.save_models:
            os.makedirs(self.output_path, exist_ok=True)
            os.makedirs(f"{self.output_path}/visualizations", exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Setup logger if not provided."""
        logger = logging.getLogger("ModelBuilder")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def build_models(self, models_dict: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Build and optimize multiple models.

        Args:
            models_dict: Dictionary mapping model names to their configurations.
                        Example:
                        {
                            'LogisticRegression': {
                                'model': LogisticRegression,
                                'population_size': 50,
                                'max_generations': 3,
                                'C_range': (0.001, 100),
                                'penalty_params': ('l1', 'l2')
                            }
                        }

        Returns:
            Dictionary with best model info and complete history
        """
        if models_dict is None:
            models_dict = self._get_default_models()

        self.logger.info(f"Building {len(models_dict)} models...")

        # Initialize optimizer
        optimizer = ModelOptimizer(
            self.X_train_scaled,
            self.X_test_scaled,
            self.y_train,
            self.y_test,
            is_classification=self.is_classification,
            logger=self.logger,
            verbose=self.verbose,
        )

        for model_name, config in models_dict.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing: {model_name}")
            self.logger.info(f"{'='*60}")

            model_class = config.pop("model")
            optimizer_params = {k: v for k, v in config.items() if k != "model"}

            try:
                # Optimize hyperparameters
                opt_results = self._optimize_model(optimizer, model_class, model_name, optimizer_params)

                if opt_results is None:
                    self.logger.warning(f"Optimization failed for {model_name}")
                    continue

                best_params, best_model, opt_time = opt_results

                # Cross-validation
                cv_results = self._cross_validate_model(best_model, model_name)

                # Test set evaluation
                test_results = self._evaluate_test_set(best_model, model_name)

                # Store history
                self.history[model_name] = {
                    "best_params": best_params,
                    "optimization_time": opt_time,
                    "cv_metrics": cv_results,
                    "test_metrics": test_results,
                }

                # Check if best model
                self._check_best_model(model_name, best_model, test_results)

                # Save model if requested
                if self.save_models:
                    self._save_model(model_name, best_model, best_params, cv_results, test_results)

                self.models_built.append((model_name, best_model, test_results))

            except Exception as e:
                self.logger.error(f"Error building {model_name}: {str(e)}")
                continue

        # Generate summary report
        self._generate_summary_report()

        return {
            "best_model": self.best_model,
            "best_model_name": self.best_model_name,
            "best_model_params": self.best_model_params,
            "best_score": self.best_score,
            "history": self.history,
            "all_models": self.models_built,
        }

    def _optimize_model(
        self,
        optimizer: ModelOptimizer,
        model_identifier,
        model_name: str,
        params: Dict[str, Any],
    ) -> Optional[Tuple[Dict, Any, float]]:
        """Optimize a single model's hyperparameters."""
        self.logger.info(f"Optimizing {model_name}...")

        start_time = time.time()

        # Handle deep learning by name to avoid importing torch unless requested
        if (isinstance(model_identifier, str) and model_identifier == "DeepLearningClassifier") or (
            hasattr(model_identifier, "__name__") and model_identifier.__name__ == "DeepLearningClassifier"
        ):
            result = self._train_final_deep_learning_model(**params)
        elif model_identifier in [LogisticRegression, LinearRegression]:
            result = optimizer.optimize_linear_model(model_class=model_identifier, **params)
        elif model_identifier in [SVC, SVR]:
            result = optimizer.optimize_svm(model_class=model_identifier, **params)
        elif model_identifier in [RandomForestClassifier, RandomForestRegressor]:
            result = optimizer.optimize_random_forest(model_class=model_identifier, **params)
        elif model_identifier in [GradientBoostingClassifier, GradientBoostingRegressor]:
            result = optimizer.optimize_gradient_boosting(model_class=model_identifier, **params)
        elif model_identifier in [KNeighborsClassifier, KNeighborsRegressor]:
            result = optimizer.optimize_knn(model_class=model_identifier, **params)
        elif model_identifier is SGDClassifier:
            result = self._optimize_sgd(**params)
        elif model_identifier in [MLPClassifier, MLPRegressor]:
            result = optimizer.optimize_mlp(model_class=model_identifier, **params)
        else:
            self.logger.warning(f"Unsupported model identifier: {model_identifier}")
            return None

        opt_time = time.time() - start_time

        if result is None:
            return None

        best_params, best_model = result
        return best_params, best_model, opt_time

    def _cross_validate_model(self, model, model_name: str) -> Dict[str, float]:
        """Perform k-fold cross-validation."""
        self.logger.info(f"Cross-validating {model_name} ({self.k_folds} folds)...")

        # Deep learning uses a custom CV routine (detected by presence of evaluate method)
        if hasattr(model, "evaluate") and callable(getattr(model, "evaluate")):
            return self._cross_validate_deep_learning(model)

        cv = KFoldCrossValidator(
            k=self.k_folds,
            is_classification=self.is_classification,
            logger=self.logger,
            verbose=self.verbose,
        )

        _, cv_metrics = cv.cross_validate(model, self.X_train_scaled, self.y_train)
        return cv_metrics

    def _evaluate_test_set(self, model, model_name: str) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.logger.info(f"Evaluating {model_name} on test set...")
        if hasattr(model, "evaluate") and callable(getattr(model, "evaluate")):
            return model.evaluate(self.X_test_scaled, self.y_test)

        y_pred = model.predict(self.X_test_scaled)

        cv = KFoldCrossValidator(
            k=self.k_folds,
            is_classification=self.is_classification,
            logger=self.logger,
            verbose=False,
        )

        metrics = cv.evaluate_on_test_set(y_pred, self.y_test)
        return metrics

    def _cross_validate_deep_learning(self, base_model: Any) -> Dict[str, Tuple[float, float]]:
        """Perform K-fold CV for deep learning model with per-fold retraining."""

        # Lazy import to avoid torch import unless needed
        try:
            from modeling.deep_learner import DeepLearningClassifier as _DLClass
        except Exception as e:
            raise RuntimeError(
                "Deep learning components are unavailable. Ensure torch is installed and working."
            ) from e

        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        metrics_list = []
        times = []
        X = self.X_train_scaled
        y = self.y_train

        # infer classes
        num_classes = len(np.unique(y))
        input_dim = X.shape[1]

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # instantiate a fresh model with same config as base_model
            dl = _DLClass(
                input_dim=input_dim,
                num_classes=num_classes,
                batch_size=getattr(base_model, "batch_size", 32),
                output_path=self.output_path,
                logger=self.logger,
                verbose=False,
            )

            start = time.time()
            dl.fit(
                X_tr,
                y_tr,
                X_val,
                y_val,
                num_epochs=30,
                learning_rate=0.001,
                optimizer_name="adam",
                criterion_name=None,
                early_stopping_patience=5,
            )
            times.append(time.time() - start)

            fold_metrics = dl.evaluate(X_val, y_val)
            metrics_list.append(fold_metrics)

        # aggregate metrics
        keys = metrics_list[0].keys()
        summary: Dict[str, Tuple[float, float]] = {}
        for k in keys:
            vals = np.array([m[k] for m in metrics_list], dtype=float)
            summary[k] = (float(vals.mean()), float(vals.std()))
        summary["training_time"] = (float(np.mean(times)), float(np.std(times)))

        return summary

    def _optimize_sgd(self, n_candidates: int = 20, random_state: int = 42, **kwargs):
        """Simple random search for SGDClassifier hyperparameters."""
        rng = np.random.RandomState(random_state)

        best_score = -np.inf
        best_params = None
        best_model = None

        # small 3-fold CV for speed
        cv = KFoldCrossValidator(
            k=min(3, self.k_folds),
            is_classification=True,
            logger=self.logger,
            verbose=False,
        )

        loss_options = ["log_loss", "hinge"]
        penalty_options = ["l2", "l1", "elasticnet"]

        for _ in range(n_candidates):
            params = {
                "loss": rng.choice(loss_options),
                "penalty": rng.choice(penalty_options),
                "alpha": 10 ** rng.uniform(-5, -3),
                "learning_rate": "optimal",
                "max_iter": 1000,
                "tol": 1e-3,
                "early_stopping": True,
                "n_jobs": None,
            }

            # ElasticNet requires l1_ratio
            if params["penalty"] == "elasticnet":
                params["l1_ratio"] = rng.uniform(0.05, 0.95)

            model = SGDClassifier(**params)
            _, metrics = cv.cross_validate(model, self.X_train_scaled, self.y_train)
            score = metrics.get("accuracy", (0.0, 0.0))[0]

            if score > best_score:
                best_score = score
                best_params = params
                best_model = model

        # fit best on full train
        best_model.fit(self.X_train_scaled, self.y_train)
        return best_params, best_model

    def _train_final_deep_learning_model(
        self,
        epochs: int = 30,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        **kwargs,
    ):
        """Train final deep learning classifier with a small validation split for early stopping."""
        # Lazy import to avoid torch import unless needed
        try:
            from modeling.deep_learner import DeepLearningClassifier as _DLClass
        except Exception as e:
            raise RuntimeError(
                "Deep learning components are unavailable. Ensure torch is installed and working."
            ) from e

        num_classes = len(np.unique(self.y_train))
        input_dim = self.X_train_scaled.shape[1]

        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train_scaled, self.y_train, test_size=0.15, random_state=42, stratify=self.y_train
        )

        dl = _DLClass(
            input_dim=input_dim,
            num_classes=num_classes,
            batch_size=batch_size,
            output_path=self.output_path,
            logger=self.logger,
            verbose=False,
        )

        start = time.time()
        dl.fit(
            X_tr,
            y_tr,
            X_val,
            y_val,
            num_epochs=epochs,
            learning_rate=learning_rate,
            optimizer_name="adam",
            criterion_name=None,
            early_stopping_patience=8,
        )
        opt_time = time.time() - start

        used_params = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }
        return used_params, dl

    def _check_best_model(self, model_name: str, model, metrics: Dict[str, float]):
        """Update best model if current model is better."""
        metric_key = "accuracy" if self.is_classification else "mse"
        current_score = metrics.get(metric_key, None)

        if current_score is None:
            return

        is_better = (self.is_classification and current_score > self.best_score) or (
            not self.is_classification and current_score < self.best_score
        )

        if is_better:
            self.best_model = model
            self.best_model_name = model_name
            self.best_score = current_score
            self.logger.info(f"New best model: {model_name} " f"({metric_key}={current_score:.4f})")

    def _save_model(
        self, model_name: str, model, params: Dict[str, Any], cv_metrics: Dict[str, Any], test_metrics: Dict[str, Any]
    ):
        """Save model artifacts, params, and metrics in a structured run directory and update best tracking."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_path, model_name, timestamp)
        os.makedirs(run_dir, exist_ok=True)

        # Primary artifact paths
        model_path = os.path.join(run_dir, "model.pkl")
        prep_path = os.path.join(run_dir, "preprocessor.pkl")
        metrics_path = os.path.join(run_dir, "metrics.json")
        params_path = os.path.join(run_dir, "params.json")
        info_path = os.path.join(run_dir, "info.json")

        try:
            # Save estimator
            joblib.dump(model, model_path)
            self.logger.info(f"Model saved: {model_path}")

            # Save the fitted preprocessor (if any)
            if getattr(self, "preprocessor", None) is not None:
                joblib.dump(self.preprocessor, prep_path)
                self.logger.info(f"Preprocessor saved: {prep_path}")

            # Persist metrics and params
            import json

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "cv_metrics": cv_metrics,
                        "test_metrics": test_metrics,
                    },
                    f,
                    indent=2,
                )

            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2)

            # Additional run info for reproducibility
            feature_names = None
            if isinstance(self.X_train, pd.DataFrame):
                feature_names = list(self.X_train.columns)

            info = {
                "timestamp": timestamp,
                "model_name": model_name,
                "is_classification": self.is_classification,
                "k_folds": self.k_folds,
                "auto_preprocess": getattr(self, "auto_preprocess", True),
                "input_dim": int(self.X_train_scaled.shape[1]),
                "n_train_samples": int(len(self.y_train)),
                "n_test_samples": int(len(self.y_test)),
                "feature_names": feature_names,
                "artifacts": {
                    "model": os.path.relpath(model_path, self.output_path),
                    "preprocessor": (
                        os.path.relpath(prep_path, self.output_path)
                        if getattr(self, "preprocessor", None) is not None
                        else None
                    ),
                    "metrics": os.path.relpath(metrics_path, self.output_path),
                    "params": os.path.relpath(params_path, self.output_path),
                },
            }
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)

            # Update per-model best tracking
            primary_metric = "accuracy" if self.is_classification else "mse"
            current_score = test_metrics.get(primary_metric)
            if current_score is not None:
                best_file = os.path.join(self.output_path, model_name, "best.json")
                best = None
                if os.path.exists(best_file):
                    try:
                        with open(best_file, "r", encoding="utf-8") as f:
                            best = json.load(f)
                    except Exception:
                        best = None

                def is_better(curr, prev):
                    if prev is None:
                        return True
                    prev_score = prev.get("score")
                    if prev_score is None:
                        return True
                    return (curr > prev_score) if self.is_classification else (curr < prev_score)

                if is_better(current_score, best):
                    best = {
                        "metric": primary_metric,
                        "score": float(current_score),
                        "run_dir": os.path.relpath(run_dir, self.output_path),
                        "model_path": os.path.relpath(model_path, self.output_path),
                        "timestamp": timestamp,
                    }
                    with open(best_file, "w", encoding="utf-8") as f:
                        json.dump(best, f, indent=2)
                    self.logger.info(f"Updated best {model_name} by {primary_metric}: {current_score:.4f}")

                # Update global best tracking across all models
                overall_file = os.path.join(self.output_path, "best_overall.json")
                overall = None
                if os.path.exists(overall_file):
                    try:
                        with open(overall_file, "r", encoding="utf-8") as f:
                            overall = json.load(f)
                    except Exception:
                        overall = None

                def overall_is_better(curr_score: float, prev_entry: dict | None) -> bool:
                    if prev_entry is None:
                        return True
                    prev_metric = prev_entry.get("metric")
                    prev_score = prev_entry.get("score")
                    # Prefer same-metric comparison; if metrics differ, fall back to accuracy> or mse< semantics
                    if prev_score is None:
                        return True
                    if self.is_classification or (prev_metric == "accuracy"):
                        return curr_score > prev_score
                    # regression
                    return curr_score < prev_score

                if overall_is_better(current_score, overall):
                    overall = {
                        "metric": primary_metric,
                        "score": float(current_score),
                        "model_name": model_name,
                        "run_dir": os.path.relpath(run_dir, self.output_path),
                        "model_path": os.path.relpath(model_path, self.output_path),
                        "timestamp": timestamp,
                    }
                    with open(overall_file, "w", encoding="utf-8") as f:
                        json.dump(overall, f, indent=2)
                    self.logger.info(f"Updated best_overall.json: {model_name} {primary_metric}={current_score:.4f}")

        except Exception as e:
            self.logger.error(f"Failed to save model artifacts: {e}")

    def _generate_summary_report(self):
        """Generate and display summary report."""
        if not self.history:
            self.logger.warning("No models built successfully")
            return

        self.logger.info("\n" + "=" * 80)
        self.logger.info("MODEL BUILDING SUMMARY")
        self.logger.info("=" * 80)

        for model_name, results in self.history.items():
            self.logger.info(f"\n{model_name}:")
            self.logger.info(f"  Optimization Time: {results['optimization_time']:.2f}s")

            cv_metrics = results["cv_metrics"]
            self.logger.info(f"  CV Metrics (μ±σ):")
            for metric, (mean, std) in cv_metrics.items():
                self.logger.info(f"    {metric}: {mean:.4f} ± {std:.4f}")

            test_metrics = results["test_metrics"]
            self.logger.info(f"  Test Metrics:")
            for metric, value in test_metrics.items():
                self.logger.info(f"    {metric}: {value:.4f}")

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"BEST MODEL: {self.best_model_name}")
        self.logger.info(f"BEST SCORE: {self.best_score:.4f}")
        self.logger.info(f"{'='*80}\n")

    def _get_default_models(self) -> Dict[str, Dict[str, Any]]:
        """Get default model configurations."""
        if self.is_classification:
            return {
                "LogisticRegression": {
                    "model": LogisticRegression,
                    "population_size": 30,
                    "max_generations": 3,
                    "C_range": (0.001, 100),
                    "penalty_params": ("l1", "l2"),
                },
                "KNN": {
                    "model": KNeighborsClassifier,
                    "population_size": 20,
                    "max_generations": 2,
                },
                "RandomForestClassifier": {
                    "model": RandomForestClassifier,
                    "population_size": 20,
                    "max_generations": 2,
                    "n_estimators_range": (10, 100),
                    "max_depth_range": (3, 20),
                },
                "SVC": {
                    "model": SVC,
                    "population_size": 10,
                    "max_generations": 2,
                    "kernel_params": ("linear", "rbf"),
                    "C_range": (0.1, 10),
                },
                "SGDClassifier": {
                    "model": SGDClassifier,
                    "n_candidates": 20,
                },
                "DeepLearningClassifier": {
                    "model": DeepLearningClassifier,
                    "epochs": 30,
                    "learning_rate": 0.001,
                    "batch_size": 64,
                },
            }
        else:
            return {
                "LinearRegression": {
                    "model": LinearRegression,
                    "population_size": 20,
                    "max_generations": 2,
                },
                "RandomForestRegressor": {
                    "model": RandomForestRegressor,
                    "population_size": 20,
                    "max_generations": 2,
                    "n_estimators_range": (10, 100),
                    "max_depth_range": (3, 20),
                },
                "SVR": {
                    "model": SVR,
                    "population_size": 20,
                    "max_generations": 2,
                    "kernel_params": ("linear", "rbf", "poly"),
                    "C_range": (0.1, 100),
                },
            }

    def plot_summary(self):
        """Generate summary visualization plots."""
        if not self.history:
            self.logger.warning("No models to plot")
            return

        model_names = list(self.history.keys())

        # Classification or regression metric sets
        if self.is_classification:
            cv_metrics_list = ["accuracy", "precision", "recall", "f1"]
            test_metrics_list = ["accuracy", "precision", "recall", "f1"]
        else:
            cv_metrics_list = ["mse", "rmse", "mae", "r2"]
            test_metrics_list = ["mse", "rmse", "mae", "r2"]

        # 1) Training time (mean ± std)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        means = []
        stds = []
        for model_name in model_names:
            tt = self.history[model_name]["cv_metrics"].get("training_time", (0.0, 0.0))
            means.append(tt[0])
            stds.append(tt[1])
        ax1.bar(model_names, means, yerr=stds, capsize=5, color="lightgray", edgecolor="black")
        ax1.set_title("Training Time (CV Mean ± Std)")
        ax1.set_ylabel("Seconds")
        ax1.tick_params(axis="x", rotation=45)
        fig1.tight_layout()
        if self.save_models:
            path1 = f"{self.output_path}/visualizations/training_time.png"
            plt.savefig(path1, dpi=300)
            self.logger.info(f"Saved: {path1}")
        plt.close(fig1)

        # 2) CV metrics (mean ± std) - one subplot per metric
        rows = 2
        cols = int(np.ceil(len(cv_metrics_list) / rows))
        fig2, axes2 = plt.subplots(rows, cols, figsize=(14, 6))
        axes2 = axes2.flatten() if isinstance(axes2, np.ndarray) else [axes2]

        for idx, metric in enumerate(cv_metrics_list):
            ax = axes2[idx]
            means = []
            stds = []
            for model_name in model_names:
                m = self.history[model_name]["cv_metrics"].get(metric, (None, None))
                means.append(m[0] if m[0] is not None else 0.0)
                stds.append(m[1] if m[1] is not None else 0.0)
            ax.bar(model_names, means, yerr=stds, capsize=5, color="skyblue", edgecolor="navy")
            ax.set_title(f"CV {metric.upper()} (μ±σ)")
            ax.tick_params(axis="x", rotation=45)
        # Hide any unused subplots
        for j in range(len(cv_metrics_list), len(axes2)):
            fig2.delaxes(axes2[j])
        fig2.tight_layout()
        if self.save_models:
            path2 = f"{self.output_path}/visualizations/cv_metrics.png"
            plt.savefig(path2, dpi=300)
            self.logger.info(f"Saved: {path2}")
        plt.close(fig2)

        # 3) Test metrics (single bars, no std)
        rows = 2
        cols = int(np.ceil(len(test_metrics_list) / rows))
        fig3, axes3 = plt.subplots(rows, cols, figsize=(14, 6))
        axes3 = axes3.flatten() if isinstance(axes3, np.ndarray) else [axes3]
        for idx, metric in enumerate(test_metrics_list):
            ax = axes3[idx]
            values = []
            for model_name in model_names:
                v = self.history[model_name]["test_metrics"].get(metric, None)
                values.append(v if v is not None else 0.0)
            ax.bar(model_names, values, color="lightgreen", edgecolor="darkgreen")
            ax.set_title(f"Test {metric.upper()}")
            ax.tick_params(axis="x", rotation=45)
            for i, v in enumerate(values):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        for j in range(len(test_metrics_list), len(axes3)):
            fig3.delaxes(axes3[j])
        fig3.tight_layout()
        if self.save_models:
            path3 = f"{self.output_path}/visualizations/test_metrics.png"
            plt.savefig(path3, dpi=300)
            self.logger.info(f"Saved: {path3}")
        plt.close(fig3)
