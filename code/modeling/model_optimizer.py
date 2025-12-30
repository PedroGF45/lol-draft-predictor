"""
Model Optimizer for League of Legends Draft Prediction.

Hyperparameter optimization using genetic algorithms (GA) for various sklearn models.
Supports both classification and regression tasks.
"""

import random
import copy
import time
import logging
import warnings
import numpy as np
import inspect
from typing import Dict, Any, Optional, Tuple, Callable

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


class ModelOptimizer:
    """
    Hyperparameter optimizer using genetic algorithm.

    Optimizes hyperparameters for multiple sklearn model types using a genetic algorithm
    approach with selection, crossover, and mutation operators.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        is_classification: bool = True,
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
        random_state: int = 42,
    ):
        """
        Initialize ModelOptimizer.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            is_classification: True for classification, False for regression
            logger: Logger instance
            verbose: Print detailed GA progress
            random_state: Random seed for reproducibility
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.is_classification = is_classification
        self.logger = logger or self._setup_logger()
        self.verbose = verbose
        self.random_state = random_state

        random.seed(random_state)
        np.random.seed(random_state)

    def _setup_logger(self) -> logging.Logger:
        """Setup logger if not provided."""
        logger = logging.getLogger("ModelOptimizer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def optimize_linear_model(
        self,
        model_class,
        population_size: int = 30,
        max_generations: int = 3,
        C_range: Tuple[float, float] = (0.001, 100),
        penalty_params: Tuple[str] = ("l1", "l2"),
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.3,
        tournament_size: int = 3,
        elitism_count: int = 1,
        **kwargs,
    ) -> Optional[Tuple[Dict[str, Any], Any]]:
        """Optimize linear regression or logistic regression."""

        is_logistic = model_class == LogisticRegression
        model_type = "LogisticRegression" if is_logistic else "LinearRegression"

        if self.verbose:
            self.logger.info(f"Optimizing {model_type}")

        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {
                "C": random.uniform(C_range[0], C_range[1]) if is_logistic else None,
                "penalty": random.choice(penalty_params) if is_logistic else None,
                "fit_intercept": random.choice([True, False]),
            }
            population.append(individual)

        best_individual = None
        best_fitness = -float("inf") if self.is_classification else float("inf")
        best_model = None

        for generation in range(max_generations):
            if self.verbose:
                self.logger.info(f"\nGeneration {generation + 1}/{max_generations}")

            # Evaluate fitness
            fitnesses = []
            models = []

            for individual in population:
                fitness, model, _ = self._evaluate_individual(individual, model_class, model_type)
                fitnesses.append(fitness)
                models.append(model)

                # Track best
                is_better = (self.is_classification and fitness > best_fitness) or (
                    not self.is_classification and fitness < best_fitness
                )
                if is_better:
                    best_fitness = fitness
                    best_individual = copy.deepcopy(individual)
                    best_model = model

            if self.verbose:
                self.logger.info(f"  Best fitness: {best_fitness:.4f} | " f"Avg fitness: {np.mean(fitnesses):.4f}")

            # Selection, crossover, mutation
            new_population = []

            # Elitism
            if best_individual is not None:
                new_population.append(copy.deepcopy(best_individual))

            # Generate new individuals
            while len(new_population) < population_size:
                parent1 = self._tournament_selection(population, fitnesses, tournament_size)
                parent2 = self._tournament_selection(population, fitnesses, tournament_size)

                if random.random() < crossover_rate:
                    child = self._crossover_linear(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)

                if random.random() < mutation_rate:
                    child = self._mutate_linear(child, C_range, penalty_params)

                new_population.append(child)

            population = new_population[:population_size]

        return (
            (
                best_individual,
                best_model,
            )
            if best_individual is not None
            else None
        )

    def optimize_svm(
        self,
        model_class,
        population_size: int = 10,
        max_generations: int = 2,
        kernel_params: Tuple[str] = ("linear", "rbf"),
        C_range: Tuple[float, float] = (0.1, 10),
        gamma_range: Tuple[float, float] = (0.001, 0.1),
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.3,
        tournament_size: int = 3,
        elitism_count: int = 1,
        **kwargs,
    ) -> Optional[Tuple[Dict[str, Any], Any]]:
        """Optimize SVM (SVC or SVR)."""

        model_type = "SVC" if model_class == SVC else "SVR"

        if self.verbose:
            self.logger.info(f"Optimizing {model_type}")

        population = []
        for _ in range(population_size):
            individual = {
                "kernel": random.choice(kernel_params),
                "C": random.uniform(C_range[0], C_range[1]),
                "gamma": random.uniform(gamma_range[0], gamma_range[1]),
            }
            population.append(individual)

        best_individual = None
        best_fitness = -float("inf") if self.is_classification else float("inf")
        best_model = None

        for generation in range(max_generations):
            if self.verbose:
                self.logger.info(f"\nGeneration {generation + 1}/{max_generations}")

            fitnesses = []
            models = []

            for individual in population:
                fitness, model, _ = self._evaluate_individual(individual, model_class, model_type)
                fitnesses.append(fitness)
                models.append(model)

                is_better = (self.is_classification and fitness > best_fitness) or (
                    not self.is_classification and fitness < best_fitness
                )
                if is_better:
                    best_fitness = fitness
                    best_individual = copy.deepcopy(individual)
                    best_model = model

            if self.verbose:
                self.logger.info(f"  Best fitness: {best_fitness:.4f} | " f"Avg fitness: {np.mean(fitnesses):.4f}")

            new_population = []
            if best_individual is not None:
                new_population.append(copy.deepcopy(best_individual))

            while len(new_population) < population_size:
                parent1 = self._tournament_selection(population, fitnesses, tournament_size)
                parent2 = self._tournament_selection(population, fitnesses, tournament_size)

                if random.random() < crossover_rate:
                    child = self._crossover_svm(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)

                if random.random() < mutation_rate:
                    child = self._mutate_svm(child, kernel_params, C_range, gamma_range)

                new_population.append(child)

            population = new_population[:population_size]

        return (best_individual, best_model) if best_individual is not None else None

    def optimize_random_forest(
        self,
        model_class,
        population_size: int = 20,
        max_generations: int = 2,
        n_estimators_range: Tuple[int, int] = (10, 100),
        max_depth_range: Tuple[int, int] = (3, 20),
        min_samples_split_range: Tuple[int, int] = (2, 10),
        min_samples_leaf_range: Tuple[int, int] = (1, 5),
        max_features_params: Tuple[str] = ("sqrt", "log2"),
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.3,
        tournament_size: int = 3,
        **kwargs,
    ) -> Optional[Tuple[Dict[str, Any], Any]]:
        """Optimize Random Forest (Classifier or Regressor)."""

        model_type = "RandomForestClassifier" if model_class == RandomForestClassifier else "RandomForestRegressor"

        if self.verbose:
            self.logger.info(f"Optimizing {model_type}")

        population = []
        for _ in range(population_size):
            individual = {
                "n_estimators": random.randint(n_estimators_range[0], n_estimators_range[1]),
                "max_depth": random.randint(max_depth_range[0], max_depth_range[1]),
                "min_samples_split": random.randint(min_samples_split_range[0], min_samples_split_range[1]),
                "min_samples_leaf": random.randint(min_samples_leaf_range[0], min_samples_leaf_range[1]),
                "max_features": random.choice(max_features_params),
            }
            population.append(individual)

        best_individual = None
        best_fitness = -float("inf") if self.is_classification else float("inf")
        best_model = None

        for generation in range(max_generations):
            if self.verbose:
                self.logger.info(f"\nGeneration {generation + 1}/{max_generations}")

            fitnesses = []
            models = []

            for individual in population:
                fitness, model, _ = self._evaluate_individual(individual, model_class, model_type)
                fitnesses.append(fitness)
                models.append(model)

                is_better = (self.is_classification and fitness > best_fitness) or (
                    not self.is_classification and fitness < best_fitness
                )
                if is_better:
                    best_fitness = fitness
                    best_individual = copy.deepcopy(individual)
                    best_model = model

            if self.verbose:
                self.logger.info(f"  Best fitness: {best_fitness:.4f} | " f"Avg fitness: {np.mean(fitnesses):.4f}")

            new_population = []
            if best_individual is not None:
                new_population.append(copy.deepcopy(best_individual))

            while len(new_population) < population_size:
                parent1 = self._tournament_selection(population, fitnesses, tournament_size)
                parent2 = self._tournament_selection(population, fitnesses, tournament_size)

                if random.random() < crossover_rate:
                    child = self._crossover_rf(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)

                if random.random() < mutation_rate:
                    child = self._mutate_rf(
                        child,
                        n_estimators_range,
                        max_depth_range,
                        min_samples_split_range,
                        min_samples_leaf_range,
                        max_features_params,
                    )

                new_population.append(child)

            population = new_population[:population_size]

        return (best_individual, best_model) if best_individual is not None else None

    def optimize_gradient_boosting(
        self,
        model_class,
        population_size: int = 15,
        max_generations: int = 2,
        n_estimators_range: Tuple[int, int] = (50, 200),
        learning_rate_range: Tuple[float, float] = (0.01, 0.2),
        max_depth_range: Tuple[int, int] = (2, 8),
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.3,
        tournament_size: int = 3,
        **kwargs,
    ) -> Optional[Tuple[Dict[str, Any], Any]]:
        """Optimize Gradient Boosting (Classifier or Regressor)."""

        model_type = (
            "GradientBoostingClassifier" if model_class == GradientBoostingClassifier else "GradientBoostingRegressor"
        )

        if self.verbose:
            self.logger.info(f"Optimizing {model_type}")

        population = []
        for _ in range(population_size):
            individual = {
                "n_estimators": random.randint(n_estimators_range[0], n_estimators_range[1]),
                "learning_rate": random.uniform(learning_rate_range[0], learning_rate_range[1]),
                "max_depth": random.randint(max_depth_range[0], max_depth_range[1]),
                "subsample": random.uniform(0.5, 1.0),
            }
            population.append(individual)

        best_individual = None
        best_fitness = -float("inf") if self.is_classification else float("inf")
        best_model = None

        for generation in range(max_generations):
            if self.verbose:
                self.logger.info(f"\nGeneration {generation + 1}/{max_generations}")

            fitnesses = []
            models = []

            for individual in population:
                fitness, model, _ = self._evaluate_individual(individual, model_class, model_type)
                fitnesses.append(fitness)
                models.append(model)

                is_better = (self.is_classification and fitness > best_fitness) or (
                    not self.is_classification and fitness < best_fitness
                )
                if is_better:
                    best_fitness = fitness
                    best_individual = copy.deepcopy(individual)
                    best_model = model

            if self.verbose:
                self.logger.info(f"  Best fitness: {best_fitness:.4f} | " f"Avg fitness: {np.mean(fitnesses):.4f}")

            new_population = []
            if best_individual is not None:
                new_population.append(copy.deepcopy(best_individual))

            while len(new_population) < population_size:
                parent1 = self._tournament_selection(population, fitnesses, tournament_size)
                parent2 = self._tournament_selection(population, fitnesses, tournament_size)

                if random.random() < crossover_rate:
                    child = self._crossover_gb(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)

                if random.random() < mutation_rate:
                    child = self._mutate_gb(
                        child,
                        n_estimators_range,
                        learning_rate_range,
                        max_depth_range,
                    )

                new_population.append(child)

            population = new_population[:population_size]

        return (best_individual, best_model) if best_individual is not None else None

    def optimize_knn(
        self,
        model_class,
        population_size: int = 20,
        max_generations: int = 2,
        n_neighbors_range: Tuple[int, int] = (1, 20),
        weights_params: Tuple[str] = ("uniform", "distance"),
        algorithm_params: Tuple[str] = ("ball_tree", "kd_tree", "brute"),
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.3,
        tournament_size: int = 3,
        **kwargs,
    ) -> Optional[Tuple[Dict[str, Any], Any]]:
        """Optimize KNN (Classifier or Regressor)."""

        model_type = "KNeighborsClassifier" if model_class == KNeighborsClassifier else "KNeighborsRegressor"

        if self.verbose:
            self.logger.info(f"Optimizing {model_type}")

        population = []
        for _ in range(population_size):
            individual = {
                "n_neighbors": random.randint(n_neighbors_range[0], n_neighbors_range[1]),
                "weights": random.choice(weights_params),
                "algorithm": random.choice(algorithm_params),
            }
            population.append(individual)

        best_individual = None
        best_fitness = -float("inf") if self.is_classification else float("inf")
        best_model = None

        for generation in range(max_generations):
            if self.verbose:
                self.logger.info(f"\nGeneration {generation + 1}/{max_generations}")

            fitnesses = []
            models = []

            for individual in population:
                fitness, model, _ = self._evaluate_individual(individual, model_class, model_type)
                fitnesses.append(fitness)
                models.append(model)

                is_better = (self.is_classification and fitness > best_fitness) or (
                    not self.is_classification and fitness < best_fitness
                )
                if is_better:
                    best_fitness = fitness
                    best_individual = copy.deepcopy(individual)
                    best_model = model

            if self.verbose:
                self.logger.info(f"  Best fitness: {best_fitness:.4f} | " f"Avg fitness: {np.mean(fitnesses):.4f}")

            new_population = []
            if best_individual is not None:
                new_population.append(copy.deepcopy(best_individual))

            while len(new_population) < population_size:
                parent1 = self._tournament_selection(population, fitnesses, tournament_size)
                parent2 = self._tournament_selection(population, fitnesses, tournament_size)

                if random.random() < crossover_rate:
                    child = self._crossover_knn(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)

                if random.random() < mutation_rate:
                    child = self._mutate_knn(
                        child,
                        n_neighbors_range,
                        weights_params,
                        algorithm_params,
                    )

                new_population.append(child)

            population = new_population[:population_size]

        return (best_individual, best_model) if best_individual is not None else None

    def optimize_mlp(
        self,
        model_class,
        population_size: int = 15,
        max_generations: int = 2,
        hidden_layer_sizes: Tuple[Tuple[int]] = ((64, 32), (128, 64, 32), (256, 128)),
        learning_rate_init_range: Tuple[float, float] = (0.0001, 0.01),
        alpha_range: Tuple[float, float] = (0.0001, 0.01),
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.3,
        tournament_size: int = 3,
        **kwargs,
    ) -> Optional[Tuple[Dict[str, Any], Any]]:
        """Optimize MLP (Classifier or Regressor)."""

        model_type = "MLPClassifier" if model_class == MLPClassifier else "MLPRegressor"

        if self.verbose:
            self.logger.info(f"Optimizing {model_type}")

        population = []
        for _ in range(population_size):
            individual = {
                "hidden_layer_sizes": random.choice(hidden_layer_sizes),
                "learning_rate_init": random.uniform(learning_rate_init_range[0], learning_rate_init_range[1]),
                "alpha": random.uniform(alpha_range[0], alpha_range[1]),
                "activation": random.choice(["relu", "tanh"]),
            }
            population.append(individual)

        best_individual = None
        best_fitness = -float("inf") if self.is_classification else float("inf")
        best_model = None

        for generation in range(max_generations):
            if self.verbose:
                self.logger.info(f"\nGeneration {generation + 1}/{max_generations}")

            fitnesses = []
            models = []

            for individual in population:
                fitness, model, _ = self._evaluate_individual(individual, model_class, model_type)
                fitnesses.append(fitness)
                models.append(model)

                is_better = (self.is_classification and fitness > best_fitness) or (
                    not self.is_classification and fitness < best_fitness
                )
                if is_better:
                    best_fitness = fitness
                    best_individual = copy.deepcopy(individual)
                    best_model = model

            if self.verbose:
                self.logger.info(f"  Best fitness: {best_fitness:.4f} | " f"Avg fitness: {np.mean(fitnesses):.4f}")

            new_population = []
            if best_individual is not None:
                new_population.append(copy.deepcopy(best_individual))

            while len(new_population) < population_size:
                parent1 = self._tournament_selection(population, fitnesses, tournament_size)
                parent2 = self._tournament_selection(population, fitnesses, tournament_size)

                if random.random() < crossover_rate:
                    child = self._crossover_mlp(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)

                if random.random() < mutation_rate:
                    child = self._mutate_mlp(
                        child,
                        hidden_layer_sizes,
                        learning_rate_init_range,
                        alpha_range,
                    )

                new_population.append(child)

            population = new_population[:population_size]

        return (best_individual, best_model) if best_individual is not None else None

    # ==================== Helper Methods ====================

    def _evaluate_individual(
        self,
        individual: Dict[str, Any],
        model_class,
        model_type: str,
    ) -> Tuple[float, Any, float]:
        """Evaluate fitness of an individual."""
        start_time = time.time()

        try:
            # Clean parameters (remove None values)
            params = {k: v for k, v in individual.items() if v is not None}

            # Handle special cases
            if model_type == "LogisticRegression":
                if params.get("penalty") == "l1":
                    params["solver"] = "liblinear"
                else:
                    params["solver"] = "lbfgs"

            if model_type == "MLPClassifier" or model_type == "MLPRegressor":
                params["max_iter"] = 500
                # Set random_state only if supported
                try:
                    sig = inspect.signature(model_class.__init__)
                    if "random_state" in sig.parameters:
                        params["random_state"] = self.random_state
                except Exception:
                    pass
            else:
                # Set random_state only if supported by the model
                try:
                    sig = inspect.signature(model_class.__init__)
                    if "random_state" in sig.parameters:
                        params["random_state"] = self.random_state
                except Exception:
                    pass

            # Build and train model
            model = model_class(**params)
            model.fit(self.X_train, self.y_train)

            # Evaluate
            y_pred = model.predict(self.X_test)

            if self.is_classification:
                fitness = accuracy_score(self.y_test, y_pred)
            else:
                fitness = -mean_squared_error(self.y_test, y_pred)  # Negate MSE for maximization

            training_time = time.time() - start_time
            return fitness, model, training_time

        except Exception as e:
            if self.verbose:
                self.logger.debug(f"Error evaluating individual: {e}")
            fitness = -float("inf") if self.is_classification else float("inf")
            return fitness, None, time.time() - start_time

    def _tournament_selection(
        self,
        population: list,
        fitnesses: list,
        tournament_size: int,
    ) -> Dict[str, Any]:
        """Select individual via tournament."""
        indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return copy.deepcopy(population[best_idx])

    # ==================== Crossover Methods ====================

    def _crossover_linear(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover for linear models."""
        child = {}
        for key in parent1.keys():
            child[key] = random.choice([parent1[key], parent2[key]])
        return child

    def _crossover_svm(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover for SVM."""
        child = {}
        for key in parent1.keys():
            child[key] = random.choice([parent1[key], parent2[key]])
        return child

    def _crossover_rf(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover for Random Forest."""
        child = {}
        for key in parent1.keys():
            if isinstance(parent1[key], (int, float)):
                # Blend numeric values
                child[key] = (parent1[key] + parent2[key]) / 2
                if isinstance(parent1[key], int):
                    child[key] = int(child[key])
            else:
                child[key] = random.choice([parent1[key], parent2[key]])
        return child

    def _crossover_gb(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover for Gradient Boosting."""
        child = {}
        for key in parent1.keys():
            if isinstance(parent1[key], (int, float)):
                child[key] = (parent1[key] + parent2[key]) / 2
                if isinstance(parent1[key], int):
                    child[key] = int(child[key])
            else:
                child[key] = random.choice([parent1[key], parent2[key]])
        return child

    def _crossover_knn(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover for KNN."""
        child = {}
        for key in parent1.keys():
            child[key] = random.choice([parent1[key], parent2[key]])
        return child

    def _crossover_mlp(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover for MLP."""
        child = {}
        for key in parent1.keys():
            if key == "hidden_layer_sizes":
                child[key] = random.choice([parent1[key], parent2[key]])
            else:
                child[key] = (parent1[key] + parent2[key]) / 2
        return child

    # ==================== Mutation Methods ====================

    def _mutate_linear(
        self,
        individual: Dict,
        C_range: Tuple[float, float],
        penalty_params: Tuple[str],
    ) -> Dict:
        """Mutate linear model individual."""
        mutant = copy.deepcopy(individual)
        if random.random() < 0.5 and mutant["C"] is not None:
            mutant["C"] = random.uniform(C_range[0], C_range[1])
        if random.random() < 0.5 and mutant["penalty"] is not None:
            mutant["penalty"] = random.choice(penalty_params)
        return mutant

    def _mutate_svm(
        self,
        individual: Dict,
        kernel_params: Tuple[str],
        C_range: Tuple[float, float],
        gamma_range: Tuple[float, float],
    ) -> Dict:
        """Mutate SVM individual."""
        mutant = copy.deepcopy(individual)
        if random.random() < 0.3:
            mutant["kernel"] = random.choice(kernel_params)
        if random.random() < 0.3:
            mutant["C"] = random.uniform(C_range[0], C_range[1])
        if random.random() < 0.3:
            mutant["gamma"] = random.uniform(gamma_range[0], gamma_range[1])
        return mutant

    def _mutate_rf(
        self,
        individual: Dict,
        n_estimators_range: Tuple[int, int],
        max_depth_range: Tuple[int, int],
        min_samples_split_range: Tuple[int, int],
        min_samples_leaf_range: Tuple[int, int],
        max_features_params: Tuple[str],
    ) -> Dict:
        """Mutate Random Forest individual."""
        mutant = copy.deepcopy(individual)
        if random.random() < 0.25:
            mutant["n_estimators"] = random.randint(n_estimators_range[0], n_estimators_range[1])
        if random.random() < 0.25:
            mutant["max_depth"] = random.randint(max_depth_range[0], max_depth_range[1])
        if random.random() < 0.25:
            mutant["min_samples_split"] = random.randint(min_samples_split_range[0], min_samples_split_range[1])
        if random.random() < 0.25:
            mutant["min_samples_leaf"] = random.randint(min_samples_leaf_range[0], min_samples_leaf_range[1])
        if random.random() < 0.25:
            mutant["max_features"] = random.choice(max_features_params)
        return mutant

    def _mutate_gb(
        self,
        individual: Dict,
        n_estimators_range: Tuple[int, int],
        learning_rate_range: Tuple[float, float],
        max_depth_range: Tuple[int, int],
    ) -> Dict:
        """Mutate Gradient Boosting individual."""
        mutant = copy.deepcopy(individual)
        if random.random() < 0.25:
            mutant["n_estimators"] = random.randint(n_estimators_range[0], n_estimators_range[1])
        if random.random() < 0.25:
            mutant["learning_rate"] = random.uniform(learning_rate_range[0], learning_rate_range[1])
        if random.random() < 0.25:
            mutant["max_depth"] = random.randint(max_depth_range[0], max_depth_range[1])
        if random.random() < 0.25:
            mutant["subsample"] = random.uniform(0.5, 1.0)
        return mutant

    def _mutate_knn(
        self,
        individual: Dict,
        n_neighbors_range: Tuple[int, int],
        weights_params: Tuple[str],
        algorithm_params: Tuple[str],
    ) -> Dict:
        """Mutate KNN individual."""
        mutant = copy.deepcopy(individual)
        if random.random() < 0.33:
            mutant["n_neighbors"] = random.randint(n_neighbors_range[0], n_neighbors_range[1])
        if random.random() < 0.33:
            mutant["weights"] = random.choice(weights_params)
        if random.random() < 0.33:
            mutant["algorithm"] = random.choice(algorithm_params)
        return mutant

    def _mutate_mlp(
        self,
        individual: Dict,
        hidden_layer_sizes: Tuple[Tuple[int]],
        learning_rate_init_range: Tuple[float, float],
        alpha_range: Tuple[float, float],
    ) -> Dict:
        """Mutate MLP individual."""
        mutant = copy.deepcopy(individual)
        if random.random() < 0.33:
            mutant["hidden_layer_sizes"] = random.choice(hidden_layer_sizes)
        if random.random() < 0.33:
            mutant["learning_rate_init"] = random.uniform(learning_rate_init_range[0], learning_rate_init_range[1])
        if random.random() < 0.33:
            mutant["alpha"] = random.uniform(alpha_range[0], alpha_range[1])
        return mutant
