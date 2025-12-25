"""
Deep Learning Models for League of Legends Draft Prediction.

Implements PyTorch-based deep neural networks for both classification and regression.
Includes training, validation, early stopping, and comprehensive evaluation.
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Default network architecture
DEFAULT_LAYER_CONFIG: List[Tuple[int, float]] = [
    (256, 0.3),
    (128, 0.3),
    (64, 0.2),
    (32, 0.1),
]


class DeepLearningModel(nn.Module, ABC):
    """
    Abstract base class for deep learning models.
    
    Provides framework for building, training, evaluating, and visualizing
    PyTorch-based neural networks for draft prediction.
    """

    def __init__(
        self,
        input_dim: int,
        num_outputs: int,
        layer_config: Optional[List[Tuple[int, float]]] = None,
        batch_size: int = 32,
        output_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
        require_cuda: bool = True,
    ):
        """
        Initialize DeepLearningModel.
        
        Args:
            input_dim: Number of input features
            num_outputs: Number of output units (classes for classifier, 1 for regressor)
            layer_config: List of (hidden_units, dropout_rate) tuples
            batch_size: Batch size for training and validation
            output_path: Directory to save model and plots
            logger: Logger instance
            verbose: Print detailed progress
        """
        super(DeepLearningModel, self).__init__()

        self.input_dim = input_dim
        self.num_outputs = num_outputs
        self.layer_config = layer_config or DEFAULT_LAYER_CONFIG
        self.batch_size = batch_size
        self.output_path = output_path
        self.logger = logger or self._setup_logger()
        self.verbose = verbose
        self.require_cuda = require_cuda

        # Detect device
        self.device = self._select_device()
        if self.require_cuda and self.device.type != "cuda":
            raise RuntimeError(
                "CUDA GPU is required for DeepLearningModel but not available. Install CUDA-enabled PyTorch and ensure a GPU is accessible."
            )

        # Early stopping attributes
        self._es_best_val_metric = None
        self._es_epochs_no_improve = 0
        self._es_best_model_state = None
        self.best_epoch = -1

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metric": [],
            "val_metric": [],
        }

        # Build network
        self._build_network()

        # Move to device
        self.to(self.device)

    def _setup_logger(self) -> logging.Logger:
        """Setup logger if not provided."""
        logger = logging.getLogger("DeepLearningModel")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _select_device(self) -> torch.device:
        """Select GPU if available, else CPU."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            if self.verbose:
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            if self.verbose:
                self.logger.info("Using CPU")
        return device

    def _build_network(self):
        """Build neural network layers."""
        layers = []
        prev_dim = self.input_dim

        # Hidden layers
        for units, dropout_rate in self.layer_config:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = units

        # Output layer
        layers.append(nn.Linear(prev_dim, self.num_outputs))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def prepare_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """Prepare data loaders."""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        
        # For classification: labels should be 1D LongTensor (class indices)
        # For regression: labels should be 2D FloatTensor (predictions)
        if hasattr(self, 'num_classes'):  # Classification (DeepLearningClassifier)
            y_train_tensor = torch.LongTensor(y_train.astype(np.int64)).to(self.device)
            y_val_tensor = torch.LongTensor(y_val.astype(np.int64)).to(self.device)
        else:  # Regression (DeepLearningRegressor)
            y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)

        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        # Create loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        optimizer_name: str = "adam",
        criterion_name: Optional[str] = None,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.001,
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer
            optimizer_name: Optimizer name ('adam', 'sgd', 'rmsprop')
            criterion_name: Loss function name (set by subclass if None)
            early_stopping_patience: Patience for early stopping
            early_stopping_min_delta: Minimum improvement for early stopping
        
        Returns:
            Training history dictionary
        """
        self.prepare_data(X_train, y_train, X_val, y_val)

        # Setup optimizer
        if optimizer_name.lower() == "adam":
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name.lower() == "rmsprop":
            optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Setup criterion (loss function)
        if criterion_name is None:
            criterion = self._get_default_criterion()
        else:
            criterion = self._get_criterion(criterion_name)

        if self.verbose:
            self.logger.info(f"Training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_metric = self._train_epoch(optimizer, criterion)
            self.history["train_loss"].append(train_loss)
            self.history["train_metric"].append(train_metric)

            # Validation phase
            val_loss, val_metric = self._validate_epoch(criterion)
            self.history["val_loss"].append(val_loss)
            self.history["val_metric"].append(val_metric)

            if self.verbose and (epoch + 1) % max(1, num_epochs // 10) == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                )

            # Early stopping
            if self._check_early_stopping(
                val_metric,
                early_stopping_patience,
                early_stopping_min_delta,
            ):
                if self.verbose:
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}"
                    )
                break

        # Load best model
        if self._es_best_model_state is not None:
            self.load_state_dict(self._es_best_model_state)
            if self.verbose:
                self.logger.info(f"Loaded best model from epoch {self.best_epoch}")

        return self.history

    def _train_epoch(self, optimizer, criterion) -> Tuple[float, float]:
        """Train for one epoch."""
        self.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_x, batch_y in self.train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = self.forward(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(batch_y.detach().cpu().numpy())

        epoch_loss = total_loss / len(self.train_loader.dataset)
        # Handle both 1D and 2D arrays safely
        labels_array = np.concatenate(all_labels) if all_labels[0].ndim == 1 else np.vstack(all_labels)
        preds_array = np.vstack(all_preds)  # predictions are always 2D from forward pass
        epoch_metric = self._compute_metric(labels_array, preds_array)

        return epoch_loss, epoch_metric

    def _validate_epoch(self, criterion) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                outputs = self.forward(batch_x)
                loss = criterion(outputs, batch_y)

                total_loss += loss.item() * batch_x.size(0)
                all_preds.append(outputs.detach().cpu().numpy())
                all_labels.append(batch_y.detach().cpu().numpy())

        epoch_loss = total_loss / len(self.val_loader.dataset)
        # Handle both 1D and 2D arrays safely
        labels_array = np.concatenate(all_labels) if all_labels[0].ndim == 1 else np.vstack(all_labels)
        preds_array = np.vstack(all_preds)  # predictions are always 2D from forward pass
        epoch_metric = self._compute_metric(labels_array, preds_array)

        return epoch_loss, epoch_metric

    def _check_early_stopping(
        self,
        current_metric: float,
        patience: int,
        min_delta: float,
    ) -> bool:
        """Check if early stopping criteria are met."""
        if self._es_best_val_metric is None:
            self._es_best_val_metric = current_metric
            self._es_best_model_state = self.state_dict().copy()
            self.best_epoch = len(self.history["val_loss"]) - 1
            return False

        # For classification, higher is better
        if self._should_update_best(current_metric):
            self._es_best_val_metric = current_metric
            self._es_epochs_no_improve = 0
            self._es_best_model_state = self.state_dict().copy()
            self.best_epoch = len(self.history["val_loss"]) - 1
        else:
            self._es_epochs_no_improve += 1

        return self._es_epochs_no_improve >= patience

    @abstractmethod
    def _should_update_best(self, current_metric: float) -> bool:
        """Determine if current metric is better than best."""
        pass

    @abstractmethod
    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute metric from predictions."""
        pass

    @abstractmethod
    def _get_default_criterion(self):
        """Get default loss function."""
        pass

    @abstractmethod
    def _get_criterion(self, name: str):
        """Get loss function by name."""
        pass

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test set."""
        pass

    def plot_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        axes[0].plot(self.history["train_loss"], label="Train")
        axes[0].plot(self.history["val_loss"], label="Validation")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training History - Loss")
        axes[0].legend()
        axes[0].grid()

        # Metric
        axes[1].plot(self.history["train_metric"], label="Train")
        axes[1].plot(self.history["val_metric"], label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Metric")
        axes[1].set_title("Training History - Metric")
        axes[1].legend()
        axes[1].grid()

        plt.tight_layout()

        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{self.output_path}/training_history_{timestamp}.png"
            plt.savefig(filepath, dpi=300)
            self.logger.info(f"History plot saved: {filepath}")

        plt.show()

    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        self.logger.info(f"Model saved: {path}")

    def load(self, path: str):
        """Load model from disk."""
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.logger.info(f"Model loaded: {path}")


class DeepLearningClassifier(DeepLearningModel):
    """Deep learning classifier for draft prediction."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        layer_config: Optional[List[Tuple[int, float]]] = None,
        batch_size: int = 32,
        output_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
    ):
        """
        Initialize classifier.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
        """
        self.num_classes = num_classes
        super().__init__(
            input_dim=input_dim,
            num_outputs=num_classes,
            layer_config=layer_config,
            batch_size=batch_size,
            output_path=output_path,
            logger=logger,
            verbose=verbose,
        )

    def _should_update_best(self, current_metric: float) -> bool:
        """For classification, higher metric is better."""
        return current_metric > self._es_best_val_metric

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy."""
        # y_true is already class indices (1D), y_pred is logits (2D: batch_size x num_classes)
        y_pred_classes = np.argmax(y_pred, axis=1)
        return accuracy_score(y_true, y_pred_classes)

    def _get_default_criterion(self):
        """Use cross-entropy loss for classification."""
        return nn.CrossEntropyLoss()

    def _get_criterion(self, name: str):
        """Get loss function by name."""
        loss_dict = {
            "crossentropy": nn.CrossEntropyLoss(),
            "bce": nn.BCEWithLogitsLoss(),
        }
        return loss_dict.get(name.lower(), nn.CrossEntropyLoss())

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate classifier on test set."""
        self.eval()

        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)

        with torch.no_grad():
            outputs = self.forward(X_test_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

        return metrics


class DeepLearningRegressor(DeepLearningModel):
    """Deep learning regressor for draft prediction."""

    def __init__(
        self,
        input_dim: int,
        layer_config: Optional[List[Tuple[int, float]]] = None,
        batch_size: int = 32,
        output_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
    ):
        """
        Initialize regressor.
        
        Args:
            input_dim: Number of input features
        """
        super().__init__(
            input_dim=input_dim,
            num_outputs=1,
            layer_config=layer_config,
            batch_size=batch_size,
            output_path=output_path,
            logger=logger,
            verbose=verbose,
        )

    def _should_update_best(self, current_metric: float) -> bool:
        """For regression, lower loss (higher R²) is better."""
        return current_metric > self._es_best_val_metric

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² score."""
        return r2_score(y_true.ravel(), y_pred.ravel())

    def _get_default_criterion(self):
        """Use MSE loss for regression."""
        return nn.MSELoss()

    def _get_criterion(self, name: str):
        """Get loss function by name."""
        loss_dict = {
            "mse": nn.MSELoss(),
            "mae": nn.L1Loss(),
            "smooth_l1": nn.SmoothL1Loss(),
        }
        return loss_dict.get(name.lower(), nn.MSELoss())

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate regressor on test set."""
        self.eval()

        X_test_tensor = torch.FloatTensor(X_test).to(self.device)

        with torch.no_grad():
            outputs = self.forward(X_test_tensor)
            y_pred = outputs.cpu().numpy().ravel()

        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

        return metrics
