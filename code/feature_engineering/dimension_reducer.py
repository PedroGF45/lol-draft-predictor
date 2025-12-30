import os
from datetime import datetime
from logging import Logger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_preparation.data_handler import DataHandler
from helpers.parquet_handler import ParquetHandler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler


class DimensionReducer:
    def __init__(self, logger: Logger, parquet_handler: ParquetHandler, random_state: int = 42) -> None:
        self.logger = logger
        self.parquet_handler = parquet_handler
        self.random_state = random_state
        self.pca_model: PCA | None = None
        self._pca_train: pd.DataFrame | None = None
        self._pca_test: pd.DataFrame | None = None

    def perform_dimension_reduction(
        self,
        data_handler: DataHandler,
        explain_percentage: float = 0.9,
        n_best_features: int = 20,
        pca_plot_dir: str | None = None,
        pca_save_dir: str | None = None,
        output_dir: str | None = None,
        correlation_threshold: float = 0.8,
        verbose: bool = True,
    ) -> None:
        """
        Perform dimension reduction using PCA followed by mutual-information based feature selection.

        Args:
            data_handler (DataHandler): DataHandler with train/test splits.
            explain_percentage (float): Target percentage of explained variance for PCA (0-1).
            n_best_features (int): Number of best features to select via mutual information.
            pca_plot_dir (str | None): Optional directory to save PCA variance plot (timestamped filename auto-generated).
            pca_save_dir (str | None): Optional directory to save PCA-only splits.
            output_dir (str | None): Optional directory to save final reduced splits.
            correlation_threshold (float): Correlation threshold for removing redundant features.
            verbose (bool): Verbosity mode.
        """
        if data_handler.get_data_train() is None or data_handler.get_data_test() is None:
            raise ValueError("DataHandler must contain train/test splits before reduction")

        self.logger.info(
            f"Starting dimension reduction: PCA (explain_percentage={explain_percentage}) → "
            f"Mutual Info Feature Selection (n_best={n_best_features})"
        )

        # Step 1: Perform PCA
        self._perform_pca(
            data_handler=data_handler,
            explain_percentage=explain_percentage,
            plot_dir=pca_plot_dir,
            save_dir=pca_save_dir,
            verbose=verbose,
        )

        # Step 2: Feature selection via mutual information with redundancy removal
        selected_features = self.identify_relevant_features(
            data_handler=data_handler,
            n_best=n_best_features,
            correlation_threshold=correlation_threshold,
            verbose=verbose,
        )

        # Step 3: Apply feature selection
        self._apply_feature_selection(data_handler, selected_features, verbose=verbose)

        # Step 4: Optionally save final reduced splits
        if output_dir:
            saved_dir = data_handler.save_splits(output_dir)
            self.logger.info(f"Dimension-reduced data saved via DataHandler to {saved_dir}")

    def _perform_pca(
        self,
        data_handler: DataHandler,
        explain_percentage: float = 0.9,
        plot_dir: str | None = None,
        save_dir: str | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Fit PCA to achieve target explained variance and transform data.

        Args:
            data_handler (DataHandler): DataHandler with train/test splits.
            explain_percentage (float): Target percentage of explained variance.
            plot_dir (str | None): Optional directory to save PCA variance plot (timestamped filename auto-generated).
            save_dir (str | None): Optional directory to save PCA-only splits.
            verbose (bool): Verbosity mode.
        """
        data_train = data_handler.get_data_train()
        data_test = data_handler.get_data_test()

        if data_train is None or data_test is None:
            raise ValueError("Train/Test data unavailable in DataHandler")

        if explain_percentage <= 0 or explain_percentage > 1:
            raise ValueError("explain_percentage must be between 0 and 1")

        if verbose:
            self.logger.info(f"Applying PCA to achieve {explain_percentage * 100}% explained variance...")

        # Fit PCA without specifying n_components to determine needed components
        pca_full = PCA()
        pca_full.fit(data_train)

        # Calculate cumulative explained variance
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= explain_percentage) + 1
        n_components = min(n_components, data_train.shape[1])

        if verbose:
            self.logger.info(f"Determined n_components={n_components} to achieve {explain_percentage * 100}% variance")

        # Record input feature names used for PCA
        self.pca_input_features = list(data_train.columns)

        # Fit PCA with determined components
        self.pca_model = PCA(n_components=n_components, random_state=self.random_state)
        train_pca = self.pca_model.fit_transform(data_train)
        test_pca = self.pca_model.transform(data_test)

        # Normalize PCA results
        # Scale PCA outputs and persist scaler
        self.pca_output_scaler = StandardScaler()
        train_pca_scaled = self.pca_output_scaler.fit_transform(train_pca)
        test_pca_scaled = self.pca_output_scaler.transform(test_pca)

        # Create PCA component names
        pca_columns = [f"pca_component_{i+1}" for i in range(n_components)]

        # Store PCA results as DataFrames
        self._pca_train = pd.DataFrame(train_pca_scaled, columns=pca_columns, index=data_train.index)
        self._pca_test = pd.DataFrame(test_pca_scaled, columns=pca_columns, index=data_test.index)

        # Concatenate PCA components with original data (features will be selected later)
        data_handler.set_data_train(pd.concat([data_train, self._pca_train], axis=1))
        data_handler.set_data_test(pd.concat([data_test, self._pca_test], axis=1))

        explained_var = self.pca_model.explained_variance_ratio_.sum()
        if verbose:
            self.logger.info(
                f"PCA complete with {n_components} components. Cumulative explained variance: {explained_var:.4f}"
            )

        if plot_dir:
            self._plot_pca_variance(plot_dir)

        if save_dir:
            saved_dir = data_handler.save_splits(save_dir)
            self.logger.info(f"PCA-transformed splits saved via DataHandler to {saved_dir}")

    def _plot_pca_variance(self, output_dir: str) -> None:
        """
        Plot PCA component importance and cumulative explained variance.

        Args:
            output_dir (str): Directory to save the plot with auto-generated timestamped filename.
        """
        if not self.pca_model:
            raise ValueError("PCA model not fitted. Run _perform_pca first.")

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_pca_variance.png"
        output_path = os.path.join(output_dir, filename)

        explained_var = self.pca_model.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Bar plot for individual explained variance
        x = range(1, len(explained_var) + 1)
        ax.bar(x, explained_var, alpha=0.7, color="blue", label="Individual explained variance")

        # Line plot for cumulative explained variance
        ax.plot(x, cumsum_var, marker="o", linestyle="-", color="red", label="Cumulative explained variance")

        # Add horizontal line for 90% threshold
        ax.axhline(y=0.9, color="green", linestyle="--", label="90% threshold")

        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance")
        ax.set_title("PCA: Component Importance and Explained Variance")
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        self.logger.info(f"PCA explained variance plot saved to {output_path}")

    def identify_relevant_features(
        self,
        data_handler: DataHandler,
        n_best: int = 20,
        correlation_threshold: float = 0.8,
        verbose: bool = True,
    ) -> list:
        """
        Identify relevant features using mutual information with redundancy removal.

        Args:
            data_handler (DataHandler): DataHandler with train/test data.
            n_best (int): Number of best features to select.
            correlation_threshold (float): Correlation threshold for redundancy.
            verbose (bool): Verbosity mode.

        Returns:
            list: List of selected feature names.
        """
        data_train = data_handler.get_data_train()
        labels_train = data_handler.get_labels_train()

        if data_train is None or labels_train is None:
            raise ValueError("Training data or labels not available in DataHandler")

        if verbose:
            self.logger.info(f"Performing mutual information based feature selection (n_best={n_best})...")

        try:
            # Calculate mutual information for all features
            is_classification = len(np.unique(labels_train)) <= 20  # Heuristic
            if is_classification:
                mutual_info_scores = mutual_info_classif(data_train, labels_train, random_state=self.random_state)
            else:
                mutual_info_scores = mutual_info_regression(data_train, labels_train, random_state=self.random_state)

            # Rank features by mutual information
            feature_scores = pd.Series(mutual_info_scores, index=data_train.columns)
            ranked_features = feature_scores.sort_values(ascending=False)

            if verbose:
                problem_type = "classification" if is_classification else "regression"
                self.logger.info(f"Top 10 features by mutual information ({problem_type}):")
                self.logger.info(ranked_features.head(10).to_string())

            # Greedy redundancy removal
            initial_selected = ranked_features.head(len(ranked_features)).index.tolist()
            final_selected = []
            remaining = initial_selected[:]

            if verbose and len(initial_selected) > 1:
                self.logger.info(f"Removing redundant features (correlation threshold={correlation_threshold})...")

            while remaining and len(final_selected) < n_best:
                current_feature = remaining.pop(0)
                is_redundant = False

                if final_selected:
                    # Check correlation with already selected features
                    correlation = (
                        data_train[final_selected + [current_feature]]
                        .corr()[current_feature]
                        .abs()
                        .drop(current_feature)
                    )
                    if (correlation > correlation_threshold).any():
                        if verbose:
                            corr_features = correlation[correlation > correlation_threshold].index.tolist()
                            self.logger.info(f"Skipping '{current_feature}' (correlated with {corr_features})")
                        is_redundant = True

                if not is_redundant:
                    final_selected.append(current_feature)

            if verbose:
                self.logger.info(f"Final selected features: {final_selected}")

            return final_selected[:n_best]

        except Exception as e:
            self.logger.error(f"Error during feature selection: {e}")
            return []

    def _apply_feature_selection(
        self, data_handler: DataHandler, selected_features: list, verbose: bool = True
    ) -> None:
        """
        Apply feature selection by keeping only selected features.

        Args:
            data_handler (DataHandler): DataHandler to update.
            selected_features (list): List of feature names to keep.
            verbose (bool): Verbosity mode.
        """
        if not selected_features:
            self.logger.warning("No features selected. Keeping original data.")
            return

        try:
            data_train = data_handler.get_data_train()
            data_test = data_handler.get_data_test()

            # Filter to only selected features that exist
            available_features = [f for f in selected_features if f in data_train.columns]

            if verbose:
                self.logger.info(
                    f"Before feature selection: train shape={data_train.shape}, test shape={data_test.shape}"
                )

            data_handler.set_data_train(data_train[available_features])
            data_handler.set_data_test(data_test[available_features])

            if verbose:
                self.logger.info(
                    f"After feature selection: train shape={data_train[available_features].shape}, test shape={data_test[available_features].shape}"
                )
                self.logger.info(f"Selected features: {available_features}")

        except Exception as e:
            self.logger.error(f"Error applying feature selection: {e}")
