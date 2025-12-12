from helpers.parquet_handler import ParquetHandler
from logging import Logger
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance

class DataVisualizer:
    """
    Utility class for performing exploratory data analysis (EDA) and generating visualizations.
    
    This class provides methods for creating various plots and statistical analyses
    to understand data distributions, correlations, and feature importance.
    """

    def __init__(self, logger: Logger, parquet_handler: ParquetHandler, random_state: int = 42) -> None:
        """
        Initialize the DataVisualizer.
        
        Args:
            logger (Logger): Logger instance for tracking operations.
            parquet_handler (ParquetHandler): Handler for reading parquet files.
            random_state (int): Random seed for reproducibility. Defaults to 42.
        """
        self.logger = logger
        self.parquet_handler = parquet_handler
        self.random_state = random_state

    def perform_eda(self,
                    data_input_path: str,
                    data_output_path: str,
                    mode: str,
                    plots: list = ["summary_statistics", "correlation_heatmap", "histogram", "violin", "scatter", "all_distributions", "skewness", "kurtosis", "feature_importance"],
                    prefix: str = "data",
                    figsize: tuple = (10, 8),
                    cmap: str = "coolwarm",
                    annot: bool = True,
                    fmt: str = ".2f",
                    sample_size: int = 10000,
                    dpi: int = 100,
                    n_estimators: int = 5,
                    n_repeats: int = 2
                    ) -> None:
        """
        Perform selected exploratory data analysis (EDA) visualizations.
        
        Generates various plots and statistical analyses based on the specified plot types.
        All methods use the sample_size parameter to limit data for performance.

        Args:
            data_input_path (str): Path to the input parquet data file.
            data_output_path (str): Directory path to save the output plots.
            plots (list): List of plot types to generate. Options: "summary_statistics",
                "correlation_heatmap", "histogram", "violin", "scatter", "all_distributions",
                "skewness", "kurtosis", "feature_importance". Defaults to all plots.
            prefix (str): Prefix for the output file names. Defaults to "data".
            figsize (tuple): Figure size (width, height). Defaults to (10, 8).
            cmap (str): Colormap for the heatmap. Defaults to "coolwarm".
            annot (bool): If True, write the data value in each cell for the heatmap. Defaults to True.
            fmt (str): String formatting code to use when annot is True for the heatmap. Defaults to ".2f".
            sample_size (int): Maximum number of samples to use for performance optimization. Defaults to 10000.
            dpi (int): Dots per inch for the saved images. Defaults to 100.
            n_estimators (int): Number of trees for RandomForestRegressor. Defaults to 5.
            n_repeats (int): Number of times to permute a feature for importance analysis. Defaults to 2.
        """

        if mode not in ["matches", "players"]:
            self.logger.error(f"Invalid mode '{mode}'. Choose either 'matches' or 'players'.")
            return

        # Branch output path based on mode, similar to parquet cleaning pipeline
        if mode == "matches":
            data_output_path = os.path.join(data_output_path, "matches")
        else:
            data_output_path = os.path.join(data_output_path, "players")

        data = self.parquet_handler.read_parquet(file_path=data_input_path, load_percentage=1.0)

        if self.parquet_handler.check_directory_exists(data_output_path) is False:
            os.makedirs(data_output_path, exist_ok=True)

        # Add timestamp to prefix for unique file naming, matching parquet convention
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        prefix = f"{timestamp}_{prefix}"

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if mode == "matches" and "team1_win" not in data.columns:
            self.logger.error("Target column 'team1_win' not found in the data for matches mode.")
            return
        
        if mode == "players" and "win_rate" not in data.columns:
            self.logger.error("Target column 'win_rate' not found in the data for players mode.")
            return
        
        # remove win_rate from numeric_cols to avoid leakage
        if mode == "players" and "win_rate" in numeric_cols:
            numeric_cols.remove("win_rate")

        if "summary_statistics" in plots:
            self.create_summary_statistics_to_csv(data[numeric_cols], 
                                                 output_path=f"{data_output_path}/{prefix}_summary_statistics.csv",
                                                 sample_size=sample_size)

        if "correlation_heatmap" in plots:
            self.plot_correlation_heatmap(data[numeric_cols], 
                                     output_path=f"{data_output_path}/{prefix}_correlation_heatmap.png", 
                                     figsize=figsize, 
                                     cmap=cmap, 
                                     annot=annot, 
                                     fmt=fmt,
                                     sample_size=sample_size)
        if "histogram" in plots:
            self.plot_all_histogram(data=data, 
                                 numeric_columns=numeric_cols,
                               output_path=f"{data_output_path}/{prefix}_histogram_plots.png", 
                               bins=30, 
                               figsize=figsize,
                               sample_size=sample_size)
            
        if "violin" in plots:
            self.plot_all_violins(data=data,
                                numeric_columns=numeric_cols, 
                              output_path=f"{data_output_path}/{prefix}_violin_plots.png", 
                              figsize=figsize,
                              sample_size=sample_size)
            
        if "scatter" in plots:
            self.plot_all_scatters(data, 
                                 numeric_columns=numeric_cols,
                               output_path=f"{data_output_path}/{prefix}_scatter_plots.png", 
                               sample_size=sample_size, 
                               dpi=dpi, 
                               figsize=figsize)
            
        if "all_distributions" in plots:
            self.plot_distribution_plots(data, 
                                        numeric_columns=numeric_cols,
                                        output_path=f"{data_output_path}/{prefix}_distribution_plots.png", 
                                        bins=30, 
                                        figsize=figsize,
                                        sample_size=sample_size)
            
        if "skewness" in plots:
            self.analyze_skewness(data, 
                                numeric_columns=numeric_cols,
                                output_path=f"{data_output_path}/{prefix}_skewness_values.csv",
                                sample_size=sample_size)
            
        if "kurtosis" in plots:
            self.analyze_kurtosis(data, 
                              numeric_columns=numeric_cols,
                              output_path=f"{data_output_path}/{prefix}_kurtosis_values.csv",
                              sample_size=sample_size)
            
        if "feature_importance" in plots:
            if mode == "matches":
                target_column = "team1_win"
            else:
                target_column = "win_rate"
            self.analyze_feature_importance(data=data, 
                                        target_column=target_column,
                                        numeric_columns=numeric_cols, 
                                        output_path=f"{data_output_path}/{prefix}_feature_importance.png", 
                                        n_estimators=n_estimators, 
                                        n_repeats=n_repeats,
                                        sample_size=sample_size,
                                        mode=mode)

    def create_summary_statistics_to_csv(self, data: pd.DataFrame, output_path: str, sample_size: int = 10000) -> None:
        """
        Generate and save summary statistics of the dataframe.

        Args:
            data (pd.DataFrame): Input dataframe.
            output_path (str): Path to save the summary statistics.
            sample_size (int): Maximum number of samples to use. Defaults to 10000.
        """
        sampled_data = data.sample(n=min(sample_size, len(data)), random_state=self.random_state)
        summary = sampled_data.describe(include='all')
        summary.to_csv(output_path)
        self.logger.info(f"Summary statistics saved to {output_path}")

    def plot_correlation_heatmap(self, data: pd.DataFrame, output_path: str, figsize: tuple = (10, 8), cmap: str = "coolwarm", annot: bool = True, fmt: str = ".2f", sample_size: int = 10000) -> None:
        """
        Generate and save a correlation heatmap.

        Args:
            data (pd.DataFrame): Input dataframe.
            output_path (str): Path to save the heatmap image.
            figsize (tuple): Figure size (width, height). Defaults to (10, 8).
            cmap (str): Colormap for the heatmap. Defaults to "coolwarm".
            annot (bool): If True, write the data value in each cell. Defaults to True.
            fmt (str): String formatting code to use when annot is True. Defaults to ".2f".
            sample_size (int): Maximum number of samples to use. Defaults to 10000.
        """
        sampled_data = data.sample(n=min(sample_size, len(data)), random_state=self.random_state)
        correlation_matrix = sampled_data.corr()
        
        # hide the upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        plt.figure(figsize=figsize)

        ax = sns.heatmap(correlation_matrix,
                            center=0,
                            annot=annot,
                            cmap=cmap,
                            fmt=fmt,
                            mask=mask,
                            xticklabels=correlation_matrix.columns,
                            yticklabels=correlation_matrix.columns,
                            annot_kws={"size": 10},
                            linewidths=0.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.title("Correlation Heatmap", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        self.logger.info(f"Correlation heatmap saved to {output_path}")
    

    def plot_all_histogram(self, data: pd.DataFrame, numeric_columns: list[str], output_path: str, bins: int = 30, figsize: tuple = (8, 6), sample_size: int = 10000) -> None:
        """
        Generate and save histograms for all numeric columns.

        Args:
            data (pd.DataFrame): Input dataframe.
            numeric_columns (list[str]): List of numeric column names.
            output_path (str): Path to save the histogram image.
            bins (int): Number of bins for the histogram. Defaults to 30.
            figsize (tuple): Figure size (width, height). Defaults to (8, 6).
            sample_size (int): Maximum number of samples to use. Defaults to 10000.
        """
        sampled_data = data.sample(n=min(sample_size, len(data)), random_state=self.random_state)
        fig, axes = plt.subplots(figsize=figsize)
        sampled_data[numeric_columns].hist(bins=bins, ax=axes)
        plt.suptitle("Histograms for Numeric Columns")
        plt.tight_layout()
        plt.savefig(output_path)
        self.logger.info(f"Histogram plots saved to {output_path}")


    def plot_all_violins(self, data: pd.DataFrame, numeric_columns: list[str], output_path: str, figsize: tuple = (10, 8), sample_size: int = 10000) -> None:
        """
        Generate and save violin plots for all numeric columns.

        Args:
            data (pd.DataFrame): Input dataframe.
            numeric_columns (list[str]): List of numeric column names.
            output_path (str): Path to save the violin plot image.
            figsize (tuple): Figure size (width, height). Defaults to (10, 8).
            sample_size (int): Maximum number of samples to use. Defaults to 10000.
        """
        sampled_data = data.sample(n=min(sample_size, len(data)), random_state=self.random_state)
        fig, axes = plt.subplots(figsize=figsize)

        axes.set_xticks(range(len(numeric_columns)))
        axes.set_xticklabels(numeric_columns, rotation=45, ha='right')

        sns.violinplot(data=sampled_data[numeric_columns], ax=axes, inner="quartile")
        axes.set_title("Violin Plots for Numeric Columns")
        axes.set_ylabel("Value")
        axes.set_xlabel("Feature")
        plt.tight_layout()
        plt.savefig(output_path)

        self.logger.info(f"Violin plots saved to {output_path}")

    def plot_all_scatters(self, data: pd.DataFrame, numeric_columns: list[str], output_path: str, sample_size: int = 1000, dpi: int = 100, figsize: tuple = (8, 6)) -> None:
        """
        Generate and save scatter plots for all numeric columns against each other.

        Args:
            data (pd.DataFrame): Input dataframe.
            numeric_columns (list[str]): List of numeric column names.
            output_path (str): Path to save the scatter plot image.
            sample_size (int): Maximum number of samples to use. Defaults to 1000.
            dpi (int): Dots per inch for the saved image. Defaults to 100.
            figsize (tuple): Figure size (width, height). Defaults to (8, 6).
        """

        sampled_data = data[numeric_columns].sample(n=min(sample_size, len(data)), random_state=self.random_state)

        num_cols = len(numeric_columns)
        fig, axes = plt.subplots(num_cols, num_cols, figsize=(figsize[0]*num_cols, figsize[1]*num_cols), dpi=dpi)

        for i, col_x in enumerate(numeric_columns):
            for j, col_y in enumerate(numeric_columns):
                ax = axes[i, j]
                if i == j:
                    ax.hist(sampled_data[col_x].dropna(), bins=30, color='blue', alpha=0.7)
                    ax.set_title(f'Histogram of {col_x}')
                else:
                    ax.scatter(sampled_data[col_x], sampled_data[col_y], alpha=0.5)
                    ax.set_title(f'Scatter: {col_x} vs {col_y}')
                if i == num_cols - 1:
                    ax.set_xlabel(col_x)
                if j == 0:
                    ax.set_ylabel(col_y)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        self.logger.info(f"Scatter plots saved to {output_path}")


    def plot_distribution_plots(self, data: pd.DataFrame, numeric_columns: list[str], output_path: str, bins: int = 30, figsize: tuple = (8, 6), sample_size: int = 10000) -> None:
        """
        Generate and save distribution plots for all numeric columns.

        Args:
            data (pd.DataFrame): Input dataframe.
            numeric_columns (list[str]): List of numeric column names.
            output_path (str): Path to save the distribution plot image.
            bins (int): Number of bins for the histogram. Defaults to 30.
            figsize (tuple): Figure size (width, height). Defaults to (8, 6).
            sample_size (int): Maximum number of samples to use. Defaults to 10000.
        """
        sampled_data = data.sample(n=min(sample_size, len(data)), random_state=self.random_state)
        for column in numeric_columns:
            plt.figure(figsize=figsize)
            sns.histplot(sampled_data[column].dropna(), bins=bins, kde=True, color='blue', stat="density", alpha=0.7)
            plt.title(f'Distribution Plot of {column}')
            plt.xlabel(column)
            plt.ylabel('Density')
            plt.tight_layout()
            column_output_path = output_path.replace(".png", f"_{column}.png")
            plt.savefig(column_output_path)
            plt.close()
            self.logger.info(f"Distribution plot for {column} saved to {column_output_path}")


    def analyze_skewness(self, data: pd.DataFrame, numeric_columns: list[str], output_path: str, sample_size: int = 10000) -> None:
        """
        Analyze and save skewness values for numeric columns.

        Args:
            data (pd.DataFrame): Input dataframe.
            numeric_columns (list[str]): List of numeric column names.
            output_path (str): Path to save the skewness values.
            sample_size (int): Maximum number of samples to use. Defaults to 10000.
        """
        sampled_data = data.sample(n=min(sample_size, len(data)), random_state=self.random_state)
        skewness_values = sampled_data[numeric_columns].skew()
        skewness_values.to_csv(output_path, header=['Skewness'])
        self.logger.info(f"Skewness values saved to {output_path}")

    def analyze_kurtosis(self, data: pd.DataFrame, numeric_columns: list[str], output_path: str, sample_size: int = 10000) -> None:
        """
        Analyze and save kurtosis values for numeric columns.

        Args:
            data (pd.DataFrame): Input dataframe.
            numeric_columns (list[str]): List of numeric column names.
            output_path (str): Path to save the kurtosis values.
            sample_size (int): Maximum number of samples to use. Defaults to 10000.
        """
        sampled_data = data.sample(n=min(sample_size, len(data)), random_state=self.random_state)
        kurtosis_values = sampled_data[numeric_columns].kurtosis()
        kurtosis_values.to_csv(output_path, header=['Kurtosis'])
        self.logger.info(f"Kurtosis values saved to {output_path}")


    def analyze_feature_importance(self, data: pd.DataFrame, target_column: str, numeric_columns: list[str], output_path: str, n_estimators: int = 100, n_repeats: int = 5, sample_size: int = 10000, mode: str = "matches") -> None:
        """
        Analyze and save feature importance using permutation importance.

        Args:
            data (pd.DataFrame): Input dataframe.
            target_column (str): Target column name.
            numeric_columns (list[str]): List of numeric column names.
            output_path (str): Path to save the feature importance plot.
            n_estimators (int): Number of trees for RandomForestRegressor. Defaults to 100.
            n_repeats (int): Number of times to permute a feature. Defaults to 5.
            sample_size (int): Maximum number of samples to use. Defaults to 10000.
        """
        sampled_data = data.sample(n=min(sample_size, len(data)), random_state=self.random_state)

        X = sampled_data.drop(columns=[target_column])[numeric_columns]
        y = sampled_data[target_column]

        if mode == "players":
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=self.random_state)
        else:
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=self.random_state, class_weight='balanced')
        
        model.fit(X, y)

        result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=self.random_state)

        sorted_idx = result.importances_mean.argsort()

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
        plt.xlabel("Permutation Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        self.logger.info(f"Feature importance plot saved to {output_path}")



        