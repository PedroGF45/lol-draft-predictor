from data_extraction.requester import Requester
from data_preparation.data_handler import DataHandler
from helpers.parquet_handler import ParquetHandler
from helpers.champion_ids import fetch_latest_champion_ids

from logging import Logger
import pandas as pd
import os


class DataCleaner:
    """
    Cleans raw match or player parquet data by removing duplicates, handling missing values,
    and filtering out-of-range records using domain-specific rules.

    Attributes:
        requester (Requester): HTTP client (unused in cleaning but kept for future enrichment hooks).
        logger (Logger): Logger for status and error messages.
        parquet_handler (ParquetHandler): Helper to read parquet inputs.
        load_percentage (float): Fraction of rows to load from parquet.
        random_state (int): Seed for deterministic sampling.
    """

    def __init__(
        self,
        requester: Requester,
        logger: Logger,
        data_handler: DataHandler,
        parquet_handler: ParquetHandler,
        load_percentage: float = 1.0,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the cleaner with IO helpers and sampling options.

        Args:
            requester (Requester): API client (not currently used during cleaning).
            logger (Logger): Logger instance for progress/error reporting.
            parquet_handler (ParquetHandler): Parquet IO helper.
            load_percentage (float): Fraction of parquet rows to load. Defaults to 1.0.
            random_state (int): Seed used when sampling rows. Defaults to 42.
        """
        self.requester = requester
        self.logger = logger
        self.data_handler = data_handler
        self.parquet_handler = parquet_handler
        self.load_percentage = load_percentage
        self.random_state = random_state

    def clean_data(self, output_path: str = None) -> None:
        """
        Run the full cleaning pipeline for match or player data.

        Args:
            output_path (str, optional): Directory to save cleaned data. If None, data is not saved.
        """

        self.logger.info("Starting data cleaning")

        # validate data integrity before processing
        try:
            self.data_handler.validate_data()
            self.logger.info("Pre-cleaning validation passed")
        except ValueError as e:
            self.logger.error(f"Pre-cleaning validation failed: {e}")
            raise

        # remove duplicates
        self._remove_duplicates()

        # identify feature types for downstream processing
        self._update_feature_types()

        # handle missing values
        self._handle_missing_values()

        # handle out of range values
        self._handle_out_of_range_values()

        # refresh feature type sets after all cleaning steps
        self._update_feature_types()

        # log feature summary
        feature_summary = self.data_handler.get_feature_summary()
        self.logger.info(f"Cleaning completed. Feature summary: {feature_summary}")

        # save cleaned data if output path is provided
        if output_path:
            saved_dir = self.data_handler.save_splits(output_path)
            self.logger.info(f"Cleaned data saved to {saved_dir}")

        self.logger.info("Data cleaning completed.")

    def _remove_unwanted_columns(self, columns_to_remove: list) -> None:

        try:
            self.logger.info("Removing unwanted columns.")
            data_train = self.data_handler.get_data_train()
            data_train_dropped = data_train.drop(columns=columns_to_remove, errors="ignore")
            self.data_handler.set_data_train(data_train_dropped)

        except Exception as e:
            self.logger.error(f"Error removing unwanted columns: {e}")
            raise e

    def _remove_duplicates(self) -> None:
        """
        Drop duplicate rows and log how many were removed.

        Args:
            data (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe without duplicate rows.
        """
        self.logger.info("Removing duplicate records.")
        initial_count = len(self.data_handler.get_data_train())

        data_train_without_duplicates = self.data_handler.get_data_train().drop_duplicates()
        self.data_handler.set_data_train(data_train_without_duplicates)

        labels_train_without_duplicates = self.data_handler.get_labels_train().loc[data_train_without_duplicates.index]
        self.data_handler.set_labels_train(labels_train_without_duplicates)

        final_count = len(data_train_without_duplicates)
        self.logger.info(f"Removed {initial_count - final_count} duplicate records.")

    def _handle_missing_values(self, strategy: str = "mean") -> None:
        """
        Handle missing values using the specified strategy.

        Args:
            strategy (str): Strategy for handling missing values.
                - 'drop': Remove rows with any missing values (NOT RECOMMENDED - will delete all data)
                - 'mean': Fill numerical with mean, categorical with mode (DEFAULT)
                - 'median': Fill numerical with median, categorical with mode
                - 'mode': Fill all columns with mode

        Raises:
            ValueError: If strategy is not recognized.
        """
        valid_strategies = {"drop", "mean", "median", "mode"}
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")

        self.logger.info(f"Handling missing values using '{strategy}' strategy.")

        if (
            self.data_handler.get_data_train().isnull().sum().sum() == 0
            and self.data_handler.get_data_test().isnull().sum().sum() == 0
        ):
            self.logger.info("No missing values found.")
            return

        self.logger.info(
            f"There are {self.data_handler.get_data_train().isnull().sum().sum()} missing values in training data and {self.data_handler.get_data_test().isnull().sum().sum()} missing values in testing data."
        )

        # ===== Custom imputation for ranked features BEFORE general strategy =====
        data_train = self.data_handler.get_data_train().copy()
        data_test = self.data_handler.get_data_test().copy()

        # Fill ranked_tier with "UNRANKED" for players without ranked data
        ranked_tier_cols = [col for col in data_train.columns if "ranked_tier" in col]
        if ranked_tier_cols:
            for col in ranked_tier_cols:
                train_missing = data_train[col].isnull().sum()
                test_missing = data_test[col].isnull().sum()
                if train_missing > 0 or test_missing > 0:
                    data_train[col] = data_train[col].fillna("UNRANKED")
                    data_test[col] = data_test[col].fillna("UNRANKED")
            self.logger.info(f"Filled {len(ranked_tier_cols)} ranked_tier columns with 'UNRANKED'")

        # Fill ranked_rank with "IV" (lowest division) for consistency
        ranked_rank_cols = [col for col in data_train.columns if "ranked_rank" in col]
        if ranked_rank_cols:
            for col in ranked_rank_cols:
                train_missing = data_train[col].isnull().sum()
                test_missing = data_test[col].isnull().sum()
                if train_missing > 0 or test_missing > 0:
                    data_train[col] = data_train[col].fillna("IV")
                    data_test[col] = data_test[col].fillna("IV")
            self.logger.info(f"Filled {len(ranked_rank_cols)} ranked_rank columns with 'IV'")

        # Update data_handler with imputed ranked features
        self.data_handler.set_data_train(data_train)
        self.data_handler.set_data_test(data_test)

        initial_train_count = len(self.data_handler.get_data_train())
        initial_test_count = len(self.data_handler.get_data_test())

        if strategy == "drop":
            data_train_without_missing = self.data_handler.get_data_train().dropna()
            self.data_handler.set_data_train(data_train_without_missing)

            data_test_without_missing = self.data_handler.get_data_test().dropna()
            self.data_handler.set_data_test(data_test_without_missing)

            labels_train_without_missing = self.data_handler.get_labels_train().loc[data_train_without_missing.index]
            self.data_handler.set_labels_train(labels_train_without_missing)

            labels_test_without_missing = self.data_handler.get_labels_test().loc[data_test_without_missing.index]
            self.data_handler.set_labels_test(labels_test_without_missing)

            final_train_count = len(data_train_without_missing)
            final_test_count = len(data_test_without_missing)
            self.logger.info(
                f"Dropped {initial_train_count - final_train_count} training and {initial_test_count - final_test_count} testing records with missing values."
            )

        elif strategy == "mean":
            data_train = self.data_handler.get_data_train().copy()
            data_test = self.data_handler.get_data_test().copy()

            for column in self.data_handler.get_numerical_features():
                mean_value = data_train[column].mean()
                data_train[column] = data_train[column].fillna(mean_value)
                data_test[column] = data_test[column].fillna(mean_value)

            for column in self.data_handler.get_categorical_features():
                if column in data_train.columns and not data_train[column].mode().empty:
                    mode_value = data_train[column].mode()[0]
                    data_train[column] = data_train[column].fillna(mode_value)
                    data_test[column] = data_test[column].fillna(mode_value)

            self.data_handler.set_data_train(data_train)
            self.data_handler.set_data_test(data_test)
            self.logger.info("Filled missing values: numerical with mean, categorical with mode.")

        elif strategy == "median":
            data_train = self.data_handler.get_data_train().copy()
            data_test = self.data_handler.get_data_test().copy()

            for column in self.data_handler.get_numerical_features():
                median_value = data_train[column].median()
                data_train[column] = data_train[column].fillna(median_value)
                data_test[column] = data_test[column].fillna(median_value)

            for column in self.data_handler.get_categorical_features():
                if column in data_train.columns and not data_train[column].mode().empty:
                    mode_value = data_train[column].mode()[0]
                    data_train[column] = data_train[column].fillna(mode_value)
                    data_test[column] = data_test[column].fillna(mode_value)

            self.data_handler.set_data_train(data_train)
            self.data_handler.set_data_test(data_test)
            self.logger.info("Filled missing values: numerical with median, categorical with mode.")

        elif strategy == "mode":
            data_train = self.data_handler.get_data_train().copy()
            data_test = self.data_handler.get_data_test().copy()

            for column in data_train.columns:
                if not data_train[column].mode().empty:
                    mode_value = data_train[column].mode()[0]
                    data_train[column] = data_train[column].fillna(mode_value)
                    data_test[column] = data_test[column].fillna(mode_value)

            self.data_handler.set_data_train(data_train)
            self.data_handler.set_data_test(data_test)
            self.logger.info("Filled missing values: all columns with mode.")

    def _update_feature_types(self) -> None:
        """
        Derive categorical and numerical feature sets from the training dataframe.
        Automatically detects feature types based on pandas dtypes.
        """

        data_train = self.data_handler.get_data_train()
        if data_train is None or data_train.empty:
            self.logger.warning("Cannot update feature types: training data is empty.")
            return

        categorical_cols = set(data_train.select_dtypes(include=["object", "category", "bool"]).columns)
        numerical_cols = set(data_train.select_dtypes(include=["number"], exclude=["bool"]).columns)

        self.data_handler.set_categorical_features(categorical_cols)
        self.data_handler.set_numerical_features(numerical_cols)

        self.logger.info(
            f"Feature types updated: {len(categorical_cols)} categorical, {len(numerical_cols)} numerical."
        )

    def _handle_out_of_range_values(self) -> None:
        """
        Validate and filter records in the combined match+player dataframe.

        This method handles the joined dataset structure where:
        - Match metadata: queue_id, game_duration, team1_win
        - Draft features: bans and picks (champion IDs)
        - Individual player features: champion_id, ranked_tier, ranked_rank per role
        - Aggregated team features: team stats (all should be non-negative)
        """
        self.logger.info("Validating combined match-player data for out-of-range values.")

        data_train = self.data_handler.get_data_train()
        data_test = self.data_handler.get_data_test()
        initial_train_count = len(data_train)
        initial_test_count = len(data_test)

        valid_train = data_train.copy()
        valid_test = data_test.copy()

        # ===== Match Metadata Validation =====
        if "queue_id" in valid_train.columns:
            valid_queue_ids = {400, 420, 430, 440, 450}
            valid_train = valid_train[valid_train["queue_id"].isin(valid_queue_ids)]
            valid_test = valid_test[valid_test["queue_id"].isin(valid_queue_ids)]
            self.logger.info("Validated queue_id")

        if "game_duration" in valid_train.columns:
            valid_train = valid_train[(valid_train["game_duration"] > 0) & (valid_train["game_duration"] < 7200)]
            valid_test = valid_test[(valid_test["game_duration"] > 0) & (valid_test["game_duration"] < 7200)]
            self.logger.info("Validated game_duration (0 < duration < 7200)")

        # ===== Draft Features Validation =====
        valid_champion_ids = fetch_latest_champion_ids()

        # Validate ban champion IDs
        ban_columns = [f"team{team}_ban{ban}" for team in [1, 2] for ban in range(1, 6)]
        for col in ban_columns:
            if col in valid_train.columns:
                valid_train = valid_train[valid_train[col].isin(valid_champion_ids)]
                valid_test = valid_test[valid_test[col].isin(valid_champion_ids)]
        self.logger.info(f"Validated {len([c for c in ban_columns if c in valid_train.columns])} ban columns")

        # Validate pick champion IDs
        pick_columns = [
            f"team{team}_pick_{role}" for team in [1, 2] for role in ["top", "jungle", "mid", "adc", "support"]
        ]
        for col in pick_columns:
            if col in valid_train.columns:
                valid_train = valid_train[valid_train[col].isin(valid_champion_ids)]
                valid_test = valid_test[valid_test[col].isin(valid_champion_ids)]
        self.logger.info(f"Validated {len([c for c in pick_columns if c in valid_train.columns])} pick columns")

        # ===== Individual Player Features Validation =====
        roles = ["top", "jungle", "mid", "adc", "support"]
        teams = [1, 2]

        # Validate individual champion_id per role
        for team in teams:
            for role in roles:
                col = f"team{team}_{role}_champion_id"
                if col in valid_train.columns:
                    valid_train = valid_train[valid_train[col].isin(valid_champion_ids)]
                    valid_test = valid_test[valid_test[col].isin(valid_champion_ids)]
        self.logger.info(f"Validated {len(roles) * len(teams)} individual player champion_id columns")

        # Validate ranked_tier per role (no NaN; allow explicit 'UNRANKED')
        valid_tiers = {
            "IRON",
            "BRONZE",
            "SILVER",
            "GOLD",
            "PLATINUM",
            "EMERALD",
            "DIAMOND",
            "MASTER",
            "GRANDMASTER",
            "CHALLENGER",
            "UNRANKED",
        }
        for team in teams:
            for role in roles:
                col = f"team{team}_{role}_ranked_tier"
                if col in valid_train.columns:
                    # Disallow NaN; only valid tiers or explicit 'UNRANKED'
                    mask_train = valid_train[col].isin(valid_tiers)
                    mask_test = valid_test[col].isin(valid_tiers)
                    valid_train = valid_train[mask_train]
                    valid_test = valid_test[mask_test]
        self.logger.info(f"Validated {len(roles) * len(teams)} ranked_tier columns (allowing 'UNRANKED', no NaN)")

        # Validate ranked_rank per role (no NaN)
        valid_ranks = {"I", "II", "III", "IV"}
        for team in teams:
            for role in roles:
                col = f"team{team}_{role}_ranked_rank"
                if col in valid_train.columns:
                    # Disallow NaN; only valid ranks
                    mask_train = valid_train[col].isin(valid_ranks)
                    mask_test = valid_test[col].isin(valid_ranks)
                    valid_train = valid_train[mask_train]
                    valid_test = valid_test[mask_test]
        self.logger.info(f"Validated {len(roles) * len(teams)} ranked_rank columns (no NaN)")

        # ===== Aggregated Team Features Validation =====
        # All aggregated numeric features should be non-negative
        agg_prefixes = ["team1_avg_", "team2_avg_"]
        numeric_agg_cols = [
            col for col in valid_train.columns if any(col.startswith(prefix) for prefix in agg_prefixes)
        ]

        for col in numeric_agg_cols:
            if col in valid_train.columns:
                valid_train = valid_train[valid_train[col] >= 0]
                valid_test = valid_test[valid_test[col] >= 0]

        if numeric_agg_cols:
            self.logger.info(f"Validated {len(numeric_agg_cols)} aggregated team statistic columns (non-negative)")

        # Update data_handler with validated data
        self.data_handler.set_data_train(valid_train)
        self.data_handler.set_data_test(valid_test)

        # Update labels to match filtered indices
        labels_train = self.data_handler.get_labels_train().loc[valid_train.index]
        labels_test = self.data_handler.get_labels_test().loc[valid_test.index]
        self.data_handler.set_labels_train(labels_train)
        self.data_handler.set_labels_test(labels_test)

        final_train_count = len(valid_train)
        final_test_count = len(valid_test)
        train_removed = initial_train_count - final_train_count
        test_removed = initial_test_count - final_test_count

        # Calculate percentages with zero division protection
        train_pct = (train_removed / initial_train_count * 100) if initial_train_count > 0 else 0.0
        test_pct = (test_removed / initial_test_count * 100) if initial_test_count > 0 else 0.0

        self.logger.info(
            f"Removed {train_removed} training ({train_pct:.2f}%) and {test_removed} testing ({test_pct:.2f}%) records with out-of-range values."
        )
