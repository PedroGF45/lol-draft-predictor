from logging import Logger

import numpy as np
import pandas as pd
from data_preparation.data_handler import DataHandler
from helpers.parquet_handler import ParquetHandler
from sklearn.preprocessing import QuantileTransformer, StandardScaler


class FeatureEngineer:
    def __init__(self, logger: Logger, parquet_handler: ParquetHandler, random_state: int = 42):
        self.logger = logger
        self.parquet_handler = parquet_handler
        self.random_state = random_state

        # Store fitted transformers for consistent train/test transformation
        self.quantile_transformer = None
        self.standard_scaler = None

    def perform_feature_engineering(
        self,
        data_handler: DataHandler,
        output_dir: str | None = None,
    ):
        """
        Perform complete feature engineering pipeline on train/test data.

        Args:
            data_handler (DataHandler): DataHandler instance containing train/test splits.
            output_dir (str | None): Optional directory to save engineered train/test/labels parquet files.
        """
        self.logger.info("Starting feature engineering pipeline")

        # Get train and test data
        data_train = data_handler.get_data_train().copy()
        data_test = data_handler.get_data_test().copy()

        self.logger.info(f"Initial train shape: {data_train.shape}, test shape: {data_test.shape}")

        # Step 1: Generate new features
        data_train = self.generate_new_features(data_train)
        data_test = self.generate_new_features(data_test)

        # Update DataHandler with new features
        data_handler.set_data_train(data_train)
        data_handler.set_data_test(data_test)

        # Step 2: Update feature types after new features are added
        self._update_feature_types(data_handler)

        # Step 2.1: Remove non-predictive identifier columns
        # Drop columns that should not be used as features (identifiers or meta)
        drop_cols = [col for col in ["game_version", "match_id"] if col in data_train.columns]
        if drop_cols:
            self.logger.info(f"Dropping non-predictive columns: {drop_cols}")
            data_train = data_train.drop(columns=drop_cols, errors="ignore")
            data_test = data_test.drop(columns=drop_cols, errors="ignore")
            data_handler.set_data_train(data_train)
            data_handler.set_data_test(data_test)
            # Update feature types post-drop
            self._update_feature_types(data_handler)

        # Step 3: One-hot encode categorical features
        data_train, data_test = self.one_hot_encode(data_train, data_test, data_handler)
        data_handler.set_data_train(data_train)
        data_handler.set_data_test(data_test)

        # Step 4: Update feature types after one-hot encoding
        self._update_feature_types(data_handler)

        # Step 5: Transform numerical features with QuantileTransformer
        data_train, data_test = self.transform_numerical_features(data_train, data_test, data_handler)
        data_handler.set_data_train(data_train)
        data_handler.set_data_test(data_test)

        # Step 6: Normalize features with StandardScaler
        data_train, data_test = self.normalize_features(data_train, data_test, data_handler)
        data_handler.set_data_train(data_train)
        data_handler.set_data_test(data_test)

        # Optionally persist engineered datasets using DataHandler's save method
        if output_dir:
            saved_dir = data_handler.save_splits(output_dir)
            self.logger.info(f"Engineered data saved via DataHandler to {saved_dir}")

        self.logger.info(
            f"Feature engineering completed. Final train shape: {data_train.shape}, test shape: {data_test.shape}"
        )

    def generate_new_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Generate new features based on domain knowledge of League of Legends.

        Creates features such as:
        - Rank score differences between teams
        - Team composition metrics
        - Ban priority features

        Args:
            dataframe (pd.DataFrame): Input dataframe with raw features.

        Returns:
            pd.DataFrame: Dataframe with additional engineered features.
        """
        self.logger.info("Generating new features based on domain knowledge")
        df = dataframe.copy()

        # ===== Rank Score Conversion =====
        # Convert ranked tier and rank to numerical score
        tier_scores = {
            "IRON": 0,
            "BRONZE": 4,
            "SILVER": 8,
            "GOLD": 12,
            "PLATINUM": 16,
            "EMERALD": 20,
            "DIAMOND": 24,
            "MASTER": 28,
            "GRANDMASTER": 29,
            "CHALLENGER": 30,
        }
        rank_scores = {"IV": 0, "III": 1, "II": 2, "I": 3}

        roles = ["top", "jungle", "mid", "adc", "support"]
        teams = [1, 2]

        # Calculate rank score for each player
        for team in teams:
            for role in roles:
                tier_col = f"team{team}_{role}_ranked_tier"
                rank_col = f"team{team}_{role}_ranked_rank"
                score_col = f"team{team}_{role}_rank_score"

                if tier_col in df.columns and rank_col in df.columns:
                    df[score_col] = df[tier_col].map(tier_scores).fillna(0) + df[rank_col].map(rank_scores).fillna(0)

        # ===== Team Rank Score Statistics =====
        # Calculate team-level rank statistics
        for team in teams:
            rank_score_cols = [
                f"team{team}_{role}_rank_score" for role in roles if f"team{team}_{role}_rank_score" in df.columns
            ]

            if rank_score_cols:
                df[f"team{team}_avg_rank_score"] = df[rank_score_cols].mean(axis=1)
                df[f"team{team}_min_rank_score"] = df[rank_score_cols].min(axis=1)
                df[f"team{team}_max_rank_score"] = df[rank_score_cols].max(axis=1)
                df[f"team{team}_rank_score_std"] = df[rank_score_cols].std(axis=1).fillna(0)

        # ===== Rank Difference Features =====
        if "team1_avg_rank_score" in df.columns and "team2_avg_rank_score" in df.columns:
            df["rank_score_diff"] = df["team1_avg_rank_score"] - df["team2_avg_rank_score"]

        # ===== Win Rate Difference Features =====
        # Calculate win rate differences if available
        if "team1_avg_wins" in df.columns and "team2_avg_wins" in df.columns:
            if "team1_avg_losses" in df.columns and "team2_avg_losses" in df.columns:
                team1_total = df["team1_avg_wins"] + df["team1_avg_losses"]
                team2_total = df["team2_avg_wins"] + df["team2_avg_losses"]

                df["team1_win_rate"] = df["team1_avg_wins"] / team1_total.replace(0, 1)
                df["team2_win_rate"] = df["team2_avg_wins"] / team2_total.replace(0, 1)
                df["win_rate_diff"] = df["team1_win_rate"] - df["team2_win_rate"]

        # ===== KDA Features =====
        # Calculate KDA if kill/death/assist data available
        for team in teams:
            if all(f"team{team}_avg_{stat}" in df.columns for stat in ["kills", "deaths", "assists"]):
                df[f"team{team}_kda"] = (df[f"team{team}_avg_kills"] + df[f"team{team}_avg_assists"]) / df[
                    f"team{team}_avg_deaths"
                ].replace(0, 1)

        if "team1_kda" in df.columns and "team2_kda" in df.columns:
            df["kda_diff"] = df["team1_kda"] - df["team2_kda"]

        # ===== Champion Pick Order Features =====
        # Count number of bans per team (some bans might be -1 for no ban)
        for team in teams:
            ban_cols = [f"team{team}_ban{i}" for i in range(1, 6) if f"team{team}_ban{i}" in df.columns]
            if ban_cols:
                # Count non-negative bans (assuming -1 means no ban)
                df[f"team{team}_total_bans"] = (df[ban_cols] >= 0).sum(axis=1)

        # ===== Role Diversity Features =====
        # Check if team has unique champions (no duplicates due to data errors)
        for team in teams:
            pick_cols = [f"team{team}_pick_{role}" for role in roles if f"team{team}_pick_{role}" in df.columns]
            if pick_cols:
                df[f"team{team}_unique_picks"] = df[pick_cols].nunique(axis=1)

        # ===== Team Aggregate Difference Features =====
        # Create robust difference features for known aggregates when available
        # This leverages per-player historical KPIs aggregated at team level
        metric_pairs = [
            "avg_wins",
            "avg_losses",
            "avg_kills",
            "avg_deaths",
            "avg_assists",
            "avg_cs",
            "avg_gold_earned",
            "avg_damage_dealt",
            "avg_damage_taken",
            "avg_vision_score",
            "avg_healing_done",
            "cs_per_minute",
            "gold_per_minute",
            "damage_per_minute",
            "vision_score_per_minute",
            "healing_per_minute",
            "win_rate",
            "ranked_league_points",
            "summoner_level",
            "champion_mastery_level",
            "champion_total_mastery_score",
        ]

        created_diffs = 0
        for m in metric_pairs:
            c1 = f"team1_{m}"
            c2 = f"team2_{m}"
            out = f"{m}_diff"
            if c1 in df.columns and c2 in df.columns and out not in df.columns:
                # Replace division-by-zero style issues by safe replacement if needed later
                try:
                    df[out] = df[c1] - df[c2]
                    created_diffs += 1
                except Exception:
                    pass

        if created_diffs:
            self.logger.info(f"Added {created_diffs} aggregate team difference features")

        self.logger.info(f"Generated {df.shape[1] - dataframe.shape[1]} new features")
        return df

    def _update_feature_types(self, data_handler: DataHandler) -> None:
        """
        Update categorical and numerical feature sets in DataHandler.

        Args:
            data_handler (DataHandler): DataHandler to update.
        """
        data_train = data_handler.get_data_train()

        categorical_cols = set(data_train.select_dtypes(include=["object", "category", "bool"]).columns)
        numerical_cols = set(data_train.select_dtypes(include=["number"], exclude=["bool"]).columns)

        data_handler.set_categorical_features(categorical_cols)
        data_handler.set_numerical_features(numerical_cols)

        self.logger.info(f"Updated feature types: {len(categorical_cols)} categorical, {len(numerical_cols)} numerical")

    def one_hot_encode(
        self, data_train: pd.DataFrame, data_test: pd.DataFrame, data_handler: DataHandler
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply one-hot encoding to categorical features.

        Args:
            data_train (pd.DataFrame): Training data.
            data_test (pd.DataFrame): Test data.
            data_handler (DataHandler): DataHandler to get categorical features.

        Returns:
            tuple: (encoded_train, encoded_test)
        """
        categorical_features = data_handler.get_categorical_features()

        if not categorical_features:
            self.logger.info("No categorical features to encode")
            return data_train, data_test

        self.logger.info(f"One-hot encoding {len(categorical_features)} categorical features")

        # Get only existing categorical columns
        existing_categorical = [col for col in categorical_features if col in data_train.columns]

        if not existing_categorical:
            self.logger.info("No existing categorical features found in dataframe")
            return data_train, data_test

        # Perform one-hot encoding
        train_encoded = pd.get_dummies(data_train, columns=existing_categorical, drop_first=True, dtype=int)
        test_encoded = pd.get_dummies(data_test, columns=existing_categorical, drop_first=True, dtype=int)

        # Align columns between train and test (test might have missing categories)
        train_encoded, test_encoded = train_encoded.align(test_encoded, join="left", axis=1, fill_value=0)

        self.logger.info(
            f"One-hot encoding completed. New shape: train={train_encoded.shape}, test={test_encoded.shape}"
        )
        return train_encoded, test_encoded

    def transform_numerical_features(
        self, data_train: pd.DataFrame, data_test: pd.DataFrame, data_handler: DataHandler
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform numerical features using QuantileTransformer to handle outliers and skewed distributions.

        Args:
            data_train (pd.DataFrame): Training data.
            data_test (pd.DataFrame): Test data.
            data_handler (DataHandler): DataHandler to get numerical features.

        Returns:
            tuple: (transformed_train, transformed_test)
        """
        numerical_features = data_handler.get_numerical_features()

        if not numerical_features:
            self.logger.info("No numerical features to transform")
            return data_train, data_test

        # Get only existing numerical columns
        existing_numerical = [col for col in numerical_features if col in data_train.columns]

        if not existing_numerical:
            self.logger.info("No existing numerical features found in dataframe")
            return data_train, data_test

        self.logger.info(f"Applying QuantileTransformer to {len(existing_numerical)} numerical features")

        # Initialize and fit transformer on training data only
        self.quantile_transformer = QuantileTransformer(
            n_quantiles=min(1000, len(data_train)), output_distribution="normal", random_state=self.random_state
        )

        # Transform train and test data
        train_transformed = data_train.copy()
        test_transformed = data_test.copy()

        train_transformed[existing_numerical] = self.quantile_transformer.fit_transform(data_train[existing_numerical])
        test_transformed[existing_numerical] = self.quantile_transformer.transform(data_test[existing_numerical])

        self.logger.info("QuantileTransformer applied successfully")
        return train_transformed, test_transformed

    def normalize_features(
        self, data_train: pd.DataFrame, data_test: pd.DataFrame, data_handler: DataHandler
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize numerical features using StandardScaler (z-score normalization).

        Args:
            data_train (pd.DataFrame): Training data.
            data_test (pd.DataFrame): Test data.
            data_handler (DataHandler): DataHandler to get numerical features.

        Returns:
            tuple: (normalized_train, normalized_test)
        """
        numerical_features = data_handler.get_numerical_features()

        if not numerical_features:
            self.logger.info("No numerical features to normalize")
            return data_train, data_test

        # Get only existing numerical columns
        existing_numerical = [col for col in numerical_features if col in data_train.columns]

        if not existing_numerical:
            self.logger.info("No existing numerical features found in dataframe")
            return data_train, data_test

        self.logger.info(f"Applying StandardScaler to {len(existing_numerical)} numerical features")

        # Initialize and fit scaler on training data only
        self.standard_scaler = StandardScaler()

        # Normalize train and test data
        train_normalized = data_train.copy()
        test_normalized = data_test.copy()

        train_normalized[existing_numerical] = self.standard_scaler.fit_transform(data_train[existing_numerical])
        test_normalized[existing_numerical] = self.standard_scaler.transform(data_test[existing_numerical])

        self.logger.info("StandardScaler applied successfully")
        return train_normalized, test_normalized
