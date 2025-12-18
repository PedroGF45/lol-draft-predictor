from helpers.parquet_handler import ParquetHandler
from logging import Logger
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from datetime import datetime
import glob

class DataHandler:
    def __init__(self, logger: Logger, parquet_handler: ParquetHandler, target_feature: str, test_size: float = 0.2, random_state: int = 42):
        self.logger = logger
        self.parquet_handler = parquet_handler
        self.target_feature = target_feature
        self.test_size = test_size
        self.random_state = random_state

        self.dataframe = None
        self.data_train = None
        self.data_test = None
        self.labels_train = None
        self.labels_test = None

        # Use sets to track feature names
        self.categorical_features = set()
        self.numerical_features = set()

    def get_data_train(self):
        return self.data_train
    
    def set_data_train(self, data_train):
        self.data_train = data_train
    
    def get_data_test(self):
        return self.data_test
    
    def set_data_test(self, data_test):
        self.data_test = data_test
    
    def get_labels_train(self):
        return self.labels_train
    
    def set_labels_train(self, labels_train):
        self.labels_train = labels_train
    
    def get_labels_test(self):
        return self.labels_test
    
    def set_labels_test(self, labels_test):
        self.labels_test = labels_test

    def set_categorical_features(self, categorical_features: set):
        self.categorical_features = categorical_features

    def set_numerical_features(self, numerical_features: set):
        self.numerical_features = numerical_features

    def get_categorical_features(self):
        return self.categorical_features
    
    def get_numerical_features(self):
        return self.numerical_features
    
    def add_categorical_feature(self, feature_name):
        self.categorical_features.add(feature_name)

    def add_numerical_feature(self, feature_name):
        self.numerical_features.add(feature_name)

    def _categorize_features(self) -> dict:
        """
        Automatically categorize features in the dataframe into different types.
        
        Returns:
            dict: Dictionary with feature categories:
                - 'match_meta': Non-feature columns (queue_id, game_version, game_duration)
                - 'draft_bans': Ban features (team1_ban1-5, team2_ban1-5)
                - 'draft_picks': Pick features (team1_pick_*, team2_pick_*)
                - 'player_individual': Individual player features per role (champion_id, ranked_tier, ranked_rank)
                - 'team_aggregated': Aggregated team statistics (avg_*)
                - 'target': Target variable
        """
        if self.dataframe is None:
            raise ValueError("No dataframe loaded. Call load_data or join_match_and_player_data first.")
        
        cols = set(self.dataframe.columns)
        
        categorized = {
            'match_meta': [],
            'draft_bans': [],
            'draft_picks': [],
            'player_individual': [],
            'team_aggregated': [],
            'target': []
        }
        
        for col in cols:
            if col == self.target_feature:
                categorized['target'].append(col)
            elif col in ['queue_id', 'game_version', 'game_duration']:
                categorized['match_meta'].append(col)
            elif '_ban' in col:
                categorized['draft_bans'].append(col)
            elif '_pick_' in col:
                categorized['draft_picks'].append(col)
            elif any(x in col for x in ['_top_', '_jungle_', '_mid_', '_adc_', '_support_']) and \
                 any(x in col for x in ['champion_id', 'ranked_tier', 'ranked_rank']):
                categorized['player_individual'].append(col)
            elif col.startswith(('team1_avg_', 'team2_avg_')):
                categorized['team_aggregated'].append(col)
        
        return categorized

    def get_feature_summary(self) -> dict:
        """
        Get a summary of features in the current dataframe.
        
        Returns:
            dict: Summary statistics about the loaded features.
        """
        if self.dataframe is None:
            raise ValueError("No dataframe loaded.")
        
        categorized = self._categorize_features()
        
        summary = {
            'total_features': len(self.dataframe.columns) - 1,  # Exclude target
            'total_rows': len(self.dataframe),
            'feature_breakdown': {
                'match_metadata': len(categorized['match_meta']),
                'draft_bans': len(categorized['draft_bans']),
                'draft_picks': len(categorized['draft_picks']),
                'individual_player': len(categorized['player_individual']),
                'aggregated_team': len(categorized['team_aggregated'])
            },
            'categorical_count': len(self.categorical_features),
            'numerical_count': len(self.numerical_features),
            'target': self.target_feature
        }
        
        return summary

    def load_data(self, parquet_path: str, load_percentage: float = 1.0):
        """
        Load a single parquet file into the dataframe.
        
        Args:
            parquet_path (str): Path to parquet file.
            load_percentage (float): Fraction of data to load (for testing). Defaults to 1.0.
        """
        self.logger.info(f"Loading data from {parquet_path}")
        self.dataframe = self.parquet_handler.read_parquet(parquet_path, load_percentage)
        self.logger.info(f"Data loaded with shape: {self.dataframe.shape}")
        
        # Log feature summary if we can categorize
        try:
            summary = self.get_feature_summary()
            self.logger.info(f"Feature summary: {summary['total_features']} total features")
            self.logger.info(f"  Breakdown: {summary['feature_breakdown']}")
        except:
            pass  # Skip if categorization not applicable

    def join_match_and_player_data(self, match_parquet_path: str, player_parquet_path: str, 
                                    aggregation_strategy: str = "mean", load_percentage: float = 1.0) -> None:
        """
        Load and join match-level data with player-level data, preserving individual player info + team aggregates.
        
        This method is designed for scenarios where you have:
        - Match data: 1 row per match with draft info (bans, picks) and outcome
        - Player data: 10 rows per match (5 per team) with historical player stats
        
        Creates features at two levels:
        1. Individual player features: champion_id, ranked_tier, ranked_rank per role per team
           (e.g., team1_top_champion_id, team1_jungle_ranked_tier, team2_mid_ranked_rank)
        2. Aggregated team features: mean/median/sum of numeric stats per team
           (e.g., team1_avg_kda, team2_avg_win_rate)
        
        Args:
            match_parquet_path (str): Path to match-level parquet file.
            player_parquet_path (str): Path to player-level parquet file.
            aggregation_strategy (str): How to aggregate player stats per team. 
                                       Options: 'mean', 'median', 'sum', 'min', 'max'.
                                       Defaults to 'mean'.
            load_percentage (float): Fraction of data to load (for testing). Defaults to 1.0.
        """
        self.logger.info("Loading match and player data for joining...")
        
        # Load both datasets
        matches_df = self.parquet_handler.read_parquet(match_parquet_path, load_percentage)
        players_df = self.parquet_handler.read_parquet(player_parquet_path, load_percentage)
        
        self.logger.info(f"Loaded {len(matches_df)} matches and {len(players_df)} player records")
        
        # Map team_id to team name for clarity
        players_df['team'] = players_df['team_id'].map({100: 'team1', 200: 'team2'})
        
        # Normalize role names to lowercase for consistency
        players_df['role'] = players_df['role'].str.lower()
        role_mapping = {'bottom': 'adc', 'utility': 'support'}
        players_df['role'] = players_df['role'].replace(role_mapping)
        
        # ====== Part 1: Pivot individual player features by role ======
        individual_cols = ['champion_id', 'ranked_tier', 'ranked_rank']
        
        player_features_list = []
        for col in individual_cols:
            # Pivot: rows=match_id, columns=team+role, values=col
            pivoted = players_df.pivot_table(
                index='match_id',
                columns=['team', 'role'],
                values=col,
                aggfunc='first'  # Take first value (should be only one per match-team-role)
            )
            # Flatten multi-level columns: (team1, top) -> team1_top_champion_id
            pivoted.columns = [f'{team}_{role}_{col}' for team, role in pivoted.columns]
            player_features_list.append(pivoted)
        
        # Combine all pivoted individual features
        player_features_df = pd.concat(player_features_list, axis=1).reset_index()
        self.logger.info(f"Created {player_features_df.shape[1] - 1} individual player features (champion_id, ranked_tier, ranked_rank per role)")
        
        # ====== Part 2: Aggregate numeric stats per team ======
        exclude_cols = {'match_id', 'puuid', 'team_id', 'team', 'role', 'champion_id', 'ranked_tier', 'ranked_rank'}
        numeric_cols = [col for col in players_df.select_dtypes(include=['number']).columns 
                       if col not in exclude_cols]
        
        self.logger.info(f"Aggregating {len(numeric_cols)} numeric player features per team using '{aggregation_strategy}'")
        
        # Aggregate player stats by match_id and team_id
        agg_dict = {col: aggregation_strategy for col in numeric_cols}
        team_stats = players_df.groupby(['match_id', 'team_id']).agg(agg_dict).reset_index()
        
        # Pivot to get team1 (100) and team2 (200) stats as separate columns
        team1_stats = team_stats[team_stats['team_id'] == 100].drop('team_id', axis=1)
        team2_stats = team_stats[team_stats['team_id'] == 200].drop('team_id', axis=1)
        
        # Rename columns to indicate team and aggregation
        team1_stats = team1_stats.rename(columns={col: f'team1_avg_{col}' for col in numeric_cols})
        team2_stats = team2_stats.rename(columns={col: f'team2_avg_{col}' for col in numeric_cols})
        
        # ====== Part 3: Join everything together ======
        combined_df = matches_df.copy()
        combined_df = combined_df.merge(player_features_df, on='match_id', how='left')
        combined_df = combined_df.merge(team1_stats, on='match_id', how='left')
        combined_df = combined_df.merge(team2_stats, on='match_id', how='left')
        
        # Drop match_id as it's not a feature
        if 'match_id' in combined_df.columns:
            combined_df = combined_df.drop('match_id', axis=1)
        
        self.dataframe = combined_df
        self.logger.info(f"Combined dataframe shape: {self.dataframe.shape}")
        self.logger.info(f"Total features: {self.dataframe.shape[1] - 1} (excluding target)")
        self.logger.info(f"  - Match features: {matches_df.shape[1] - 1}")
        self.logger.info(f"  - Individual player features: {player_features_df.shape[1] - 1}")
        self.logger.info(f"  - Aggregated team features: {len(numeric_cols) * 2}")

    def validate_data(self) -> bool:
        """
        Validate the loaded dataframe before processing.
        
        Checks:
        - Dataframe is not None
        - Target feature exists
        - No completely empty columns
        - Reasonable data types
        
        Returns:
            bool: True if validation passes.
            
        Raises:
            ValueError: If validation fails with specific error message.
        """
        if self.dataframe is None:
            raise ValueError("No dataframe loaded. Call load_data or join_match_and_player_data first.")
        
        if self.target_feature not in self.dataframe.columns:
            raise ValueError(f"Target feature '{self.target_feature}' not found in dataframe. "
                           f"Available columns: {list(self.dataframe.columns)}")
        
        # Check for completely empty columns
        empty_cols = self.dataframe.columns[self.dataframe.isnull().all()].tolist()
        if empty_cols:
            self.logger.warning(f"Found {len(empty_cols)} completely empty columns: {empty_cols}")
        
        # Check for duplicate columns
        duplicate_cols = self.dataframe.columns[self.dataframe.columns.duplicated()].tolist()
        if duplicate_cols:
            raise ValueError(f"Found duplicate column names: {duplicate_cols}")
        
        self.logger.info("Data validation passed")
        return True

    def split_data(self):
        """
        Split the dataframe into training and testing sets.
        
        Validates data before splitting and separates features from target.
        """
        # Validate before splitting
        self.validate_data()

        self.logger.info("Splitting data into training and testing sets")
        X = self.dataframe.drop(columns=[self.target_feature])
        y = self.dataframe[self.target_feature]

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )

        y_train = pd.Series(y_train, name=self.target_feature)
        y_test = pd.Series(y_test, name=self.target_feature)

        self.data_train = x_train.reset_index(drop=True)
        self.data_test = x_test.reset_index(drop=True)
        self.labels_train = y_train.reset_index(drop=True)
        self.labels_test = y_test.reset_index(drop=True)

        self.logger.info(f"Training data shape: {self.data_train.shape}")
        self.logger.info(f"Testing data shape: {self.data_test.shape}")
        self.logger.info(f"Training labels shape: {self.labels_train.shape}")
        self.logger.info(f"Testing labels shape: {self.labels_test.shape}")

    def save_cleaned_data(self, output_dir: str) -> str:
        """
        Save cleaned train/test data and labels as separate parquet files in a timestamped subdirectory.

        Args:
            output_dir (str): Base directory path for cleaned data (e.g., 'data/cleaned/matches').
            
        Returns:
            str: Path to the created timestamped subdirectory.
        """

        # Generate timestamp for versioning
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create timestamped subdirectory
        timestamped_dir = os.path.join(output_dir, timestamp)
        os.makedirs(timestamped_dir, exist_ok=True)

        # Save each component without timestamp prefix (directory already has it)
        train_data_path = os.path.join(timestamped_dir, "data_train.parquet")
        test_data_path = os.path.join(timestamped_dir, "data_test.parquet")
        train_labels_path = os.path.join(timestamped_dir, "labels_train.parquet")
        test_labels_path = os.path.join(timestamped_dir, "labels_test.parquet")

        self.parquet_handler.write_parquet(self.data_train, train_data_path)
        self.parquet_handler.write_parquet(self.data_test, test_data_path)
        self.parquet_handler.write_parquet(self.labels_train.to_frame(), train_labels_path)
        self.parquet_handler.write_parquet(self.labels_test.to_frame(), test_labels_path)

        self.logger.info(f"Cleaned data saved to {timestamped_dir}:")
        self.logger.info(f"  - Train data: {train_data_path}")
        self.logger.info(f"  - Test data: {test_data_path}")
        self.logger.info(f"  - Train labels: {train_labels_path}")
        self.logger.info(f"  - Test labels: {test_labels_path}")
        
        return timestamped_dir

    def load_cleaned_data(self, input_dir: str, timestamp: str = None) -> None:
        """
        Load cleaned train/test data and labels from separate parquet files in a timestamped subdirectory.

        Args:
            input_dir (str): Base directory path for cleaned data (e.g., 'data/cleaned/matches').
            timestamp (str): Optional timestamp subdirectory name to load specific version (e.g., '20251218_153045').
                           If None, loads from the most recent timestamped subdirectory.
        """
        
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        # Find timestamped subdirectory
        if timestamp:
            timestamped_dir = os.path.join(input_dir, timestamp)
            if not os.path.exists(timestamped_dir):
                raise FileNotFoundError(f"Timestamped subdirectory not found: {timestamped_dir}")
        else:
            # Get all subdirectories that look like timestamps
            subdirs = [d for d in os.listdir(input_dir) 
                      if os.path.isdir(os.path.join(input_dir, d)) and 
                      d.replace('_', '').isdigit() and len(d) == 15]  # Format: YYYYMMDD_HHMMSS
            
            if not subdirs:
                raise FileNotFoundError(f"No timestamped subdirectories found in {input_dir}")
            
            # Get most recent timestamp
            timestamp = max(subdirs)
            timestamped_dir = os.path.join(input_dir, timestamp)

        # Construct file paths
        train_data_path = os.path.join(timestamped_dir, "data_train.parquet")
        test_data_path = os.path.join(timestamped_dir, "data_test.parquet")
        train_labels_path = os.path.join(timestamped_dir, "labels_train.parquet")
        test_labels_path = os.path.join(timestamped_dir, "labels_test.parquet")

        # Verify all files exist
        for path in [train_data_path, test_data_path, train_labels_path, test_labels_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")

        # Load data
        self.data_train = self.parquet_handler.read_parquet(train_data_path, load_percentage=1.0)
        self.data_test = self.parquet_handler.read_parquet(test_data_path, load_percentage=1.0)
        
        labels_train_df = self.parquet_handler.read_parquet(train_labels_path, load_percentage=1.0)
        labels_test_df = self.parquet_handler.read_parquet(test_labels_path, load_percentage=1.0)
        
        # Convert to Series
        self.labels_train = labels_train_df[labels_train_df.columns[0]]
        self.labels_test = labels_test_df[labels_test_df.columns[0]]

        self.logger.info(f"Cleaned data loaded from {timestamped_dir}:")
        self.logger.info(f"  - Train data shape: {self.data_train.shape}")
        self.logger.info(f"  - Test data shape: {self.data_test.shape}")
        self.logger.info(f"  - Train labels shape: {self.labels_train.shape}")
        self.logger.info(f"  - Test labels shape: {self.labels_test.shape}")
