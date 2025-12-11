from data_extraction.requester import Requester
from helpers.parquet_handler import ParquetHandler
from helpers.champion_ids import fetch_latest_champion_ids

from logging import Logger
import pandas as pd


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

    def __init__(self, requester: Requester, logger: Logger, parquet_handler: ParquetHandler, load_percentage: float = 1.0, random_state: int = 42) -> None:
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
        self.parquet_handler = parquet_handler
        self.load_percentage = load_percentage
        self.random_state = random_state

    def clean_data(self, raw_data_path: str, cleaned_data_path: str, mode: str) -> None:
        """
        Run the full cleaning pipeline for match or player data.

        Args:
            raw_data_path (str): Path to the raw parquet input.
            cleaned_data_path (str): Destination path for cleaned parquet.
            mode (str): 'matches' to clean match data or 'players' for player history data.

        Raises:
            ValueError: If mode is not 'matches' or 'players'.
        """

        if mode not in {"matches", "players"}:
            raise ValueError("mode must be either 'matches' or 'players'")

        self.logger.info(f"Starting data cleaning from {raw_data_path} to {cleaned_data_path}")
        
        raw_data = self.parquet_handler.read_parquet(raw_data_path, load_percentage=self.load_percentage)
        
        # remove duplicates
        cleaned_data_without_duplicates = self._remove_duplicates(raw_data)

        # handle missing values
        cleaned_data_without_missing = self._handle_missing_values(cleaned_data_without_duplicates)
        
        # handle out of range values
        if mode == "matches":
            cleaned_data_without_missing = self._handle_matches_out_of_range_values(cleaned_data_without_missing)
        else:  # mode == "players"
            cleaned_data_without_missing = self._handle_players_out_of_range_values(cleaned_data_without_missing)

        self.logger.info("Data cleaning completed.")

        self.parquet_handler.write_parquet(cleaned_data_without_missing, cleaned_data_path)

    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Drop duplicate rows and log how many were removed.

        Args:
            data (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe without duplicate rows.
        """
        self.logger.info("Removing duplicate records.")
        initial_count = len(data)
        cleaned_data = data.drop_duplicates()
        final_count = len(cleaned_data)
        self.logger.info(f"Removed {initial_count - final_count} duplicate records.")
        return cleaned_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows containing any missing values.

        Args:
            data (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with missing-value rows dropped.
        """
        self.logger.info("Handling missing values.")
        initial_count = len(data)
        cleaned_data = data.dropna()
        final_count = len(cleaned_data)
        self.logger.info(f"Removed {initial_count - final_count} records with missing values.")
        return cleaned_data

    def _handle_matches_out_of_range_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter match records using queue, duration, boolean, and champion ID validity checks.

        Args:
            data (pd.DataFrame): Match dataframe to validate.

        Returns:
            pd.DataFrame: Filtered dataframe containing only valid match rows.
        """

        self.logger.info("Handling out-of-range values for match data.")
        initial_count = len(data)

        # queue_id should be in a predefined set
        valid_queue_ids = {400, 420, 430, 440, 450}
        valid_data = data[data['queue_id'].isin(valid_queue_ids)]

        # game_duration should be positive and less than 2 hours (7200 seconds)
        valid_data = valid_data[(valid_data['game_duration'] > 0) & (valid_data['game_duration'] < 7200)]

        # team1_win should be boolean
        valid_data = valid_data[valid_data['team1_win'].apply(lambda x: isinstance(x, bool))]

        # bans and picks should be non-negative integers and valid champion ids
        champion_id_columns = [
            'team1_ban1', 'team1_ban2', 'team1_ban3', 'team1_ban4', 'team1_ban5',
            'team2_ban1', 'team2_ban2', 'team2_ban3', 'team2_ban4', 'team2_ban5',
            'team1_pick_top', 'team1_pick_jungle', 'team1_pick_mid', 'team1_pick_adc', 'team1_pick_support',
            'team2_pick_top', 'team2_pick_jungle', 'team2_pick_mid', 'team2_pick_adc', 'team2_pick_support'
        ]

        valid_champion_ids = fetch_latest_champion_ids()

        for col in champion_id_columns:
            valid_data = valid_data[valid_data[col].isin(valid_champion_ids)]

        final_count = len(valid_data)
        self.logger.info(f"Removed {initial_count - final_count} records with out-of-range values.")
        return valid_data


    def _handle_players_out_of_range_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply validation rules to player history rows (IDs, roles, tiers, and numeric ranges).

        Args:
            data (pd.DataFrame): Player history dataframe to validate.

        Returns:
            pd.DataFrame: Filtered dataframe with invalid rows removed.
        """
        
        self.logger.info("Handling out-of-range values for player history data.")
        initial_count = len(data)

        # puuid should be strings of length 78
        valid_data = data[data['puuid'].apply(lambda x: isinstance(x, str) and len(x) == 78)]
        valid_data['puuid'] = valid_data['puuid'].astype(str)
        puuid_removed = initial_count - len(valid_data)
        self.logger.info(f"Removed {puuid_removed} records with invalid puuid.")

        # team_id should be either 100 or 200
        valid_data = valid_data[valid_data['team_id'].isin([100, 200])]
        valid_data['team_id'] = valid_data['team_id'].astype(int)
        team_id_removed = initial_count - puuid_removed - len(valid_data)
        self.logger.info(f"Removed {team_id_removed} records with invalid team_id.")

        # role should be one of predefined roles
        valid_roles = {"TOP", "JUNGLE", "MID", "BOTTOM", "UTILITY"}
        valid_data = valid_data[valid_data['role'].isin(valid_roles)]
        valid_data['role'] = valid_data['role'].astype(str)
        role_removed = initial_count - puuid_removed - team_id_removed - len(valid_data)
        self.logger.info(f"Removed {role_removed} records with invalid role.")

        # champion_id should be valid champion ids
        valid_champion_ids = fetch_latest_champion_ids()
        valid_data = valid_data[valid_data['champion_id'].isin(valid_champion_ids)]
        valid_data['champion_id'] = valid_data['champion_id'].astype(int)
        champion_id_removed = initial_count - puuid_removed - team_id_removed - role_removed - len(valid_data)
        self.logger.info(f"Removed {champion_id_removed} records with invalid champion_id.")

        # summoner_level should be positive integers
        valid_data = valid_data[valid_data['summoner_level'].apply(lambda x: isinstance(x, int) and x > 0)]
        valid_data['summoner_level'] = valid_data['summoner_level'].astype(int)
        summoner_level_removed = initial_count - puuid_removed - team_id_removed - role_removed - champion_id_removed - len(valid_data)
        self.logger.info(f"Removed {summoner_level_removed} records with invalid summoner_level.")

        # champion_mastery_level should be non-negative integers
        valid_data = valid_data[valid_data['champion_mastery_level'].apply(lambda x: isinstance(x, int) and x >= 0)]
        valid_data['champion_mastery_level'] = valid_data['champion_mastery_level'].astype(int)
        champion_mastery_level_removed = initial_count  - puuid_removed - team_id_removed - role_removed - champion_id_removed - summoner_level_removed - len(valid_data)
        self.logger.info(f"Removed {champion_mastery_level_removed} records with invalid champion.")

        # champion_total_mastery_score should be non-negative integers
        valid_data = valid_data[valid_data['champion_total_mastery_score'].apply(lambda x: isinstance(x, int) and x >= 0)]
        valid_data['champion_total_mastery_score'] = valid_data['champion_total_mastery_score'].astype(int)
        champion_total_mastery_score_removed = initial_count - puuid_removed - team_id_removed - role_removed - champion_id_removed - summoner_level_removed - champion_mastery_level_removed - len(valid_data)
        self.logger.info(f"Removed {champion_total_mastery_score_removed} records with invalid champion_total_mastery_score.")
        
        # ranked_tier should be one of predefined tiers
        valid_tiers = {"IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"}
        valid_data = valid_data[valid_data['ranked_tier'].isin(valid_tiers)]
        valid_data['ranked_tier'] = valid_data['ranked_tier'].astype(str)
        ranked_tier_removed = initial_count - puuid_removed - team_id_removed - role_removed - champion_id_removed - summoner_level_removed - champion_mastery_level_removed - champion_total_mastery_score_removed - len(valid_data)
        self.logger.info(f"Removed {ranked_tier_removed} records with invalid ranked_tier.")

        # ranked_rank should be one of predefined ranks
        valid_ranks = {"I", "II", "III", "IV"}
        valid_data = valid_data[valid_data['ranked_rank'].isin(valid_ranks)]
        valid_data['ranked_rank'] = valid_data['ranked_rank'].astype(str)
        ranked_rank_removed = initial_count - puuid_removed - team_id_removed - role_removed - champion_id_removed - summoner_level_removed - champion_mastery_level_removed - champion_total_mastery_score_removed - ranked_tier_removed - len(valid_data)
        self.logger.info(f"Removed {ranked_rank_removed} records with invalid ranked_rank.")

        # ranked_league_points should be non-negative integers
        valid_data = valid_data[valid_data['ranked_league_points'] >= 0]
        valid_data['ranked_league_points'] = valid_data['ranked_league_points'].astype(int)
        ranked_league_points_removed = initial_count - puuid_removed - team_id_removed - role_removed - champion_id_removed - summoner_level_removed - champion_mastery_level_removed - champion_total_mastery_score_removed - ranked_tier_removed - ranked_rank_removed - len(valid_data)
        self.logger.info(f"Removed {ranked_league_points_removed} records with invalid ranked_league_points.")
        
        # win_rate should be between 0 and 1
        valid_data = valid_data[(valid_data['win_rate'] >= 0) & (valid_data['win_rate'] <= 1)]
        valid_data['win_rate'] = valid_data['win_rate'].astype(float)
        win_rate_removed = initial_count - puuid_removed - team_id_removed - role_removed - champion_id_removed - summoner_level_removed - champion_mastery_level_removed - champion_total_mastery_score_removed - ranked_tier_removed - ranked_rank_removed - ranked_league_points_removed - len(valid_data)
        self.logger.info(f"Removed {win_rate_removed} records with invalid win_rate.")

        # numeric columns should be non-negative
        numeric_columns = [
            'all_in_pings', 'assist_me_pings', 'command_pings', 'enemy_missing_pings',
            'enemy_vision_pings', 'hold_pings', 'get_back_pings','need_vision_pings',
            'on_my_way_pings', 'push_pings', 'vision_cleared_pings', 'average_wards_placed',
            'average_ward_kills', 'average_sight_wards_bought', 'average_detector_wards_placed',
            'average_killing_sprees', 'average_largest_killing_spree', 'average_largest_multi_kill',
            'average_double_kills', 'average_triple_kills', 'average_quadra_kills', 'average_penta_kills',
            'average_baron_kills', 'average_dragon_kills', 'average_inhibitor_kills', 'average_inhibitor_takedowns',
            'average_inhibitors_lost', 'average_turret_kills', 'average_turret_takedowns', 'average_turrets_lost',
            'average_objectives_stolen', 'average_objectives_stolen_assists', 'average_total_damage_dealt_to_champions',
            'average_physical_damage_dealt_to_champions', 'average_magic_damage_dealt_to_champions',
            'average_true_damage_dealt_to_champions', 'average_total_damage_taken', 'average_damage_self_mitigated',
            'average_total_heal', 'average_total_heals_on_teammates', 'average_kills', 'average_deaths',
            'average_assists', 'kda_ratio', 'win_rate', 'average_cs_per_minute', 'average_kills_per_minute',
            'average_deaths_per_minute', 'average_assists_per_minute', 'average_total_minions_killed',
            'average_neutral_minions_killed', 'average_gold_earned', 'average_gold_spent',
            'average_items_purchased', 'average_time_ccing_others', 'average_total_time_cc_dealt',
            'average_longest_time_spent_living', 'average_total_time_spent_dead', 'average_vision_score'
        ]

        for col in numeric_columns:
            valid_data = valid_data[valid_data[col] >= 0]
            valid_data[col] = valid_data[col].astype(float)
        numeric_columns_removed = initial_count - puuid_removed - team_id_removed - role_removed - champion_id_removed - summoner_level_removed - champion_mastery_level_removed - champion_total_mastery_score_removed - ranked_tier_removed - ranked_rank_removed - ranked_league_points_removed - win_rate_removed - len(valid_data)
        self.logger.info(f"Removed {numeric_columns_removed} records with invalid numeric column values.")

        final_count = len(valid_data)
        self.logger.info(f"Removed {initial_count - final_count} records with out-of-range values.")
        return valid_data


