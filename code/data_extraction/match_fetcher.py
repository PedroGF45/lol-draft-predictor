from data_extraction.requester import Requester
from data_extraction.schemas import MATCH_SCHEMA, PLAYER_HISTORY_SCHEMA
from helpers.parquet_handler import ParquetHandler
from helpers.checkpoint import save_checkpoint, load_checkpoint

import os
import logging
import pandas as pd
from typing import Any
from tqdm import tqdm

class MatchFetcher:
    """
    Fetches and enriches match data from the Riot API, combining match details with player historical KPIs.

    This class orchestrates the enrichment pipeline: for each match, it fetches draft information (bans, picks, roles),
    player state (summoner level, mastery, rank), and aggregated historical statistics from prior games. Results are
    saved as two parquet tables (matches and player_history) for downstream ML training. Supports checkpoint-based
    resumption to recover from API interruptions.

    Attributes:
        requester (Requester): HTTP client for Riot API calls with retry/backoff.
        logger (logging.Logger): Logger for status and error messages.
        dataframe_target_path (str): Directory to save output parquet files.
        checkpoint_loading_path (str, optional): Path to checkpoint file for resumable processing.
        processed_matches (set[str]): Set of match IDs already processed (persisted to checkpoint).
        final_match_df_list (list): Accumulated match records (one per match).
        final_player_history_df_list (list): Accumulated player history records (10 per match).
    """

    def __init__(self, 
                 requester: Requester, 
                 logger: logging.Logger, 
                 parquet_handler: ParquetHandler, 
                 dataframe_target_path: str, 
                 checkpoint_loading_path: str = None,
                 load_percentage: float = 1.0,
                 random_state: int = 42) -> None:
        """
        Initialize MatchFetcher with API client, logging, and optional checkpoint recovery.

        Args:
            requester (Requester): Configured HTTP client for API calls.
            logger (logging.Logger): Logger instance for status/error messages.
            parquet_handler (ParquetHandler): Handler for reading/writing parquet files.
            dataframe_target_path (str): Directory path for output parquet files.
            checkpoint_loading_path (str, optional): Path to checkpoint file for resumption. Defaults to None.
            load_percentage (float, optional): Percentage of data to load from parquet. Defaults to 1.0.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        
        self.requester = requester
        self.parquet_handler = parquet_handler
        self.logger = logger
        self.load_percentage = load_percentage
        self.random_state = random_state

        self.dataframe_target_path = os.path.join(dataframe_target_path, "preprocess")
        
        if not self.parquet_handler.check_directory_exists(self.dataframe_target_path):
            self.parquet_handler.create_directory(self.dataframe_target_path)
        
        # Initialize checkpoint state
        self.checkpoint_loading_path = checkpoint_loading_path
        self.processed_matches = set()
        self.final_match_df_list = []
        self.final_player_history_df_list = []
        
        if checkpoint_loading_path and os.path.exists(checkpoint_loading_path):
            checkpoint_state = load_checkpoint(logger=self.logger, path=checkpoint_loading_path)
            if checkpoint_state:
                self.processed_matches = checkpoint_state.get("processed_matches", set())
                self.final_match_df_list = checkpoint_state.get("final_match_df_list", [])
                self.final_player_history_df_list = checkpoint_state.get("final_player_history_df_list", [])
                self.logger.info(f"Resumed from checkpoint: {len(self.processed_matches)} matches already processed")
      
    def fetch_match_data(self, parquet_path: str, keep_remakes: bool = False, queue: list[int] | None = None, match_limit_per_player: int = 50, checkpoint_save_interval: int = 10) -> None:
        """
        Orchestrate the enrichment pipeline: fetch match details, player state, and KPIs for all matches.

        Processes each match in the input parquet file (loading draft features, summoner level, mastery, rank, and
        historical KPIs). Skips remakes, non-ranked queues, and already-processed matches (from checkpoint). Saves
        two output parquet files: matches.parquet (1 row per match) and player_history.parquet (10 rows per match).
        Periodically checkpoints progress for resumable processing.

        Args:
            parquet_path (str): Path to input parquet with match_id column.
            keep_remakes (bool): If False (default), skip matches < 300 seconds.
            queue (list[int] | None): Queue IDs to include. Defaults to [420, 440] if None.
            match_limit_per_player (int): Number of prior matches per player to fetch for KPI aggregation (default: 50).
            checkpoint_save_interval (int): Save checkpoint every N matches processed (default: 10).
        """

        if queue is None:
            queue = [420, 440]

        if not os.path.exists(parquet_path):
            self.logger.error(f'Parquet path must be a valid path but got {parquet_path}')
        
        match_df = self.parquet_handler.read_parquet(file_path=parquet_path, load_percentage=self.load_percentage)

        # Add progress bar for match processing
        total_matches = len(match_df)
        self.logger.info(f'Processing {total_matches} matches...')
        
        checkpoint_counter = 0
        for match_id in tqdm(match_df.itertuples(index=False), total=total_matches, desc="Processing matches", unit="match"):
            match_id = match_id.match_id
            
            # Skip already-processed matches
            if match_id in self.processed_matches:
                self.logger.debug(f'Skipping already-processed match: {match_id}')
                continue
            
            # fetch match details
            match_pre_features = self.fetch_match_pre_features(match_id=match_id)

            if match_pre_features is None:
                self.logger.info(f'Skipping match {match_id} due to incomplete data')
                self.processed_matches.add(match_id)
                continue

            # check if game is a remake
            if not keep_remakes and match_pre_features.get("game_duration") < 300:
                self.logger.info(f'Skipping remake match: {match_id}')
                self.processed_matches.add(match_id)
                continue

            # filter by queue
            if match_pre_features.get("queue_id") not in queue:
                self.logger.info(f'Skipping match {match_id} due to queue filter. Queue ID: {match_pre_features.get("queue_id")}')
                self.processed_matches.add(match_id)
                continue

            participants = match_pre_features.get("team1_participants") + match_pre_features.get("team2_participants")

            # Fetch summoner level for each participant
            summoner_levels = self.fetch_summoner_level_data(participants=participants)
            self.logger.debug(f'Summoner Levels: {summoner_levels}')

            # fetch champion mastery of the champion played in the match for each participant
            champion_picks = match_pre_features.get("team1_picks") + match_pre_features.get("team2_picks")
            champion_masteries = self.fetch_champion_mastery_data(participants=participants, champion_picks=champion_picks)
            self.logger.debug(f'Champion Masteries: {champion_masteries}')

            # fetch the total mastery score for each participant
            champion_total_mastery_scores = self.fetch_total_mastery_score(participants=participants)
            self.logger.debug(f'Total Mastery Scores: {champion_total_mastery_scores}')

            # fetch rank queue data for each participant
            rank_queue_data = self.fetch_rank_queue_data(participants=participants)
            self.logger.debug(f'Rank Queue Data: {rank_queue_data}')

            # get raw kpis for each participant on last N matches before this match
            current_match_timestamp_creation = match_pre_features.get("game_creation")
            kpis_data = self.fetch_raw_player_kpis(participants=participants, match_limit_per_player=match_limit_per_player, before_timestamp=current_match_timestamp_creation)
            self.logger.debug(f'KPIs Data: {kpis_data}')

            # create match record with match schema
            match_record = self.create_match_record(match_id=match_id, match_pre_features=match_pre_features)
            self.final_match_df_list.append(match_record)

            # create player history records
            for puuid in participants:
                team_id = 100 if puuid in match_pre_features.get("team1_participants") else 200
                role = match_pre_features.get("team1_roles")[match_pre_features.get("team1_participants").index(puuid)] if team_id == 100 else match_pre_features.get("team2_roles")[match_pre_features.get("team2_participants").index(puuid)]
                champion_id = match_pre_features.get("team1_picks")[match_pre_features.get("team1_participants").index(puuid)] if team_id == 100 else match_pre_features.get("team2_picks")[match_pre_features.get("team2_participants").index(puuid)]
                player_history_record = self.create_player_history_record(
                    match_id=match_id,
                    puuid=puuid,
                    team_id=team_id,
                    role=role,
                    champion_id=champion_id,
                    summoner_level=summoner_levels.get(puuid),
                    champion_mastery=champion_masteries.get(puuid),
                    champion_total_mastery_score=champion_total_mastery_scores.get(puuid),
                    rank_queue_data=rank_queue_data,
                    kpis_data=kpis_data.get(puuid)
                )
                self.final_player_history_df_list.append(player_history_record)
            
            # Mark match as processed
            self.processed_matches.add(match_id)
            
            # Periodically save checkpoint
            checkpoint_counter += 1
            if checkpoint_counter % checkpoint_save_interval == 0 and self.checkpoint_loading_path:
                self._save_checkpoint()

        final_match_df = pd.DataFrame(self.final_match_df_list)
        final_player_history_df = pd.DataFrame(self.final_player_history_df_list)

        self.logger.info(f'Final Dataframe Head: {final_match_df.head()}')
        self.logger.info(f'Final Dataframde description: {final_match_df.describe()}')

        self.logger.info(f'Final Player History Dataframe Head: {final_player_history_df.head()}')
        self.logger.info(f'Final Player History Dataframde description: {final_player_history_df.describe()}')
        
        # suffix with detailed timestamp, number of matches saved, number of matches_per_player
        current_date = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        match_path_sufix = f'{current_date}_{len(final_match_df)}_matches_{match_limit_per_player}_players_per_match'
        match_output_path = os.path.join(self.dataframe_target_path, "matches", f'{match_path_sufix}.parquet')
        self.parquet_handler.write_parquet(data=final_match_df, file_path=match_output_path)
        self.logger.info(f'Match Dataframe saved to {match_output_path}')

        player_history_output_path = os.path.join(self.dataframe_target_path, "players", f'{match_path_sufix}.parquet')
        self.parquet_handler.write_parquet(data=final_player_history_df, file_path=player_history_output_path)
        self.logger.info(f'Player History Dataframe saved to {player_history_output_path}')
        
        # Clear checkpoint after successful completion
        if self.checkpoint_loading_path and os.path.exists(self.checkpoint_loading_path):
            try:
                os.remove(self.checkpoint_loading_path)
                self.logger.info('Checkpoint cleared after successful completion')
            except Exception as e:
                self.logger.warning(f'Failed to clear checkpoint: {e}')

    def _save_checkpoint(self) -> None:
        """
        Save current progress (processed matches and dataframe rows) to checkpoint file.

        Allows the pipeline to resume from the exact point of interruption without reprocessing.
        """
        if not self.checkpoint_loading_path:
            self.logger.warning('Checkpoint loading path not set; skipping checkpoint save.')
            return

        checkpoint_dir = (
            os.path.dirname(self.checkpoint_loading_path)
            if os.path.splitext(self.checkpoint_loading_path)[1]
            else self.checkpoint_loading_path
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_filename = f"{timestamp}_{len(self.processed_matches)}_matches_checkpoint.pkl"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        checkpoint_state = {
            "processed_matches": self.processed_matches,
            "final_match_df_list": self.final_match_df_list,
            "final_player_history_df_list": self.final_player_history_df_list
        }
        save_checkpoint(logger=self.logger, state=checkpoint_state, path=checkpoint_path)

        # Keep latest path for cleanup
        self.checkpoint_loading_path = checkpoint_path

    def fetch_match_pre_features(self, match_id: str) -> dict[str, Any]:
        """
        Fetch draft-related features for a single match: bans, picks, roles, and outcome.

        Args:
            match_id (str): Riot match ID (e.g., 'EUW1_7586168110').

        Returns:
            dict[str, Any]: Match features including game metadata, bans, picks, team roles, and outcome. 
                           Returns None if match data is invalid or incomplete (e.g., missing teams).
        """
        
        match_details_endpoint = f'/lol/match/v5/matches/{match_id}'
        match_details = self.requester.make_request(is_v5=True, endpoint_url=match_details_endpoint)

        # Guard against invalid or missing match details
        if not isinstance(match_details, dict) or not match_details.get("metadata") or not match_details.get("info"):
            self.logger.info(f'Skipping match {match_id} due to invalid match details response')
            return None

        data_version = match_details.get("metadata").get("dataVersion") # version of the data schema
        game_creation = match_details.get("info").get("gameCreation") # timestamp of game creation
        game_duration = match_details.get("info").get("gameDuration") # in seconds
        game_mode = match_details.get("info").get("gameMode") # CLASSIC, ARAM, URF, DOOMBOTSTEEMO, ONEFORALL5x5, ASCENSION, etc
        game_type = match_details.get("info").get("gameType") # MATCHED_GAME, CUSTOM_GAME
        queue_id = match_details.get("info").get("queueId") #420, 430, 440, 450, 700, 720, 900, 910, 920, 940, 950, 960
        game_version = match_details.get("info").get("gameVersion") #11.18.387.1024
        platform_id = match_details.get("info").get("platformId") #EUW1

        # Validate that both teams exist
        teams = match_details.get("info", {}).get("teams", [])
        if len(teams) < 2:
            self.logger.info(f'Skipping match {match_id} due to incomplete team data (only {len(teams)} team(s) found)')
            return None

        # list of bans by championId for each team
        team1_bans = [ban.get('championId') for ban in teams[0].get("bans")]
        team2_bans = [ban.get('championId') for ban in teams[1].get("bans")]

        team1_picks = []
        team2_picks = []
        team1_roles = []
        team2_roles = []
        team1_participants = [p.get('puuid') for p in match_details.get("info").get("participants") if p.get("teamId") == 100]
        team2_participants = [p.get('puuid') for p in match_details.get("info").get("participants") if p.get("teamId") == 200]
        team1_participants_data = [p for p in match_details.get("info").get("participants") if p.get("teamId") == 100]
        team2_participants_data = [p for p in match_details.get("info").get("participants") if p.get("teamId") == 200]

        role_order = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

        for participant in team1_participants_data:
            team1_picks.append(participant.get("championId"))
            team1_roles.append(participant.get("individualPosition") if participant.get("individualPosition") in role_order else "UNKNOWN")
        for participant in team2_participants_data:
            team2_picks.append(participant.get("championId"))
            team2_roles.append(participant.get("individualPosition") if participant.get("individualPosition") in role_order else "UNKNOWN")
        match_outcome = teams[0].get("win")  # True if team 1 won, False otherwise

        self.logger.debug(f'Match ID: {match_id}')
        self.logger.debug(f'Data Version: {data_version}')
        self.logger.debug(f'Game Creation Timestamp: {game_creation}')
        self.logger.debug(f'Game Duration: {game_duration} seconds')
        self.logger.debug(f'Game Mode: {game_mode}')
        self.logger.debug(f'Game Type: {game_type}')
        self.logger.debug(f'Queue ID: {queue_id}')
        self.logger.debug(f'Game Version: {game_version}')
        self.logger.debug(f'Platform ID: {platform_id}')
        self.logger.debug(f'Team 1 Bans: {team1_bans}')
        self.logger.debug(f'Team 2 Bans: {team2_bans}')
        self.logger.debug(f'Team 1 Picks: {team1_picks}')
        self.logger.debug(f'Team 2 Picks: {team2_picks}')
        self.logger.debug(f'Team 1 Roles: {team1_roles}')
        self.logger.debug(f'Team 2 Roles: {team2_roles}')
        self.logger.debug(f'Team 1 Participants: {team1_participants}')
        self.logger.debug(f'Team 2 Participants: {team2_participants}')
        self.logger.debug(f'Match Outcome (Team 1 Win): {match_outcome}')

        match_pre_features = {
            "data_version": data_version,
            "game_creation": game_creation,
            "game_duration": game_duration,
            "game_mode": game_mode,
            "game_type": game_type,
            "queue_id": queue_id,
            "game_version": game_version,
            "platform_id": platform_id,
            "team1_bans": team1_bans,
            "team2_bans": team2_bans,
            "team1_picks": team1_picks,
            "team2_picks": team2_picks,
            "team1_roles": team1_roles,
            "team2_roles": team2_roles,
            "team1_participants": team1_participants,
            "team2_participants": team2_participants,
            "match_outcome": match_outcome
        }

        return match_pre_features
    
    def fetch_summoner_level_data(self, participants: list[str]) -> dict[str, Any]:
        """
        Fetch summoner level for each participant.

        Args:
            participants (list[str]): List of player PUUIDs.

        Returns:
            dict[str, Any]: Mapping of PUUID → summoner level (int).
        """
        
        summoner_levels = {}
        for puuid in participants:
            summoner_endpoint = f'/lol/summoner/v4/summoners/by-puuid/{puuid}'
            summoner_data = self.requester.make_request(is_v5=False, endpoint_url=summoner_endpoint)

            summoner_level = summoner_data.get("summonerLevel")
            self.logger.debug(f'Summoner PUUID: {puuid}, Level: {summoner_level}')
            summoner_levels[puuid] = summoner_level

        return summoner_levels
    
    def fetch_champion_mastery_data(self, participants: list[str], champion_picks: list[int]) -> dict[str, Any]:
        """
        Fetch champion mastery stats (level, points, last played) for each participant's picked champion.

        Args:
            participants (list[str]): List of player PUUIDs.
            champion_picks (list[int]): List of champion IDs picked (ordered same as participants).

        Returns:
            dict[str, Any]: Mapping of PUUID → {lastPlayTime, championLevel, championPoints}.
        """
        champion_masteries = {}
        
        for puuid in participants:
            champion_id = champion_picks[participants.index(puuid)]
            champion_mastery_endpoint = f'/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/by-champion/{champion_id}'
            mastery_data = self.requester.make_request(is_v5=False, endpoint_url=champion_mastery_endpoint)

            last_played_time = mastery_data.get("lastPlayTime")
            champion_level = mastery_data.get("championLevel")
            champion_points = mastery_data.get("championPoints")
            

            champion_masteries[puuid] = {
                "lastPlayTime": last_played_time,
                "championLevel": champion_level,
                "championPoints": champion_points
            }

        return champion_masteries
    
    def fetch_total_mastery_score(self, participants: list[str]) -> dict[str, int]:
        """
        Fetch total champion mastery score for each participant (across all champions).

        Args:
            participants (list[str]): List of player PUUIDs.

        Returns:
            dict[str, int]: Mapping of PUUID → total mastery score (int).
        """
        total_mastery_scores = {}
        for puuid in participants:
            total_mastery_endpoint = f'/lol/champion-mastery/v4/scores/by-puuid/{puuid}'
            total_score = self.requester.make_request(is_v5=False, endpoint_url=total_mastery_endpoint)

            total_mastery_scores[puuid] = total_score

        return total_mastery_scores
    
    def fetch_rank_queue_data(self, participants: list[str]) -> dict[str, Any]:
        """
        Fetch ranked queue info (tier, rank, LP, wins/losses) for solo and flex queues.

        Args:
            participants (list[str]): List of player PUUIDs.

        Returns:
            dict[str, Any]: Mapping of PUUID_solo/PUUID_flex → {tier, rank, leaguePoints, wins, losses, hotStreak}.
        """
        rank_queue_data = {}
        for puuid in participants:
            summoner_endpoint = f'/lol/league/v4/entries/by-puuid/{puuid}'
            rank_data = self.requester.make_request(is_v5=False, endpoint_url=summoner_endpoint)

            # Guard against None or non-list responses (e.g., API errors)
            if not isinstance(rank_data, list):
                self.logger.debug(f'Rank data for {puuid} is not a list: {type(rank_data).__name__}')
                continue

            # Find the rank data for the RANKED_SOLO_5x5 queue
            solo_queue_data = next((entry for entry in rank_data if entry.get("queueType") == "RANKED_SOLO_5x5"), None)
            if solo_queue_data:
                rank_queue_data[puuid + "_solo"] = {
                    "tier": solo_queue_data.get("tier"),
                    "rank": solo_queue_data.get("rank"),
                    "leaguePoints": solo_queue_data.get("leaguePoints"),
                    "wins": solo_queue_data.get("wins"),
                    "losses": solo_queue_data.get("losses"),
                    "hotStreak": solo_queue_data.get("hotStreak")
                }
            
            flex_queue_data = next((entry for entry in rank_data if entry.get("queueType") == "RANKED_FLEX_SR"), None)
            if flex_queue_data:
                rank_queue_data[puuid + "_flex"] = {
                    "tier": flex_queue_data.get("tier"),
                    "rank": flex_queue_data.get("rank"),
                    "leaguePoints": flex_queue_data.get("leaguePoints"),
                    "wins": flex_queue_data.get("wins"),
                    "losses": flex_queue_data.get("losses"),
                    "hotStreak": flex_queue_data.get("hotStreak")
                }

        return rank_queue_data
    
    def fetch_raw_player_kpis(self, participants: list[str], match_limit_per_player: int = 50, before_timestamp: int = None) -> dict[str, Any]:
        """
        Fetch raw KPI data from the last N matches for each participant, filtered by timestamp.

        For each participant, retrieves their last `match_limit_per_player` match IDs before `before_timestamp`, 
        then fetches detailed stats (kills, deaths, CS, gold, damage, etc.) from each match.

        Args:
            participants (list[str]): List of player PUUIDs.
            match_limit_per_player (int): Number of prior matches to fetch per player (default: 50).
            before_timestamp (int, optional): Unix timestamp to filter matches (only matches before this time). 

        Returns:
            dict[str, Any]: Mapping of PUUID → list of match KPI dicts (70+ stat fields per match).
        """
        kpis_data = {}
        for puuid in participants:
            kpis_endpoint = f'/lol/match/v5/matches/by-puuid/{puuid}/ids?count={match_limit_per_player}'
            if before_timestamp:
                kpis_endpoint += f'&endTime={before_timestamp}'
            kpis_match_ids = self.requester.make_request(is_v5=True, endpoint_url=kpis_endpoint)

            # Guard against None or non-list responses (e.g., API errors)
            if not isinstance(kpis_match_ids, list):
                self.logger.debug(f'KPI match IDs for {puuid} is not a list: {type(kpis_match_ids).__name__}')
                continue

            kpis_list = []
            for kpis_match_id in kpis_match_ids:
                match_details_endpoint = f'/lol/match/v5/matches/{kpis_match_id}'
                match_details = self.requester.make_request(is_v5=True, endpoint_url=match_details_endpoint)

                # Guard against None or invalid match details
                if not isinstance(match_details, dict) or not match_details.get("info"):
                    self.logger.debug(f'Invalid match details for {kpis_match_id}')
                    continue

                participant_data = next((p for p in match_details.get("info").get("participants", []) if p.get("puuid") == puuid), None)
                if participant_data:
                    # Extract game duration from match info
                    game_duration_seconds = match_details.get("info").get("gameDuration", 0)
                    
                    self.logger.debug(f'Participant Data for PUUID {puuid} in Match {kpis_match_id}')
                    
                    # get relevant kpis
                    kpis = {
                        "allInPings": participant_data.get("allInPings"),
                        "assistmePings": participant_data.get("assistmePings"),
                        "assists": participant_data.get("assists"),
                        "baronKills": participant_data.get("baronKills"),
                        "bountyLevel": participant_data.get("bountyLevel"),
                        "commandPings": participant_data.get("commandPings"),
                        "consumablesPurchased": participant_data.get("consumablesPurchased"),
                        "damageDealtToBuildings": participant_data.get("damageDealtToBuildings"),
                        "damageDealtToObjectives": participant_data.get("damageDealtToObjectives"),
                        "damageDealtToTurrets": participant_data.get("damageDealtToTurrets"),
                        "damageSelfMitigated": participant_data.get("damageSelfMitigated"),
                        "deaths": participant_data.get("deaths"),
                        "detectorWardsPlaced": participant_data.get("detectorWardsPlaced"),
                        "doubleKills": participant_data.get("doubleKills"),
                        "dragonKills": participant_data.get("dragonKills"),
                        "enemyMissingPings": participant_data.get("enemyMissingPings"),
                        "enemyVisionPings": participant_data.get("enemyVisionPings"),
                        "firstBloodAssist": participant_data.get("firstBloodAssist"),
                        "firstBloodKill": participant_data.get("firstBloodKill"),
                        "firstTowerAssist": participant_data.get("firstTowerAssist"),
                        "firstTowerKill": participant_data.get("firstTowerKill"),
                        "holdPings": participant_data.get("holdPings"),
                        "getbackPings": participant_data.get("getbackPings"),
                        "goldEarned": participant_data.get("goldEarned"),
                        "goldSpent": participant_data.get("goldSpent"),
                        "inhibitorKills": participant_data.get("inhibitorKills"),
                        "inhibitorTakedowns": participant_data.get("inhibitorTakedowns"),
                        "inhibitorsLost": participant_data.get("inhibitorsLost"),
                        "itemsPurchased": participant_data.get("itemsPurchased"),
                        "killingSprees": participant_data.get("killingSprees"),
                        "kills": participant_data.get("kills"),
                        "largestCriticalStrike": participant_data.get("largestCriticalStrike"),
                        "largeKillingSpree": participant_data.get("largeKillingSpree"),
                        "largestMultiKill": participant_data.get("largestMultiKill"),
                        "longestTimeSpentLiving": participant_data.get("longestTimeSpentLiving"),
                        "magicDamageDealt": participant_data.get("magicDamageDealt"),
                        "magicDamageDealtToChampions": participant_data.get("magicDamageDealtToChampions"),
                        "magicDamageTaken": participant_data.get("magicDamageTaken"),
                        "neutralMinionsKilled": participant_data.get("neutralMinionsKilled"),
                        "needVisionPings": participant_data.get("needVisionPings"),
                        "objectivesStolen": participant_data.get("objectivesStolen"),
                        "objectivesStolenAssists": participant_data.get("objectivesStolenAssists"),
                        "onMyWayPings": participant_data.get("onMyWayPings"),
                        "pentakills": participant_data.get("pentakills"),
                        "physicalDamageDealt": participant_data.get("physicalDamageDealt"),
                        "physicalDamageDealtToChampions": participant_data.get("physicalDamageDealtToChampions"),
                        "physicalDamageTaken": participant_data.get("physicalDamageTaken"),
                        "pushPings": participant_data.get("pushPings"),
                        "quadraKills": participant_data.get("quadraKills"),
                        "sightWardsBoughtInGame": participant_data.get("sightWardsBoughtInGame"),
                        "timeCCingOthers": participant_data.get("timeCCingOthers"),
                        "totalDamageDealt": participant_data.get("totalDamageDealt"),
                        "totalDamageDealtToChampions": participant_data.get("totalDamageDealtToChampions"),
                        "totalDamageShieldedOnTeammates": participant_data.get("totalDamageShieldedOnTeammates"),
                        "totalDamageTaken": participant_data.get("totalDamageTaken"),
                        "totalHeal": participant_data.get("totalHeal"),
                        "totalHealsOnTeammates": participant_data.get("totalHealsOnTeammates"),
                        "totalMinionsKilled": participant_data.get("totalMinionsKilled"),
                        "totalTimeCCDealt": participant_data.get("totalTimeCCDealt"),
                        "totalTimeSpentDead": participant_data.get("totalTimeSpentDead"),
                        "totalUnitsHealed": participant_data.get("totalUnitsHealed"),
                        "tripleKills": participant_data.get("tripleKills"),
                        "trueDamageDealt": participant_data.get("trueDamageDealt"),
                        "trueDamageDealtToChampions": participant_data.get("trueDamageDealtToChampions"),
                        "trueDamageTaken": participant_data.get("trueDamageTaken"),
                        "turretKills": participant_data.get("turretKills"),
                        "turretTakedowns": participant_data.get("turretTakedowns"),
                        "turretsLost": participant_data.get("turretsLost"),
                        "unrealKills": participant_data.get("unrealKills"),
                        "visionScore": participant_data.get("visionScore"),
                        "visionClearedPings": participant_data.get("visionClearedPings"),
                        "visionWardsBoughtInGame": participant_data.get("visionWardsBoughtInGame"),
                        "wardKills": participant_data.get("wardKills"),
                        "wardsPlaced": participant_data.get("wardsPlaced"),
                        "win": participant_data.get("win"),
                        "gameDuration": game_duration_seconds  # Add game duration for per-minute calculations
                    }

                    kpis_list.append(kpis)

            kpis_data[puuid] = kpis_list

        return kpis_data
    
    def create_match_record(self, match_id: str, match_pre_features: dict[str, Any]) -> dict[str, Any]:
        """
        Construct a single match record for the matches.parquet output table.

        Flattens draft bans/picks into 20 champion ID columns and includes match metadata and outcome.

        Args:
            match_id (str): Riot match ID.
            match_pre_features (dict[str, Any]): Match features dict with bans, picks, metadata, outcome.

        Returns:
            dict[str, Any]: Flat dict with 25 columns (match_id, queue_id, game_version, game_duration, team1_win, 10 bans, 10 picks).
        """
        match_record = {
            "match_id": match_id,
            "queue_id": match_pre_features.get("queue_id"),
            "game_version": match_pre_features.get("game_version"),
            "game_duration": match_pre_features.get("game_duration"),
            "team1_win": match_pre_features.get("match_outcome"),

            # bans
            "team1_ban1": match_pre_features.get("team1_bans")[0] if len(match_pre_features.get("team1_bans")) > 0 else None,
            "team1_ban2": match_pre_features.get("team1_bans")[1] if len(match_pre_features.get("team1_bans")) > 1 else None,
            "team1_ban3": match_pre_features.get("team1_bans")[2] if len(match_pre_features.get("team1_bans")) > 2 else None,
            "team1_ban4": match_pre_features.get("team1_bans")[3] if len(match_pre_features.get("team1_bans")) > 3 else None,
            "team1_ban5": match_pre_features.get("team1_bans")[4] if len(match_pre_features.get("team1_bans")) > 4 else None,
            "team2_ban1": match_pre_features.get("team2_bans")[0] if len(match_pre_features.get("team2_bans")) > 0 else None,
            "team2_ban2": match_pre_features.get("team2_bans")[1] if len(match_pre_features.get("team2_bans")) > 1 else None,
            "team2_ban3": match_pre_features.get("team2_bans")[2] if len(match_pre_features.get("team2_bans")) > 2 else None,
            "team2_ban4": match_pre_features.get("team2_bans")[3] if len(match_pre_features.get("team2_bans")) > 3 else None,
            "team2_ban5": match_pre_features.get("team2_bans")[4] if len(match_pre_features.get("team2_bans")) > 4 else None,
            # picks
            "team1_pick_top": match_pre_features.get("team1_picks")[0],
            "team1_pick_jungle": match_pre_features.get("team1_picks")[1],
            "team1_pick_mid": match_pre_features.get("team1_picks")[2],
            "team1_pick_adc": match_pre_features.get("team1_picks")[3],
            "team1_pick_support": match_pre_features.get("team1_picks")[4],
            "team2_pick_top": match_pre_features.get("team2_picks")[0],
            "team2_pick_jungle": match_pre_features.get("team2_picks")[1],
            "team2_pick_mid": match_pre_features.get("team2_picks")[2],
            "team2_pick_adc": match_pre_features.get("team2_picks")[3],
            "team2_pick_support": match_pre_features.get("team2_picks")[4]
        }
        return match_record
    
    def create_player_history_record(self, 
                                    match_id: str, 
                                    puuid: str,
                                    team_id: int,
                                    role: str,
                                    champion_id: int,
                                    summoner_level: int, 
                                    champion_mastery: dict[str, Any],
                                    champion_total_mastery_score: int, 
                                    rank_queue_data: dict[str, Any],
                                    kpis_data: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Construct a single player history record for the player_history.parquet output table.

        Aggregates player state (summoner level, mastery, rank) and KPIs (win rate, KDA, per-minute stats) 
        from prior matches into a single row per player per match.

        Args:
            match_id (str): Riot match ID.
            puuid (str): Player PUUID.
            team_id (int): Team ID (100 or 200).
            role (str): Position in match (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY).
            champion_id (int): Champion ID picked.
            summoner_level (int): Summoner account level.
            champion_mastery (dict[str, Any]): Mastery level, points, last played time.
            champion_total_mastery_score (int): Total mastery across all champions.
            rank_queue_data (dict[str, Any]): Rank tier, LP, wins/losses for solo/flex queues.
            kpis_data (list[dict[str, Any]]): List of KPI dicts from prior matches.

        Returns:
            dict[str, Any]: Flat dict with 27 columns (state + aggregated KPIs).
        """
        player_history_record = {
            'match_id': match_id,
            'puuid': puuid,
            'team_id': team_id,
            'role': role,
            'champion_id': champion_id,

            # state of the summoner
            'summoner_level': summoner_level,
            'champion_mastery_level': champion_mastery.get("championLevel") if champion_mastery else None,
            'champion_total_mastery_score': champion_total_mastery_score,
            'ranked_tier': rank_queue_data.get(puuid + "_solo", {}).get("tier") if rank_queue_data.get(puuid + "_solo") else None,
            'ranked_rank': rank_queue_data.get(puuid + "_solo", {}).get("rank") if rank_queue_data.get(puuid + "_solo") else None,
            'ranked_league_points': rank_queue_data.get(puuid + "_solo", {}).get("leaguePoints") if rank_queue_data.get(puuid + "_solo") else None,

            # pings
            'all_in_pings': self._calculate_average(kpis_data, 'allInPings'),
            'assist_me_pings': self._calculate_average(kpis_data, 'assistmePings'),
            'command_pings': self._calculate_average(kpis_data, 'commandPings'),
            'enemy_missing_pings': self._calculate_average(kpis_data, 'enemyMissingPings'),
            'enemy_vision_pings': self._calculate_average(kpis_data, 'enemyVisionPings'),
            'hold_pings': self._calculate_average(kpis_data, 'holdPings'),
            'get_back_pings': self._calculate_average(kpis_data, 'getbackPings'),
            'need_vision_pings': self._calculate_average(kpis_data, 'needVisionPings'),
            'on_my_way_pings': self._calculate_average(kpis_data, 'onMyWayPings'),
            'push_pings': self._calculate_average(kpis_data, 'pushPings'),
            'vision_cleared_pings': self._calculate_average(kpis_data, 'visionClearedPings'),

            # wards
            'average_wards_placed': self._calculate_average(kpis_data, 'wardsPlaced'),
            'average_ward_kills': self._calculate_average(kpis_data, 'wardKills'),
            'average_sight_wards_bought': self._calculate_average(kpis_data, 'sightWardsBoughtInGame'),
            'average_detector_wards_placed': self._calculate_average(kpis_data, 'detectorWardsPlaced'),

            # kill streaks
            'average_killing_sprees': self._calculate_average(kpis_data, 'killingSprees'),
            'average_largest_killing_spree': self._calculate_average(kpis_data, 'largeKillingSpree'),
            'average_largest_multi_kill': self._calculate_average(kpis_data, 'largestMultiKill'),
            'average_double_kills': self._calculate_average(kpis_data, 'doubleKills'),
            'average_triple_kills': self._calculate_average(kpis_data, 'tripleKills'),
            'average_quadra_kills': self._calculate_average(kpis_data, 'quadraKills'),
            'average_penta_kills': self._calculate_average(kpis_data, 'pentakills'),

            # objectives
            'average_baron_kills': self._calculate_average(kpis_data, 'baronKills'),
            'average_dragon_kills': self._calculate_average(kpis_data, 'dragonKills'),
            'average_inhibitor_kills': self._calculate_average(kpis_data, 'inhibitorKills'),
            'average_inhibitor_takedowns': self._calculate_average(kpis_data, 'inhibitorTakedowns'),
            'average_inhibitors_lost': self._calculate_average(kpis_data, 'inhibitorsLost'),
            'average_turret_kills': self._calculate_average(kpis_data, 'turretKills'),
            'average_turret_takedowns': self._calculate_average(kpis_data, 'turretTakedowns'),
            'average_turrets_lost': self._calculate_average(kpis_data, 'turretsLost'),
            'average_objectives_stolen': self._calculate_average(kpis_data, 'objectivesStolen'),
            'average_objectives_stolen_assists': self._calculate_average(kpis_data, 'objectivesStolenAssists'),

            # damage stats
            'average_total_damage_dealt_to_champions': self._calculate_average(kpis_data, 'totalDamageDealtToChampions'),
            'average_physical_damage_dealt_to_champions': self._calculate_average(kpis_data, 'physicalDamageDealtToChampions'),
            'average_magic_damage_dealt_to_champions': self._calculate_average(kpis_data, 'magicDamageDealtToChampions'),
            'average_true_damage_dealt_to_champions': self._calculate_average(kpis_data, 'trueDamageDealtToChampions'),
            'average_total_damage_taken': self._calculate_average(kpis_data, 'totalDamageTaken'),
            'average_damage_self_mitigated': self._calculate_average(kpis_data, 'damageSelfMitigated'),
            'average_total_heal': self._calculate_average(kpis_data, 'totalHeal'),
            'average_total_heals_on_teammates': self._calculate_average(kpis_data, 'totalHealsOnTeammates'),

            # combat stats
            'average_kills': self._calculate_average(kpis_data, 'kills'),
            'average_deaths': self._calculate_average(kpis_data, 'deaths'),
            'average_assists': self._calculate_average(kpis_data, 'assists'),
            'kda_ratio': self._calculate_kda_ratio(kpis_data),
            'win_rate': self._calculate_win_rate(kpis_data),
            'average_cs_per_minute': self._calculate_per_minute(kpis_data, 'totalMinionsKilled'),
            'average_kills_per_minute': self._calculate_per_minute(kpis_data, 'kills'),
            'average_deaths_per_minute': self._calculate_per_minute(kpis_data, 'deaths'),
            'average_assists_per_minute': self._calculate_per_minute(kpis_data, 'assists'),

            # cs
            'average_total_minions_killed': self._calculate_average(kpis_data, 'totalMinionsKilled'),
            'average_neutral_minions_killed': self._calculate_average(kpis_data, 'neutralMinionsKilled'),

            # gold
            'average_gold_earned': self._calculate_average(kpis_data, 'goldEarned'),
            'average_gold_spent': self._calculate_average(kpis_data, 'goldSpent'),
            'average_items_purchased': self._calculate_average(kpis_data, 'itemsPurchased'),

            # cc
            'average_time_ccing_others': self._calculate_average(kpis_data, 'timeCCingOthers'),
            'average_total_time_cc_dealt': self._calculate_average(kpis_data, 'totalTimeCCDealt'),

            # survival
            'average_longest_time_spent_living': self._calculate_average(kpis_data, 'longestTimeSpentLiving'),
            'average_total_time_spent_dead': self._calculate_average(kpis_data, 'totalTimeSpentDead'),

            # others
            'average_vision_score': self._calculate_average(kpis_data, 'visionScore')
        }

        return player_history_record

    def _calculate_win_rate(self, kpis_data: list[dict[str, Any]]) -> float:
        """
        Calculate win rate from a list of match KPIs.

        Args:
            kpis_data (list[dict[str, Any]]): List of KPI dicts from prior matches.

        Returns:
            float: Win rate as a fraction [0, 1]. Returns 0.0 if list is empty.
        """
        if not kpis_data:
            return 0.0
        wins = sum(1 for kpis in kpis_data if kpis.get("win"))
        return wins / len(kpis_data)
    
    def _calculate_average(self, kpis_data: list[dict[str, Any]], key: str) -> float:
        """
        Calculate the average value of a stat across prior matches.

        Args:
            kpis_data (list[dict[str, Any]]): List of KPI dicts.
            key (str): Stat key to average (e.g., 'kills', 'totalMinionsKilled').

        Returns:
            float: Average value. Returns 0.0 if list is empty.
        """
        if not kpis_data:
            return 0.0

        total = sum(self._safe_number(kpis.get(key)) for kpis in kpis_data)
        return total / len(kpis_data)
    
    def _safe_number(self, value: Any) -> float:
        """
        Convert a value to float if numeric; otherwise return 0.0.

        Args:
            value (Any): Potentially numeric value.

        Returns:
            float: Numeric value or 0.0 for None/non-numeric inputs.
        """
        return value if isinstance(value, (int, float)) else 0.0
    
    def _calculate_kda_ratio(self, kpis_data: list[dict[str, Any]]) -> float:
        """
        Calculate the KDA (Kill+Death Assist / Death) ratio from prior matches.

        Args:
            kpis_data (list[dict[str, Any]]): List of KPI dicts.

        Returns:
            float: KDA ratio. Returns total kills + assists if deaths == 0; returns 0.0 if list is empty.
        """
        if not kpis_data:
            return 0.0
        total_kills = sum((kpis.get("kills") or 0) for kpis in kpis_data)
        total_assists = sum((kpis.get("assists") or 0) for kpis in kpis_data)
        total_deaths = sum((kpis.get("deaths") or 0) for kpis in kpis_data)
        if total_deaths == 0:
            return total_kills + total_assists
        return (total_kills + total_assists) / total_deaths
    
    def _calculate_per_minute(self, kpis_data: list[dict[str, Any]], key: str) -> float:
        """
        Calculate the per-minute rate of a statistic from prior matches.

        Args:
            kpis_data (list[dict[str, Any]]): List of KPI dicts containing 'gameDuration' and stat keys.
            key (str): The statistic key to aggregate (e.g., 'kills', 'cs').

        Returns:
            float: The per-minute rate. Returns 0.0 if kpis_data is empty or total_minutes is zero.
        """
        if not kpis_data:
            return 0.0
        total = sum((kpis.get(key) or 0) for kpis in kpis_data)
        total_minutes = sum(((kpis.get("gameDuration") or 0) / 60) for kpis in kpis_data)
        if total_minutes == 0:
            return 0.0
        return total / total_minutes