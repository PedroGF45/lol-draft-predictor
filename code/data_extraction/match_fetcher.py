import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import pandas as pd
from data_extraction.requester import Requester
from data_extraction.schemas import MATCH_SCHEMA, PLAYER_HISTORY_SCHEMA
from helpers.checkpoint import load_checkpoint, save_checkpoint
from helpers.master_data_registry import MasterDataRegistry
from helpers.parquet_handler import ParquetHandler
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

    def __init__(
        self,
        requester: Requester,
        logger: logging.Logger,
        parquet_handler: ParquetHandler,
        dataframe_target_path: str,
        checkpoint_loading_path: str = None,
        load_percentage: float = 1.0,
        random_state: int = 42,
        master_registry: Optional[MasterDataRegistry] = None,
        max_workers: int = 8,
    ) -> None:
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
        self.master_registry = master_registry
        self.max_workers = max_workers
        self.backoff_cooldown_counter = 0  # Counter to manage backoff cooldown (allow increase after N iterations)

        self.dataframe_target_path = dataframe_target_path

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

        # Simple in-memory caches to reduce repeated API calls across matches
        self._cache_summoner_level: dict[str, Any] = {}
        self._cache_total_mastery: dict[str, Any] = {}
        self._cache_rank_entries: dict[str, Any] = {}
        self._cache_champion_mastery: dict[tuple[str, int], Any] = {}
        self._cache_kpis_ids: dict[str, list[str]] = {}

    def _process_single_match(
        self, match_id: str, match_limit_per_player: int = 50
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Process a single match fetched from the Riot API into structured records.

        This method extracts match-level and player-level features from raw match data,
        including draft information, summoner levels, champion masteries, ranks, and historical KPIs.

        Args:
            match_id (str): Riot match ID (e.g., 'EUW1_7586168110')
            match_limit_per_player (int): Number of prior matches per player to fetch for KPI aggregation (default: 50)

        Returns:
            tuple[dict, list[dict]]: A tuple containing:
                - match_record (dict): Single match record with draft and metadata
                - player_history_records (list[dict]): List of 10 player history records (one per participant)
        """
        # Extract pre-features for the match
        self.logger.info(f"[predict] Fetching pre-features for match {match_id}")
        match_pre_features = self.fetch_match_pre_features(match_id=match_id)

        if match_pre_features is None:
            raise ValueError(f"Unable to fetch match pre-features for match {match_id}")

        participants = match_pre_features.get("team1_participants") + match_pre_features.get("team2_participants")
        self.logger.info(f"[predict] Participants fetched: {len(participants)}")

        # Fetch summoner level for each participant
        self.logger.info(f"[predict] Fetching summoner levels")
        summoner_levels = self.fetch_summoner_level_data(participants=participants)
        self.logger.debug(f"Summoner Levels: {summoner_levels}")

        # Fetch champion mastery of the champion played in the match for each participant
        champion_picks = match_pre_features.get("team1_picks") + match_pre_features.get("team2_picks")
        self.logger.info(f"[predict] Fetching champion masteries for picks")
        champion_masteries = self.fetch_champion_mastery_data(participants=participants, champion_picks=champion_picks)
        self.logger.debug(f"Champion Masteries: {champion_masteries}")

        # Fetch the total mastery score for each participant
        self.logger.info(f"[predict] Fetching total mastery scores")
        champion_total_mastery_scores = self.fetch_total_mastery_score(participants=participants)
        self.logger.debug(f"Total Mastery Scores: {champion_total_mastery_scores}")

        # Fetch rank queue data for each participant
        self.logger.info(f"[predict] Fetching rank queue data")
        rank_queue_data = self.fetch_rank_queue_data(participants=participants)
        self.logger.debug(f"Rank Queue Data: {rank_queue_data}")

        # Get raw KPIs for each participant on last N matches before this match
        current_match_timestamp_creation = match_pre_features.get("game_creation")
        self.logger.info(f"[predict] Fetching raw player KPIs (limit per player={match_limit_per_player})")
        kpis_data = self.fetch_raw_player_kpis(
            participants=participants,
            match_limit_per_player=match_limit_per_player,
            before_timestamp=current_match_timestamp_creation,
        )
        self.logger.debug(f"KPIs Data: {kpis_data}")

        # Create match record with match schema
        self.logger.info(f"[predict] Creating match record")
        match_record = self.create_match_record(match_id=match_id, match_pre_features=match_pre_features)

        # Create player history records
        self.logger.info(f"[predict] Creating player history records")
        player_history_records = []
        for puuid in participants:
            team_id = 100 if puuid in match_pre_features.get("team1_participants") else 200
            role = (
                match_pre_features.get("team1_roles")[match_pre_features.get("team1_participants").index(puuid)]
                if team_id == 100
                else match_pre_features.get("team2_roles")[match_pre_features.get("team2_participants").index(puuid)]
            )
            champion_id = (
                match_pre_features.get("team1_picks")[match_pre_features.get("team1_participants").index(puuid)]
                if team_id == 100
                else match_pre_features.get("team2_picks")[match_pre_features.get("team2_participants").index(puuid)]
            )
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
                kpis_data=kpis_data.get(puuid),
            )
            player_history_records.append(player_history_record)

        self.logger.info(f"[predict] Single-match processing complete for {match_id}")
        return match_record, player_history_records

    def fetch_match_data(
        self,
        parquet_path: str,
        keep_remakes: bool = False,
        queue: list[int] | None = None,
        match_limit_per_player: int = 50,
        checkpoint_save_interval: int = 10,
    ) -> None:
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
            self.logger.error(f"Parquet path must be a valid path but got {parquet_path}")

        match_df = self.parquet_handler.read_parquet(file_path=parquet_path, load_percentage=self.load_percentage)

        # Filter through master registry if available
        if self.master_registry:
            initial_count = len(match_df)
            # Filter out matches already in registry
            if "game_version" in match_df.columns:
                match_df["_composite_key"] = list(
                    zip(match_df["match_id"].astype(str), match_df["game_version"].astype(str))
                )
                match_df = match_df[~match_df["_composite_key"].isin(self.master_registry.match_registry.keys())]
                match_df = match_df.drop(columns=["_composite_key"])
            else:
                # If no game_version yet, just filter by match_id (less precise but still helpful)
                registered_match_ids = {k[0] for k in self.master_registry.match_registry.keys()}
                match_df = match_df[~match_df["match_id"].isin(registered_match_ids)]

            filtered_count = initial_count - len(match_df)
            self.logger.info(
                f"Registry filter: {len(match_df)} new matches to process, {filtered_count} already in registry"
            )

        # Add progress bar for match processing
        total_matches = len(match_df)
        self.logger.info(f"Processing {total_matches} matches...")

        checkpoint_counter = 0
        for match_id in tqdm(
            match_df.itertuples(index=False), total=total_matches, desc="Processing matches", unit="match"
        ):
            match_id = match_id.match_id

            # Skip already-processed matches
            if match_id in self.processed_matches:
                self.logger.debug(f"Skipping already-processed match: {match_id}")
                continue

            # fetch match details
            match_pre_features = self.fetch_match_pre_features(match_id=match_id)

            if match_pre_features is None:
                self.logger.info(f"Skipping match {match_id} due to incomplete data")
                self.processed_matches.add(match_id)
                continue

            # check if game is a remake
            if not keep_remakes and match_pre_features.get("game_duration") < 300:
                self.logger.info(f"Skipping remake match: {match_id}")
                self.processed_matches.add(match_id)
                continue

            # filter by queue
            if match_pre_features.get("queue_id") not in queue:
                self.logger.info(
                    f'Skipping match {match_id} due to queue filter. Queue ID: {match_pre_features.get("queue_id")}'
                )
                self.processed_matches.add(match_id)
                continue

            participants = match_pre_features.get("team1_participants") + match_pre_features.get("team2_participants")

            # Fetch summoner level for each participant
            summoner_levels = self.fetch_summoner_level_data(participants=participants)
            self.logger.debug(f"Summoner Levels: {summoner_levels}")

            # fetch champion mastery of the champion played in the match for each participant
            champion_picks = match_pre_features.get("team1_picks") + match_pre_features.get("team2_picks")
            champion_masteries = self.fetch_champion_mastery_data(
                participants=participants, champion_picks=champion_picks
            )
            self.logger.debug(f"Champion Masteries: {champion_masteries}")

            # fetch the total mastery score for each participant
            champion_total_mastery_scores = self.fetch_total_mastery_score(participants=participants)
            self.logger.debug(f"Total Mastery Scores: {champion_total_mastery_scores}")

            # fetch rank queue data for each participant
            rank_queue_data = self.fetch_rank_queue_data(participants=participants)
            self.logger.debug(f"Rank Queue Data: {rank_queue_data}")

            # get raw kpis for each participant on last N matches before this match
            current_match_timestamp_creation = match_pre_features.get("game_creation")
            kpis_data = self.fetch_raw_player_kpis(
                participants=participants,
                match_limit_per_player=match_limit_per_player,
                before_timestamp=current_match_timestamp_creation,
            )
            self.logger.debug(f"KPIs Data: {kpis_data}")

            # create match record with match schema
            match_record = self.create_match_record(match_id=match_id, match_pre_features=match_pre_features)
            self.final_match_df_list.append(match_record)

            # create player history records
            for puuid in participants:
                team_id = 100 if puuid in match_pre_features.get("team1_participants") else 200
                role = (
                    match_pre_features.get("team1_roles")[match_pre_features.get("team1_participants").index(puuid)]
                    if team_id == 100
                    else match_pre_features.get("team2_roles")[
                        match_pre_features.get("team2_participants").index(puuid)
                    ]
                )
                champion_id = (
                    match_pre_features.get("team1_picks")[match_pre_features.get("team1_participants").index(puuid)]
                    if team_id == 100
                    else match_pre_features.get("team2_picks")[
                        match_pre_features.get("team2_participants").index(puuid)
                    ]
                )
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
                    kpis_data=kpis_data.get(puuid),
                )
                self.final_player_history_df_list.append(player_history_record)

            # Mark match as processed
            self.processed_matches.add(match_id)

            # Periodically save checkpoint
            checkpoint_counter += 1
            if checkpoint_counter % checkpoint_save_interval == 0 and self.checkpoint_loading_path:
                self._save_checkpoint()

            # Check for rate limiting and apply backoff
            if self.requester.should_backoff(threshold=3):
                # Rate limit detected: reduce worker count
                self.max_workers = max(1, self.max_workers // 2)
                self.logger.warning(f"Rate limit backoff: reduced max_workers to {self.max_workers}")
                self.backoff_cooldown_counter = (
                    0  # Reset cooldown to require many successful iterations before increase
                )
            else:
                # Gradually increase workers back up if we're not hitting rate limits
                self.backoff_cooldown_counter += 1
                if (
                    self.backoff_cooldown_counter >= 50 and self.max_workers < 8
                ):  # Increase after 50 successful iterations
                    self.max_workers = min(8, self.max_workers + 1)
                    self.logger.info(f"Rate limit recovery: increased max_workers to {self.max_workers}")
                    self.backoff_cooldown_counter = 0

        final_match_df = pd.DataFrame(self.final_match_df_list)
        final_player_history_df = pd.DataFrame(self.final_player_history_df_list)

        self.logger.info(f"Final Dataframe Head: {final_match_df.head()}")
        self.logger.info(f"Final Dataframde description: {final_match_df.describe()}")

        self.logger.info(f"Final Player History Dataframe Head: {final_player_history_df.head()}")
        self.logger.info(f"Final Player History Dataframde description: {final_player_history_df.describe()}")

        # Register matches with master registry if available
        if self.master_registry and len(final_match_df) > 0:
            self.logger.info(f"Registering {len(final_match_df)} matches with master registry")
            collection_metadata = {
                "source": "match_fetcher",
                "match_limit_per_player": match_limit_per_player,
                "queues": queue,
            }
            _, duplicates = self.master_registry.register_matches(
                final_match_df[["match_id", "game_version"]], collection_metadata=collection_metadata
            )
            self.logger.info(f"Registry registration complete: {duplicates} duplicates detected")

        # suffix with detailed timestamp, number of matches saved, number of matches_per_player
        current_date = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        match_path_sufix = f"{current_date}_{len(final_match_df)}_matches_{match_limit_per_player}_players_per_match"
        match_output_path = os.path.join(self.dataframe_target_path, "matches", f"{match_path_sufix}.parquet")
        self.parquet_handler.write_parquet(data=final_match_df, file_path=match_output_path)
        self.logger.info(f"Match Dataframe saved to {match_output_path}")

        player_history_output_path = os.path.join(self.dataframe_target_path, "players", f"{match_path_sufix}.parquet")
        self.parquet_handler.write_parquet(data=final_player_history_df, file_path=player_history_output_path)
        self.logger.info(f"Player History Dataframe saved to {player_history_output_path}")

        # Clear checkpoint after successful completion
        if self.checkpoint_loading_path and os.path.exists(self.checkpoint_loading_path):
            try:
                os.remove(self.checkpoint_loading_path)
                self.logger.info("Checkpoint cleared after successful completion")
            except Exception as e:
                self.logger.warning(f"Failed to clear checkpoint: {e}")

    def _save_checkpoint(self) -> None:
        """
        Save current progress (processed matches and dataframe rows) to checkpoint file.

        Allows the pipeline to resume from the exact point of interruption without reprocessing.
        """
        if not self.checkpoint_loading_path:
            self.logger.warning("Checkpoint loading path not set; skipping checkpoint save.")
            return

        checkpoint_dir = (
            os.path.dirname(self.checkpoint_loading_path)
            if os.path.splitext(self.checkpoint_loading_path)[1]
            else self.checkpoint_loading_path
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_filename = f"{timestamp}_{len(self.processed_matches)}_matches_checkpoint.pkl"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        checkpoint_state = {
            "processed_matches": self.processed_matches,
            "final_match_df_list": self.final_match_df_list,
            "final_player_history_df_list": self.final_player_history_df_list,
        }
        save_checkpoint(logger=self.logger, state=checkpoint_state, path=checkpoint_path)

        # Keep latest path for cleanup
        self.checkpoint_loading_path = checkpoint_path

    def fetch_active_game_pre_features(self, game_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch draft-related features for an active/live game using V5 spectator endpoint.

        Args:
            game_id (str): The active game ID from spectator v5 endpoint (e.g., '7672924003')

        Returns:
            dict[str, Any]: Live game features including bans, picks, team roles (no outcome)
                           Returns None if game data is invalid
        """
        # Use V5 spectator-game-summary endpoint with gameId (no deprecated V4 endpoints)
        self.logger.info(f"Fetching spectator game summary for gameId: {game_id}")
        spectator_game = self.requester.make_request(
            is_v5=True, endpoint_url=f"/lol/spectator/v5/spectator-game-summary/{game_id}"
        )

        if not spectator_game or not isinstance(spectator_game, dict):
            self.logger.warning(
                f"Failed to fetch spectator data for gameId {game_id}. "
                f"Response was None or non-dict (possibly 403/404 from Riot API). "
                f"Check API key permissions and ensure the game is observable."
            )
            return None

        # Some non-200 responses return a JSON body with a status code; treat them as invalid
        status_payload = spectator_game.get("status") if isinstance(spectator_game, dict) else None
        if isinstance(status_payload, dict) and status_payload.get("status_code"):
            self.logger.info(
                "Spectator API returned error for gameId %s: %s",
                game_id,
                status_payload.get("message") or status_payload.get("status_code"),
            )
            return None

        self.logger.info(f"Successfully fetched spectator game data for gameId: {game_id}")

        # Parse spectator game data into match_pre_features format
        # V5 spectator-game-summary response structure
        try:
            game_mode = spectator_game.get("gameMode")
            game_type = spectator_game.get("gameType")
            queue_id = spectator_game.get("gameQueueConfigId")
            game_start_time = spectator_game.get("gameStartTime", 0)  # Unix timestamp in ms
            game_length = spectator_game.get("gameLength", 0)  # Seconds since game start
            platform_id = spectator_game.get("platformId")

            participants_raw = spectator_game.get("participants") or []
            if not isinstance(participants_raw, list) or len(participants_raw) < 10:
                self.logger.info(
                    "Invalid spectator data for gameId %s: expected 10 participants, got %s",
                    game_id,
                    len(participants_raw) if isinstance(participants_raw, list) else "non-list",
                )
                return None

            # Filter to entries with both teamId and puuid present to avoid downstream failures
            participants = [p for p in participants_raw if p.get("teamId") in (100, 200) and p.get("puuid")]
            if len(participants) < 10:
                self.logger.info(
                    "Spectator data missing participant identifiers for gameId %s (got %d with puuid)",
                    game_id,
                    len(participants),
                )
                return None

            banned_champions = spectator_game.get("bannedChampions", []) or []

            # Sort participants by team (100=blue, 200=red)
            team1_participants = [p["puuid"] for p in participants if p.get("teamId") == 100]
            team2_participants = [p["puuid"] for p in participants if p.get("teamId") == 200]

            team1_picks = [p.get("championId", -1) for p in participants if p.get("teamId") == 100]
            team2_picks = [p.get("championId", -1) for p in participants if p.get("teamId") == 200]

            # Validate we have complete team compositions
            if (
                len(team1_participants) != 5
                or len(team2_participants) != 5
                or len(team1_picks) != 5
                or len(team2_picks) != 5
            ):
                self.logger.info(
                    "Spectator data incomplete for gameId %s (team sizes: %d/%d, picks: %d/%d)",
                    game_id,
                    len(team1_participants),
                    len(team2_participants),
                    len(team1_picks),
                    len(team2_picks),
                )
                return None

            # Extract bans by team
            team1_bans = [b.get("championId", -1) for b in banned_champions if b.get("teamId") == 100]
            team2_bans = [b.get("championId", -1) for b in banned_champions if b.get("teamId") == 200]

            # Pad bans to 5 (some games may have fewer)
            team1_bans += [-1] * (5 - len(team1_bans))
            team2_bans += [-1] * (5 - len(team2_bans))
            team1_bans = team1_bans[:5]
            team2_bans = team2_bans[:5]

            # For live games, we don't have role assignments from Riot
            # We'll assign placeholder roles (can be improved with role detection logic)
            team1_roles = ["UNKNOWN"] * len(team1_participants)
            team2_roles = ["UNKNOWN"] * len(team2_participants)

            # Try to infer roles from spell1Id (Smite=11 for jungle, etc.)
            # This is a simple heuristic and not perfect
            for i, p in enumerate([p for p in participants if p.get("teamId") == 100]):
                spell1 = p.get("spell1Id")
                spell2 = p.get("spell2Id")
                # Smite = jungle
                if spell1 == 11 or spell2 == 11:
                    team1_roles[i] = "JUNGLE"

            for i, p in enumerate([p for p in participants if p.get("teamId") == 200]):
                spell1 = p.get("spell1Id")
                spell2 = p.get("spell2Id")
                if spell1 == 11 or spell2 == 11:
                    team2_roles[i] = "JUNGLE"

            return {
                "match_id": f"LIVE_{game_id}",
                "game_creation": game_start_time,
                "game_duration": game_length,
                "game_mode": game_mode,
                "game_type": game_type,
                "queue_id": queue_id,
                "game_version": "LIVE",
                "platform_id": platform_id,
                "team1_bans": team1_bans,
                "team2_bans": team2_bans,
                "team1_picks": team1_picks,
                "team2_picks": team2_picks,
                "team1_roles": team1_roles,
                "team2_roles": team2_roles,
                "team1_participants": team1_participants,
                "team2_participants": team2_participants,
                "match_outcome": None,  # Unknown for live games
                "is_live": True,
            }
        except Exception as e:
            self.logger.error(f"Failed to parse active game data: {e}")
            return None

    def fetch_match_pre_features(self, match_id: str) -> dict[str, Any]:
        """
        Fetch draft-related features for a single match: bans, picks, roles, and outcome.

        Args:
            match_id (str): Riot match ID (e.g., 'EUW1_7586168110').

        Returns:
            dict[str, Any]: Match features including game metadata, bans, picks, team roles, and outcome.
                           Returns None if match data is invalid or incomplete (e.g., missing teams).
        """

        match_details_endpoint = f"/lol/match/v5/matches/{match_id}"
        match_details = self.requester.make_request(is_v5=True, endpoint_url=match_details_endpoint)

        # Guard against invalid or missing match details
        if not isinstance(match_details, dict) or not match_details.get("metadata") or not match_details.get("info"):
            self.logger.info(f"Skipping match {match_id} due to invalid match details response")
            return None

        data_version = match_details.get("metadata").get("dataVersion")  # version of the data schema
        game_creation = match_details.get("info").get("gameCreation")  # timestamp of game creation
        game_duration = match_details.get("info").get("gameDuration")  # in seconds
        game_mode = match_details.get("info").get(
            "gameMode"
        )  # CLASSIC, ARAM, URF, DOOMBOTSTEEMO, ONEFORALL5x5, ASCENSION, etc
        game_type = match_details.get("info").get("gameType")  # MATCHED_GAME, CUSTOM_GAME
        queue_id = match_details.get("info").get(
            "queueId"
        )  # 420, 430, 440, 450, 700, 720, 900, 910, 920, 940, 950, 960
        game_version = match_details.get("info").get("gameVersion")  # 11.18.387.1024
        platform_id = match_details.get("info").get("platformId")  # EUW1

        # Validate that both teams exist
        teams = match_details.get("info", {}).get("teams", [])
        if len(teams) < 2:
            self.logger.info(f"Skipping match {match_id} due to incomplete team data (only {len(teams)} team(s) found)")
            return None

        # list of bans by championId for each team
        team1_bans = [ban.get("championId") for ban in teams[0].get("bans")]
        team2_bans = [ban.get("championId") for ban in teams[1].get("bans")]

        team1_picks = []
        team2_picks = []
        team1_roles = []
        team2_roles = []
        team1_participants = [
            p.get("puuid") for p in match_details.get("info").get("participants") if p.get("teamId") == 100
        ]
        team2_participants = [
            p.get("puuid") for p in match_details.get("info").get("participants") if p.get("teamId") == 200
        ]
        team1_participants_data = [p for p in match_details.get("info").get("participants") if p.get("teamId") == 100]
        team2_participants_data = [p for p in match_details.get("info").get("participants") if p.get("teamId") == 200]

        role_order = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

        for participant in team1_participants_data:
            team1_picks.append(participant.get("championId"))
            team1_roles.append(
                participant.get("individualPosition")
                if participant.get("individualPosition") in role_order
                else "UNKNOWN"
            )
        for participant in team2_participants_data:
            team2_picks.append(participant.get("championId"))
            team2_roles.append(
                participant.get("individualPosition")
                if participant.get("individualPosition") in role_order
                else "UNKNOWN"
            )
        match_outcome = teams[0].get("win")  # True if team 1 won, False otherwise

        self.logger.debug(f"Match ID: {match_id}")
        self.logger.debug(f"Data Version: {data_version}")
        self.logger.debug(f"Game Creation Timestamp: {game_creation}")
        self.logger.debug(f"Game Duration: {game_duration} seconds")
        self.logger.debug(f"Game Mode: {game_mode}")
        self.logger.debug(f"Game Type: {game_type}")
        self.logger.debug(f"Queue ID: {queue_id}")
        self.logger.debug(f"Game Version: {game_version}")
        self.logger.debug(f"Platform ID: {platform_id}")
        self.logger.debug(f"Team 1 Bans: {team1_bans}")
        self.logger.debug(f"Team 2 Bans: {team2_bans}")
        self.logger.debug(f"Team 1 Picks: {team1_picks}")
        self.logger.debug(f"Team 2 Picks: {team2_picks}")
        self.logger.debug(f"Team 1 Roles: {team1_roles}")
        self.logger.debug(f"Team 2 Roles: {team2_roles}")
        self.logger.debug(f"Team 1 Participants: {team1_participants}")
        self.logger.debug(f"Team 2 Participants: {team2_participants}")
        self.logger.debug(f"Match Outcome (Team 1 Win): {match_outcome}")

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
            "match_outcome": match_outcome,
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

        def _fetch_level(puuid: str) -> tuple[str, Any]:
            if puuid in self._cache_summoner_level:
                return puuid, self._cache_summoner_level[puuid]
            summoner_endpoint = f"/lol/summoner/v4/summoners/by-puuid/{puuid}"
            summoner_data = self.requester.make_request(is_v5=False, endpoint_url=summoner_endpoint) or {}
            level = summoner_data.get("summonerLevel")
            self._cache_summoner_level[puuid] = level
            return puuid, level

        summoner_levels: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_fetch_level, puuid): puuid for puuid in participants}
            for future in as_completed(futures):
                puuid, level = future.result()
                self.logger.debug(f"Summoner PUUID: {puuid}, Level: {level}")
                summoner_levels[puuid] = level
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

        def _fetch_mastery(puuid: str, champion_id: int) -> tuple[str, dict[str, Any]]:
            cache_key = (puuid, champion_id)
            if cache_key in self._cache_champion_mastery:
                return puuid, self._cache_champion_mastery[cache_key]
            endpoint = f"/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/by-champion/{champion_id}"
            mastery_data = self.requester.make_request(is_v5=False, endpoint_url=endpoint) or {}
            record = {
                "lastPlayTime": mastery_data.get("lastPlayTime"),
                "championLevel": mastery_data.get("championLevel"),
                "championPoints": mastery_data.get("championPoints"),
            }
            self._cache_champion_mastery[cache_key] = record
            return puuid, record

        champion_masteries: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_fetch_mastery, puuid, champion_picks[participants.index(puuid)]): puuid
                for puuid in participants
            }
            for future in as_completed(futures):
                puuid, record = future.result()
                champion_masteries[puuid] = record
        return champion_masteries

    def fetch_total_mastery_score(self, participants: list[str]) -> dict[str, int]:
        """
        Fetch total champion mastery score for each participant (across all champions).

        Args:
            participants (list[str]): List of player PUUIDs.

        Returns:
            dict[str, int]: Mapping of PUUID → total mastery score (int).
        """

        def _fetch_total_score(puuid: str) -> tuple[str, int]:
            if puuid in self._cache_total_mastery:
                return puuid, self._cache_total_mastery[puuid]
            endpoint = f"/lol/champion-mastery/v4/scores/by-puuid/{puuid}"
            total_score = self.requester.make_request(is_v5=False, endpoint_url=endpoint)
            self._cache_total_mastery[puuid] = total_score
            return puuid, total_score

        total_mastery_scores: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_fetch_total_score, puuid): puuid for puuid in participants}
            for future in as_completed(futures):
                puuid, score = future.result()
                total_mastery_scores[puuid] = score
        return total_mastery_scores

    def fetch_rank_queue_data(self, participants: list[str]) -> dict[str, Any]:
        """
        Fetch ranked queue info (tier, rank, LP, wins/losses) for solo and flex queues.

        Args:
            participants (list[str]): List of player PUUIDs.

        Returns:
            dict[str, Any]: Mapping of PUUID_solo/PUUID_flex → {tier, rank, leaguePoints, wins, losses, hotStreak}.
        """

        def _fetch_rank(puuid: str) -> tuple[str, dict[str, Any]]:
            if puuid in self._cache_rank_entries:
                return puuid, self._cache_rank_entries[puuid]
            endpoint = f"/lol/league/v4/entries/by-puuid/{puuid}"
            rank_data = self.requester.make_request(is_v5=False, endpoint_url=endpoint)
            self._cache_rank_entries[puuid] = rank_data
            return puuid, rank_data

        rank_queue_data: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_fetch_rank, puuid): puuid for puuid in participants}
            for future in as_completed(futures):
                puuid, rank_data = future.result()
                if not isinstance(rank_data, list):
                    self.logger.debug(f"Rank data for {puuid} is not a list: {type(rank_data).__name__}")
                    continue
                solo_queue_data = next(
                    (entry for entry in rank_data if entry.get("queueType") == "RANKED_SOLO_5x5"), None
                )
                if solo_queue_data:
                    rank_queue_data[puuid + "_solo"] = {
                        "tier": solo_queue_data.get("tier"),
                        "rank": solo_queue_data.get("rank"),
                        "leaguePoints": solo_queue_data.get("leaguePoints"),
                        "wins": solo_queue_data.get("wins"),
                        "losses": solo_queue_data.get("losses"),
                        "hotStreak": solo_queue_data.get("hotStreak"),
                    }
                flex_queue_data = next(
                    (entry for entry in rank_data if entry.get("queueType") == "RANKED_FLEX_SR"), None
                )
                if flex_queue_data:
                    rank_queue_data[puuid + "_flex"] = {
                        "tier": flex_queue_data.get("tier"),
                        "rank": flex_queue_data.get("rank"),
                        "leaguePoints": flex_queue_data.get("leaguePoints"),
                        "wins": flex_queue_data.get("wins"),
                        "losses": flex_queue_data.get("losses"),
                        "hotStreak": flex_queue_data.get("hotStreak"),
                    }
        return rank_queue_data

    def fetch_raw_player_kpis(
        self, participants: list[str], match_limit_per_player: int = 50, before_timestamp: int = None
    ) -> dict[str, Any]:
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
        kpis_data: dict[str, Any] = {}

        def _fetch_kpis_ids(puuid: str) -> tuple[str, list[str]]:
            if puuid in self._cache_kpis_ids:
                return puuid, self._cache_kpis_ids[puuid]
            endpoint = f"/lol/match/v5/matches/by-puuid/{puuid}/ids?count={match_limit_per_player}"
            if before_timestamp:
                endpoint += f"&endTime={before_timestamp}"
            ids = self.requester.make_request(is_v5=True, endpoint_url=endpoint)
            ids = ids if isinstance(ids, list) else []
            self._cache_kpis_ids[puuid] = ids
            return puuid, ids

        # Fetch KPI match IDs concurrently per participant
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures_ids = {executor.submit(_fetch_kpis_ids, puuid): puuid for puuid in participants}
            for future in as_completed(futures_ids):
                puuid, kpis_match_ids = future.result()
                if not kpis_match_ids:
                    self.logger.debug(f"No KPI match IDs for {puuid}")
                    kpis_data[puuid] = []
                    continue

                # Fetch match details for KPI IDs with bounded concurrency
                def _fetch_match_kpi(kpis_match_id: str) -> Optional[dict[str, Any]]:
                    match_details_endpoint = f"/lol/match/v5/matches/{kpis_match_id}"
                    match_details = self.requester.make_request(is_v5=True, endpoint_url=match_details_endpoint)
                    if not isinstance(match_details, dict) or not match_details.get("info"):
                        self.logger.debug(f"Invalid match details for {kpis_match_id}")
                        return None
                    participant_data = next(
                        (p for p in match_details.get("info").get("participants", []) if p.get("puuid") == puuid), None
                    )
                    if not participant_data:
                        return None
                    game_duration_seconds = match_details.get("info").get("gameDuration", 0)
                    return {
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
                        "gameDuration": game_duration_seconds,
                    }

                kpis_list: list[dict[str, Any]] = []
                with ThreadPoolExecutor(max_workers=min(self.max_workers, 6)) as exec_matches:
                    futures_kpi = {exec_matches.submit(_fetch_match_kpi, mid): mid for mid in kpis_match_ids}
                    for f in as_completed(futures_kpi):
                        record = f.result()
                        if record:
                            kpis_list.append(record)
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

        # Defensive helper to avoid index errors when upstream data is incomplete (e.g., live games)
        def _safe_pick(picks: list[Any], idx: int) -> Any:
            return picks[idx] if isinstance(picks, list) and len(picks) > idx else None

        match_record = {
            "match_id": match_id,
            "queue_id": match_pre_features.get("queue_id"),
            "game_version": match_pre_features.get("game_version"),
            "game_duration": match_pre_features.get("game_duration"),
            "team1_win": match_pre_features.get("match_outcome"),
            # bans
            "team1_ban1": (
                match_pre_features.get("team1_bans")[0] if len(match_pre_features.get("team1_bans")) > 0 else None
            ),
            "team1_ban2": (
                match_pre_features.get("team1_bans")[1] if len(match_pre_features.get("team1_bans")) > 1 else None
            ),
            "team1_ban3": (
                match_pre_features.get("team1_bans")[2] if len(match_pre_features.get("team1_bans")) > 2 else None
            ),
            "team1_ban4": (
                match_pre_features.get("team1_bans")[3] if len(match_pre_features.get("team1_bans")) > 3 else None
            ),
            "team1_ban5": (
                match_pre_features.get("team1_bans")[4] if len(match_pre_features.get("team1_bans")) > 4 else None
            ),
            "team2_ban1": (
                match_pre_features.get("team2_bans")[0] if len(match_pre_features.get("team2_bans")) > 0 else None
            ),
            "team2_ban2": (
                match_pre_features.get("team2_bans")[1] if len(match_pre_features.get("team2_bans")) > 1 else None
            ),
            "team2_ban3": (
                match_pre_features.get("team2_bans")[2] if len(match_pre_features.get("team2_bans")) > 2 else None
            ),
            "team2_ban4": (
                match_pre_features.get("team2_bans")[3] if len(match_pre_features.get("team2_bans")) > 3 else None
            ),
            "team2_ban5": (
                match_pre_features.get("team2_bans")[4] if len(match_pre_features.get("team2_bans")) > 4 else None
            ),
            # picks
            "team1_pick_top": _safe_pick(match_pre_features.get("team1_picks"), 0),
            "team1_pick_jungle": _safe_pick(match_pre_features.get("team1_picks"), 1),
            "team1_pick_mid": _safe_pick(match_pre_features.get("team1_picks"), 2),
            "team1_pick_adc": _safe_pick(match_pre_features.get("team1_picks"), 3),
            "team1_pick_support": _safe_pick(match_pre_features.get("team1_picks"), 4),
            "team2_pick_top": _safe_pick(match_pre_features.get("team2_picks"), 0),
            "team2_pick_jungle": _safe_pick(match_pre_features.get("team2_picks"), 1),
            "team2_pick_mid": _safe_pick(match_pre_features.get("team2_picks"), 2),
            "team2_pick_adc": _safe_pick(match_pre_features.get("team2_picks"), 3),
            "team2_pick_support": _safe_pick(match_pre_features.get("team2_picks"), 4),
        }
        return match_record

    def create_player_history_record(
        self,
        match_id: str,
        puuid: str,
        team_id: int,
        role: str,
        champion_id: int,
        summoner_level: int,
        champion_mastery: dict[str, Any],
        champion_total_mastery_score: int,
        rank_queue_data: dict[str, Any],
        kpis_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
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
            "match_id": match_id,
            "puuid": puuid,
            "team_id": team_id,
            "role": role,
            "champion_id": champion_id,
            # state of the summoner
            "summoner_level": summoner_level,
            "champion_mastery_level": champion_mastery.get("championLevel") if champion_mastery else None,
            "champion_total_mastery_score": champion_total_mastery_score,
            "ranked_tier": (
                rank_queue_data.get(puuid + "_solo", {}).get("tier") if rank_queue_data.get(puuid + "_solo") else None
            ),
            "ranked_rank": (
                rank_queue_data.get(puuid + "_solo", {}).get("rank") if rank_queue_data.get(puuid + "_solo") else None
            ),
            "ranked_league_points": (
                rank_queue_data.get(puuid + "_solo", {}).get("leaguePoints")
                if rank_queue_data.get(puuid + "_solo")
                else None
            ),
            # pings
            "all_in_pings": self._calculate_average(kpis_data, "allInPings"),
            "assist_me_pings": self._calculate_average(kpis_data, "assistmePings"),
            "command_pings": self._calculate_average(kpis_data, "commandPings"),
            "enemy_missing_pings": self._calculate_average(kpis_data, "enemyMissingPings"),
            "enemy_vision_pings": self._calculate_average(kpis_data, "enemyVisionPings"),
            "hold_pings": self._calculate_average(kpis_data, "holdPings"),
            "get_back_pings": self._calculate_average(kpis_data, "getbackPings"),
            "need_vision_pings": self._calculate_average(kpis_data, "needVisionPings"),
            "on_my_way_pings": self._calculate_average(kpis_data, "onMyWayPings"),
            "push_pings": self._calculate_average(kpis_data, "pushPings"),
            "vision_cleared_pings": self._calculate_average(kpis_data, "visionClearedPings"),
            # wards
            "average_wards_placed": self._calculate_average(kpis_data, "wardsPlaced"),
            "average_ward_kills": self._calculate_average(kpis_data, "wardKills"),
            "average_sight_wards_bought": self._calculate_average(kpis_data, "sightWardsBoughtInGame"),
            "average_detector_wards_placed": self._calculate_average(kpis_data, "detectorWardsPlaced"),
            # kill streaks
            "average_killing_sprees": self._calculate_average(kpis_data, "killingSprees"),
            "average_largest_killing_spree": self._calculate_average(kpis_data, "largeKillingSpree"),
            "average_largest_multi_kill": self._calculate_average(kpis_data, "largestMultiKill"),
            "average_double_kills": self._calculate_average(kpis_data, "doubleKills"),
            "average_triple_kills": self._calculate_average(kpis_data, "tripleKills"),
            "average_quadra_kills": self._calculate_average(kpis_data, "quadraKills"),
            "average_penta_kills": self._calculate_average(kpis_data, "pentakills"),
            # objectives
            "average_baron_kills": self._calculate_average(kpis_data, "baronKills"),
            "average_dragon_kills": self._calculate_average(kpis_data, "dragonKills"),
            "average_inhibitor_kills": self._calculate_average(kpis_data, "inhibitorKills"),
            "average_inhibitor_takedowns": self._calculate_average(kpis_data, "inhibitorTakedowns"),
            "average_inhibitors_lost": self._calculate_average(kpis_data, "inhibitorsLost"),
            "average_turret_kills": self._calculate_average(kpis_data, "turretKills"),
            "average_turret_takedowns": self._calculate_average(kpis_data, "turretTakedowns"),
            "average_turrets_lost": self._calculate_average(kpis_data, "turretsLost"),
            "average_objectives_stolen": self._calculate_average(kpis_data, "objectivesStolen"),
            "average_objectives_stolen_assists": self._calculate_average(kpis_data, "objectivesStolenAssists"),
            # damage stats
            "average_total_damage_dealt_to_champions": self._calculate_average(
                kpis_data, "totalDamageDealtToChampions"
            ),
            "average_physical_damage_dealt_to_champions": self._calculate_average(
                kpis_data, "physicalDamageDealtToChampions"
            ),
            "average_magic_damage_dealt_to_champions": self._calculate_average(
                kpis_data, "magicDamageDealtToChampions"
            ),
            "average_true_damage_dealt_to_champions": self._calculate_average(kpis_data, "trueDamageDealtToChampions"),
            "average_total_damage_taken": self._calculate_average(kpis_data, "totalDamageTaken"),
            "average_damage_self_mitigated": self._calculate_average(kpis_data, "damageSelfMitigated"),
            "average_total_heal": self._calculate_average(kpis_data, "totalHeal"),
            "average_total_heals_on_teammates": self._calculate_average(kpis_data, "totalHealsOnTeammates"),
            # combat stats
            "average_kills": self._calculate_average(kpis_data, "kills"),
            "average_deaths": self._calculate_average(kpis_data, "deaths"),
            "average_assists": self._calculate_average(kpis_data, "assists"),
            "kda_ratio": self._calculate_kda_ratio(kpis_data),
            "win_rate": self._calculate_win_rate(kpis_data),
            "average_cs_per_minute": self._calculate_per_minute(kpis_data, "totalMinionsKilled"),
            "average_kills_per_minute": self._calculate_per_minute(kpis_data, "kills"),
            "average_deaths_per_minute": self._calculate_per_minute(kpis_data, "deaths"),
            "average_assists_per_minute": self._calculate_per_minute(kpis_data, "assists"),
            # cs
            "average_total_minions_killed": self._calculate_average(kpis_data, "totalMinionsKilled"),
            "average_neutral_minions_killed": self._calculate_average(kpis_data, "neutralMinionsKilled"),
            # gold
            "average_gold_earned": self._calculate_average(kpis_data, "goldEarned"),
            "average_gold_spent": self._calculate_average(kpis_data, "goldSpent"),
            "average_items_purchased": self._calculate_average(kpis_data, "itemsPurchased"),
            # cc
            "average_time_ccing_others": self._calculate_average(kpis_data, "timeCCingOthers"),
            "average_total_time_cc_dealt": self._calculate_average(kpis_data, "totalTimeCCDealt"),
            # survival
            "average_longest_time_spent_living": self._calculate_average(kpis_data, "longestTimeSpentLiving"),
            "average_total_time_spent_dead": self._calculate_average(kpis_data, "totalTimeSpentDead"),
            # others
            "average_vision_score": self._calculate_average(kpis_data, "visionScore"),
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
