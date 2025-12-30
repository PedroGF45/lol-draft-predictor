import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from data_extraction.requester import Requester
from data_extraction.schemas import MATCHES_SCHEMA, PLAYERS_SCHEMA
from helpers.checkpoint import load_checkpoint, save_checkpoint
from helpers.master_data_registry import MasterDataRegistry
from helpers.parquet_handler import ParquetHandler
from tqdm import tqdm


class InvalidPatientZeroError(Exception):
    """Raised when the provided patient zero account is invalid or not found."""


class DataMiner:
    """
    Orchestrates breadth-first search (BFS) discovery of matches and players from Riot API.

    Implements a resumable crawl through League of Legends match history and participant data,
    starting from a patient-zero summoner and expanding outward. Stores discovered players and
    matches while maintaining checkpoint state for recovery from interruptions.

    Attributes:
        logger (logging.Logger): Logger instance for tracking discovery progress.
        requester (Requester): API client for making League of Legends API calls.
        raw_data_path (str): Directory path for storing checkpoint and parquet files.
        patient_zero_game_name (str): Summoner name of the starting player.
        patient_zero_tag_line (str): Tag line (e.g., 'NA1') of the starting player.
        players_queue (deque): FIFO queue of player PUUIDs awaiting match discovery.
        seen_players (set): Set of already-processed player PUUIDs (prevents duplicate work).
        seen_matches (set): Set of already-processed match IDs (prevents duplicate work).
    """

    def __init__(
        self,
        logger: logging.Logger,
        requester: Requester,
        parquet_handler: ParquetHandler,
        raw_data_path: str,
        patient_zero_game_name: str,
        patient_zero_tag_line: str,
        checkpoint_loading_path: str = None,
        checkpoint_save_path: str = None,
        master_registry: Optional[MasterDataRegistry] = None,
        max_workers: int = 4,
    ) -> None:
        """
        Initialize DataMiner with a patient zero summoner and optional checkpoint recovery.

        Validates the patient zero summoner exists in the API, initializes BFS queues, and optionally
        restores crawl state from a checkpoint file (for resumable discovery).

        Args:
            logger (logging.Logger): Logger instance for tracking discovery progress.
            requester (Requester): Configured Requester for API calls.
            raw_data_path (str): Directory path for storing match/player parquet output files.
            patient_zero_game_name (str): Summoner name of the starting player.
            patient_zero_tag_line: (str): Tag line of the starting player (e.g., 'NA1').
            checkpoint_loading_path (str, optional): Path to checkpoint file for recovery. If None, starts fresh.
            checkpoint_save_path (str, optional): Directory path for saving new checkpoints. Required to save checkpoints.

        Raises:
            InvalidPatientZeroError: If patient_zero summoner is not found or API returns error.
        """

        self.logger = logger
        self.requester = requester
        self.parquet_handler = parquet_handler
        self.master_registry = master_registry
        self.max_workers = max_workers
        self.backoff_cooldown_counter = 0  # Counter to manage backoff cooldown (allow increase after N iterations)

        self.raw_data_path = raw_data_path
        self.checkpoint_save_path = checkpoint_save_path

        self.patient_zero_game_name = patient_zero_game_name
        self.patient_zero_tag_line = patient_zero_tag_line

        if checkpoint_loading_path is None:
            self.players_queue = deque()
            self.seen_players = set()
            self.seen_matches = set()
            self._cache_match_players: dict[str, List[str]] = {}
        else:
            checkpoint_state = load_checkpoint(logger=self.logger, path=checkpoint_loading_path)
            self.players_queue = checkpoint_state.get("players_queue")
            self.seen_players = checkpoint_state.get("seen_players")
            self.seen_matches = checkpoint_state.get("seen_matches")
            self.logger.info(
                f"Loaded:{len(self.players_queue)} for the players queue \n{len(self.seen_players)} for the players set \n{len(self.seen_matches)} for the matches set \n"
            )
            self._cache_match_players = {}

        if not self._is_patient_zero_valid():
            raise InvalidPatientZeroError(
                f"Invalid patient zero: '{self.patient_zero_game_name}#{self.patient_zero_tag_line}' not found or inaccessible"
            )

    def _is_patient_zero_valid(self) -> bool:
        """
        Validate that the patient zero summoner exists in the API and add them to discovery queue.

        Args:
            (none - uses self.patient_zero_game_name and self.patient_zero_tag_line)

        Returns:
            bool: True if patient zero summoner found and added to queue, False otherwise.
        """
        endpoint_url = (
            f"/riot/account/v1/accounts/by-riot-id/{self.patient_zero_game_name}/{self.patient_zero_tag_line}"
        )
        response = self.requester.make_request(is_v5=True, endpoint_url=endpoint_url)

        if response and response.get("puuid"):
            self.players_queue.append(response["puuid"])
            self.seen_players.add(response["puuid"])
            return True
        return False

    def start_search(
        self, search_mode: str = "players", target_number_of_players: int = 100, target_number_of_matches: int = 100
    ) -> None:
        """
        Execute breadth-first search discovery of matches and players.

        Expands outward from patient zero, discovering new matches and participants based on search mode,
        and continues until the target is reached. Periodically saves checkpoints for recovery.

        Args:
            search_mode (str): Type of search ('players', 'matches', or 'both'). Defaults to 'players'.
            target_number_of_players (int): Target number of unique players to discover. Defaults to 100.
            target_number_of_matches (int): Target number of unique matches to discover. Defaults to 100.

        Returns:
            None. Results stored in seen_players and seen_matches sets.
        """

        if search_mode not in ["players", "matches"]:
            self.logger.warning(
                f'Search mode should be "players", "matches" or "both" but got {search_mode}. Sticking to default "players" search.'
            )
            search_mode = "players"

        number_of_matches = 0
        if search_mode == "matches":
            number_of_matches = 100
        elif search_mode == "both":
            number_of_matches = 50
        else:
            number_of_matches = 20

        start = time.time()

        # Setup tqdm progress bar based on search mode
        if search_mode == "matches":
            target_total = target_number_of_matches
            current_count = len(self.seen_matches)
            desc = "Discovering matches"
        else:  # default to players
            target_total = target_number_of_players
            current_count = len(self.seen_players)
            desc = "Discovering players"

        pbar = tqdm(total=target_total, initial=min(current_count, target_total), desc=desc, unit="items")

        # Calculate checkpoint interval based on target size
        if target_total < 1000:
            checkpoint_interval = 100
        elif target_total < 10000:
            checkpoint_interval = 1000
        else:
            checkpoint_interval = 10000

        last_checkpoint_count = current_count  # Track last checkpoint count
        self.logger.info(f"Checkpoint interval set to {checkpoint_interval} (target: {target_total})")

        while len(self.players_queue) > 0 and not self._has_reached_target(
            mode=search_mode, target_players=target_number_of_players, target_matches=target_number_of_matches
        ):

            self.logger.info(
                f"Number of current players: {len(self.seen_players)}\n Number of current matches: {len(self.seen_matches)}"
            )

            player_to_use = self.players_queue.popleft()
            matches_of_player = self.get_last_matches(puuid=player_to_use, number_of_matches=number_of_matches)

            # Process unseen matches concurrently to speed up discovery
            unseen_matches = [m for m in matches_of_player if m not in self.seen_matches]
            for m in unseen_matches:
                self.seen_matches.add(m)

            def _fetch_match_players(mid: str) -> tuple[str, List[str]]:
                if mid in self._cache_match_players:
                    return mid, self._cache_match_players[mid]
                players = self.get_players_from_match(match_id=mid)
                self._cache_match_players[mid] = players
                return mid, players

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_fetch_match_players, mid): mid for mid in unseen_matches}
                for future in as_completed(futures):
                    if self._has_reached_target(
                        mode=search_mode,
                        target_players=target_number_of_players,
                        target_matches=target_number_of_matches,
                    ):
                        break
                    _, new_players = future.result()
                    for player in new_players:
                        if self._has_reached_target(
                            mode=search_mode,
                            target_players=target_number_of_players,
                            target_matches=target_number_of_matches,
                        ):
                            break
                        if player not in self.seen_players:
                            self.seen_players.add(player)
                            self.players_queue.append(player)

            self.logger.info(
                f"Number of players after requests: {len(self.seen_players)}\n Number of matches after requests: {len(self.seen_matches)}"
            )

            # Update progress bar
            if search_mode == "matches":
                new_count = len(self.seen_matches)
            else:
                new_count = len(self.seen_players)
            # Clamp to target_total to avoid over-updating
            delta = max(0, min(new_count, target_total) - min(current_count, target_total))
            if delta > 0:
                pbar.update(delta)
                current_count = new_count
            pbar.set_postfix(players=len(self.seen_players), matches=len(self.seen_matches), workers=self.max_workers)

            # Only save checkpoint at intervals
            items_since_last_checkpoint = new_count - last_checkpoint_count
            if items_since_last_checkpoint >= checkpoint_interval:
                checkpoint_dict = {}
                checkpoint_dict["players_queue"] = self.players_queue
                checkpoint_dict["seen_players"] = self.seen_players
                checkpoint_dict["seen_matches"] = self.seen_matches

                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

                if self.checkpoint_save_path:
                    os.makedirs(self.checkpoint_save_path, exist_ok=True)
                    checkpoint_path = os.path.join(
                        self.checkpoint_save_path,
                        f"{timestamp}_{len(self.seen_players)}_players_{len(self.seen_matches)}_matches.pkl",
                    )
                else:
                    raise ValueError("checkpoint_save_path must be provided to save checkpoints")

                save_checkpoint(logger=self.logger, state=checkpoint_dict, path=checkpoint_path)
                last_checkpoint_count = new_count
                self.logger.info(f"Checkpoint saved at {new_count} items")

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
                    self.backoff_cooldown_counter >= 30 and self.max_workers < 4
                ):  # Increase after 30 successful iterations
                    self.max_workers = min(4, self.max_workers + 1)
                    self.logger.info(f"Rate limit recovery: increased max_workers to {self.max_workers}")
                    self.backoff_cooldown_counter = 0

        players_dataframe = self.convert_to_dataframe(set_to_save=self.seen_players, mode="players")
        matches_dataframe = self.convert_to_dataframe(set_to_save=self.seen_matches, mode="matches")
        # Filter through master registry if available to avoid duplicates
        if self.master_registry:
            self.logger.info(f"Filtering {len(matches_dataframe)} matches through master registry")
            # Note: DataMiner only has match_ids, not game_versions yet
            # The registry will track these matches, and MatchFetcher will add game_version
            collection_metadata = {
                "source": "data_miner",
                "search_mode": search_mode,
                "patient_zero": f"{self.patient_zero_game_name}#{self.patient_zero_tag_line}",
            }
            # We'll pass game_version as 'pending' here since we don't have it yet
            matches_dataframe["game_version"] = "pending"
            matches_dataframe, duplicates = self.master_registry.register_matches(
                matches_dataframe, collection_metadata=collection_metadata
            )
            self.logger.info(
                f"After registry filter: {len(matches_dataframe)} new matches, {duplicates} duplicates skipped"
            )

        # suffix with detailed timestamp and number of players/matches saved
        current_date = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        player_prefix = f"{current_date}_{len(players_dataframe)}_players"
        player_data_path = os.path.join(self.raw_data_path, f"exploration\\players\\{player_prefix}.parquet")
        self.parquet_handler.write_parquet(data=players_dataframe, file_path=player_data_path)

        match_prefix = f"{current_date}_{len(matches_dataframe)}_matches"
        match_data_path = os.path.join(self.raw_data_path, f"exploration\\matches\\{match_prefix}.parquet")
        self.parquet_handler.write_parquet(data=matches_dataframe, file_path=match_data_path)

        # Ensure progress bar completes and closes
        try:
            # Force to 100% if we reached or exceeded the target
            if current_count < target_total:
                pbar.update(target_total - current_count)
        finally:
            pbar.close()

        end = time.time()
        self.logger.info(
            f"Players length is {len(self.players_queue)} and set players length is {len(self.seen_players)}"
        )
        self.logger.info(f"Matches length is {len(self.seen_matches)}")
        self.logger.info(f"It took {end - start} seconds")

    def get_last_matches(self, puuid: str, number_of_matches: int = 100) -> List[str]:
        """
        Fetch the last N match IDs for a given player.

        Args:
            puuid (str): The player's PUUID.
            number_of_matches (int): Number of recent matches to fetch. Defaults to 100.

        Returns:
            List[str]: List of match IDs. Returns empty list if API request fails.
        """
        endpoint_url = f"/lol/match/v5/matches/by-puuid/{puuid}/ids?count={number_of_matches}"
        response = self.requester.make_request(is_v5=True, endpoint_url=endpoint_url)

        if response:
            return response
        self.logger.warning(f"Matches weren't fetch for player with puuid of {puuid}")
        return []

    def get_players_from_match(self, match_id: str) -> List[str]:
        """
        Extract all participant PUUIDs from a match.

        Args:
            match_id (str): The ID of the match.

        Returns:
            List[str]: List of participant PUUIDs. Returns empty list if API request fails.
        """
        endpoint_url = f"/lol/match/v5/matches/{match_id}"
        response = self.requester.make_request(is_v5=True, endpoint_url=endpoint_url)

        if response and response.get("metadata").get("participants"):
            return response.get("metadata").get("participants")
        self.logger.warning(f"Players weren't fetch for the match with the id of {match_id}")
        return []

    def _has_reached_target(self, mode: str, target_players: int, target_matches: int) -> bool:
        """
        Check if discovery target is reached based on search mode.

        Args:
            mode (str): Search mode ('players', 'matches', or 'both').
            target_players (int): Target number of players to discover.
            target_matches (int): Target number of matches to discover.

        Returns:
            bool: True if the target for the given mode has been reached, False otherwise.
        """
        if mode == "players":
            return len(self.seen_players) >= target_players
        elif mode == "matches":
            return len(self.seen_matches) >= target_matches
        else:
            raise ValueError(f"Unknown search_mode: {mode}")

    def convert_to_dataframe(self, set_to_save: set, mode: str) -> pd.DataFrame:
        """
        Convert a set of player PUUIDs or match IDs to a pandas DataFrame.

        Args:
            set_to_save (set): Set of strings (PUUIDs or match IDs) to convert.
            mode (str): Type of data ('players' for PUUIDs or 'matches' for match IDs).

        Returns:
            pd.DataFrame: DataFrame with single column ('puuid' or 'match_id'). Returns None if conversion fails.
        """

        if set_to_save is None:
            self.logger.error("No set provided. Unable to convert dataframe to parquet.")
            return None

        try:
            columns = "puuid" if mode == "players" else "match_id"
            dataframe = pd.DataFrame(list(set_to_save), columns=[columns])

        except Exception as e:
            self.logger.error(f"Error trying to create a pandas Dataframe: {e}")
            return None

        return dataframe
