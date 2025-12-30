"""
Master Data Registry - Single Source of Truth for Match Data

This module provides a centralized registry to track all match data collected,
ensuring uniqueness, preventing duplicates, and avoiding data leakage across
incremental data collection sessions.

Key Features:
- Composite key tracking (match_id + game_version) for true uniqueness
- Persistent state across sessions using pickle
- Train/test split tracking to prevent data leakage
- Incremental data collection support
- Query capabilities for duplicate detection
"""

import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


class MasterDataRegistry:
    """
    Maintains a persistent registry of all matches collected to ensure data integrity.

    This class serves as the single source of truth for tracking which matches have been
    collected and how they've been split for training/testing. It prevents:
    - Duplicate match collection
    - Data leakage between train/test sets
    - Inconsistent data states across incremental fetches

    Attributes:
        registry_path (str): Path to the persistent registry file
        logger (logging.Logger): Logger for status messages
        match_registry (Dict[Tuple[str, str], Dict]): Maps (match_id, game_version) to metadata
        train_matches (Set[Tuple[str, str]]): Set of (match_id, game_version) in training set
        test_matches (Set[Tuple[str, str]]): Set of (match_id, game_version) in test set
        collection_history (List[Dict]): History of data collection sessions
    """

    def __init__(self, registry_path: str, logger: logging.Logger):
        """
        Initialize or load existing master data registry.

        Args:
            registry_path (str): Path where registry pickle file should be stored
            logger (logging.Logger): Logger instance for tracking operations
        """
        self.registry_path = registry_path
        self.logger = logger

        # Core data structures
        self.match_registry: Dict[Tuple[str, str], Dict] = {}  # (match_id, game_version) -> metadata
        self.train_matches: Set[Tuple[str, str]] = set()
        self.test_matches: Set[Tuple[str, str]] = set()
        self.collection_history: List[Dict] = []

        # Metadata
        self.created_at: Optional[datetime] = None
        self.last_updated: Optional[datetime] = None
        self.version: str = "1.0.0"

        # Load existing registry if it exists
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        """Load existing registry from disk or create a new one."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "rb") as f:
                    state = pickle.load(f)
                    self.match_registry = state.get("match_registry", {})
                    self.train_matches = state.get("train_matches", set())
                    self.test_matches = state.get("test_matches", set())
                    self.collection_history = state.get("collection_history", [])
                    self.created_at = state.get("created_at")
                    self.last_updated = state.get("last_updated")
                    self.version = state.get("version", "1.0.0")

                self.logger.info(f"Loaded existing registry with {len(self.match_registry)} matches")
                self.logger.info(f"  - Train set: {len(self.train_matches)} matches")
                self.logger.info(f"  - Test set: {len(self.test_matches)} matches")
            except Exception as e:
                self.logger.error(f"Failed to load registry: {e}")
                self.logger.warning("Initializing fresh registry")
                self._initialize_fresh()
        else:
            self._initialize_fresh()

    def _initialize_fresh(self) -> None:
        """Initialize a fresh registry."""
        self.match_registry = {}
        self.train_matches = set()
        self.test_matches = set()
        self.collection_history = []
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.logger.info("Initialized fresh data registry")

    def save(self) -> None:
        """Persist the current registry state to disk."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)

            self.last_updated = datetime.now()

            state = {
                "match_registry": self.match_registry,
                "train_matches": self.train_matches,
                "test_matches": self.test_matches,
                "collection_history": self.collection_history,
                "created_at": self.created_at,
                "last_updated": self.last_updated,
                "version": self.version,
            }

            # Atomic write using temp file
            temp_path = f"{self.registry_path}.tmp"
            with open(temp_path, "wb") as f:
                pickle.dump(state, f)

            os.replace(temp_path, self.registry_path)
            self.logger.info(f"Registry saved to {self.registry_path}")

        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
            raise

    def register_matches(
        self, matches_df: pd.DataFrame, collection_metadata: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, int]:
        """
        Register new matches and return only the unique ones not already in registry.

        Args:
            matches_df (pd.DataFrame): DataFrame with at least 'match_id' and 'game_version' columns
            collection_metadata (Dict, optional): Metadata about this collection session

        Returns:
            Tuple[pd.DataFrame, int]: (DataFrame of new unique matches, count of duplicates filtered)
        """
        if "match_id" not in matches_df.columns:
            raise ValueError("DataFrame must contain 'match_id' column")

        # Create game_version if it doesn't exist (for backward compatibility)
        if "game_version" not in matches_df.columns:
            self.logger.warning("'game_version' not found. Using 'unknown' as default.")
            matches_df["game_version"] = "unknown"

        initial_count = len(matches_df)
        new_matches = []
        duplicates = 0

        for _, row in matches_df.iterrows():
            match_key = (str(row["match_id"]), str(row["game_version"]))

            if match_key not in self.match_registry:
                # New match - register it
                self.match_registry[match_key] = {
                    "match_id": match_key[0],
                    "game_version": match_key[1],
                    "first_seen": datetime.now(),
                    "collection_session": len(self.collection_history),
                    "split_assigned": None,  # Will be assigned during train/test split
                }
                new_matches.append(row)
            else:
                duplicates += 1

        # Record collection session
        session_info = {
            "timestamp": datetime.now(),
            "total_matches_provided": initial_count,
            "new_matches_added": len(new_matches),
            "duplicates_filtered": duplicates,
            "metadata": collection_metadata or {},
        }
        self.collection_history.append(session_info)

        self.logger.info(f"Match registration: {len(new_matches)} new, {duplicates} duplicates filtered")

        # Save registry after registration
        self.save()

        return pd.DataFrame(new_matches) if new_matches else pd.DataFrame(columns=matches_df.columns), duplicates

    def is_match_registered(self, match_id: str, game_version: str) -> bool:
        """
        Check if a match is already in the registry.

        Args:
            match_id (str): Match ID to check
            game_version (str): Game version to check

        Returns:
            bool: True if match exists in registry
        """
        return (str(match_id), str(game_version)) in self.match_registry

    def filter_new_matches(
        self, match_ids: List[str], game_versions: Optional[List[str]] = None
    ) -> List[Tuple[str, str]]:
        """
        Filter a list of match IDs to return only those not in registry.

        Args:
            match_ids (List[str]): List of match IDs to filter
            game_versions (List[str], optional): Corresponding game versions. If None, uses 'unknown'

        Returns:
            List[Tuple[str, str]]: List of (match_id, game_version) tuples for new matches
        """
        if game_versions is None:
            game_versions = ["unknown"] * len(match_ids)

        if len(match_ids) != len(game_versions):
            raise ValueError("match_ids and game_versions must have same length")

        new_matches = []
        for match_id, game_version in zip(match_ids, game_versions):
            match_key = (str(match_id), str(game_version))
            if match_key not in self.match_registry:
                new_matches.append(match_key)

        return new_matches

    def assign_splits(self, train_matches: Set[Tuple[str, str]], test_matches: Set[Tuple[str, str]]) -> None:
        """
        Assign matches to train or test splits. This should be called once after initial split.

        Args:
            train_matches (Set[Tuple[str, str]]): Set of (match_id, game_version) for training
            test_matches (Set[Tuple[str, str]]): Set of (match_id, game_version) for testing
        """
        self.train_matches = train_matches
        self.test_matches = test_matches

        # Update registry metadata
        for match_key in train_matches:
            if match_key in self.match_registry:
                self.match_registry[match_key]["split_assigned"] = "train"

        for match_key in test_matches:
            if match_key in self.match_registry:
                self.match_registry[match_key]["split_assigned"] = "test"

        self.logger.info(f"Split assignments: {len(train_matches)} train, {len(test_matches)} test")
        self.save()

    def get_train_match_keys(self) -> Set[Tuple[str, str]]:
        """Get all match keys assigned to training set."""
        return self.train_matches.copy()

    def get_test_match_keys(self) -> Set[Tuple[str, str]]:
        """Get all match keys assigned to test set."""
        return self.test_matches.copy()

    def validate_no_leakage(self) -> bool:
        """
        Validate that there's no overlap between train and test sets.

        Returns:
            bool: True if no leakage detected

        Raises:
            ValueError: If leakage is detected
        """
        overlap = self.train_matches.intersection(self.test_matches)
        if overlap:
            self.logger.error(f"Data leakage detected! {len(overlap)} matches in both train and test")
            raise ValueError(f"Data leakage: {len(overlap)} matches overlap between train/test sets")

        self.logger.info("No data leakage detected - train and test sets are disjoint")
        return True

    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the registry.

        Returns:
            Dict: Statistics about registered matches and splits
        """
        stats = {
            "total_matches": len(self.match_registry),
            "train_matches": len(self.train_matches),
            "test_matches": len(self.test_matches),
            "unassigned_matches": len(self.match_registry) - len(self.train_matches) - len(self.test_matches),
            "collection_sessions": len(self.collection_history),
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "version": self.version,
        }

        # Game version breakdown
        version_counts = {}
        for (_, version), _ in self.match_registry.items():
            version_counts[version] = version_counts.get(version, 0) + 1
        stats["game_versions"] = version_counts

        return stats

    def get_collection_summary(self) -> pd.DataFrame:
        """
        Get a summary of all collection sessions.

        Returns:
            pd.DataFrame: DataFrame with collection history
        """
        if not self.collection_history:
            return pd.DataFrame()

        summary_data = []
        for i, session in enumerate(self.collection_history):
            summary_data.append(
                {
                    "session_id": i,
                    "timestamp": session["timestamp"],
                    "total_provided": session["total_matches_provided"],
                    "new_added": session["new_matches_added"],
                    "duplicates_filtered": session["duplicates_filtered"],
                    "cumulative_total": sum(s["new_matches_added"] for s in self.collection_history[: i + 1]),
                }
            )

        return pd.DataFrame(summary_data)

    def export_master_dataset(self, output_path: str) -> None:
        """
        Export the complete registry as a CSV for inspection.

        Args:
            output_path (str): Path where to save the CSV file
        """
        if not self.match_registry:
            self.logger.warning("Registry is empty, nothing to export")
            return

        records = []
        for (match_id, game_version), metadata in self.match_registry.items():
            records.append(
                {
                    "match_id": match_id,
                    "game_version": game_version,
                    "first_seen": metadata["first_seen"],
                    "collection_session": metadata["collection_session"],
                    "split_assigned": metadata["split_assigned"],
                }
            )

        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Exported {len(df)} matches to {output_path}")

    def reset_splits(self) -> None:
        """
        Reset train/test split assignments. Use with caution!

        This should only be called if you need to re-split your data.
        """
        self.train_matches = set()
        self.test_matches = set()

        for match_key in self.match_registry:
            self.match_registry[match_key]["split_assigned"] = None

        self.logger.warning("All split assignments have been reset")
        self.save()

    def __repr__(self) -> str:
        """String representation of registry status."""
        return (
            f"MasterDataRegistry(total={len(self.match_registry)}, "
            f"train={len(self.train_matches)}, test={len(self.test_matches)}, "
            f"sessions={len(self.collection_history)})"
        )
