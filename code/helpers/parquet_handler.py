import pandas as pd
import os
from logging import Logger


class ParquetHandler:
    """
    Utility for reading and writing parquet files with optional subsampling.

    Attributes:
        logger (Logger): Logger for status and error messages.
        random_state (int): Seed used when sampling a percentage of rows.
    """
    def __init__(self, logger: Logger, random_state: int = 42) -> None:
        """
        Initialize the parquet handler.

        Args:
            logger (Logger): Logger instance for reporting progress/errors.
            random_state (int): Seed for deterministic sampling. Defaults to 42.
        """
        self.logger = logger
        self.random_state = random_state

    def read_parquet(self, file_path: str, load_percentage: float = 1.0) -> pd.DataFrame:
        """
        Read a parquet file and optionally sample a percentage of rows.

        Args:
            file_path (str): Path to the parquet file.
            load_percentage (float): Fraction of rows to load (0 < p <= 1). Defaults to 1.0.

        Returns:
            pd.DataFrame: Loaded (and possibly sampled) dataframe.

        Raises:
            Exception: Propagates any read/parquet errors after logging.
        """
        try:
            data = pd.read_parquet(file_path)
            self.logger.info(f"Successfully read parquet file from {file_path}")
            if 0 < load_percentage < 1.0:
                sample_size = int(len(data) * load_percentage)
                data = data.sample(n=sample_size, random_state=self.random_state).reset_index(drop=True)
                self.logger.info(f"Loaded {sample_size} records ({load_percentage*100}%) from the parquet file")
            return data
        except Exception as e:
            self.logger.error(f"Error reading parquet file from {file_path}: {e}")
            raise

    def write_parquet(self, data: pd.DataFrame, file_path: str) -> None:
        """
        Write a dataframe to parquet, creating parent directories if missing.

        Args:
            data (pd.DataFrame): Dataframe to persist.
            file_path (str): Destination parquet path.

        Raises:
            Exception: Propagates any write/parquet errors after logging.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            data.to_parquet(file_path, index=False)
            self.logger.info(f"Successfully wrote parquet file to {file_path}")
        except Exception as e:
            self.logger.error(f"Error writing parquet file to {file_path}: {e}")
            raise
