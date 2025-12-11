import pandas as pd
import os
from logging import Logger


class ParquetHandler:
    def __init__(self, logger: Logger) -> None:
        self.logger = logger

    def read_parquet(self, file_path: str) -> pd.DataFrame:
        try:
            data = pd.read_parquet(file_path)
            self.logger.info(f"Successfully read parquet file from {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"Error reading parquet file from {file_path}: {e}")
            raise

    def write_parquet(self, data: pd.DataFrame, file_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            data.to_parquet(file_path, index=False)
            self.logger.info(f"Successfully wrote parquet file to {file_path}")
        except Exception as e:
            self.logger.error(f"Error writing parquet file to {file_path}: {e}")
            raise
