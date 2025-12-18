from helpers.parquet_handler import ParquetHandler
from logging import Logger

class FeatureEngineer:
    def __init__(self, logger: Logger, parquet_handler: ParquetHandler, random_state: int = 42):
        self.logger = logger
        self.parquet_handler = parquet_handler
        self.random_state = random_state

    def perform_feature_engineering(self, input_parquet_path: str, output_parquet_path: str):
        self.logger.info(f"Starting feature engineering on {input_parquet_path}")
        
        # Load data
        dataframe = self.parquet_handler.read_parquet(input_parquet_path)

        self.generate_features(dataframe)
        self.one_hot_encode(dataframe)
        self.transform_numerical_features(dataframe)
        self.normalize_features(dataframe)
        
        self.parquet_handler.write_parquet(dataframe, output_parquet_path)
        self.logger.info(f"Feature engineering completed. Data saved to {output_parquet_path}")