from helpers.parquet_handler import ParquetHandler
from logging import Logger
from sklearn.model_selection import train_test_split
import pandas as pd

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

        self.categorical_features = {}
        self.numerical_features = {}

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

    def load_data(self, parquet_path: str, load_percentage: float = 1.0):
        self.logger.info(f"Loading data from {parquet_path}")
        self.dataframe = self.parquet_handler.read_parquet(parquet_path, load_percentage)
        self.logger.info(f"Data loaded with shape: {self.dataframe.shape}")

    def split_data(self):

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

        self.logger.info(f"Training data shape: {self.data_train.shape}, Testing data shape: {self.data_test.shape}")

