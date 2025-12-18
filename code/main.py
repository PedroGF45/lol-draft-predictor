from data_extraction.requester import Requester
from data_extraction.data_miner import DataMiner, InvalidPatientZeroError
from data_extraction.match_fetcher import MatchFetcher
from data_preparation.data_cleaner import DataCleaner
from data_preparation.data_handler import DataHandler
from feature_engineering.feature_engineer import FeatureEngineer
from feature_engineering.dimension_reducer import DimensionReducer
from helpers.parquet_handler import ParquetHandler
from helpers.champion_ids import fetch_latest_champion_ids
from visualization.data_visualizer import DataVisualizer

from dotenv import load_dotenv
import os

import logging

logging.basicConfig(
    level=logging.INFO,  # use DEBUG to see response snippets
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

load_dotenv()

RIOT_API_KEY = os.getenv("RIOT_API_KEY")
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH")
EXPLORATION_DATA_PATH = os.getenv("EXPLORATION_DATA_PATH")
CLEANED_DATA_PATH = os.getenv("CLEANED_DATA_PATH")
PREPROCESSED_DATA_PATH = os.getenv("PREPROCESSED_DATA_PATH")
FEATURE_ENGINEER_DATA_PATH = os.getenv("FEATURE_ENGINEER_DATA_PATH")
VISUALIZATIONS_PATH = os.getenv("VISUALIZATIONS_PATH")
MATCHES_PARQUET_PATH = os.getenv("MATCHES_PARQUET_PATH")
PLAYERS_PARQUET_PATH = os.getenv("PLAYERS_PARQUET_PATH")
RANDOM_SEED = 42

print("RIOT_API_KEY: ", RIOT_API_KEY)

region_v4 = "euw1"
region_v5 = "europe"
base_url_v4 = f'https://{region_v4}.api.riotgames.com'
base_url_v5 = f'https://{region_v5}.api.riotgames.com'
headers = {
    "X-Riot-Token": RIOT_API_KEY
}

requester = Requester(base_url_v4=base_url_v4, base_url_v5=base_url_v5, headers=headers, logger=logger)

# get puuid of summoner
game_name = "PedroGF45"
tag_line = "EUW"
parquet_handler = ParquetHandler(logger=logger, random_state=RANDOM_SEED)

""" # Phase 1: Discover matches (uncomment to run discovery)
data_miner = DataMiner(logger=logger, 
                       parquet_handler=parquet_handler,
                       requester=requester, 
                        raw_data_path=DATA_PATH, 
                       patient_zero_game_name=game_name, 
                        patient_zero_tag_line=tag_line)
response = data_miner.start_search(search_mode="matches", 
                                          target_number_of_players=1000,
                                           target_number_of_matches=100) 

# Phase 2: Fetch and enrich matches

match_fetcher = MatchFetcher(requester=requester, 
                             logger=logger, 
                             parquet_handler=parquet_handler, 
                             dataframe_target_path=DATA_PATH, 
                             checkpoint_loading_path=CHECKPOINT_PATH,
                             load_percentage=100,
                             random_state=RANDOM_SEED)
match_fetcher.fetch_match_data(parquet_path=PARQUET_PATH, match_limit_per_player=10)
"""

data_handler = DataHandler(logger=logger, parquet_handler=parquet_handler, target_feature="team1_win", random_state=RANDOM_SEED)
data_handler.load_splits(input_dir=CLEANED_DATA_PATH, timestamp="20251218_210913")


""" data_handler.join_match_and_player_data(
    match_parquet_path=MATCHES_PARQUET_PATH,
    player_parquet_path=PLAYERS_PARQUET_PATH
)

data_handler.split_data()

data_cleaner = DataCleaner(requester=requester, logger=logger, data_handler=data_handler, parquet_handler=parquet_handler)
data_cleaner.clean_data(output_path=CLEANED_DATA_PATH) 


data_visualizer = DataVisualizer(logger=logger, parquet_handler=parquet_handler, random_state=RANDOM_SEED)
data_visualizer.perform_eda(
    plots="feature_importance",
    prefix="player_history",
    figsize=(12, 10),
    cmap="viridis",
    annot=True,
    fmt=".2f",
    sample_size=10000,
    dpi=100,
    n_estimators=5,
    n_repeats=2,
    data_handler=data_handler,
    data_output_path="F:\\Code\\lol-draft-predictor\\data\\visualizations"
)
"""

feature_engineer = FeatureEngineer(logger=logger, parquet_handler=parquet_handler, random_state=RANDOM_SEED)
feature_engineer.perform_feature_engineering(data_handler=data_handler, output_dir=FEATURE_ENGINEER_DATA_PATH)

dimension_reducer = DimensionReducer(logger=logger, parquet_handler=parquet_handler)
dimension_reducer.perform_dimension_reduction(
    data_handler=data_handler,
    pca_plot_dir=VISUALIZATIONS_PATH,
    output_dir=FEATURE_ENGINEER_DATA_PATH
)

data_visualizer = DataVisualizer(logger=logger, parquet_handler=parquet_handler, random_state=RANDOM_SEED)
data_visualizer.perform_eda(
    prefix="player_history",
    figsize=(12, 10),
    cmap="viridis",
    annot=True,
    fmt=".2f",
    sample_size=10000,
    dpi=100,
    n_estimators=5,
    n_repeats=2,
    data_handler=data_handler,
    data_output_path=VISUALIZATIONS_PATH
)