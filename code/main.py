from data_extraction.requester import Requester
from data_extraction.data_miner import DataMiner, InvalidPatientZeroError
from data_extraction.match_fetcher import MatchFetcher
from data_preparation.data_cleaner import DataCleaner
from helpers.parquet_handler import ParquetHandler
from helpers.champion_ids import fetch_latest_champion_ids

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
data_path = "F:\\Code\\lol-draft-predictor\\data"
checkpoint_file_path = "F:\\Code\\lol-draft-predictor\\data\\pickle\\checkpoint.pkl"
parquet_handler = ParquetHandler(logger=logger)

""" # Phase 1: Discover matches (uncomment to run discovery)
data_miner = DataMiner(logger=logger, 
                       requester=requester, 
                        raw_data_path=data_path, 
                       patient_zero_game_name=game_name, 
                        patient_zero_tag_line=tag_line)
response = data_miner.start_search(search_mode="matches", 
                                          target_number_of_players=1000,
                                           target_number_of_matches=1000) """

# Phase 2: Fetch and enrich matches
parquet_file_path = "F:\\Code\\lol-draft-predictor\\data\\matches_id\\1000_games.parquet"
checkpoint_path = os.path.join(data_path, "pickle", "match_fetcher_checkpoint.pkl")
match_fetcher = MatchFetcher(requester=requester, logger=logger, parquet_handler=parquet_handler, dataframe_target_path=data_path, checkpoint_loading_path=checkpoint_path)
match_fetcher.fetch_match_data(parquet_path=parquet_file_path, match_limit=10)



#data_cleaner = DataCleaner(requester=requester, logger=logger, parquet_handler=parquet_handler)
#data_cleaner.clean_data(raw_data_path=parquet_file_path, cleaned_data_path="F:\\Code\\lol-draft-predictor\\data\\cleaned_matches.parquet", mode="players")

