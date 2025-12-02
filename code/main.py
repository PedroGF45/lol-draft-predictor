from data_extraction.requester import Requester
from data_extraction.data_miner import DataMiner, InvalidPatientZeroError

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

region = "europe"
base_url = f'https://{region}.api.riotgames.com'
headers = {
    "X-Riot-Token": RIOT_API_KEY
}

requester = Requester(base_url=base_url, headers=headers, logger=logger)

# get puuid of summoner
game_name = "PedroGF45"
tag_line = "EUW"

data_miner = DataMiner(logger=logger, requester=requester, patient_zero_game_name=game_name, patient_zero_tag_line=tag_line)
response = data_miner.start_player_search(target_number_of_players=100)
