from data_extraction.requester import Requester

from dotenv import load_dotenv
import os

import logging

logging.basicConfig(
    level=logging.INFO,  # use DEBUG to see response snippets
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

load_dotenv()

RIOT_API_KEY = os.getenv("RIOT_API_KEY")

print("RIOT_API_KEY: ", RIOT_API_KEY)

region = "europe"
base_url = f'https://{region}.api.riotgames.com'
headers = {
    "X-Riot-Token": RIOT_API_KEY
}

requester = Requester(base_url=base_url, headers=headers)

# get puuid of summoner
game_name = "PedroGF45"
tag_line = "EUW"
end_point = f"/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
response = requester.make_request(endpoint_url=end_point)
print(response)