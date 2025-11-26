from data_extraction.requester import Requester

from dotenv import load_dotenv
import os

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