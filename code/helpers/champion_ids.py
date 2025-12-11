def fetch_latest_champion_ids():
    """
    Retrieve the current set of valid champion IDs from Data Dragon.

    Returns:
        set[int]: Set of champion numeric IDs for the latest patch.
    """
    import requests

    url = "https://ddragon.leagueoflegends.com/api/versions.json"
    response = requests.get(url)
    versions = response.json()
    latest_version = versions[0]

    url = f"https://ddragon.leagueoflegends.com/cdn/{latest_version}/data/en_US/champion.json"
    response = requests.get(url)
    champions_data = response.json()

    champion_ids = {int(champion_info['key']) for _, champion_info in champions_data['data'].items()}

    return champion_ids