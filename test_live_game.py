"""
Test script for live game detection feature.

Usage:
    python test_live_game.py <game_name> <tag_line>
    
Example:
    python test_live_game.py Unefraise KARAP
"""

import sys
import os
import logging

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from dotenv import load_dotenv
from data_extraction.requester import Requester
from data_extraction.match_fetcher import MatchFetcher
from helpers.parquet_handler import ParquetHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def test_live_game_detection(game_name: str, tag_line: str):
    """Test the live game detection workflow."""
    
    # Initialize components
    riot_api_key = os.getenv("RIOT_API_KEY")
    if not riot_api_key:
        logger.error("RIOT_API_KEY not set in .env file")
        return
    
    region_v4 = "euw1"
    region_v5 = "europe"
    base_url_v4 = f"https://{region_v4}.api.riotgames.com"
    base_url_v5 = f"https://{region_v5}.api.riotgames.com"
    headers = {"X-Riot-Token": riot_api_key}
    
    requester = Requester(
        base_url_v4=base_url_v4,
        base_url_v5=base_url_v5,
        headers=headers,
        logger=logger
    )
    
    parquet_handler = ParquetHandler(logger=logger, random_state=42)
    
    match_fetcher = MatchFetcher(
        requester=requester,
        logger=logger,
        parquet_handler=parquet_handler,
        dataframe_target_path="./data/temp",
        checkpoint_loading_path=None,
        load_percentage=1.0,
        random_state=42,
        master_registry=None,
        max_workers=4,
    )
    
    # Test account lookup
    logger.info(f"Testing account lookup for {game_name}#{tag_line}")
    account = requester.get_account_by_riot_id(game_name, tag_line)
    
    if not account:
        logger.error(f"Account not found for {game_name}#{tag_line}")
        return
    
    logger.info(f"✓ Account found: PUUID = {account.get('puuid')}")
    
    # Test summoner lookup
    puuid = account['puuid']
    summoner = requester.get_summoner_by_puuid(puuid)
    
    if not summoner:
        logger.error(f"Summoner not found for PUUID {puuid}")
        return
    
    logger.info(f"✓ Summoner found: ID = {summoner.get('id')}, Level = {summoner.get('summonerLevel')}")
    
    # Test active game check
    encrypted_id = summoner['id']
    active_game = requester.get_active_game(encrypted_id)
    
    if not active_game:
        logger.warning(f"⚠ No active game found for {game_name}#{tag_line}")
        logger.info("This is normal if the player is not currently in a game.")
        return
    
    logger.info(f"✓ Active game found: Game ID = {active_game.get('gameId')}")
    logger.info(f"  Queue: {active_game.get('gameQueueConfigId')}")
    logger.info(f"  Mode: {active_game.get('gameMode')}")
    logger.info(f"  Participants: {len(active_game.get('participants', []))}")
    
    # Test fetch_active_game_pre_features
    logger.info(f"\nTesting fetch_active_game_pre_features...")
    pre_features = match_fetcher.fetch_active_game_pre_features(game_name, tag_line)
    
    if not pre_features:
        logger.error("Failed to fetch active game pre-features")
        return
    
    logger.info(f"✓ Pre-features extracted successfully:")
    logger.info(f"  Match ID: {pre_features.get('match_id')}")
    logger.info(f"  Team 1 (Blue) picks: {pre_features.get('team1_picks')}")
    logger.info(f"  Team 2 (Red) picks: {pre_features.get('team2_picks')}")
    logger.info(f"  Team 1 bans: {pre_features.get('team1_bans')}")
    logger.info(f"  Team 2 bans: {pre_features.get('team2_bans')}")
    logger.info(f"  Participants: {len(pre_features.get('team1_participants', []))} vs {len(pre_features.get('team2_participants', []))}")
    
    logger.info("\n✅ All tests passed! Live game detection is working.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    
    game_name = sys.argv[1]
    tag_line = sys.argv[2]
    
    test_live_game_detection(game_name, tag_line)
