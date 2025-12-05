from data_extraction.requester import Requester

import os
import logging
import pandas as pd

class MatchFetcher():

    def __init__(self, requester: Requester, logger: logging.Logger,  dataframe_target_path: str) -> None:
        
        self.requester = requester
        self.logger = logger

        self.dataframe_target_path = dataframe_target_path
        if not os.path.exists(dataframe_target_path):
            logger.warning(f'Dataframe target path is not valid. Defaulting to "data/players"')
            self.dataframe_target_path = "./data/players"
      
    def get_dataframe_from_parquet(self, parquet_path: str) -> pd.DataFrame:

        try:
            match_df = pd.read_parquet(parquet_path)
            self.logger.info(f'Match data frame successfully loaded: {match_df.info()}')

        except Exception as e:
            match_df = pd.DataFrame()
            self.logger.error(f'Error while converting a dataframe from a parque path: {e}')

        return match_df

    def fetch_match_data(self, parquet_path: str):

        if not os.path.exists(parquet_path):
            self.logger.error(f'Parquet path must be a valid path but got {parquet_path}')
        
        match_df = self.get_dataframe_from_parquet(parquet_path=parquet_path)

        for _index, match_id in match_df.itertuples():
            
            # get match important pre-game features
            match_details_endpoint = f'/lol/match/v5/matches/{match_id}'
            match_details = self.requester.make_request(endpoint_url=match_details_endpoint)

            data_version = match_details.get("metadata").get("dataVersion") # version of the data schema
            game_duration = match_details.get("info").get("gameDuration") # in seconds
            game_mode = match_details.get("info").get("gameMode") # CLASSIC, ARAM, URF, DOOMBOTSTEEMO, ONEFORALL5x5, ASCENSION, etc
            game_type = match_details.get("info").get("gameType") # MATCHED_GAME, CUSTOM_GAME
            queue_id = match_details.get("info").get("queueId") #420, 430, 440, 450, 700, 720, 900, 910, 920, 940, 950, 960
            game_version = match_details.get("info").get("gameVersion") #11.18.387.1024
            platform_id = match_details.get("info").get("platformId") #EUW1

            team1_bans = match_details.get("info").get("teams")[0].get("bans") # list of banned champions for team 1 and pick turn
            team2_bans = match_details.get("info").get("teams")[1].get("bans") # list of banned champions for team 2 and pick turn
            
            team1_picks = []
            team2_picks = []
            team1_roles = []
            team2_roles = []
            team1_participants = []
            team2_participants = []

            for participant in match_details.get("info").get("participants"):
                if participant.get("teamId") == 100:
                    team1_picks.append(participant.get("championName"))
                    team1_roles.append(participant.get("individualPosition"))
                    team1_participants.append(participant.get("puuid"))
                else:
                    team2_picks.append(participant.get("championName"))
                    team2_roles.append(participant.get("individualPosition"))
                    team2_participants.append(participant.get("puuid"))
            match_outcome = match_details.get("info").get("teams")[0].get("win")  # True if team 1 won, False otherwise

            self.logger.info(f'Match ID: {match_id}')
            self.logger.info(f'Data Version: {data_version}')
            self.logger.info(f'Game Duration: {game_duration} seconds')
            self.logger.info(f'Game Mode: {game_mode}')
            self.logger.info(f'Game Type: {game_type}')
            self.logger.info(f'Queue ID: {queue_id}')
            self.logger.info(f'Game Version: {game_version}')
            self.logger.info(f'Platform ID: {platform_id}')
            self.logger.info(f'Team 1 Bans: {team1_bans}')
            self.logger.info(f'Team 2 Bans: {team2_bans}')
            self.logger.info(f'Team 1 Picks: {team1_picks}')
            self.logger.info(f'Team 2 Picks: {team2_picks}')
            self.logger.info(f'Team 1 Roles: {team1_roles}')
            self.logger.info(f'Team 2 Roles: {team2_roles}')
            self.logger.info(f'Team 1 Participants: {team1_participants}')
            self.logger.info(f'Team 2 Participants: {team2_participants}')
            self.logger.info(f'Match Outcome (Team 1 Win): {match_outcome}')

            break


            # get the players
           


    def fetch_player_data(self):
        pass
        