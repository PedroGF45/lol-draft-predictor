from data_extraction.requester import Requester

import os
import logging
import pandas as pd
from typing import Any

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

    def fetch_match_data(self, parquet_path: str, keep_remakes: bool = False, queue: list[int] = [420, 440], match_limit: int = 50) -> None:

        if not os.path.exists(parquet_path):
            self.logger.error(f'Parquet path must be a valid path but got {parquet_path}')
        
        match_df = self.get_dataframe_from_parquet(parquet_path=parquet_path)

        final_df = []

        for _index, match_id in match_df.itertuples(index=False):
            
            # fetch match details
            match_pre_features = self.fetch_match_pre_features(match_id=match_id)

            # check if game is a remake
            if not keep_remakes and match_pre_features.get("game_duration") < 300:
                self.logger.info(f'Skipping remake match: {match_id}')
                continue

            # filter by queue
            if match_pre_features.get("queue_id") not in queue:
                self.logger.info(f'Skipping match {match_id} due to queue filter. Queue ID: {match_pre_features.get("queue_id")}')
                continue

            participants = match_pre_features.get("team1_participants") + match_pre_features.get("team2_participants")

            # Fetch summoner level for each participant
            summoner_levels = self.fetch_summoner_level_data(participants=participants)
            self.logger.info(f'Summoner Levels: {summoner_levels}')

            # fetch champion mastery of the champion played in the match for each participant
            champion_picks = match_pre_features.get("team1_picks") + match_pre_features.get("team2_picks")
            champion_masteries = self.fetch_champion_mastery_data(participants=participants, champion_picks=champion_picks)
            self.logger.info(f'Champion Masteries: {champion_masteries}')

            # fetch the total mastery score for each participant
            champion_total_mastery_scores = self.fetch_total_mastery_score(participants=participants)
            self.logger.info(f'Total Mastery Scores: {champion_total_mastery_scores}')

            # fetch rank queue data for each participant
            rank_queue_data = self.fetch_rank_queue_data(participants=participants)
            self.logger.info(f'Rank Queue Data: {rank_queue_data}')

            # get raw kpis for each participant on last 100 matches before this match
            kpis_data = self.fetch_raw_player_kpis(participants=participants, match_limit=match_limit)
            self.logger.info(f'KPIs Data: {kpis_data}')

            # combine all data into a single record
            record = {
                "match_id": match_id,
                "match_pre_features": match_pre_features,
                "summoner_levels": summoner_levels,
                "champion_masteries": champion_masteries,
                "champion_total_mastery_scores": champion_total_mastery_scores,
                "rank_queue_data": rank_queue_data,
                "kpis_data": kpis_data
            }

            final_df.append(record)

        final_df = pd.DataFrame(final_df)

        self.logger.info(f'Final Dataframe Info: {final_df.info()}')
        self.logger.info(f'Final Dataframe Head: {final_df.head()}')
        self.logger.info(f'Final Dataframde description: {final_df.describe()}')
        # save final dataframe to parquet
        output_parquet_path = os.path.join(self.dataframe_target_path, f'match_data.parquet')
        final_df.to_parquet(output_parquet_path)


    def fetch_match_pre_features(self, match_id: str) -> dict[str, Any]:
    
        match_details_endpoint = f'/lol/match/v5/matches/{match_id}'
        match_details = self.requester.make_request(is_v5=True, endpoint_url=match_details_endpoint)

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
                team1_picks.append(participant.get("championId"))
                team1_roles.append(participant.get("individualPosition"))
                team1_participants.append(participant.get("puuid"))
            else:
                team2_picks.append(participant.get("championId"))
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

        match_pre_features = {
            "data_version": data_version,
            "game_duration": game_duration,
            "game_mode": game_mode,
            "game_type": game_type,
            "queue_id": queue_id,
            "game_version": game_version,
            "platform_id": platform_id,
            "team1_bans": team1_bans,
            "team2_bans": team2_bans,
            "team1_picks": team1_picks,
            "team2_picks": team2_picks,
            "team1_roles": team1_roles,
            "team2_roles": team2_roles,
            "team1_participants": team1_participants,
            "team2_participants": team2_participants,
            "match_outcome": match_outcome
        }

        return match_pre_features
    
    def fetch_summoner_level_data(self, participants: list[str]) -> dict[str, Any]:
        
        summoner_levels = {}
        for puuid in participants:
            summoner_endpoint = f'/lol/summoner/v4/summoners/by-puuid/{puuid}'
            summoner_data = self.requester.make_request(is_v5=False, endpoint_url=summoner_endpoint)

            summoner_level = summoner_data.get("summonerLevel")
            self.logger.info(f'Summoner PUUID: {puuid}, Level: {summoner_level}')
            summoner_levels[puuid] = summoner_level

        return summoner_levels
    
    def fetch_champion_mastery_data(self, participants: list[str], champion_picks: list[int]) -> dict[str, Any]:
        champion_masteries = {}
        
        for puuid in participants:
            champion_id = champion_picks[participants.index(puuid)]
            champion_mastery_endpoint = f'/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/by-champion/{champion_id}'
            mastery_data = self.requester.make_request(is_v5=False, endpoint_url=champion_mastery_endpoint)

            last_played_time = mastery_data.get("lastPlayTime")
            champion_level = mastery_data.get("championLevel")
            champion_points = mastery_data.get("championPoints")
            

            champion_masteries[puuid] = {
                "lastPlayTime": last_played_time,
                "championLevel": champion_level,
                "championPoints": champion_points
            }

        return champion_masteries
    
    def fetch_total_mastery_score(self, participants: list[str]) -> dict[str, int]:
        total_mastery_scores = {}
        for puuid in participants:
            total_mastery_endpoint = f'/lol/champion-mastery/v4/scores/by-puuid/{puuid}'
            total_score = self.requester.make_request(is_v5=False, endpoint_url=total_mastery_endpoint)

            total_mastery_scores[puuid] = total_score

        return total_mastery_scores
    
    def fetch_rank_queue_data(self, participants: list[str]) -> dict[str, Any]:
        rank_queue_data = {}
        for puuid in participants:
            summoner_endpoint = f'/lol/league/v4/entries/by-puuid/{puuid}'
            rank_data = self.requester.make_request(is_v5=False, endpoint_url=summoner_endpoint)

            # Find the rank data for the RANKED_SOLO_5x5 queue
            solo_queue_data = next((entry for entry in rank_data if entry.get("queueType") == "RANKED_SOLO_5x5"), None)
            if solo_queue_data:
                rank_queue_data[puuid + "_solo"] = {
                    "tier": solo_queue_data.get("tier"),
                    "rank": solo_queue_data.get("rank"),
                    "leaguePoints": solo_queue_data.get("leaguePoints"),
                    "wins": solo_queue_data.get("wins"),
                    "losses": solo_queue_data.get("losses"),
                    "hotStreak": solo_queue_data.get("hotStreak")
                }
            
            flex_queue_data = next((entry for entry in rank_data if entry.get("queueType") == "RANKED_FLEX_SR"), None)
            if flex_queue_data:
                rank_queue_data[puuid + "_flex"] = {
                    "tier": flex_queue_data.get("tier"),
                    "rank": flex_queue_data.get("rank"),
                    "leaguePoints": flex_queue_data.get("leaguePoints"),
                    "wins": flex_queue_data.get("wins"),
                    "losses": flex_queue_data.get("losses"),
                    "hotStreak": flex_queue_data.get("hotStreak")
                }

        return rank_queue_data
    
    def fetch_raw_player_kpis(self, participants: list[str], match_limit: int = 50) -> dict[str, Any]:
        kpis_data = {}
        for puuid in participants:
            kpis_endpoint = f'/lol/match/v5/matches/by-puuid/{puuid}/ids?count={match_limit}'
            kpis_match_ids = self.requester.make_request(is_v5=True, endpoint_url=kpis_endpoint)

            kpis_list = []
            for kpis_match_id in kpis_match_ids:
                match_details_endpoint = f'/lol/match/v5/matches/{kpis_match_id}'
                match_details = self.requester.make_request(is_v5=True, endpoint_url=match_details_endpoint)

                participant_data = next((p for p in match_details.get("info").get("participants") if p.get("puuid") == puuid), None)
                if participant_data:

                    print(f'Participant Data for PUUID {puuid} in Match {kpis_match_id}')
                    
                    # get relevant kpis
                    kpis = {
                        "allInPings": participant_data.get("allInPings"),
                        "assistmePings": participant_data.get("assistmePings"),
                        "assists": participant_data.get("assists"),
                        "baronKills": participant_data.get("baronKills"),
                        "bountyLevel": participant_data.get("bountyLevel"),
                        "commandPings": participant_data.get("commandPings"),
                        "consumablesPurchased": participant_data.get("consumablesPurchased"),
                        "damageDealtToBuildings": participant_data.get("damageDealtToBuildings"),
                        "damageDealtToObjectives": participant_data.get("damageDealtToObjectives"),
                        "damageDealtToTurrets": participant_data.get("damageDealtToTurrets"),
                        "damageSelfMitigated": participant_data.get("damageSelfMitigated"),
                        "deaths": participant_data.get("deaths"),
                        "detectorWardsPlaced": participant_data.get("detectorWardsPlaced"),
                        "doubleKills": participant_data.get("doubleKills"),
                        "dragonKills": participant_data.get("dragonKills"),
                        "enemyMissingPings": participant_data.get("enemyMissingPings"),
                        "enemyVisionPings": participant_data.get("enemyVisionPings"),
                        "firstBloodAssist": participant_data.get("firstBloodAssist"),
                        "firstBloodKill": participant_data.get("firstBloodKill"),
                        "firstTowerAssist": participant_data.get("firstTowerAssist"),
                        "firstTowerKill": participant_data.get("firstTowerKill"),
                        "holdPings": participant_data.get("holdPings"),
                        "getbackPings": participant_data.get("getbackPings"),
                        "goldEarned": participant_data.get("goldEarned"),
                        "goldSpent": participant_data.get("goldSpent"),
                        "inhibitorKills": participant_data.get("inhibitorKills"),
                        "inhibitorTakedowns": participant_data.get("inhibitorTakedowns"),
                        "inhibitorsLost": participant_data.get("inhibitorsLost"),
                        "itemsPurchased": participant_data.get("itemsPurchased"),
                        "killingSprees": participant_data.get("killingSprees"),
                        "kills": participant_data.get("kills"),
                        "largestCriticalStrike": participant_data.get("largestCriticalStrike"),
                        "largeKillingSpree": participant_data.get("largeKillingSpree"),
                        "largestMultiKill": participant_data.get("largestMultiKill"),
                        "longestTimeSpentLiving": participant_data.get("longestTimeSpentLiving"),
                        "magicDamageDealt": participant_data.get("magicDamageDealt"),
                        "magicDamageDealtToChampions": participant_data.get("magicDamageDealtToChampions"),
                        "magicDamageTaken": participant_data.get("magicDamageTaken"),
                        "neutralMinionsKilled": participant_data.get("neutralMinionsKilled"),
                        "needVisionPings": participant_data.get("needVisionPings"),
                        "objectivesStolen": participant_data.get("objectivesStolen"),
                        "objectivesStolenAssists": participant_data.get("objectivesStolenAssists"),
                        "onMyWayPings": participant_data.get("onMyWayPings"),
                        "pentakills": participant_data.get("pentakills"),
                        "physicalDamageDealt": participant_data.get("physicalDamageDealt"),
                        "physicalDamageDealtToChampions": participant_data.get("physicalDamageDealtToChampions"),
                        "physicalDamageTaken": participant_data.get("physicalDamageTaken"),
                        "pushPings": participant_data.get("pushPings"),
                        "quadraKills": participant_data.get("quadraKills"),
                        "sightWardsBoughtInGame": participant_data.get("sightWardsBoughtInGame"),
                        "timeCCingOthers": participant_data.get("timeCCingOthers"),
                        "totalDamageDealt": participant_data.get("totalDamageDealt"),
                        "totalDamageDealtToChampions": participant_data.get("totalDamageDealtToChampions"),
                        "totalDamageShieldedOnTeammates": participant_data.get("totalDamageShieldedOnTeammates"),
                        "totalDamageTaken": participant_data.get("totalDamageTaken"),
                        "totalHeal": participant_data.get("totalHeal"),
                        "totalHealsOnTeammates": participant_data.get("totalHealsOnTeammates"),
                        "totalMinionsKilled": participant_data.get("totalMinionsKilled"),
                        "totalTimeCCDealt": participant_data.get("totalTimeCCDealt"),
                        "totalTimeSpentDead": participant_data.get("totalTimeSpentDead"),
                        "totalUnitsHealed": participant_data.get("totalUnitsHealed"),
                        "tripleKills": participant_data.get("tripleKills"),
                        "trueDamageDealt": participant_data.get("trueDamageDealt"),
                        "trueDamageDealtToChampions": participant_data.get("trueDamageDealtToChampions"),
                        "trueDamageTaken": participant_data.get("trueDamageTaken"),
                        "turretKills": participant_data.get("turretKills"),
                        "turretTakedowns": participant_data.get("turretTakedowns"),
                        "turretsLost": participant_data.get("turretsLost"),
                        "unrealKills": participant_data.get("unrealKills"),
                        "visionScore": participant_data.get("visionScore"),
                        "visionClearedPings": participant_data.get("visionClearedPings"),
                        "visionWardsBoughtInGame": participant_data.get("visionWardsBoughtInGame"),
                        "wardKills": participant_data.get("wardKills"),
                        "wardsPlaced": participant_data.get("wardsPlaced")
                    }

                    kpis_list.append(kpis)

            kpis_data[puuid] = kpis_list

        return kpis_data