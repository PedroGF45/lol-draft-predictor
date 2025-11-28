from data_extraction.requester import Requester
from typing import List
from collections import deque
import time

class InvalidPatientZeroError(Exception):
    """Raised when the provided patient zero account is invalid or not found."""

class DataMiner():
    """
    Docstring for DataMiner
    """

    def __init__(self, requester: Requester, patient_zero_game_name: str, patient_zero_tag_line: str) -> None:
        self.requester = requester

        self.patient_zero_game_name = patient_zero_game_name
        self.patient_zero_tag_line = patient_zero_tag_line

        self.players_queue = deque()
        self.seen_players = set()
        self.seen_matches = set()

        if not self._is_patient_zero_valid():
            raise InvalidPatientZeroError(
                f"Invalid patient zero: '{self.patient_zero_game_name}#{self.patient_zero_tag_line}' not found or inaccessible"
            )
        
    def _is_patient_zero_valid(self) -> bool:
        """
        Docstring for _is_patient_zero_valid
        
        :param self: Description
        :return: Description
        :rtype: bool
        """
        endpoint_url = f"/riot/account/v1/accounts/by-riot-id/{self.patient_zero_game_name}/{self.patient_zero_tag_line}"
        response = self.requester.make_request(endpoint_url=endpoint_url)

        if response and response.get("puuid"):
            self.players_queue.append(response["puuid"])
            self.seen_players.add(response["puuid"])
            return True
        return False

    def start_player_search(self, target_number_of_players: int = 100) -> None:
        """
        Docstring for start_player_search
        
        :param self: Description
        :param target_number_of_players: Description
        :type target_number_of_players: int
        """
        start = time.time()

        while len(self.players_queue) > 0 and len(self.seen_players) < target_number_of_players:

            print("------------")
            
            print(f'Number of current players: {len(self.seen_players)}')
            
            player_to_use = self.players_queue.popleft()

            matches_of_player = self.get_last_matches(puuid=player_to_use)

            for match in matches_of_player:
                if len(self.seen_players) >= target_number_of_players:
                            break
                
                if match not in self.seen_matches:
                    self.seen_matches.add(match)
                    new_players = self.get_players_from_match(match_id=match)

                    for player in new_players:
                        if len(self.seen_players) >= target_number_of_players:
                            break

                        if player not in self.seen_players:
                            self.seen_players.add(player)
                            self.players_queue.append(player)

            print(f'Number of players after requests {len(self.seen_players)}')

        end = time.time()
        print("In the end nothing even matters")
        print(f'Players length is {len(self.players_queue)} and set players length is {len(self.seen_players)}')
        print(f'Matches length is {len(self.seen_matches)}')
        print(f'It took {end - start} seconds')

    def get_last_matches(self, puuid: str, number_of_matches: int = 100) -> List[str]:
        """
        Docstring for get_last_matches
        
        :param self: Description
        :param puuid: Description
        :type puuid: str
        :param number_of_matches: Description
        :type number_of_matches: int
        :return: Description
        :rtype: Any
        """
        endpoint_url = f'/lol/match/v5/matches/by-puuid/{puuid}/ids?count={number_of_matches}'
        response = self.requester.make_request(endpoint_url=endpoint_url)

        if response:
            return response
        return []

    def get_players_from_match(self, match_id: str) -> List[str]:
        """
        Docstring for get_players_from_match
        
        :param self: Description
        :param match_id: Description
        :type match_id: str
        :return: Description
        :rtype: Any
        """
        endpoint_url = f'/lol/match/v5/matches/{match_id}'
        response = self.requester.make_request(endpoint_url=endpoint_url)

        if response and response.get("metadata").get("participants"):
            return response.get("metadata").get("participants")
        return []
