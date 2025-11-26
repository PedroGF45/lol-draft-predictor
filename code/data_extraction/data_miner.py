from data_extraction.requester import Requester


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

        endpoint_url = (
            f"/riot/account/v1/accounts/by-riot-id/{self.patient_zero_game_name}/{self.patient_zero_tag_line}"
        )
        response = self.requester.make_request(endpoint_url=endpoint_url)
        return bool(response and response.get("puuid"))

    
        