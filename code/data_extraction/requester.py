import requests
from typing import Any, Optional, Dict
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class Requester:

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str],
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
    ) -> None:
        # Normalize base URL to always end with a single slash for safe joining
        self.base_url = base_url.rstrip("/") + "/"
        self.headers = headers
        self.default_timeout = timeout

        # Reuse a single session with retry/backoff to handle transient errors and 429s
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def make_request(
        self,
        endpoint_url: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Any]:

        # Build a correct URL regardless of trailing/leading slashes
        url = urljoin(self.base_url, endpoint_url.lstrip("/"))

        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.default_timeout if timeout is None else timeout,
            )
        except requests.exceptions.RequestException as e:
            print(f"Network error when calling {url}: {e}")
            return None

        # Handle common status codes with project-relevant messages
        if response.status_code == 200:
            return self._safe_json(response)
        elif response.status_code == 403:
            print("Error 403: Forbidden. API key might be invalid or expired.")
        elif response.status_code == 404:
            print("Error 404: Not Found. The requested resource may not exist in this region.")
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            print(f"Error 429: Rate limited. Retry-After={retry_after or 'unknown'} seconds.")
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(f"Response content (first 500 chars): {response.text[:500]}")

        # Attempt to return any JSON error body for callers to inspect; may be None
        return self._safe_json(response)

    def _safe_json(self, response: requests.Response) -> Optional[Any]:
        try:
            return response.json()
        except ValueError:
            return None
    
