import logging
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class Requester:
    """
    Lightweight HTTP client for Riot API endpoints with retries and timeouts.

    This client normalizes the base URL, safely joins endpoint paths, reuses a
    configured `requests.Session` with retry/backoff for transient failures and
    rate limits (429), and parses JSON responses defensively.

    Attributes:
        base_url: Normalized base URL that always ends with a single '/'.
        headers: Default HTTP headers applied to every request.
        default_timeout: Default per-request timeout (seconds).
        session: A `requests.Session` configured with retry/backoff policy.

    Example:
        requester = Requester(
            base_url="https://europe.api.riotgames.com",
            headers={"X-Riot-Token": "<api-key>"},
        )
        data = requester.make_request("/riot/account/v1/accounts/by-riot-id/Name/TAG")
    """

    def __init__(
        self,
        logger: logging.Logger,
        base_url_v4: str,
        base_url_v5: str,
        headers: dict[str, str],
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
    ) -> None:
        """
        Initialize a session-enabled client with retry/backoff and timeouts.

        Args:
            base_url: Base URL for the API (e.g., "https://europe.api.riotgames.com").
            headers: Default headers to include on every request (e.g., X-Riot-Token).
            timeout: Default per-request timeout in seconds. Defaults to 10.0.
            max_retries: Max retries for transient/network/status errors. Defaults to 3.
            backoff_factor: Exponential backoff factor between retries. Defaults to 0.3.

        Notes:
            - Retries apply to status codes in [429, 500, 502, 503, 504] and respect
              the `Retry-After` header when present.
            - The session is mounted for both HTTP and HTTPS with the same policy.
        """

        self.logger = logger

        # Normalize base URL to always end with a single slash for safe joining
        self.base_url_v4: str = base_url_v4.rstrip("/") + "/"
        self.base_url_v5: str = base_url_v5.rstrip("/") + "/"
        self.headers: dict[str, str] = headers
        self.default_timeout: float = timeout

        # Rate limit tracking for dynamic backoff
        self.rate_limit_hits: int = 0  # Cumulative 429 error count

        # Reuse a single session with retry/backoff to handle transient errors and 429s
        self.session: requests.Session = requests.Session()
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
        is_v5: bool,
        endpoint_url: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Any]:
        """
        Execute a GET request to the given endpoint and parse JSON safely.

        Args:
            endpoint_url: Endpoint path (with or without leading '/').
            params: Optional query string parameters to include in the request.
            timeout: Override the default timeout for this call (seconds).

        Returns:
            The JSON-decoded response (dict, list, etc.) if available; otherwise None.

        Behavior:
            - On HTTP 200, returns parsed JSON.
            - On 403, 404, 429 or other non-2xx, logs a concise message and returns
              the parsed JSON error payload if present; otherwise None.
            - On network/timeout errors, returns None.
        """

        base_url = self.base_url_v5 if is_v5 else self.base_url_v4

        # Build a correct URL regardless of trailing/leading slashes
        url = urljoin(base_url, endpoint_url.lstrip("/"))

        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.default_timeout if timeout is None else timeout,
            )
        except requests.exceptions.RequestException as e:
            self.logger.error("Network error when calling %s: %s", url, e)
            return None

        # Handle common status codes with project-relevant messages
        if response.status_code == 200:
            return self._safe_json(response)

        # Log and return None for non-successful responses so callers do not
        # accidentally treat an error payload (e.g., 403/404 JSON) as valid data.
        if response.status_code == 403:
            self.logger.warning("Error 403: Forbidden. API key might be invalid or expired.")
        elif response.status_code == 404:
            self.logger.warning("Error 404: Not Found. The requested resource may not exist in this region.")
        elif response.status_code == 429:
            self.rate_limit_hits += 1
            retry_after = response.headers.get("Retry-After")
            self.logger.warning(
                "Error 429: Rate limited (hit #%d). Retry-After=%s seconds.",
                self.rate_limit_hits,
                retry_after or "unknown",
            )
        else:
            self.logger.error("Request failed with status code: %s", response.status_code)
            self.logger.debug("Response content (first 500 chars): %s", response.text[:500])

        return None

    def _safe_json(self, response: requests.Response) -> Optional[Any]:
        """
        Attempt to decode a response body as JSON, returning None on failure.

        Args:
            response: A completed `requests.Response` instance.

        Returns:
            The JSON-decoded object or None if the body is not valid JSON.
        """

        try:
            return response.json()
        except ValueError:
            return None

    def should_backoff(self, threshold: int = 3) -> bool:
        """
        Check if rate limiting has been detected (≥ threshold hits) and reset counter if true.

        Args:
            threshold: Number of 429 hits to trigger backoff. Defaults to 3.

        Returns:
            True if rate_limit_hits >= threshold (and counter is reset), False otherwise.
        """
        if self.rate_limit_hits >= threshold:
            self.logger.warning(
                "Rate limit backoff triggered (hits=%d >= threshold=%d). Resetting counter.",
                self.rate_limit_hits,
                threshold,
            )
            self.rate_limit_hits = 0
            return True
        return False
