"""
Rate limiting service using Redis with in-memory fallback.
Implements fixed-window rate limiting with configurable limits per endpoint.
"""

import logging
import os
import time
from typing import Dict, List, Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RateLimitService:
    """Handles rate limiting for API endpoints."""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.memory_store: Dict[str, List[float]] = {}
        self.window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        self.default_limit = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "30"))
        self.match_limit = int(os.getenv("MATCH_RATE_LIMIT_MAX_REQUESTS", "10"))
        self.batch_limit = int(os.getenv("BATCH_RATE_LIMIT_MAX_REQUESTS", "5"))

    async def init(self) -> bool:
        """Initialize Redis connection."""
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            logger.info("Redis not configured, using in-memory rate limiting")
            return False

        try:
            self.redis_client = await redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            await self.redis_client.ping()
            logger.info("Rate limit service connected to Redis")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
            return False

    async def is_allowed(self, client_id: str, endpoint: str = "default") -> bool:
        """Check if request is allowed under rate limit."""
        # Determine limit based on endpoint
        if endpoint == "match":
            limit = self.match_limit
        elif endpoint == "batch":
            limit = self.batch_limit
        else:
            limit = self.default_limit

        key = f"ratelimit:{client_id}:{endpoint}"

        # Try Redis first
        if self.redis_client:
            return await self._redis_check(key, limit)
        else:
            return self._memory_check(key, limit)

    async def _redis_check(self, key: str, limit: int) -> bool:
        """Check rate limit using Redis."""
        try:
            count = await self.redis_client.incr(key)
            if count == 1:
                await self.redis_client.expire(key, self.window)
            return count <= limit
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            return True  # Allow on error

    def _memory_check(self, key: str, limit: int) -> bool:
        """Check rate limit using in-memory storage."""
        current_time = time.time()

        # Clean old entries
        if key in self.memory_store:
            self.memory_store[key] = [ts for ts in self.memory_store[key] if current_time - ts < self.window]
        else:
            self.memory_store[key] = []

        # Check limit
        if len(self.memory_store[key]) >= limit:
            return False

        self.memory_store[key].append(current_time)
        return True

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Rate limit service closed")
