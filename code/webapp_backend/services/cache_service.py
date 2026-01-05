"""
Cache service for handling prediction caching with Redis.
Implements TTL-based caching with fallback to in-memory storage.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class CacheService:
    """Handles caching of predictions and model data."""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache: Dict[str, tuple] = {}  # (value, expiry_time)
        self.is_connected = False

    async def init(self) -> bool:
        """Initialize Redis connection."""
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            logger.info("Redis not configured, using in-memory cache")
            return False

        try:
            self.redis_client = await redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            await self.redis_client.ping()
            self.is_connected = True
            logger.info("Redis connected successfully")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory cache")
            self.redis_client = None
            return False

    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        try:
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    logger.debug(f"Cache hit: {key}")
                    return value
            else:
                # In-memory fallback
                if key in self.memory_cache:
                    value, expiry = self.memory_cache[key]
                    if expiry > time.time():
                        logger.debug(f"Memory cache hit: {key}")
                        return value
                    else:
                        del self.memory_cache[key]
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None

    async def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        """Set value in cache with TTL."""
        try:
            if self.redis_client:
                await self.redis_client.setex(key, ttl, value)
                logger.debug(f"Cached: {key} (TTL: {ttl}s)")
                return True
            else:
                # In-memory fallback
                self.memory_cache[key] = (value, time.time() + ttl)
                logger.debug(f"Memory cached: {key} (TTL: {ttl}s)")
                return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            if self.redis_client:
                await self.redis_client.delete(key)
                logger.debug(f"Cache deleted: {key}")
                return True
            else:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    logger.debug(f"Memory cache deleted: {key}")
                    return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
        return False

    async def clear(self) -> bool:
        """Clear all cache."""
        try:
            if self.redis_client:
                await self.redis_client.flushdb()
                logger.info("Cache cleared")
                return True
            else:
                self.memory_cache.clear()
                logger.info("Memory cache cleared")
                return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
        return False

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.is_connected = False
            logger.info("Redis connection closed")
