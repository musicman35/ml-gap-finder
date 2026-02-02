"""Redis cache client for ML Gap Finder."""

import json
from typing import Any

import redis.asyncio as redis

from config.settings import settings


class RedisCache:
    """Async Redis client for caching."""

    def __init__(self, url: str | None = None):
        """Initialize Redis client.

        Args:
            url: Redis connection URL. Defaults to settings.redis_url.
        """
        self.url = url or settings.redis_url
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        """Establish Redis connection."""
        self._client = redis.from_url(self.url, decode_responses=True)

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> "RedisCache":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    async def get(self, key: str) -> str | None:
        """Get value by key.

        Args:
            key: Cache key.

        Returns:
            Cached value or None.
        """
        if not self._client:
            raise RuntimeError("Not connected to Redis")
        return await self._client.get(key)

    async def get_json(self, key: str) -> Any | None:
        """Get JSON value by key.

        Args:
            key: Cache key.

        Returns:
            Parsed JSON value or None.
        """
        value = await self.get(key)
        if value:
            return json.loads(value)
        return None

    async def set(
        self,
        key: str,
        value: str,
        expire: int | None = None,
    ) -> None:
        """Set a key-value pair.

        Args:
            key: Cache key.
            value: Value to cache.
            expire: Optional TTL in seconds.
        """
        if not self._client:
            raise RuntimeError("Not connected to Redis")
        if expire:
            await self._client.setex(key, expire, value)
        else:
            await self._client.set(key, value)

    async def set_json(
        self,
        key: str,
        value: Any,
        expire: int | None = None,
    ) -> None:
        """Set a JSON value.

        Args:
            key: Cache key.
            value: Value to serialize and cache.
            expire: Optional TTL in seconds.
        """
        await self.set(key, json.dumps(value), expire)

    async def delete(self, key: str) -> None:
        """Delete a key.

        Args:
            key: Cache key to delete.
        """
        if not self._client:
            raise RuntimeError("Not connected to Redis")
        await self._client.delete(key)

    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Cache key to check.

        Returns:
            True if key exists.
        """
        if not self._client:
            raise RuntimeError("Not connected to Redis")
        return await self._client.exists(key) > 0

    async def incr(self, key: str) -> int:
        """Increment a counter.

        Args:
            key: Counter key.

        Returns:
            New counter value.
        """
        if not self._client:
            raise RuntimeError("Not connected to Redis")
        return await self._client.incr(key)

    async def expire(self, key: str, seconds: int) -> None:
        """Set expiration on a key.

        Args:
            key: Cache key.
            seconds: TTL in seconds.
        """
        if not self._client:
            raise RuntimeError("Not connected to Redis")
        await self._client.expire(key, seconds)

    # Rate limiting helpers
    async def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> bool:
        """Check if request is within rate limit.

        Args:
            key: Rate limit key (e.g., "ratelimit:api:user123").
            max_requests: Maximum requests allowed in window.
            window_seconds: Time window in seconds.

        Returns:
            True if request is allowed.
        """
        if not self._client:
            raise RuntimeError("Not connected to Redis")

        current = await self._client.get(key)
        if current is None:
            await self._client.setex(key, window_seconds, 1)
            return True

        if int(current) >= max_requests:
            return False

        await self._client.incr(key)
        return True

    # Caching helpers for specific use cases
    async def cache_paper_metadata(
        self,
        arxiv_id: str,
        metadata: dict[str, Any],
        expire: int = 3600,
    ) -> None:
        """Cache paper metadata.

        Args:
            arxiv_id: Paper arXiv ID.
            metadata: Paper metadata dictionary.
            expire: Cache TTL in seconds (default 1 hour).
        """
        key = f"paper:metadata:{arxiv_id}"
        await self.set_json(key, metadata, expire)

    async def get_paper_metadata(self, arxiv_id: str) -> dict[str, Any] | None:
        """Get cached paper metadata.

        Args:
            arxiv_id: Paper arXiv ID.

        Returns:
            Cached metadata or None.
        """
        key = f"paper:metadata:{arxiv_id}"
        return await self.get_json(key)

    async def cache_gap_result(
        self,
        gap_key: str,
        result: dict[str, Any],
        expire: int = 1800,
    ) -> None:
        """Cache gap detection result.

        Args:
            gap_key: Unique key for the gap query.
            result: Gap detection result.
            expire: Cache TTL in seconds (default 30 minutes).
        """
        key = f"gap:result:{gap_key}"
        await self.set_json(key, result, expire)

    async def get_gap_result(self, gap_key: str) -> dict[str, Any] | None:
        """Get cached gap result.

        Args:
            gap_key: Unique key for the gap query.

        Returns:
            Cached result or None.
        """
        key = f"gap:result:{gap_key}"
        return await self.get_json(key)
