import logging
import os

import redis.asyncio as redis
from fastapi import HTTPException, Request

from .usage_tracker import get_client_ip, hash_ip

logger = logging.getLogger(__name__)

CHECKOUT_RATE_LIMIT = int(os.getenv("CHECKOUT_RATE_LIMIT", "10"))
CHECKOUT_RATE_LIMIT_WINDOW_SECONDS = int(
    os.getenv("CHECKOUT_RATE_LIMIT_WINDOW", "3600")
)
LOOKUP_RATE_LIMIT = int(os.getenv("LOOKUP_RATE_LIMIT", "60"))
LOOKUP_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("LOOKUP_RATE_LIMIT_WINDOW", "3600"))


async def _count_in_window(redis_client: redis.Redis, key: str, window: int) -> int:
    async with redis_client.pipeline(transaction=True) as pipeline:
        pipeline.incr(key)
        pipeline.expire(key, window, nx=True)
        count, _ = await pipeline.execute()
    return count


async def enforce_checkout_rate_limit(request: Request) -> None:
    """Fixed-window per-IP rate limit for checkout creation. Fails closed."""
    redis_client = getattr(request.app.state, "redis_client", None)
    if redis_client is None:
        logger.error("Redis client unavailable for checkout rate limiting")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    ip_hash = hash_ip(get_client_ip(request))
    key = f"checkout_rl:{ip_hash}"
    try:
        count = await _count_in_window(
            redis_client, key, CHECKOUT_RATE_LIMIT_WINDOW_SECONDS
        )
    except redis.RedisError as e:
        logger.error(f"Redis error during checkout rate limiting: {e}")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    if count > CHECKOUT_RATE_LIMIT:
        logger.warning("Checkout rate limit exceeded for IP hash %s", ip_hash)
        raise HTTPException(
            status_code=429,
            detail="Too many checkout requests. Please try again later.",
        )


async def lookup_rate_limit_exceeded(request: Request) -> bool:
    """Fixed-window per-IP limit for unmetered lookups.

    Fails open: a Redis outage must not block free lookups.
    """
    redis_client = getattr(request.app.state, "redis_client", None)
    if redis_client is None:
        logger.error("Redis client unavailable for lookup rate limiting")
        return False

    ip_hash = hash_ip(get_client_ip(request))
    try:
        count = await _count_in_window(
            redis_client, f"lookup_rl:{ip_hash}", LOOKUP_RATE_LIMIT_WINDOW_SECONDS
        )
    except redis.RedisError as e:
        logger.error(f"Redis error during lookup rate limiting: {e}")
        return False

    if count > LOOKUP_RATE_LIMIT:
        logger.warning("Lookup rate limit exceeded for IP hash %s", ip_hash)
        return True
    return False
