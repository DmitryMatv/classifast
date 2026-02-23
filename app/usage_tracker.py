import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple

import httpx
import jwt
import redis.asyncio as redis
from fastapi import Request, Response

logger = logging.getLogger(__name__)

# Configuration from environment
ANON_LIMIT = int(os.getenv("ANON_LIMIT", "10"))
FREE_USER_LIMIT = int(os.getenv("FREE_USER_LIMIT", "30"))
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "default")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Constants
TRACKING_COOKIE_NAME = "cf_track"
TRACKING_COOKIE_MAX_AGE = 365 * 24 * 60 * 60  # 1 year
USAGE_TTL = 30 * 24 * 60 * 60  # 30 days
TIER_CACHE_TTL = 3600  # Cache user tier for 1 hour
NEGATIVE_TIER_CACHE_TTL = 60  # Cache failed lookups for 1 minute
GRACE_PERIOD_TTL = 300  # 5 minutes - grace period for checkout completion


@dataclass
class UsageStatus:
    allowed: bool
    remaining: int
    limit: int
    is_authenticated: bool
    is_pro: bool
    tracking_id: str | None = None


def get_client_ip(request: Request) -> str:
    """Extract client IP, handling proxies."""
    # Cloudflare header first (cannot be spoofed, CF overwrites client value)
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip
    # Fallback to X-Forwarded-For for non-Cloudflare deployments
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def hash_ip(ip: str) -> str:
    """Hash IP for privacy."""
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


async def set_checkout_grace(user_id: str, redis_client: redis.Redis | None) -> bool:
    """Set checkout grace period for user after successful return from Polar."""
    if not redis_client or not user_id:
        return False
    try:
        grace_key = f"checkout_grace:{user_id}"
        await redis_client.setex(grace_key, GRACE_PERIOD_TTL, "1")
        logger.info("Checkout grace period activated")
        return True
    except redis.RedisError as e:
        logger.error(f"Failed to set checkout grace period: {e}")
        return False


async def has_active_grace(user_id: str, redis_client: redis.Redis | None) -> bool:
    """Check if user has active checkout grace period."""
    if not redis_client or not user_id:
        return False
    try:
        grace_key = f"checkout_grace:{user_id}"
        exists = await redis_client.exists(grace_key)
        if exists:
            ttl = await redis_client.ttl(grace_key)
            logger.debug(f"Checkout grace period active with {ttl}s remaining")
        return bool(exists)
    except redis.RedisError as e:
        logger.error(f"Failed to check checkout grace period: {e}")
        return False


async def verify_checkout_token(
    checkout_token: str,
    request: Request,
    redis_client: redis.Redis | None,
) -> bool:
    """Verify checkout token and activate grace period if valid."""
    if not redis_client or not checkout_token:
        return False
    try:
        pending_key = f"checkout_pending:{checkout_token}"
        stored_user_id = await redis_client.get(pending_key)

        if stored_user_id:
            stored_user_id = (
                stored_user_id.decode()
                if isinstance(stored_user_id, bytes)
                else stored_user_id
            )

            grace_set = await set_checkout_grace(stored_user_id, redis_client)
            if grace_set:
                try:
                    await redis_client.delete(pending_key)
                except redis.RedisError as e:
                    logger.warning(f"Failed to delete pending key after grace set: {e}")
                logger.info("Checkout token verified, grace period activated")
                return True
            else:
                logger.warning("Checkout token valid but grace period failed to set")
                return False
        else:
            logger.warning("Checkout token not found or expired")
        return False
    except redis.RedisError as e:
        logger.error(f"Redis error during checkout token verification: {e}")
        return False


def get_or_create_tracking_id(request: Request) -> Tuple[str, bool]:
    """Get tracking ID from cookie or create new one."""
    existing = request.cookies.get(TRACKING_COOKIE_NAME)
    if existing and len(existing) == 36:  # UUID format
        logger.debug(f"Using existing tracking ID from cookie: {existing}")
        return existing, False
    new_id = str(uuid.uuid4())
    logger.info(f"Created new tracking ID: {new_id}")
    return new_id, True


def extract_user_info_from_token(
    request: Request,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract user_id and tier from verified JWT token.
    Returns (None, None) if token is invalid or unverifiable.
    """
    # First try: Authorization header (from Clerk JS)
    user_id, tier = extract_from_auth_header(request)
    if user_id:
        return user_id, tier

    # Second try: __session cookie (available on page load, before Clerk JS)
    user_id, tier = extract_from_session_cookie(request)
    if user_id:
        logger.debug("User authenticated via __session cookie: %s", user_id)
        return user_id, tier

    return None, None


def extract_from_auth_header(
    request: Request,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract user info from Authorization header."""
    from .dependencies import CLERK_FRONTEND_API, get_jwks_client

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return None, None

    token = auth_header[7:]

    jwks_client = get_jwks_client()
    if not jwks_client:
        return None, None

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        expected_issuer = f"https://{CLERK_FRONTEND_API}"

        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=expected_issuer,
            options={
                "verify_signature": True,
                "verify_exp": True,
            },
        )

        user_id = payload.get("sub")
        if not user_id:
            return None, None

        public_metadata = payload.get("public_metadata", {})
        tier = public_metadata.get("tier", "free") if public_metadata else "free"

        return user_id, tier
    except (jwt.PyJWTError, ValueError, KeyError):
        return None, None


def extract_from_session_cookie(
    request: Request,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract user info from Clerk's __session cookie."""
    from .dependencies import CLERK_FRONTEND_API, get_jwks_client

    session_cookie = request.cookies.get("__session")
    if not session_cookie:
        return None, None

    jwks_client = get_jwks_client()
    if not jwks_client:
        return None, None

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(session_cookie)
        expected_issuer = f"https://{CLERK_FRONTEND_API}"

        payload = jwt.decode(
            session_cookie,
            signing_key.key,
            algorithms=["RS256"],
            issuer=expected_issuer,
            options={
                "verify_signature": True,
                "verify_exp": True,
            },
        )

        user_id = payload.get("sub")
        if not user_id:
            return None, None

        public_metadata = payload.get("public_metadata", {})
        tier = public_metadata.get("tier", "free") if public_metadata else "free"

        return user_id, tier
    except (jwt.PyJWTError, ValueError, KeyError):
        return None, None


async def fetch_clerk_user_tier(user_id: str) -> str | None:
    """Fetch current tier directly from Clerk API (bypasses JWT cache)."""
    start_time = time.time()

    clerk_secret = os.getenv("CLERK_SECRET_KEY")
    if not clerk_secret or not user_id:
        return None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            api_start = time.time()
            response = await client.get(
                f"https://api.clerk.com/v1/users/{user_id}",
                headers={"Authorization": f"Bearer {clerk_secret}"},
            )
            api_duration = time.time() - api_start
            logger.debug(
                "Clerk API tier check: %.3fs, user_id=%s, status=%d",
                api_duration,
                user_id,
                response.status_code,
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("public_metadata", {}).get("tier")
    except (httpx.HTTPError, ValueError, KeyError) as e:
        elapsed = time.time() - start_time
        logger.warning(f"Failed to fetch tier from Clerk API: {e} ({elapsed:.3f}s)")
    except Exception as e:  # Fallback for unexpected errors
        elapsed = time.time() - start_time
        logger.error(f"Unexpected error fetching tier from Clerk: {e} ({elapsed:.3f}s)")
    return None


async def get_cached_user_tier(
    user_id: str, redis_client: redis.Redis | None
) -> str | None:
    """Get user tier with Redis caching to avoid hitting Clerk API on every request."""
    start_time = time.time()

    if not user_id:
        return None

    cache_key = f"user_tier:{user_id}"

    # Try cache first
    if redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                cached_value = cached.decode() if isinstance(cached, bytes) else cached
                logger.debug(
                    "Tier cache hit: user_id=%s, tier=%s", user_id, cached_value
                )
                # Check for negative result sentinel
                return None if cached_value == "none" else cached_value
        except (redis.RedisError, ValueError):
            pass

    # Cache miss - fetch from Clerk API
    tier = await fetch_clerk_user_tier(user_id)

    # Store in cache (including None to prevent hammering API)
    if redis_client:
        try:
            if tier:
                await redis_client.setex(cache_key, TIER_CACHE_TTL, tier)
                logger.debug(
                    "Tier cache set: user_id=%s, tier=%s, ttl=%d",
                    user_id,
                    tier,
                    TIER_CACHE_TTL,
                )
            else:
                await redis_client.setex(cache_key, NEGATIVE_TIER_CACHE_TTL, "none")
                logger.debug(
                    "Tier cache set (negative): user_id=%s, ttl=%d",
                    user_id,
                    NEGATIVE_TIER_CACHE_TTL,
                )
        except redis.RedisError:
            pass

    total_elapsed = time.time() - start_time
    logger.info(
        "Tier check completed: %.3fs, user_id=%s, tier=%s", total_elapsed, user_id, tier
    )
    return tier


async def check_usage(
    request: Request,
    redis_client: redis.Redis | None,
) -> UsageStatus:
    """
    Check if the user/anonymous visitor can make a classification request.
    Returns UsageStatus with allowed flag and remaining quota.
    """
    # Check if user is authenticated
    user_id, tier = extract_user_info_from_token(request)

    # Log diagnostic info for debugging mismatches between frontend and backend
    if not user_id:
        auth_header = request.headers.get("authorization", "")
        session_cookie = request.cookies.get("__session")
        if not auth_header and not session_cookie:
            logger.debug(
                "Anonymous request - no Authorization header, no __session cookie"
            )
        elif not auth_header and session_cookie:
            logger.debug(
                "__session cookie present but token extraction failed - treating as anon"
            )
        elif auth_header:
            logger.debug(
                "Auth header present but token extract failed - treating as anon"
            )

    # Check for pro user status (combines JWT, cache, and grace period)
    if user_id:
        # Check checkout grace period first (takes priority)
        if await has_active_grace(user_id, redis_client):
            logger.info(
                f"Checkout grace period active for user {user_id} - allowing unlimited access"
            )
            return UsageStatus(
                allowed=True,
                remaining=-1,
                limit=-1,
                is_authenticated=True,
                is_pro=True,
                tracking_id=user_id,
            )

        # Check JWT tier first, then verify with cache if not pro
        is_pro = tier == "pro"
        if not is_pro:
            actual_tier = await get_cached_user_tier(user_id, redis_client)
            is_pro = actual_tier == "pro"
            if is_pro:
                logger.info(f"Pro tier confirmed via cache/API for user {user_id}")
        else:
            logger.info(f"Pro user access allowed: {user_id}")

        if is_pro:
            return UsageStatus(
                allowed=True,
                remaining=-1,  # Unlimited
                limit=-1,
                is_authenticated=True,
                is_pro=True,
                tracking_id=user_id,
            )

    # If Redis is not available, allow the request (fail open)
    if not redis_client:
        logger.warning("Redis not available, allowing request")
        return UsageStatus(
            allowed=True,
            remaining=-1,
            limit=-1,
            is_authenticated=bool(user_id),
            is_pro=False,
        )

    # Determine tracking key and limit
    if user_id:
        # Authenticated free user
        key = f"user:{user_id}:usage_count"
        limit = FREE_USER_LIMIT
        tracking_id = user_id
        logger.info(f"Checking usage for authenticated free user: {user_id}")
    else:
        # Anonymous user - check BOTH cookie AND IP counters
        tracking_id, _ = get_or_create_tracking_id(request)
        ip_hash = hash_ip(get_client_ip(request))
        limit = ANON_LIMIT

        logger.info(
            f"Checking usage for anonymous user: tracking_id={tracking_id}, ip_hash={ip_hash}"
        )

        cookie_key = f"anon:{tracking_id}:usage_count"
        ip_key = f"anon:ip:{ip_hash}:usage_count"

        try:
            cookie_count = await redis_client.get(cookie_key)
            cookie_count = int(cookie_count) if cookie_count else 0

            ip_count = await redis_client.get(ip_key)
            ip_count = int(ip_count) if ip_count else 0

            # Use higher of the two (defense against cookie clearing)
            current_count = max(cookie_count, ip_count)
            remaining = max(0, limit - current_count)

            logger.info(
                f"Anonymous usage counts: cookie={cookie_count}, ip={ip_count}, current={current_count}, remaining={remaining}"
            )

            return UsageStatus(
                allowed=current_count < limit,
                remaining=remaining,
                limit=limit,
                is_authenticated=False,
                is_pro=False,
                tracking_id=tracking_id,
            )
        except redis.RedisError as e:
            logger.error(f"Redis error checking anonymous usage: {e}")
            return UsageStatus(
                allowed=True,
                remaining=-1,
                limit=-1,
                is_authenticated=False,
                is_pro=False,
                tracking_id=tracking_id,
            )

    # Authenticated free user path
    try:
        current_count = await redis_client.get(key)
        current_count = int(current_count) if current_count else 0
        remaining = max(0, limit - current_count)

        logger.info(
            f"Authenticated free user usage: {key}, current={current_count}, remaining={remaining}"
        )

        return UsageStatus(
            allowed=current_count < limit,
            remaining=remaining,
            limit=limit,
            is_authenticated=True,
            is_pro=False,
            tracking_id=tracking_id,
        )
    except redis.RedisError as e:
        logger.error(f"Redis error checking user usage: {e}")
        return UsageStatus(
            allowed=True,
            remaining=-1,
            limit=-1,
            is_authenticated=True,
            is_pro=False,
            tracking_id=tracking_id,
        )


async def increment_usage(
    request: Request,
    redis_client: redis.Redis | None,
    usage_status: UsageStatus,
) -> None:
    """Increment usage counter after successful classification."""
    if not redis_client or usage_status.is_pro:
        return

    user_id, _ = extract_user_info_from_token(request)
    ttl = USAGE_TTL  # 30 days

    try:
        if user_id:
            # Authenticated user - use user ID
            key = f"user:{user_id}:usage_count"
            await redis_client.incr(key)
            await redis_client.expire(key, ttl)
            logger.info(f"Incremented authenticated user usage: {key}")
        else:
            # Anonymous user - always increment BOTH counters
            tracking_id = usage_status.tracking_id
            ip_hash = hash_ip(get_client_ip(request))

            cookie_key = f"anon:{tracking_id}:usage_count"
            ip_key = f"anon:ip:{ip_hash}:usage_count"

            await redis_client.incr(cookie_key)
            await redis_client.incr(ip_key)
            await redis_client.expire(cookie_key, ttl)
            await redis_client.expire(ip_key, ttl)

            logger.info(
                f"Incremented anonymous user usage: tracking_id={tracking_id}, ip_hash={ip_hash}, cookie_key={cookie_key}, ip_key={ip_key}"
            )

    except redis.RedisError as e:
        logger.error(f"Redis error incrementing usage: {e}")


def set_tracking_cookie(response: Response, tracking_id: str) -> None:
    """Set the tracking cookie on response."""
    logger.info(f"Setting tracking cookie: {tracking_id}")
    response.set_cookie(
        key=TRACKING_COOKIE_NAME,
        value=tracking_id,
        max_age=TRACKING_COOKIE_MAX_AGE,
        httponly=True,
        secure=True,
        samesite="lax",
    )


def add_quota_headers(response: Response, usage_status: UsageStatus) -> None:
    """Add quota information headers to response."""
    if usage_status.remaining >= 0:
        response.headers["X-RateLimit-Remaining"] = str(usage_status.remaining)
        response.headers["X-RateLimit-Limit"] = str(usage_status.limit)
