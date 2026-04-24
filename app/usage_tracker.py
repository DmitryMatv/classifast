import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import httpx
import redis.asyncio as redis
from fastapi import Request, Response

from .clerk_auth import (
    ClerkAuthenticationError,
    ClerkInfrastructureError,
    authenticate_clerk_token_local,
    should_validate_clerk_azp,
)

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
QUOTA_FAIL_OPEN = os.getenv("QUOTA_FAIL_OPEN", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Constants
TRACKING_COOKIE_NAME = "cf_track"
ANON_USAGE_TTL = 365 * 24 * 60 * 60  # 1 year
TRACKING_COOKIE_MAX_AGE = ANON_USAGE_TTL  # Keep the anon cookie aligned with Redis
USAGE_TTL = ANON_USAGE_TTL  # 1 year for authenticated free-user usage too
TIER_CACHE_TTL = 3600  # Cache user tier for 1 hour
NEGATIVE_TIER_CACHE_TTL = 60  # Cache failed lookups for 1 minute
GRACE_PERIOD_TTL = int(
    os.getenv("CHECKOUT_GRACE_TTL", "300")
)  # 5 minutes - grace period for checkout completion


@dataclass
class UsageStatus:
    allowed: bool
    remaining: int
    limit: int
    is_authenticated: bool
    is_pro: bool
    tracking_id: str | None = None


TierResolutionStatus = Literal[
    "confirmed_pro",
    "confirmed_non_pro",
    "transient_unavailable",
    "explicit_negative",
]


@dataclass(frozen=True)
class TierResolution:
    status: TierResolutionStatus
    tier: str | None = None


TIER_CACHE_SENTINEL_NON_PRO = "__sentinel:non_pro"
TIER_CACHE_SENTINEL_EXPLICIT_NEGATIVE = "__sentinel:explicit_negative"
TIER_CACHE_SENTINEL_TRANSIENT_UNAVAILABLE = "__sentinel:transient_unavailable"


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
    if existing:
        try:
            uuid.UUID(existing)
            logger.debug(f"Using existing tracking ID from cookie: {existing}")
            return existing, False
        except ValueError:
            pass
    new_id = str(uuid.uuid4())
    logger.info(f"Created new tracking ID: {new_id}")
    return new_id, True


def quota_fail_open_enabled() -> bool:
    return QUOTA_FAIL_OPEN


async def extract_user_info_from_token(
    request: Request,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract user_id and tier from verified JWT token.
    Returns (None, None) if token is invalid or unverifiable.
    """
    # First try: Authorization header (from Clerk JS)
    user_id, tier = await extract_from_auth_header(request)
    if user_id:
        return user_id, tier

    # Second try: __session cookie (available on page load, before Clerk JS)
    user_id, tier = await extract_from_session_cookie(request)
    if user_id:
        logger.debug("User authenticated via __session cookie: %s", user_id)
        return user_id, tier

    return None, None


async def extract_from_auth_header(
    request: Request,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract user info from Authorization header."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return None, None

    token = auth_header[7:]

    try:
        return await authenticate_clerk_token_local(
            token,
            validate_azp=should_validate_clerk_azp(),
        )
    except ClerkAuthenticationError as exc:
        logger.debug("Bearer token authentication failed: %s", exc.detail)
        return None, None
    except ClerkInfrastructureError as exc:
        logger.warning(
            "Bearer token verification temporarily unavailable; falling back to anonymous quota: %s",
            exc.detail,
        )
        return None, None


async def extract_from_session_cookie(
    request: Request,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract user info from Clerk's __session cookie."""
    session_cookie = request.cookies.get("__session")
    if not session_cookie:
        return None, None

    try:
        return await authenticate_clerk_token_local(
            session_cookie,
            validate_azp=False,
        )
    except ClerkAuthenticationError as exc:
        logger.debug("Session cookie authentication failed: %s", exc.detail)
        return None, None
    except ClerkInfrastructureError as exc:
        logger.warning(
            "Session cookie verification temporarily unavailable; falling back to anonymous quota: %s",
            exc.detail,
        )
        return None, None


async def fetch_clerk_user_tier(user_id: str) -> TierResolution:
    """Fetch current tier directly from Clerk API and classify the result."""
    start_time = time.time()

    clerk_secret = os.getenv("CLERK_SECRET_KEY")
    if not clerk_secret or not user_id:
        logger.error("CLERK_SECRET_KEY missing or user_id empty during tier lookup")
        return TierResolution(status="explicit_negative")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            api_start = time.time()
            response = await client.get(
                f"https://api.clerk.com/v1/users/{user_id}",
                headers={
                    "Authorization": f"Bearer {clerk_secret}",
                    "Clerk-API-Version": "2025-11-10",
                },
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
                tier = data.get("public_metadata", {}).get("tier")
                if tier == "pro":
                    return TierResolution(status="confirmed_pro", tier="pro")
                if isinstance(tier, str) and tier:
                    return TierResolution(status="confirmed_non_pro", tier=tier)
                return TierResolution(status="confirmed_non_pro")

            if response.status_code in {401, 403, 404}:
                return TierResolution(status="explicit_negative")

            if response.status_code in {429, 500, 502, 503, 504}:
                return TierResolution(status="transient_unavailable")

            logger.warning(
                "Unexpected Clerk tier response status: user_id=%s, status=%d",
                user_id,
                response.status_code,
            )
            return TierResolution(status="explicit_negative")
    except (httpx.TimeoutException, httpx.RequestError) as e:
        elapsed = time.time() - start_time
        logger.warning(f"Failed to fetch tier from Clerk API: {e} ({elapsed:.3f}s)")
        return TierResolution(status="transient_unavailable")
    except (ValueError, KeyError, TypeError) as e:
        elapsed = time.time() - start_time
        logger.warning(f"Failed to parse tier from Clerk API: {e} ({elapsed:.3f}s)")
        return TierResolution(status="explicit_negative")
    except Exception as e:  # Fallback for unexpected errors
        elapsed = time.time() - start_time
        logger.error(f"Unexpected error fetching tier from Clerk: {e} ({elapsed:.3f}s)")
        return TierResolution(status="explicit_negative")


async def get_cached_user_tier(
    user_id: str, redis_client: redis.Redis | None
) -> TierResolution:
    """
    Preserve the distinction between confirmed non-Pro, explicit negatives,
    and transient outages so JWT Pro hints only fail open for infrastructure issues.
    """
    start_time = time.time()

    if not user_id:
        return TierResolution(status="explicit_negative")

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
                if cached_value == "pro":
                    return TierResolution(status="confirmed_pro", tier="pro")
                if cached_value == TIER_CACHE_SENTINEL_NON_PRO:
                    return TierResolution(status="confirmed_non_pro")
                if cached_value == TIER_CACHE_SENTINEL_EXPLICIT_NEGATIVE:
                    return TierResolution(status="explicit_negative")
                if cached_value == TIER_CACHE_SENTINEL_TRANSIENT_UNAVAILABLE:
                    return TierResolution(status="transient_unavailable")
                return TierResolution(status="confirmed_non_pro", tier=cached_value)
        except (redis.RedisError, ValueError):
            pass

    # Cache miss - fetch from Clerk API
    resolution = await fetch_clerk_user_tier(user_id)

    if redis_client:
        try:
            if resolution.status == "confirmed_pro":
                await redis_client.setex(cache_key, TIER_CACHE_TTL, "pro")
                logger.debug(
                    "Tier cache set: user_id=%s, status=%s, ttl=%d",
                    user_id,
                    resolution.status,
                    TIER_CACHE_TTL,
                )
            elif resolution.status == "confirmed_non_pro":
                await redis_client.setex(
                    cache_key,
                    TIER_CACHE_TTL,
                    resolution.tier or TIER_CACHE_SENTINEL_NON_PRO,
                )
                logger.debug(
                    "Tier cache set: user_id=%s, status=%s, tier=%s, ttl=%d",
                    user_id,
                    resolution.status,
                    resolution.tier,
                    TIER_CACHE_TTL,
                )
            elif resolution.status == "explicit_negative":
                await redis_client.setex(
                    cache_key,
                    NEGATIVE_TIER_CACHE_TTL,
                    TIER_CACHE_SENTINEL_EXPLICIT_NEGATIVE,
                )
                logger.debug(
                    "Tier cache set: user_id=%s, status=%s, ttl=%d",
                    user_id,
                    resolution.status,
                    NEGATIVE_TIER_CACHE_TTL,
                )
            else:
                await redis_client.setex(
                    cache_key,
                    NEGATIVE_TIER_CACHE_TTL,
                    TIER_CACHE_SENTINEL_TRANSIENT_UNAVAILABLE,
                )
                logger.debug(
                    "Tier cache set: user_id=%s, status=%s, ttl=%d",
                    user_id,
                    resolution.status,
                    NEGATIVE_TIER_CACHE_TTL,
                )
        except redis.RedisError:
            pass

    total_elapsed = time.time() - start_time
    logger.info(
        "Tier check completed: %.3fs, user_id=%s, status=%s, tier=%s",
        total_elapsed,
        user_id,
        resolution.status,
        resolution.tier,
    )
    return resolution


async def check_usage(
    request: Request,
    redis_client: redis.Redis | None,
) -> UsageStatus:
    """
    Check if the user/anonymous visitor can make a classification request.
    Returns UsageStatus with allowed flag and remaining quota.
    """
    # Check if user is authenticated
    user_id, tier = await extract_user_info_from_token(request)

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

    if not redis_client:
        if quota_fail_open_enabled():
            logger.warning(
                "Redis not available, allowing request because QUOTA_FAIL_OPEN is enabled"
            )
            return UsageStatus(
                allowed=True,
                remaining=-1,
                limit=-1,
                is_authenticated=bool(user_id),
                is_pro=False,
                tracking_id=user_id,
            )

        logger.warning("Redis not available, denying metered request")
        if user_id:
            return UsageStatus(
                allowed=False,
                remaining=0,
                limit=FREE_USER_LIMIT,
                is_authenticated=True,
                is_pro=False,
                tracking_id=user_id,
            )

        tracking_id, _ = get_or_create_tracking_id(request)
        return UsageStatus(
            allowed=False,
            remaining=0,
            limit=ANON_LIMIT,
            is_authenticated=False,
            is_pro=False,
            tracking_id=tracking_id,
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

        jwt_tier_hint = tier
        tier_resolution = await get_cached_user_tier(user_id, redis_client)
        if tier_resolution.status == "confirmed_pro":
            is_pro = True
            logger.info(f"Pro tier confirmed via Clerk for user {user_id}")
        elif (
            tier_resolution.status == "transient_unavailable" and jwt_tier_hint == "pro"
        ):
            is_pro = True
            logger.info(
                "Using JWT Pro hint for user %s because Clerk tier confirmation is temporarily unavailable",
                user_id,
            )
        else:
            is_pro = False

        if (
            tier_resolution.status == "confirmed_non_pro"
            and jwt_tier_hint == "pro"
            and not is_pro
        ):
            logger.info(
                "Ignoring stale JWT Pro hint for user %s because Clerk tier is explicitly %s",
                user_id,
                tier_resolution.tier or TIER_CACHE_SENTINEL_NON_PRO,
            )
        if (
            tier_resolution.status == "explicit_negative"
            and jwt_tier_hint == "pro"
            and not is_pro
        ):
            logger.info(
                "Ignoring JWT Pro hint for user %s because Clerk returned an explicit negative tier result",
                user_id,
            )

        if is_pro:
            return UsageStatus(
                allowed=True,
                remaining=-1,  # Unlimited
                limit=-1,
                is_authenticated=True,
                is_pro=True,
                tracking_id=user_id,
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
            if quota_fail_open_enabled():
                return UsageStatus(
                    allowed=True,
                    remaining=-1,
                    limit=-1,
                    is_authenticated=False,
                    is_pro=False,
                    tracking_id=tracking_id,
                )
            return UsageStatus(
                allowed=False,
                remaining=0,
                limit=limit,
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
        if quota_fail_open_enabled():
            return UsageStatus(
                allowed=True,
                remaining=-1,
                limit=-1,
                is_authenticated=True,
                is_pro=False,
                tracking_id=tracking_id,
            )
        return UsageStatus(
            allowed=False,
            remaining=0,
            limit=limit,
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

    try:
        # tracking_id carries the post-check identity:
        # authenticated user_id for signed-in users, cookie id for anonymous users.
        tracking_id = usage_status.tracking_id
        if not tracking_id:
            logger.warning(
                "Skipping usage increment because tracking_id is missing: authenticated=%s",
                usage_status.is_authenticated,
            )
            return

        if usage_status.is_authenticated:
            key = f"user:{tracking_id}:usage_count"
            await redis_client.incr(key)
            await redis_client.expire(key, USAGE_TTL)
            logger.info(f"Incremented authenticated user usage: {key}")
        else:
            # Anonymous user - always increment BOTH counters
            ip_hash = hash_ip(get_client_ip(request))

            cookie_key = f"anon:{tracking_id}:usage_count"
            ip_key = f"anon:ip:{ip_hash}:usage_count"

            await redis_client.incr(cookie_key)
            await redis_client.incr(ip_key)
            await redis_client.expire(cookie_key, ANON_USAGE_TTL)
            await redis_client.expire(ip_key, ANON_USAGE_TTL)

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
