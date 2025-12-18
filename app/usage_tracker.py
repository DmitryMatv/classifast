import hashlib
import logging
import os
import uuid
from dataclasses import dataclass

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
TRACKING_COOKIE_NAME = "cf_track"
TRACKING_COOKIE_MAX_AGE = 365 * 24 * 60 * 60  # 1 year
TIER_CACHE_TTL = 10  # Cache user tier for 10 seconds


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


def get_or_create_tracking_id(request: Request) -> tuple[str, bool]:
    """Get tracking ID from cookie or create new one."""
    existing = request.cookies.get(TRACKING_COOKIE_NAME)
    if existing and len(existing) == 36:  # UUID format
        logger.info(f"Using existing tracking ID from cookie: {existing}")
        return existing, False
    new_id = str(uuid.uuid4())
    logger.info(f"Created new tracking ID: {new_id}")
    return new_id, True


def extract_user_info_from_token(request: Request) -> tuple[str | None, str | None]:
    """
    Extract user_id and tier from verified JWT token.
    Returns (None, None) if token is invalid or unverifiable.
    """
    # Import here to avoid circular imports
    from .payments import CLERK_FRONTEND_API, get_jwks_client

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        # Diagnostic: log if Authorization header is missing
        if not auth_header:
            logger.debug(
                "No Authorization header provided - treating as anonymous user"
            )
        else:
            logger.debug(f"Invalid Authorization header format: {auth_header[:20]}...")
        return None, None

    token = auth_header[7:]

    # Get JWKS client for signature verification
    jwks_client = get_jwks_client()
    if not jwks_client:
        return None, None  # Treat as anonymous if JWKS not configured

    try:
        # Verify signature before trusting claims
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
        public_metadata = payload.get("public_metadata", {})
        tier = public_metadata.get("tier", "free") if public_metadata else "free"

        return user_id, tier
    except Exception:
        return None, None  # Treat invalid tokens as anonymous


async def fetch_clerk_user_tier(user_id: str) -> str | None:
    """Fetch current tier directly from Clerk API (bypasses JWT cache)."""
    clerk_secret = os.getenv("CLERK_SECRET_KEY")
    if not clerk_secret or not user_id:
        return None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"https://api.clerk.com/v1/users/{user_id}",
                headers={"Authorization": f"Bearer {clerk_secret}"},
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("public_metadata", {}).get("tier")
    except Exception as e:
        logger.warning(f"Failed to fetch tier from Clerk API: {e}")
    return None


async def get_cached_user_tier(
    user_id: str, redis_client: redis.Redis | None
) -> str | None:
    """Get user tier with Redis caching to avoid hitting Clerk API on every request.

    Caches both positive results (60s) and negative results/errors (10s) to prevent
    API hammering during outages or when tier is not set.
    """
    if not user_id:
        return None

    cache_key = f"user_tier:{user_id}"

    # Try cache first
    if redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                cached_value = cached.decode() if isinstance(cached, bytes) else cached
                # Check for negative result sentinel
                return None if cached_value == "none" else cached_value
        except Exception:
            pass

    # Cache miss - fetch from Clerk API
    tier = await fetch_clerk_user_tier(user_id)

    # Store in cache (including None with shorter TTL to prevent repeated API calls)
    if redis_client:
        try:
            if tier:
                # Positive result: cache for full TTL
                await redis_client.setex(cache_key, TIER_CACHE_TTL, tier)
            else:
                # Negative result: cache "none" sentinel for shorter TTL
                await redis_client.setex(cache_key, 10, "none")
        except Exception:
            pass

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
        # Check if there's any indication this should be an authenticated request
        auth_header = request.headers.get("authorization", "")
        if not auth_header:
            logger.debug("Anonymous request - no Authorization header")
        else:
            logger.warning(
                "Authorization header present but token extraction failed - treating as anonymous"
            )

    # Pro users have unlimited access
    if user_id and tier == "pro":
        logger.info(f"Pro user access allowed: {user_id}")
        return UsageStatus(
            allowed=True,
            remaining=-1,  # Unlimited
            limit=-1,
            is_authenticated=True,
            is_pro=True,
        )

    # If authenticated but not pro in JWT, verify with cached Clerk API check
    # This handles JWT propagation delay after checkout
    if user_id and tier != "pro":
        actual_tier = await get_cached_user_tier(user_id, redis_client)
        if actual_tier == "pro":
            logger.info(f"Pro tier confirmed via cache/API for user {user_id}")
            return UsageStatus(
                allowed=True,
                remaining=-1,
                limit=-1,
                is_authenticated=True,
                is_pro=True,
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

        try:
            cookie_key = f"anon:{tracking_id}:usage_count"
            ip_key = f"anon:ip:{ip_hash}:usage_count"

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
        except Exception as e:
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
    except Exception as e:
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
    ttl = 30 * 24 * 60 * 60  # 30 days

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

    except Exception as e:
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
