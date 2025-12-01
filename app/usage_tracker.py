import hashlib
import logging
import os
import uuid
from dataclasses import dataclass

import jwt
import redis.asyncio as redis
from fastapi import Request, Response

logger = logging.getLogger(__name__)

# Configuration from environment
ANON_DAILY_LIMIT = int(os.getenv("ANON_DAILY_LIMIT", "10"))
FREE_USER_DAILY_LIMIT = int(os.getenv("FREE_USER_DAILY_LIMIT", "30"))
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "default")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
TRACKING_COOKIE_NAME = "cf_track"
TRACKING_COOKIE_MAX_AGE = 365 * 24 * 60 * 60  # 1 year


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
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip
    return request.client.host if request.client else "unknown"


def hash_ip(ip: str) -> str:
    """Hash IP for privacy."""
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def get_or_create_tracking_id(request: Request) -> tuple[str, bool]:
    """Get tracking ID from cookie or create new one."""
    existing = request.cookies.get(TRACKING_COOKIE_NAME)
    if existing and len(existing) == 36:  # UUID format
        return existing, False
    return str(uuid.uuid4()), True


def extract_user_info_from_token(request: Request) -> tuple[str | None, str | None]:
    """
    Extract user_id and tier from verified JWT token.
    Returns (None, None) if token is invalid or unverifiable.
    """
    # Import here to avoid circular imports
    from .payments import CLERK_FRONTEND_API, get_jwks_client

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
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

    # Pro users have unlimited access
    if user_id and tier == "pro":
        return UsageStatus(
            allowed=True,
            remaining=-1,  # Unlimited
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
        key = f"user:{user_id}:daily_count"
        limit = FREE_USER_DAILY_LIMIT
        tracking_id = user_id
    else:
        # Anonymous user - use cookie-based tracking with IP as fallback
        tracking_id, is_new_cookie = get_or_create_tracking_id(request)
        limit = ANON_DAILY_LIMIT

        try:
            if not is_new_cookie:
                # Cookie exists - use cookie-based tracking ONLY (per-user limit)
                cookie_key = f"anon:{tracking_id}:daily_count"
                current_count = await redis_client.get(cookie_key)
                current_count = int(current_count) if current_count else 0
            else:
                # No cookie (first visit, private mode) - fall back to IP
                ip_hash = hash_ip(get_client_ip(request))
                ip_key = f"anon:ip:{ip_hash}:daily_count"
                current_count = await redis_client.get(ip_key)
                current_count = int(current_count) if current_count else 0

            remaining = max(0, limit - current_count)

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
    ttl = 24 * 60 * 60  # 24 hours

    try:
        if user_id:
            # Authenticated user - use user ID
            key = f"user:{user_id}:daily_count"
            await redis_client.incr(key)
            await redis_client.expire(key, ttl)
        else:
            # Anonymous user - increment only the relevant counter
            tracking_id = usage_status.tracking_id
            _, is_new_cookie = get_or_create_tracking_id(request)

            if tracking_id and not is_new_cookie:
                # Has cookie - use cookie-based tracking
                cookie_key = f"anon:{tracking_id}:daily_count"
                await redis_client.incr(cookie_key)
                await redis_client.expire(cookie_key, ttl)
            else:
                # No cookie - use IP-based tracking
                ip_hash = hash_ip(get_client_ip(request))
                ip_key = f"anon:ip:{ip_hash}:daily_count"
                await redis_client.incr(ip_key)
                await redis_client.expire(ip_key, ttl)

    except Exception as e:
        logger.error(f"Redis error incrementing usage: {e}")


def set_tracking_cookie(response: Response, tracking_id: str) -> None:
    """Set the tracking cookie on response."""
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
