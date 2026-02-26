import logging
import os
import secrets
from urllib.parse import urlparse

import httpx
import jwt
import redis.asyncio as redis
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from polar_sdk import Polar
from polar_sdk._webhooks import WebhookVerificationError, validate_event
from polar_sdk.models import (
    WebhookSubscriptionActivePayload,
    WebhookSubscriptionCanceledPayload,
    WebhookSubscriptionCreatedPayload,
    WebhookSubscriptionRevokedPayload,
    WebhookSubscriptionUpdatedPayload,
)

from .dependencies import (
    CLERK_FRONTEND_API,
    CLERK_PERMITTED_ORIGINS,
    CLERK_SECRET_KEY,
    get_jwks_client,
)

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

POLAR_ACCESS_TOKEN = os.getenv("POLAR_ACCESS_TOKEN")
POLAR_WEBHOOK_SECRET = os.getenv("POLAR_WEBHOOK_SECRET")

CHECKOUT_PENDING_TTL = 900  # 15 minutes - token validity


# Dependency to get authenticated user ID from Clerk
async def get_current_user_id(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Invalid Authorization header format"
        )
    token = authorization[7:]  # len("Bearer ") == 7

    # Get JWKS client for signature verification
    jwks_client = get_jwks_client()
    if not jwks_client:
        logger.error("CLERK_FRONTEND_API not set")
        raise HTTPException(status_code=500, detail="Server configuration error")

    try:
        # Get signing key from JWKS and verify JWT signature
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
                "verify_iat": True,
                "verify_nbf": True,
                "require": ["sid", "exp", "iat", "iss"],
            },
        )

        # Validate authorized parties (azp) claim for CSRF protection
        azp = payload.get("azp")
        permitted_origins = (
            [o.strip() for o in CLERK_PERMITTED_ORIGINS.split(",") if o.strip()]
            if CLERK_PERMITTED_ORIGINS
            else []
        )

        if permitted_origins:
            if not azp:
                logger.warning(
                    "Missing azp claim when permitted origins are configured"
                )
                raise HTTPException(status_code=401, detail="Missing token origin")
            if azp not in permitted_origins:
                logger.warning(f"Invalid azp claim: {azp}")
                raise HTTPException(status_code=401, detail="Invalid token origin")

        session_id = payload.get("sid")

        if not session_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        # Verify session with Clerk API for additional security layer
        if not CLERK_SECRET_KEY:
            logger.error("CLERK_SECRET_KEY not set")
            raise HTTPException(status_code=500, detail="Server configuration error")

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"https://api.clerk.com/v1/sessions/{session_id}",
                headers={
                    "Authorization": f"Bearer {CLERK_SECRET_KEY}",
                    "Clerk-API-Version": "2025-11-10",
                },
            )

            if response.status_code != 200:
                logger.error(f"Clerk session verification failed: {response.text}")
                raise HTTPException(status_code=401, detail="Invalid session")

            session_data = response.json()
            if session_data.get("status") != "active":
                raise HTTPException(status_code=401, detail="Session is not active")

            return session_data.get("user_id")

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {type(e).__name__}")
        raise HTTPException(status_code=401, detail="Invalid token")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auth error: {type(e).__name__}")
        raise HTTPException(status_code=401, detail="Authentication failed")


@router.post("/create-checkout")
async def create_checkout(
    request: Request, user_id: str = Depends(get_current_user_id)
):
    try:
        body = await request.json()
        product_id = body.get("product_id")
        return_url = body.get("return_url")

        if not product_id:
            raise HTTPException(status_code=400, detail="Missing product_id")

        if not POLAR_ACCESS_TOKEN:
            raise HTTPException(
                status_code=500, detail="Polar Access Token not configured"
            )

        # Validate return_url to prevent open redirect attacks
        if return_url:
            parsed = urlparse(return_url)
            allowed_hosts = [urlparse(str(request.base_url)).netloc]
            extra_hosts = os.getenv("ALLOWED_REDIRECT_HOSTS", "")
            if extra_hosts:
                allowed_hosts.extend(
                    h.strip() for h in extra_hosts.split(",") if h.strip()
                )
            if not parsed.netloc or parsed.netloc not in allowed_hosts:
                logger.warning(f"Invalid return_url rejected: {return_url[:50]}")
                raise HTTPException(status_code=400, detail="Invalid return_url")

        # Generate secure checkout token
        checkout_token = secrets.token_urlsafe(32)
        redis_client = getattr(request.app.state, "redis_client", None)

        if redis_client:
            try:
                await redis_client.setex(
                    f"checkout_pending:{checkout_token}", CHECKOUT_PENDING_TTL, user_id
                )
                logger.debug("Checkout token generated")
            except redis.RedisError as e:
                logger.error(f"Failed to store checkout token: {e}")
                raise HTTPException(
                    status_code=500, detail="Failed to initialize secure checkout"
                )
        else:
            logger.error("Redis client not available for checkout token storage")
            raise HTTPException(
                status_code=500, detail="Checkout service temporarily unavailable"
            )

        # Use return_url if provided, otherwise fallback to homepage
        success_url = return_url if return_url else str(request.base_url)
        separator = "&" if "?" in success_url else "?"
        success_url += f"{separator}checkout=success&checkout_token={checkout_token}"

        # Fetch user details from Clerk to pre-fill checkout form
        user_details = await get_clerk_user_details(user_id)

        # Initialize Polar SDK
        with Polar(access_token=POLAR_ACCESS_TOKEN) as polar:
            checkout = polar.checkouts.create(
                request={
                    "products": [product_id],
                    "metadata": {"user_id": user_id},
                    "success_url": success_url,
                    "customer_email": user_details.get("email"),
                    "customer_name": user_details.get("name"),
                }
            )

            return {"url": checkout.url}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating checkout: {e}")
        raise HTTPException(status_code=500, detail="Failed to create checkout")


@router.post("/webhooks/polar")
async def polar_webhook(request: Request):
    if not POLAR_WEBHOOK_SECRET:
        logger.error("POLAR_WEBHOOK_SECRET not set")
        raise HTTPException(status_code=500, detail="Webhook secret not configured")

    payload = await request.body()

    try:
        event = validate_event(
            body=payload,
            headers=dict(request.headers),
            secret=POLAR_WEBHOOK_SECRET,
        )
    except WebhookVerificationError as e:
        logger.warning(f"Webhook verification failed: {e.message}")
        raise HTTPException(status_code=403, detail="Invalid webhook signature")

    logger.info(f"Received Polar webhook: {event.TYPE}")

    # Handle subscription events using typed payloads
    # Note: Some state changes may trigger both specific events (e.g., Active) AND Updated events.
    # This is fine since handle_subscription_update is idempotent (sets tier to same value).
    if isinstance(
        event, (WebhookSubscriptionCreatedPayload, WebhookSubscriptionActivePayload)
    ):
        await handle_subscription_update(event.data, tier="pro")
    elif isinstance(
        event, (WebhookSubscriptionCanceledPayload, WebhookSubscriptionRevokedPayload)
    ):
        # Skip Clerk tier update on cancellation - user keeps pro tier until trial expires
        # Polar will send subscription.updated webhook when trial actually ends
        logger.info(
            "Subscription canceled - skipping tier update, waiting for trial expiry"
        )
    elif isinstance(event, WebhookSubscriptionUpdatedPayload):
        # Handle subscription updates (e.g., status transitions)
        # Polar SDK status values: incomplete, incomplete_expired, trialing, active, past_due, canceled, unpaid
        status = getattr(event.data, "status", None)
        if status in ("active", "trialing"):
            # Active and trialing subscriptions get Pro tier
            await handle_subscription_update(event.data, tier="pro")
        elif status in ("canceled", "unpaid", "past_due", "incomplete_expired"):
            # These statuses indicate subscription is not usable
            # Note: 'canceled' status here means trial has ended (not just user cancelled)
            await handle_subscription_update(event.data, tier="free")
        elif status in ("incomplete", "pending"):
            # Incomplete/pending: waiting for payment, don't change tier yet
            logger.info(f"Subscription in pending state: {status}")
        else:
            logger.warning(f"Unknown subscription status: {status}")

    return {"status": "received"}


async def handle_subscription_update(subscription, tier: str):
    """Update user tier based on subscription metadata."""
    metadata = subscription.metadata or {}
    user_id = metadata.get("user_id")

    if user_id:
        logger.info(f"Updating user {user_id} to tier {tier}")
        success = await update_clerk_user_metadata(user_id, {"tier": tier})
        if not success:
            logger.error(
                f"Failed to update Clerk metadata for user {user_id}, tier={tier}"
            )
            raise HTTPException(
                status_code=502, detail="Failed to update user metadata"
            )
    else:
        logger.warning("No user_id found in subscription metadata")


async def update_clerk_user_metadata(user_id: str, metadata: dict) -> bool:
    if not CLERK_SECRET_KEY:
        logger.error("CLERK_SECRET_KEY missing, cannot update user metadata")
        return False

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.patch(
            f"https://api.clerk.com/v1/users/{user_id}/metadata",
            headers={
                "Authorization": f"Bearer {CLERK_SECRET_KEY}",
                "Clerk-API-Version": "2025-11-10",
            },
            json={"public_metadata": metadata},
        )

        if response.status_code == 200:
            logger.info(f"Successfully updated Clerk metadata for user {user_id}")
            return True
        else:
            logger.error(f"Failed to update Clerk metadata: {response.status_code}")
            return False


async def get_clerk_user_details(user_id: str) -> dict:
    """Fetch user email and name from Clerk for checkout pre-fill"""
    if not CLERK_SECRET_KEY:
        logger.warning("CLERK_SECRET_KEY missing, cannot fetch user details")
        return {"email": None, "name": None}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"https://api.clerk.com/v1/users/{user_id}",
                headers={
                    "Authorization": f"Bearer {CLERK_SECRET_KEY}",
                    "Clerk-API-Version": "2025-11-10",
                },
            )
            if response.status_code == 200:
                user_data = response.json()
                email_addresses = user_data.get("email_addresses", [])
                primary_email_id = user_data.get("primary_email_address_id")
                primary_email = next(
                    (
                        e["email_address"]
                        for e in email_addresses
                        if e["id"] == primary_email_id
                    ),
                    email_addresses[0]["email_address"] if email_addresses else None,
                )
                first_name = user_data.get("first_name") or ""
                last_name = user_data.get("last_name") or ""
                full_name = f"{first_name} {last_name}".strip() or None

                return {"email": primary_email, "name": full_name}
            else:
                logger.warning(
                    f"Failed to fetch Clerk user details: {response.status_code}"
                )
    except Exception as e:
        logger.error(f"Error fetching Clerk user details: {e}")

    return {"email": None, "name": None}
