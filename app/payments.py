import logging
import os
from urllib.parse import urlparse

import httpx
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
from polar_sdk.models.subscription import Subscription

from .clerk_auth import (
    ClerkAuthenticationError,
    ClerkInfrastructureError,
    authenticate_clerk_token_with_session,
    should_validate_clerk_azp,
)
from .dependencies import CLERK_SECRET_KEY
from .mapping_store import get_mapping_product
from .rate_limit import enforce_checkout_rate_limit
from .usage_tracker import set_cached_user_tier, set_checkout_grace

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

POLAR_ACCESS_TOKEN = os.getenv("POLAR_ACCESS_TOKEN")
POLAR_WEBHOOK_SECRET = os.getenv("POLAR_WEBHOOK_SECRET")


def get_pro_product_id() -> str:
    product_id = os.getenv("POLAR_PRO_PRODUCT_ID", "").strip()
    if not product_id:
        raise HTTPException(status_code=500, detail="Polar Pro product not configured")
    return product_id


def _candidate_id_values_from_mapping(raw_value: dict) -> list[object]:
    return [raw_value.get(key) for key in ("id", "product_id", "polar_product_id")]


def _candidate_id_values_from_object(raw_value: object) -> list[object]:
    return [
        getattr(raw_value, attr)
        for attr in ("id", "product_id", "polar_product_id")
        if hasattr(raw_value, attr)
    ]


def _normalize_candidate_ids(raw_value) -> set[str]:
    if raw_value is None:
        return set()
    if isinstance(raw_value, str):
        values = [part.strip() for part in raw_value.split(",")]
        return {value for value in values if value}
    if isinstance(raw_value, (list, tuple, set)):
        return set().union(*(_normalize_candidate_ids(item) for item in raw_value))
    if isinstance(raw_value, dict):
        return set().union(
            *(
                _normalize_candidate_ids(value)
                for value in _candidate_id_values_from_mapping(raw_value)
            )
        )

    return set().union(
        *(
            _normalize_candidate_ids(value)
            for value in _candidate_id_values_from_object(raw_value)
        )
    )


def extract_subscription_product_ids(subscription: Subscription) -> set[str]:
    metadata = getattr(subscription, "metadata", None) or {}
    candidate_ids: set[str] = set()

    for key in ("product_id", "polar_product_id", "product_ids", "polar_product_ids"):
        if hasattr(subscription, key):
            candidate_ids.update(_normalize_candidate_ids(getattr(subscription, key)))
        if isinstance(metadata, dict):
            candidate_ids.update(_normalize_candidate_ids(metadata.get(key)))

    for attr in (
        "product",
        "products",
        "prices",
        "items",
        "subscriptions",
        "order_items",
    ):
        if hasattr(subscription, attr):
            candidate_ids.update(_normalize_candidate_ids(getattr(subscription, attr)))

    return candidate_ids


def subscription_matches_pro_entitlement(subscription: Subscription) -> bool:
    try:
        configured_product_id = get_pro_product_id()
    except HTTPException:
        logger.error("POLAR_PRO_PRODUCT_ID not configured; refusing entitlement update")
        raise

    product_ids = extract_subscription_product_ids(subscription)
    if not product_ids:
        logger.warning("Ignoring Polar subscription without product identity")
        return False

    if configured_product_id not in product_ids:
        logger.info(
            "Ignoring Polar subscription outside configured Pro product: %s",
            sorted(product_ids),
        )
        return False
    return True


def validate_return_url(request: Request, return_url: str | None) -> str | None:
    """Validate return_url against the current host allowlist."""
    if not return_url:
        return None

    parsed = urlparse(return_url)
    allowed_hosts = [urlparse(str(request.base_url)).netloc]
    extra_hosts = os.getenv("ALLOWED_REDIRECT_HOSTS", "")
    if extra_hosts:
        allowed_hosts.extend(h.strip() for h in extra_hosts.split(",") if h.strip())

    if not parsed.netloc or parsed.netloc not in allowed_hosts:
        logger.warning(f"Invalid return_url rejected: {return_url[:50]}")
        raise HTTPException(status_code=400, detail="Invalid return_url")

    return return_url


# Dependency to get authenticated user ID from Clerk
async def get_current_user_id(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Invalid Authorization header format"
        )
    token = authorization[7:]  # len("Bearer ") == 7

    try:
        user_id, _ = await authenticate_clerk_token_with_session(
            token, validate_azp=should_validate_clerk_azp()
        )
        return user_id
    except ClerkInfrastructureError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except ClerkAuthenticationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.post(
    "/create-checkout",
    dependencies=[Depends(enforce_checkout_rate_limit)],
)
async def create_checkout(
    request: Request, user_id: str = Depends(get_current_user_id)
):
    try:
        body = await request.json()
        return_url = validate_return_url(request, body.get("return_url"))

        if not POLAR_ACCESS_TOKEN:
            raise HTTPException(
                status_code=500, detail="Polar Access Token not configured"
            )

        product_id = get_pro_product_id()

        # Use return_url if provided, otherwise fallback to homepage
        success_url = return_url if return_url else str(request.base_url)
        separator = "&" if "?" in success_url else "?"
        success_url += f"{separator}checkout=success"

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


@router.post(
    "/create-mapping-checkout",
    dependencies=[Depends(enforce_checkout_rate_limit)],
)
async def create_mapping_checkout(request: Request):
    try:
        body = await request.json()
        slug = body.get("slug")
        return_url = validate_return_url(request, body.get("return_url"))

        if not slug:
            raise HTTPException(status_code=400, detail="Missing slug")

        product = get_mapping_product(slug)
        if not product:
            raise HTTPException(status_code=404, detail="Mapping product not found")

        if not POLAR_ACCESS_TOKEN:
            raise HTTPException(
                status_code=500, detail="Polar Access Token not configured"
            )

        success_url = return_url if return_url else str(request.base_url)
        separator = "&" if "?" in success_url else "?"
        success_url += f"{separator}checkout=success"

        with Polar(access_token=POLAR_ACCESS_TOKEN) as polar:
            checkout = polar.checkouts.create(
                request={
                    "products": [product.polar_product_id],
                    "metadata": {
                        "mapping_slug": product.slug,
                        "source_standard": product.source_standard,
                        "target_standard": product.target_standard,
                    },
                    "success_url": success_url,
                }
            )

        return {"url": checkout.url}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating mapping checkout: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to create mapping checkout"
        ) from e


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
        if subscription_matches_pro_entitlement(event.data):
            await handle_subscription_update(
                event.data,
                tier="pro",
                redis_client=getattr(request.app.state, "redis_client", None),
            )
    elif isinstance(
        event, (WebhookSubscriptionCanceledPayload, WebhookSubscriptionRevokedPayload)
    ):
        if not subscription_matches_pro_entitlement(event.data):
            return {"status": "received"}
        # Skip Clerk tier update on cancellation - user keeps pro tier until trial expires
        # Polar will send subscription.updated webhook when trial actually ends
        logger.info(
            "Subscription canceled - skipping tier update, waiting for trial expiry"
        )
    elif isinstance(event, WebhookSubscriptionUpdatedPayload):
        if not subscription_matches_pro_entitlement(event.data):
            return {"status": "received"}
        # Handle subscription updates (e.g., status transitions)
        # Polar SDK status values: incomplete, incomplete_expired, trialing, active, past_due, canceled, unpaid
        status = getattr(event.data, "status", None)
        if status in ("active", "trialing"):
            # Active and trialing subscriptions get Pro tier
            await handle_subscription_update(
                event.data,
                tier="pro",
                redis_client=getattr(request.app.state, "redis_client", None),
            )
        elif status in ("canceled", "unpaid", "past_due", "incomplete_expired"):
            # These statuses indicate subscription is not usable
            # Note: 'canceled' status here means trial has ended (not just user cancelled)
            await handle_subscription_update(
                event.data,
                tier="free",
                redis_client=getattr(request.app.state, "redis_client", None),
            )
        elif status in ("incomplete", "pending"):
            # Incomplete/pending: waiting for payment, don't change tier yet
            logger.info(f"Subscription in pending state: {status}")
        else:
            logger.warning(f"Unknown subscription status: {status}")

    return {"status": "received"}


async def handle_subscription_update(
    subscription: Subscription,
    tier: str,
    redis_client: redis.Redis | None = None,
):
    """Update user tier based on subscription metadata."""
    metadata = subscription.metadata or {}
    user_id = metadata.get("user_id")

    if isinstance(user_id, str) and user_id:
        logger.info(f"Updating user {user_id} to tier {tier}")
        success = await update_clerk_user_metadata(user_id, {"tier": tier})
        if not success:
            logger.error(
                f"Failed to update Clerk metadata for user {user_id}, tier={tier}"
            )
            raise HTTPException(
                status_code=502, detail="Failed to update user metadata"
            )
        await set_cached_user_tier(user_id, tier, redis_client)
        if tier == "pro":
            await set_checkout_grace(user_id, redis_client)
    elif user_id:
        logger.warning(
            "Ignoring non-string user_id in subscription metadata: %r", user_id
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
