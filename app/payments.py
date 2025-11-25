import logging
import os

import httpx
import jwt
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from polar_sdk import Polar
from polar_sdk.models import CheckoutCreate
from svix.webhooks import Webhook, WebhookVerificationError

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

POLAR_ACCESS_TOKEN = os.getenv("POLAR_ACCESS_TOKEN")
POLAR_WEBHOOK_SECRET = os.getenv("POLAR_WEBHOOK_SECRET")
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")


# Dependency to get authenticated user ID from Clerk
async def get_current_user_id(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = authorization.replace("Bearer ", "")

    try:
        # 1. Decode JWT without verification to get 'sid' (session id)
        # In a production environment with more setup, we would verify the JWT signature locally.
        # Here we rely on the subsequent Clerk API call which verifies the session validity.
        payload = jwt.decode(token, options={"verify_signature": False})
        session_id = payload.get("sid")

        if not session_id:
            # Fallback: try to see if 'sub' is the user ID and verify via other means,
            # but 'sid' is standard for Clerk session tokens.
            raise HTTPException(status_code=401, detail="Invalid token payload")

        # 2. Verify session with Clerk API
        if not CLERK_SECRET_KEY:
            logger.error("CLERK_SECRET_KEY not set")
            raise HTTPException(status_code=500, detail="Server configuration error")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.clerk.com/v1/sessions/{session_id}",
                headers={"Authorization": f"Bearer {CLERK_SECRET_KEY}"},
            )

            if response.status_code != 200:
                logger.error(f"Clerk session verification failed: {response.text}")
                raise HTTPException(status_code=401, detail="Invalid session")

            session_data = response.json()
            if session_data.get("status") != "active":
                raise HTTPException(status_code=401, detail="Session is not active")

            return session_data.get("user_id")

    except Exception as e:
        logger.error(f"Auth error: {str(e)}")
        raise HTTPException(status_code=401, detail="Authentication failed")


@router.post("/create-checkout")
async def create_checkout(
    request: Request, user_id: str = Depends(get_current_user_id)
):
    try:
        body = await request.json()
        product_id = body.get("product_id")

        if not product_id:
            raise HTTPException(status_code=400, detail="Missing product_id")

        if not POLAR_ACCESS_TOKEN:
            raise HTTPException(
                status_code=500, detail="Polar Access Token not configured"
            )

        # Initialize Polar SDK
        with Polar(access_token=POLAR_ACCESS_TOKEN) as polar:
            # Create a checkout session
            # We pass the product_id and metadata to link it to the Clerk user
            checkout = polar.checkouts.create(
                request=CheckoutCreate(
                    products=[product_id],
                    metadata={"user_id": user_id},
                    success_url=str(request.base_url) + "?checkout=success",
                )
            )

            return {"url": checkout.url}

    except Exception as e:
        logger.error(f"Error creating checkout: {e}")
        # If the SDK call fails (e.g. attribute error), we might need to adjust based on SDK version
        raise HTTPException(
            status_code=500, detail=f"Failed to create checkout: {str(e)}"
        )


@router.post("/webhooks/polar")
async def polar_webhook(request: Request):
    if not POLAR_WEBHOOK_SECRET:
        logger.error("POLAR_WEBHOOK_SECRET not set")
        raise HTTPException(status_code=500, detail="Webhook secret not configured")

    headers = request.headers
    payload = await request.body()

    try:
        # Verify webhook signature using Svix (Polar uses standard webhooks)
        wh = Webhook(POLAR_WEBHOOK_SECRET)
        # Headers must be a dict of strings
        # request.headers is a Headers object, convert to dict
        headers_dict = dict(headers)
        msg = wh.verify(payload, headers_dict)
    except WebhookVerificationError as e:
        logger.warning(f"Webhook verification failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid webhook signature")
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        raise HTTPException(status_code=400, detail="Webhook error")

    # Handle events
    event_type = msg.get("type")
    data = msg.get("data")

    logger.info(f"Received Polar webhook: {event_type}")

    if event_type == "subscription.created":
        await handle_subscription_update(data, tier="pro")
    elif event_type == "subscription.updated":
        # check status, if canceled or past_due?
        status = data.get("status")
        if status == "active":
            await handle_subscription_update(data, tier="pro")
        else:
            pass  # specific handling if needed
    elif event_type == "subscription.canceled":
        # In Polar, canceled means it will expire at end of period usually,
        # but 'revoked' might be immediate?
        # For simplicity, we'll set tier to free on cancellation or revocation.
        # You might want to check 'ends_at' logic in a real app.
        await handle_subscription_update(data, tier="free")

    return {"status": "received"}


async def handle_subscription_update(data, tier):
    # Metadata is usually at the top level of the resource or in 'metadata' field
    metadata = data.get("metadata", {})
    user_id = metadata.get("user_id")

    if user_id:
        logger.info(f"Updating user {user_id} to tier {tier}")
        await update_clerk_user_metadata(user_id, {"tier": tier})
    else:
        logger.warning("No user_id found in subscription metadata")


async def update_clerk_user_metadata(user_id, metadata):
    if not CLERK_SECRET_KEY:
        logger.error("CLERK_SECRET_KEY missing, cannot update user metadata")
        return

    async with httpx.AsyncClient() as client:
        response = await client.patch(
            f"https://api.clerk.com/v1/users/{user_id}/metadata",
            headers={"Authorization": f"Bearer {CLERK_SECRET_KEY}"},
            json={"public_metadata": metadata},
        )

        if response.status_code != 200:
            logger.error(f"Failed to update Clerk metadata: {response.text}")
