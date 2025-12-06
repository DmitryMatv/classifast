import logging
import os

import httpx
import jwt
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from jwt import PyJWKClient
from polar_sdk import Polar
from polar_sdk.webhooks import WebhookVerificationError, validate_event

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

POLAR_ACCESS_TOKEN = os.getenv("POLAR_ACCESS_TOKEN")
POLAR_WEBHOOK_SECRET = os.getenv("POLAR_WEBHOOK_SECRET")
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
CLERK_FRONTEND_API = os.getenv("CLERK_FRONTEND_API", "")
if CLERK_FRONTEND_API:
    CLERK_FRONTEND_API = (
        CLERK_FRONTEND_API.replace("https://", "").replace("http://", "").rstrip("/")
    )
CLERK_PERMITTED_ORIGINS = os.getenv("CLERK_PERMITTED_ORIGINS", "")

# Cached JWKS client for Clerk JWT verification
_jwks_client = None


def get_jwks_client():
    global _jwks_client
    if _jwks_client is None and CLERK_FRONTEND_API:
        _jwks_client = PyJWKClient(
            f"https://{CLERK_FRONTEND_API}/.well-known/jwks.json"
        )
    return _jwks_client


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

        if not product_id:
            raise HTTPException(status_code=400, detail="Missing product_id")

        if not POLAR_ACCESS_TOKEN:
            raise HTTPException(
                status_code=500, detail="Polar Access Token not configured"
            )

        # Fetch user details from Clerk to pre-fill checkout form
        user_details = await get_clerk_user_details(user_id)

        # Initialize Polar SDK
        with Polar(access_token=POLAR_ACCESS_TOKEN) as polar:
            checkout = polar.checkouts.create(
                request={
                    "products": [product_id],
                    "metadata": {"user_id": user_id},
                    "success_url": str(request.base_url) + "?checkout=success",
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

    logger.info(f"Received Polar webhook: {event.type}")

    if event.type == "subscription.created":
        await handle_subscription_update(event.data, tier="pro")
    elif event.type == "subscription.updated":
        status = getattr(event.data, "status", None)
        if status == "active":
            await handle_subscription_update(event.data, tier="pro")
    elif event.type == "subscription.canceled":
        await handle_subscription_update(event.data, tier="free")

    return {"status": "received"}


async def handle_subscription_update(data, tier):
    metadata = getattr(data, "metadata", {}) or {}
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


async def get_clerk_user_details(user_id: str) -> dict:
    """Fetch user email and name from Clerk for checkout pre-fill"""
    if not CLERK_SECRET_KEY:
        logger.warning("CLERK_SECRET_KEY missing, cannot fetch user details")
        return {"email": None, "name": None}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.clerk.com/v1/users/{user_id}",
                headers={"Authorization": f"Bearer {CLERK_SECRET_KEY}"},
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
