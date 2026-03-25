import logging

import httpx
import jwt

from .dependencies import (
    CLERK_FRONTEND_API,
    CLERK_PERMITTED_ORIGINS,
    CLERK_SECRET_KEY,
    get_jwks_client,
)

logger = logging.getLogger(__name__)


class ClerkAuthenticationError(Exception):
    def __init__(self, detail: str, status_code: int = 401):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def decode_and_verify_clerk_jwt(
    token: str,
    *,
    require_session_claims: bool = True,
    validate_azp: bool = False,
) -> dict:
    """Decode a Clerk JWT and enforce the claims required by server-side auth."""
    jwks_client = get_jwks_client()
    if not jwks_client or not CLERK_FRONTEND_API:
        logger.error("Clerk JWT verification is not configured")
        raise ClerkAuthenticationError("Server configuration error", status_code=500)

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        expected_issuer = f"https://{CLERK_FRONTEND_API}"
        required_claims = ["exp", "iat", "iss", "nbf"]
        if require_session_claims:
            required_claims.extend(["sid", "sub"])

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
                "require": required_claims,
            },
        )

        if require_session_claims and not payload.get("sid"):
            raise ClerkAuthenticationError("Invalid token payload")

        if validate_azp:
            if not CLERK_PERMITTED_ORIGINS:
                logger.error(
                    "CLERK_PERMITTED_ORIGINS not set but AZP validation requested"
                )
                raise ClerkAuthenticationError(
                    "Server configuration error", status_code=500
                )
            permitted_origins = [
                origin.strip()
                for origin in CLERK_PERMITTED_ORIGINS.split(",")
                if origin.strip()
            ]
            if not permitted_origins:
                logger.error(
                    "CLERK_PERMITTED_ORIGINS is empty but AZP validation requested"
                )
                raise ClerkAuthenticationError(
                    "Server configuration error", status_code=500
                )
            azp = payload.get("azp")
            if not azp:
                raise ClerkAuthenticationError("Missing token origin")
            if azp not in permitted_origins:
                raise ClerkAuthenticationError("Invalid token origin")

        return payload
    except ClerkAuthenticationError:
        raise
    except jwt.ExpiredSignatureError as exc:
        raise ClerkAuthenticationError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        logger.warning("Invalid Clerk JWT: %s", type(exc).__name__)
        raise ClerkAuthenticationError("Invalid token") from exc
    except Exception as exc:
        logger.error("Unexpected Clerk JWT verification error: %s", type(exc).__name__)
        raise ClerkAuthenticationError("Authentication failed") from exc


async def verify_clerk_session_active(session_id: str) -> str:
    """Check that a Clerk session is active and return its user_id."""
    if not session_id:
        raise ClerkAuthenticationError("Invalid token payload")
    if not CLERK_SECRET_KEY:
        logger.error("CLERK_SECRET_KEY not set")
        raise ClerkAuthenticationError("Server configuration error", status_code=500)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"https://api.clerk.com/v1/sessions/{session_id}",
                headers={
                    "Authorization": f"Bearer {CLERK_SECRET_KEY}",
                    "Clerk-API-Version": "2025-11-10",
                },
            )
    except httpx.HTTPError as exc:
        logger.error(
            "Clerk session verification request failed: %s", type(exc).__name__
        )
        raise ClerkAuthenticationError("Authentication failed") from exc

    if response.status_code != 200:
        logger.error("Clerk session verification failed: %s", response.status_code)
        raise ClerkAuthenticationError("Invalid session")

    try:
        session_data = response.json()
    except ValueError as exc:
        logger.error("Failed to parse Clerk session response")
        raise ClerkAuthenticationError("Authentication failed") from exc

    if session_data.get("status") != "active":
        raise ClerkAuthenticationError("Session is not active")

    user_id = session_data.get("user_id")
    if not user_id:
        raise ClerkAuthenticationError("Invalid session")

    return user_id


async def authenticate_clerk_token(
    token: str,
    *,
    validate_azp: bool,
) -> tuple[str, str | None]:
    """
    Authenticate a Clerk token and return (user_id, tier_hint).

    The returned tier comes from JWT public metadata and should be treated as a hint.
    """
    payload = decode_and_verify_clerk_jwt(
        token,
        require_session_claims=True,
        validate_azp=validate_azp,
    )
    session_id = payload.get("sid")
    if not isinstance(session_id, str):
        raise ClerkAuthenticationError("Invalid token payload")

    session_user_id = await verify_clerk_session_active(session_id)
    token_user_id = payload.get("sub")
    if not isinstance(token_user_id, str) or not token_user_id:
        raise ClerkAuthenticationError("Invalid token payload")
    if token_user_id != session_user_id:
        raise ClerkAuthenticationError("Invalid session")

    public_metadata = payload.get("public_metadata")
    tier = public_metadata.get("tier") if isinstance(public_metadata, dict) else None
    return session_user_id, tier
