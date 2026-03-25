import unittest
from unittest.mock import AsyncMock, patch

from app.clerk_auth import (
    ClerkAuthenticationError,
    authenticate_clerk_token,
    decode_and_verify_clerk_jwt,
)


class ClerkAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_sub_claim_fails_authentication(self) -> None:
        with (
            patch(
                "app.clerk_auth.decode_and_verify_clerk_jwt",
                return_value={"sid": "sess_123"},
            ),
            patch(
                "app.clerk_auth.verify_clerk_session_active",
                new=AsyncMock(return_value="user_123"),
            ),
        ):
            with self.assertRaises(ClerkAuthenticationError) as ctx:
                await authenticate_clerk_token("token", validate_azp=False)

        self.assertEqual(ctx.exception.detail, "Invalid token payload")

    def test_validate_azp_requires_configured_permitted_origins(self) -> None:
        class DummySigningKey:
            key = "signing-key"

        class DummyJwksClient:
            def get_signing_key_from_jwt(self, token: str):
                return DummySigningKey()

        with (
            patch("app.clerk_auth.CLERK_PERMITTED_ORIGINS", ""),
            patch("app.clerk_auth.CLERK_FRONTEND_API", "clerk.example.com"),
            patch("app.clerk_auth.get_jwks_client", return_value=DummyJwksClient()),
            patch("app.clerk_auth.jwt.decode", return_value={"sub": "user_123"}),
        ):
            with self.assertRaises(ClerkAuthenticationError) as ctx:
                decode_and_verify_clerk_jwt(
                    "token",
                    require_session_claims=False,
                    validate_azp=True,
                )

        self.assertEqual(ctx.exception.detail, "Server configuration error")
        self.assertEqual(ctx.exception.status_code, 500)


if __name__ == "__main__":
    unittest.main()
