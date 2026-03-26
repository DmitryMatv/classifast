import unittest
from unittest.mock import AsyncMock, patch

from app.clerk_auth import (
    ClerkAuthenticationError,
    authenticate_clerk_token,
    authenticate_clerk_token_local,
    authenticate_clerk_token_with_session,
    decode_and_verify_clerk_jwt,
    should_validate_clerk_azp,
)


class ClerkAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_auth_succeeds_without_session_claims(self) -> None:
        with patch(
            "app.clerk_auth.decode_and_verify_clerk_jwt",
            return_value={"sub": "user_123", "public_metadata": {"tier": "pro"}},
        ):
            user_id, tier = await authenticate_clerk_token_local(
                "token",
                validate_azp=False,
            )

        self.assertEqual(user_id, "user_123")
        self.assertEqual(tier, "pro")

    async def test_local_auth_does_not_verify_live_session(self) -> None:
        with (
            patch(
                "app.clerk_auth.decode_and_verify_clerk_jwt",
                return_value={"sub": "user_123"},
            ),
            patch(
                "app.clerk_auth.verify_clerk_session_active",
                new=AsyncMock(side_effect=AssertionError("should not verify session")),
            ) as verify_mock,
        ):
            user_id, tier = await authenticate_clerk_token_local(
                "token",
                validate_azp=True,
            )

        self.assertEqual(user_id, "user_123")
        self.assertIsNone(tier)
        verify_mock.assert_not_called()

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

    async def test_session_auth_rejects_subject_mismatch(self) -> None:
        with (
            patch(
                "app.clerk_auth.decode_and_verify_clerk_jwt",
                return_value={"sid": "sess_123", "sub": "user_token"},
            ),
            patch(
                "app.clerk_auth.verify_clerk_session_active",
                new=AsyncMock(return_value="user_session"),
            ),
        ):
            with self.assertRaises(ClerkAuthenticationError) as ctx:
                await authenticate_clerk_token_with_session(
                    "token",
                    validate_azp=False,
                )

        self.assertEqual(ctx.exception.detail, "Invalid session")

    async def test_backward_compatible_auth_wrapper_uses_session_auth(self) -> None:
        with (
            patch(
                "app.clerk_auth.decode_and_verify_clerk_jwt",
                return_value={"sid": "sess_123", "sub": "user_123"},
            ),
            patch(
                "app.clerk_auth.verify_clerk_session_active",
                new=AsyncMock(return_value="user_123"),
            ) as verify_mock,
        ):
            user_id, tier = await authenticate_clerk_token(
                "token",
                validate_azp=False,
            )

        self.assertEqual(user_id, "user_123")
        self.assertIsNone(tier)
        verify_mock.assert_awaited_once_with("sess_123")

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

    def test_should_validate_clerk_azp_reflects_configured_origins(self) -> None:
        with patch("app.clerk_auth.CLERK_PERMITTED_ORIGINS", " , https://a.example , "):
            self.assertTrue(should_validate_clerk_azp())

        with patch("app.clerk_auth.CLERK_PERMITTED_ORIGINS", " , , "):
            self.assertFalse(should_validate_clerk_azp())


if __name__ == "__main__":
    unittest.main()
