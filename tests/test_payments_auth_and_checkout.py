import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI
from fastapi import HTTPException
from polar_sdk._webhooks import WebhookVerificationError

from app import payments


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(payments.router, prefix="/api")
    app.state.redis_client = AsyncMock()
    return app


class CheckoutRouteTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_test_app()

    async def asyncSetUp(self) -> None:
        self.app.dependency_overrides.clear()

    async def asyncTearDown(self) -> None:
        self.app.dependency_overrides.clear()

    async def _post_json(self, path: str, payload: dict, headers: dict | None = None):
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(path, json=payload, headers=headers)

    async def test_create_checkout_requires_auth_header(self) -> None:
        response = await self._post_json(
            "/api/create-checkout",
            {"product_id": "prod_123"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Missing Authorization header")

    async def test_create_checkout_rejects_invalid_return_url(self) -> None:
        request = AsyncMock()
        request.json.return_value = {
            "product_id": "prod_123",
            "return_url": "https://evil.example/checkout-complete",
        }
        request.base_url = "https://classifast.com/"

        with (
            patch("app.payments.POLAR_ACCESS_TOKEN", "polar-token"),
            self.assertRaises(HTTPException) as ctx,
        ):
            await payments.create_checkout(request, user_id="user_123")

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "Invalid return_url")


class WebhookRouteTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_test_app()

    async def _post_webhook(self, payload: bytes = b"{}") -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/api/webhooks/polar",
                content=payload,
                headers={"content-type": "application/json"},
            )

    async def test_invalid_webhook_signature_is_rejected(self) -> None:
        with (
            patch("app.payments.POLAR_WEBHOOK_SECRET", "secret"),
            patch(
                "app.payments.validate_event",
                side_effect=WebhookVerificationError("invalid signature"),
            ),
        ):
            response = await self._post_webhook()

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Invalid webhook signature")

    async def test_trialing_subscription_update_routes_to_pro_tier(self) -> None:
        class DummyUpdatedPayload:
            TYPE = "subscription.updated"

            def __init__(self, status: str):
                self.data = SimpleNamespace(status=status, metadata={"user_id": "u1"})

        with (
            patch("app.payments.POLAR_WEBHOOK_SECRET", "secret"),
            patch("app.payments.WebhookSubscriptionUpdatedPayload", DummyUpdatedPayload),
            patch(
                "app.payments.validate_event",
                return_value=DummyUpdatedPayload("trialing"),
            ),
            patch(
                "app.payments.handle_subscription_update",
                new_callable=AsyncMock,
            ) as handler_mock,
        ):
            response = await self._post_webhook()

        self.assertEqual(response.status_code, 200)
        handler_mock.assert_awaited_once()
        _, kwargs = handler_mock.await_args
        self.assertEqual(kwargs["tier"], "pro")

    async def test_canceled_subscription_update_routes_to_free_tier(self) -> None:
        class DummyUpdatedPayload:
            TYPE = "subscription.updated"

            def __init__(self, status: str):
                self.data = SimpleNamespace(status=status, metadata={"user_id": "u1"})

        with (
            patch("app.payments.POLAR_WEBHOOK_SECRET", "secret"),
            patch("app.payments.WebhookSubscriptionUpdatedPayload", DummyUpdatedPayload),
            patch(
                "app.payments.validate_event",
                return_value=DummyUpdatedPayload("canceled"),
            ),
            patch(
                "app.payments.handle_subscription_update",
                new_callable=AsyncMock,
            ) as handler_mock,
        ):
            response = await self._post_webhook()

        self.assertEqual(response.status_code, 200)
        handler_mock.assert_awaited_once()
        _, kwargs = handler_mock.await_args
        self.assertEqual(kwargs["tier"], "free")


if __name__ == "__main__":
    unittest.main()
