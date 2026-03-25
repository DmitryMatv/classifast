import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi import FastAPI, HTTPException
from polar_sdk._webhooks import WebhookVerificationError

from app import payments
from app.mapping_store import MAPPING_PRODUCTS


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

    async def test_create_checkout_ignores_client_supplied_product_id(self) -> None:
        request = AsyncMock()
        request.json.return_value = {
            "product_id": "attacker-product",
            "return_url": "http://testserver/NAICS/",
        }
        request.base_url = "http://testserver/"
        request.app.state.redis_client = AsyncMock()
        polar_instance = MagicMock()
        polar_instance.checkouts.create.return_value = SimpleNamespace(
            url="https://polar.example/checkout"
        )
        polar_context = MagicMock()
        polar_context.__enter__.return_value = polar_instance
        polar_context.__exit__.return_value = None

        with (
            patch.dict(
                os.environ,
                {"POLAR_PRO_PRODUCT_ID": "configured-pro-product"},
                clear=False,
            ),
            patch("app.payments.POLAR_ACCESS_TOKEN", "polar-token"),
            patch("app.payments.Polar", return_value=polar_context),
            patch(
                "app.payments.get_clerk_user_details",
                new=AsyncMock(return_value={"email": None, "name": None}),
            ),
        ):
            response = await payments.create_checkout(request, user_id="user_123")

        self.assertEqual(response["url"], "https://polar.example/checkout")
        request_payload = polar_instance.checkouts.create.call_args.kwargs["request"]
        self.assertEqual(request_payload["products"], ["configured-pro-product"])
        self.assertEqual(request_payload["metadata"]["user_id"], "user_123")

    async def test_create_checkout_requires_pro_product_configuration(self) -> None:
        request = AsyncMock()
        request.json.return_value = {"return_url": "http://testserver/NAICS/"}
        request.base_url = "http://testserver/"

        with (
            patch.dict(
                os.environ,
                {"POLAR_PRO_PRODUCT_ID": "", "POLAR_PRO_PRODUCT_IDS": ""},
                clear=False,
            ),
            patch("app.payments.POLAR_ACCESS_TOKEN", "polar-token"),
            self.assertRaises(HTTPException) as ctx,
        ):
            await payments.create_checkout(request, user_id="user_123")

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(ctx.exception.detail, "Polar Pro product not configured")

    async def test_create_mapping_checkout_rejects_unknown_slug(self) -> None:
        response = await self._post_json(
            "/api/create-mapping-checkout",
            {
                "slug": "missing-product",
                "return_url": "http://testserver/mapping/missing-product/",
            },
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Mapping product not found")

    async def test_create_mapping_checkout_rejects_invalid_return_url(self) -> None:
        response = await self._post_json(
            "/api/create-mapping-checkout",
            {
                "slug": next(iter(MAPPING_PRODUCTS)),
                "return_url": "https://evil.example/mapping/redirect/",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid return_url")

    async def test_create_mapping_checkout_uses_configured_polar_product_id(
        self,
    ) -> None:
        product = next(iter(MAPPING_PRODUCTS.values()))
        polar_instance = MagicMock()
        polar_instance.checkouts.create.return_value = SimpleNamespace(
            url="https://polar.example/checkout"
        )
        polar_context = MagicMock()
        polar_context.__enter__.return_value = polar_instance
        polar_context.__exit__.return_value = None

        with (
            patch("app.payments.POLAR_ACCESS_TOKEN", "polar-token"),
            patch("app.payments.Polar", return_value=polar_context),
        ):
            response = await self._post_json(
                "/api/create-mapping-checkout",
                {
                    "slug": product.slug,
                    "return_url": f"http://testserver/mapping/{product.slug}/",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["url"], "https://polar.example/checkout")
        polar_instance.checkouts.create.assert_called_once()
        request_payload = polar_instance.checkouts.create.call_args.kwargs["request"]
        self.assertEqual(request_payload["products"], [product.polar_product_id])
        self.assertEqual(request_payload["metadata"]["mapping_slug"], product.slug)
        self.assertEqual(
            request_payload["success_url"],
            f"http://testserver/mapping/{product.slug}/?checkout=success",
        )


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

    async def test_trialing_subscription_update_routes_to_pro_tier_only_for_allowed_product(
        self,
    ) -> None:
        class DummyUpdatedPayload:
            TYPE = "subscription.updated"

            def __init__(self, status: str, product_id: str | None):
                self.data = SimpleNamespace(
                    status=status,
                    product_id=product_id,
                    metadata={"user_id": "u1"},
                )

        with (
            patch.dict(
                os.environ,
                {"POLAR_PRO_PRODUCT_ID": "allowed-product"},
                clear=False,
            ),
            patch("app.payments.POLAR_WEBHOOK_SECRET", "secret"),
            patch(
                "app.payments.WebhookSubscriptionUpdatedPayload", DummyUpdatedPayload
            ),
            patch(
                "app.payments.validate_event",
                return_value=DummyUpdatedPayload("trialing", "allowed-product"),
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

    async def test_non_allowlisted_subscription_update_is_ignored(self) -> None:
        class DummyUpdatedPayload:
            TYPE = "subscription.updated"

            def __init__(self, status: str, product_id: str | None):
                self.data = SimpleNamespace(
                    status=status,
                    product_id=product_id,
                    metadata={"user_id": "u1"},
                )

        with (
            patch.dict(
                os.environ,
                {"POLAR_PRO_PRODUCT_ID": "allowed-product"},
                clear=False,
            ),
            patch("app.payments.POLAR_WEBHOOK_SECRET", "secret"),
            patch(
                "app.payments.WebhookSubscriptionUpdatedPayload", DummyUpdatedPayload
            ),
            patch(
                "app.payments.validate_event",
                return_value=DummyUpdatedPayload("trialing", "other-product"),
            ),
            patch(
                "app.payments.handle_subscription_update",
                new_callable=AsyncMock,
            ) as handler_mock,
        ):
            response = await self._post_webhook()

        self.assertEqual(response.status_code, 200)
        handler_mock.assert_not_awaited()

    async def test_allowlisted_subscription_canceled_event_skips_tier_update(
        self,
    ) -> None:
        class DummyCanceledPayload:
            TYPE = "subscription.canceled"

            def __init__(self, product_id: str | None):
                self.data = SimpleNamespace(
                    product_id=product_id,
                    metadata={"user_id": "u1"},
                )

        with (
            patch.dict(
                os.environ,
                {"POLAR_PRO_PRODUCT_ID": "allowed-product"},
                clear=False,
            ),
            patch("app.payments.POLAR_WEBHOOK_SECRET", "secret"),
            patch(
                "app.payments.WebhookSubscriptionCanceledPayload",
                DummyCanceledPayload,
            ),
            patch(
                "app.payments.validate_event",
                return_value=DummyCanceledPayload("allowed-product"),
            ),
            patch(
                "app.payments.handle_subscription_update",
                new_callable=AsyncMock,
            ) as handler_mock,
        ):
            response = await self._post_webhook()

        self.assertEqual(response.status_code, 200)
        handler_mock.assert_not_awaited()

    async def test_allowlisted_subscription_revoked_event_skips_tier_update(
        self,
    ) -> None:
        class DummyRevokedPayload:
            TYPE = "subscription.revoked"

            def __init__(self, product_id: str | None):
                self.data = SimpleNamespace(
                    product_id=product_id,
                    metadata={"user_id": "u1"},
                )

        with (
            patch.dict(
                os.environ,
                {"POLAR_PRO_PRODUCT_ID": "allowed-product"},
                clear=False,
            ),
            patch("app.payments.POLAR_WEBHOOK_SECRET", "secret"),
            patch(
                "app.payments.WebhookSubscriptionRevokedPayload",
                DummyRevokedPayload,
            ),
            patch(
                "app.payments.validate_event",
                return_value=DummyRevokedPayload("allowed-product"),
            ),
            patch(
                "app.payments.handle_subscription_update",
                new_callable=AsyncMock,
            ) as handler_mock,
        ):
            response = await self._post_webhook()

        self.assertEqual(response.status_code, 200)
        handler_mock.assert_not_awaited()

    async def test_subscription_update_without_product_identity_is_ignored(
        self,
    ) -> None:
        class DummyUpdatedPayload:
            TYPE = "subscription.updated"

            def __init__(self, status: str):
                self.data = SimpleNamespace(status=status, metadata={"user_id": "u1"})

        with (
            patch.dict(
                os.environ,
                {"POLAR_PRO_PRODUCT_ID": "allowed-product"},
                clear=False,
            ),
            patch("app.payments.POLAR_WEBHOOK_SECRET", "secret"),
            patch(
                "app.payments.WebhookSubscriptionUpdatedPayload", DummyUpdatedPayload
            ),
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
        handler_mock.assert_not_awaited()

    async def test_allowlisted_canceled_subscription_update_routes_to_free_tier(
        self,
    ) -> None:
        class DummyUpdatedPayload:
            TYPE = "subscription.updated"

            def __init__(self, status: str, product_id: str | None):
                self.data = SimpleNamespace(
                    status=status,
                    product_id=product_id,
                    metadata={"user_id": "u1"},
                )

        with (
            patch.dict(
                os.environ,
                {"POLAR_PRO_PRODUCT_ID": "allowed-product"},
                clear=False,
            ),
            patch("app.payments.POLAR_WEBHOOK_SECRET", "secret"),
            patch(
                "app.payments.WebhookSubscriptionUpdatedPayload", DummyUpdatedPayload
            ),
            patch(
                "app.payments.validate_event",
                return_value=DummyUpdatedPayload("canceled", "allowed-product"),
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

    async def test_non_allowlisted_terminal_subscription_update_is_ignored(
        self,
    ) -> None:
        class DummyUpdatedPayload:
            TYPE = "subscription.updated"

            def __init__(self, status: str, product_id: str | None):
                self.data = SimpleNamespace(
                    status=status,
                    product_id=product_id,
                    metadata={"user_id": "u1"},
                )

        with (
            patch.dict(
                os.environ,
                {"POLAR_PRO_PRODUCT_ID": "allowed-product"},
                clear=False,
            ),
            patch("app.payments.POLAR_WEBHOOK_SECRET", "secret"),
            patch(
                "app.payments.WebhookSubscriptionUpdatedPayload", DummyUpdatedPayload
            ),
            patch(
                "app.payments.validate_event",
                return_value=DummyUpdatedPayload("past_due", "other-product"),
            ),
            patch(
                "app.payments.handle_subscription_update",
                new_callable=AsyncMock,
            ) as handler_mock,
        ):
            response = await self._post_webhook()

        self.assertEqual(response.status_code, 200)
        handler_mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
