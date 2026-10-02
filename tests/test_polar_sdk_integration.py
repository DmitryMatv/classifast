import asyncio
import base64
import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI

from app import payments
from app.mapping_store import MAPPING_PRODUCTS
from app.usage_tracker import GRACE_PERIOD_TTL, TIER_CACHE_TTL

PRO_PRODUCT_ID = "6f42124b-9738-400a-adc6-07835298f6f3"
CHECKOUT_ID = "cd1f9dfe-5c58-43b9-beb7-542f048c193d"
CUSTOMER_ID = "917a921b-2790-43fb-a76b-6848ea973939"
ORGANIZATION_ID = "ae247836-4fc1-42ae-a4a6-3313b5c123d7"
PRICE_ID = "c2905b01-177c-4e81-a38e-aa350df76e0d"
WEBHOOK_SECRET = "whsec_" + base64.b64encode(b"local-test-signing-key").decode()
CREATED_AT = "2026-10-02T09:00:00Z"


def _price(product_id):
    return {
        "created_at": CREATED_AT,
        "modified_at": None,
        "id": PRICE_ID,
        "source": "catalog",
        "amount_type": "fixed",
        "price_currency": "usd",
        "tax_behavior": "exclusive",
        "is_archived": False,
        "product_id": product_id,
        "price_amount": 1900,
    }


def _product(product_id, recurring=True):
    return {
        "id": product_id,
        "created_at": CREATED_AT,
        "modified_at": None,
        "trial_interval": None,
        "trial_interval_count": None,
        "name": "Classifast Pro" if recurring else "Classification mapping",
        "description": "Classification service",
        "visibility": "public",
        "recurring_interval": "month" if recurring else None,
        "recurring_interval_count": 1 if recurring else None,
        "meter_interval": None,
        "meter_interval_count": None,
        "is_recurring": recurring,
        "is_archived": False,
        "organization_id": ORGANIZATION_ID,
        "metadata": {},
        "is_deletable": False,
        "prices": [_price(product_id)],
        "benefits": [],
        "medias": [],
        "attached_custom_fields": [],
    }


def _checkout_payload(product_id, recurring=True):
    product = _product(product_id, recurring)
    return {
        "id": CHECKOUT_ID,
        "created_at": CREATED_AT,
        "modified_at": None,
        "payment_processor": "stripe",
        "status": "open",
        "client_secret": "local-checkout-client-secret",
        "url": f"https://checkout.polar.sh/c/{CHECKOUT_ID}",
        "expires_at": "2026-10-03T09:00:00Z",
        "success_url": "http://testserver/?checkout=success",
        "return_url": None,
        "embed_origin": None,
        "amount": 1900,
        "units": None,
        "min_units": None,
        "max_units": None,
        "discount_amount": 0,
        "net_amount": 1900,
        "tax_amount": None,
        "tax_behavior": None,
        "total_amount": 1900,
        "currency": "usd",
        "allow_trial": True,
        "active_trial_interval": None,
        "active_trial_interval_count": None,
        "trial_end": None,
        "organization_id": ORGANIZATION_ID,
        "product_id": product_id,
        "product_price_id": PRICE_ID,
        "discount_id": None,
        "allow_discount_codes": True,
        "require_billing_address": False,
        "is_discount_applicable": True,
        "is_free_product_price": False,
        "is_payment_required": True,
        "is_payment_setup_required": recurring,
        "is_payment_form_required": True,
        "customer_id": None,
        "is_business_customer": False,
        "customer_name": None,
        "customer_email": None,
        "customer_ip_address": None,
        "customer_billing_name": None,
        "customer_billing_address": None,
        "customer_tax_id": None,
        "payment_method_type": None,
        "payment_processor_metadata": {},
        "billing_address_fields": {
            "country": "required",
            "state": "optional",
            "city": "optional",
            "postal_code": "optional",
            "line1": "optional",
            "line2": "optional",
        },
        "trial_interval": None,
        "trial_interval_count": None,
        "metadata": {},
        "external_customer_id": None,
        "products": [product],
        "product": product,
        "product_price": _price(product_id),
        "prices": {product_id: [_price(product_id)]},
        "discount": None,
        "subscription_id": None,
        "attached_custom_fields": [],
        "customer_metadata": {},
    }


def _subscription_payload(event_type, status, product_id=PRO_PRODUCT_ID):
    return {
        "type": event_type,
        "timestamp": CREATED_AT,
        "api_version": "2026-10",
        "data": {
            "created_at": CREATED_AT,
            "modified_at": None,
            "id": "6dbd21e0-54c3-4e1b-9985-dded58f52403",
            "amount": 1900,
            "currency": "usd",
            "recurring_interval": "month",
            "recurring_interval_count": 1,
            "status": status,
            "current_period_start": CREATED_AT,
            "current_period_end": "2026-11-02T09:00:00Z",
            "current_meter_period_start": None,
            "current_meter_period_end": None,
            "trial_start": CREATED_AT if status == "trialing" else None,
            "trial_end": "2026-10-09T09:00:00Z" if status == "trialing" else None,
            "cancel_at_period_end": False,
            "canceled_at": CREATED_AT if status == "canceled" else None,
            "started_at": CREATED_AT,
            "ends_at": None,
            "ended_at": CREATED_AT if status == "canceled" else None,
            "pause_at_period_end": False,
            "paused_at": None,
            "resumes_at": None,
            "customer_id": CUSTOMER_ID,
            "product_id": product_id,
            "discount_id": None,
            "checkout_id": CHECKOUT_ID,
            "units": None,
            "customer_cancellation_reason": None,
            "customer_cancellation_comment": None,
            "metadata": {"user_id": "user_123"},
            "customer": {
                "id": CUSTOMER_ID,
                "created_at": CREATED_AT,
                "modified_at": None,
                "metadata": {},
                "email": "buyer@example.com",
                "email_verified": True,
                "type": "individual",
                "name": "Test Buyer",
                "billing_name": None,
                "billing_address": None,
                "tax_id": None,
                "organization_id": ORGANIZATION_ID,
                "deleted_at": None,
                "first_user_event_at": None,
                "avatar_url": None,
            },
            "product": _product(product_id),
            "discount": None,
            "prices": [_price(product_id)],
            "meters": [],
            "pending_update": None,
        },
    }


def _signed_headers(body, timestamp=None, key_encoding="base64"):
    timestamp = int(time.time()) if timestamp is None else timestamp
    signed = b"local-webhook-id." + str(timestamp).encode() + b"." + body
    key = (
        base64.b64decode(WEBHOOK_SECRET.removeprefix("whsec_"))
        if key_encoding == "base64"
        else WEBHOOK_SECRET.encode()
    )
    signature = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest())
    return {
        "content-type": "application/json",
        "webhook-id": "local-webhook-id",
        "webhook-timestamp": str(timestamp),
        "webhook-signature": "v1," + signature.decode(),
    }


@pytest.fixture
def payment_app(monkeypatch):
    app = FastAPI()
    app.include_router(payments.router, prefix="/api")
    app.state.redis_client = AsyncMock()
    app.state.redis_client.incr.return_value = 1
    monkeypatch.setenv("POLAR_PRO_PRODUCT_ID", PRO_PRODUCT_ID)
    monkeypatch.setattr(payments, "POLAR_ACCESS_TOKEN", "local-test-access-token")
    monkeypatch.setattr(payments, "POLAR_WEBHOOK_SECRET", WEBHOOK_SECRET)
    return app


async def _post(app, path, **kwargs):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as client:
        return await client.post(path, **kwargs)


@pytest.mark.parametrize("kind", ["pro", "mapping"])
def test_checkout_routes_use_real_sdk_request_and_response(payment_app, kind):
    mapping = MAPPING_PRODUCTS["unspsc-to-cpv-mapping"]
    is_pro = kind == "pro"
    product_id = PRO_PRODUCT_ID if is_pro else mapping.polar_product_id
    path = "/api/create-checkout" if is_pro else "/api/create-mapping-checkout"
    return_url = "http://testserver/NAICS/?q=bolts"
    payment_app.dependency_overrides[payments.get_current_user_id] = lambda: "user_123"
    clerk_details = AsyncMock(
        return_value={"email": "buyer@example.com", "name": "Test Buyer"}
    )
    payload = _checkout_payload(product_id, recurring=is_pro)

    with (
        patch(
            "httpx.Client.send", return_value=httpx.Response(201, json=payload)
        ) as send,
        patch("app.payments.get_clerk_user_details", clerk_details),
    ):
        response = asyncio.run(
            _post(
                payment_app,
                path,
                json={
                    "slug": mapping.slug,
                    "product_id": "untrusted-client-product",
                    "return_url": return_url,
                },
            )
        )

    assert response.status_code == 200
    assert response.json() == {"url": f"https://checkout.polar.sh/c/{CHECKOUT_ID}"}
    assert "set-cookie" not in response.headers
    send.assert_called_once()
    request = send.call_args.args[0]
    assert request.method == "POST"
    assert request.url == "https://api.polar.sh/v1/checkouts/"
    assert request.headers["Polar-Version"] == "2026-10"
    assert request.headers["Authorization"] == "Bearer local-test-access-token"
    body = json.loads(request.content)
    expected = {
        "products": [product_id],
        "success_url": "http://testserver/NAICS/?q=bolts&checkout=success",
    }
    if is_pro:
        expected.update(
            metadata={"user_id": "user_123"},
            customer_email="buyer@example.com",
            customer_name="Test Buyer",
        )
        clerk_details.assert_awaited_once_with("user_123")
    else:
        expected["metadata"] = {
            "mapping_slug": mapping.slug,
            "source_standard": mapping.source_standard,
            "target_standard": mapping.target_standard,
        }
        clerk_details.assert_not_awaited()
    assert body == expected
    payment_app.state.redis_client.setex.assert_not_awaited()


@pytest.mark.parametrize("key_encoding", ["base64", "utf8"])
@pytest.mark.parametrize(
    "event_type,status,tier",
    [
        ("subscription.created", "trialing", "pro"),
        ("subscription.active", "active", "pro"),
        ("subscription.updated", "trialing", "pro"),
        ("subscription.updated", "active", "pro"),
        ("subscription.updated", "canceled", "free"),
        ("subscription.updated", "past_due", "free"),
        ("subscription.canceled", "active", None),
        ("subscription.revoked", "canceled", None),
    ],
)
def test_signed_subscription_events_keep_entitlement_contract(
    payment_app, event_type, status, tier, key_encoding
):
    body = json.dumps(_subscription_payload(event_type, status)).encode()
    clerk_update = AsyncMock(return_value=True)
    with patch("app.payments.update_clerk_user_metadata", clerk_update):
        response = asyncio.run(
            _post(
                payment_app,
                "/api/webhooks/polar",
                content=body,
                headers=_signed_headers(body, key_encoding=key_encoding),
            )
        )

    assert response.status_code == 200
    assert response.json() == {"status": "received"}
    assert "set-cookie" not in response.headers
    redis_client = payment_app.state.redis_client
    if tier is None:
        clerk_update.assert_not_awaited()
        redis_client.setex.assert_not_awaited()
    else:
        clerk_update.assert_awaited_once_with("user_123", {"tier": tier})
        redis_client.setex.assert_any_await("user_tier:user_123", TIER_CACHE_TTL, tier)
        if tier == "pro":
            redis_client.setex.assert_any_await(
                "checkout_grace:user_123", GRACE_PERIOD_TTL, "1"
            )
            assert redis_client.setex.await_count == 2
        else:
            assert redis_client.setex.await_count == 1


def test_signed_non_pro_subscription_does_not_change_entitlements(payment_app):
    body = json.dumps(
        _subscription_payload("subscription.active", "active", "other-product")
    ).encode()
    clerk_update = AsyncMock(return_value=True)
    with patch("app.payments.update_clerk_user_metadata", clerk_update):
        response = asyncio.run(
            _post(
                payment_app,
                "/api/webhooks/polar",
                content=body,
                headers=_signed_headers(body),
            )
        )

    assert response.status_code == 200
    clerk_update.assert_not_awaited()
    payment_app.state.redis_client.setex.assert_not_awaited()


@pytest.mark.parametrize("event_type", ["subscription.active", "future.event"])
@pytest.mark.parametrize("invalid_signature", ["tampered", "missing", "stale"])
def test_invalid_signed_webhooks_do_not_write_entitlements(
    payment_app, invalid_signature, event_type
):
    body = json.dumps(_subscription_payload(event_type, "active")).encode()
    headers = (
        _signed_headers(body, int(time.time()) - 3600)
        if invalid_signature == "stale"
        else _signed_headers(body)
    )
    if invalid_signature == "tampered":
        body += b" "
    elif invalid_signature == "missing":
        del headers["webhook-signature"]
    clerk_update = AsyncMock(return_value=True)
    with patch("app.payments.update_clerk_user_metadata", clerk_update):
        response = asyncio.run(
            _post(payment_app, "/api/webhooks/polar", content=body, headers=headers)
        )

    assert response.status_code == 403
    assert response.json() == {"detail": "Invalid webhook signature"}
    assert "set-cookie" not in response.headers
    clerk_update.assert_not_awaited()
    payment_app.state.redis_client.setex.assert_not_awaited()


@pytest.mark.parametrize("key_encoding", ["base64", "utf8"])
def test_signed_unknown_webhook_is_acknowledged_without_entitlement_changes(
    payment_app, key_encoding
):
    body = json.dumps(
        {
            "type": "future.event",
            "timestamp": CREATED_AT,
            "api_version": "2026-10",
            "data": {},
        }
    ).encode()
    clerk_update = AsyncMock(return_value=True)
    with patch("app.payments.update_clerk_user_metadata", clerk_update):
        response = asyncio.run(
            _post(
                payment_app,
                "/api/webhooks/polar",
                content=body,
                headers=_signed_headers(body, key_encoding=key_encoding),
            )
        )

    assert response.status_code == 200
    assert response.json() == {"status": "received"}
    assert "set-cookie" not in response.headers
    clerk_update.assert_not_awaited()
    payment_app.state.redis_client.setex.assert_not_awaited()


@pytest.mark.parametrize(
    "body",
    [b"{", b'{"type":"subscription.updated","data":{}}', b"{}", b"[]"],
    ids=["invalid-json", "invalid-subscription", "missing-type", "non-object"],
)
def test_signed_malformed_webhook_returns_payload_error_without_entitlement_changes(
    payment_app, body
):
    clerk_update = AsyncMock(return_value=True)
    with patch("app.payments.update_clerk_user_metadata", clerk_update):
        response = asyncio.run(
            _post(
                payment_app,
                "/api/webhooks/polar",
                content=body,
                headers=_signed_headers(body),
            )
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid webhook payload"}
    assert "set-cookie" not in response.headers
    clerk_update.assert_not_awaited()
    payment_app.state.redis_client.setex.assert_not_awaited()
