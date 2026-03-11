import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import redis.asyncio as redis

from app.usage_tracker import (
    NEGATIVE_TIER_CACHE_TTL,
    TIER_CACHE_TTL,
    check_usage,
    get_cached_user_tier,
    get_client_ip,
    get_or_create_tracking_id,
)


def _build_request(
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    client_host: str = "127.0.0.1",
):
    request = Mock()
    request.headers = headers or {}
    request.cookies = cookies or {}
    request.client = SimpleNamespace(host=client_host)
    return request


class UsageTrackerHelperTests(unittest.TestCase):
    def test_cloudflare_ip_takes_precedence(self) -> None:
        request = _build_request(
            headers={
                "cf-connecting-ip": "203.0.113.10",
                "x-forwarded-for": "198.51.100.5, 198.51.100.6",
            }
        )

        self.assertEqual(get_client_ip(request), "203.0.113.10")

    def test_forwarded_for_is_used_when_cloudflare_header_missing(self) -> None:
        request = _build_request(
            headers={"x-forwarded-for": "198.51.100.5, 198.51.100.6"}
        )

        self.assertEqual(get_client_ip(request), "198.51.100.5")

    def test_existing_tracking_cookie_is_reused(self) -> None:
        tracking_id = str(uuid.uuid4())
        request = _build_request(cookies={"cf_track": tracking_id})

        value, created = get_or_create_tracking_id(request)

        self.assertEqual(value, tracking_id)
        self.assertFalse(created)

    def test_invalid_tracking_cookie_is_replaced(self) -> None:
        request = _build_request(cookies={"cf_track": "not-a-uuid"})

        value, created = get_or_create_tracking_id(request)

        uuid.UUID(value)
        self.assertTrue(created)


class UsageTrackerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_cached_user_tier_uses_negative_cache_sentinel(self) -> None:
        redis_client = AsyncMock()
        redis_client.get.return_value = b"none"

        tier = await get_cached_user_tier("user-123", redis_client)

        self.assertIsNone(tier)

    async def test_get_cached_user_tier_fetches_and_caches_on_miss(self) -> None:
        redis_client = AsyncMock()
        redis_client.get.return_value = None

        with patch("app.usage_tracker.fetch_clerk_user_tier", return_value="pro"):
            tier = await get_cached_user_tier("user-123", redis_client)

        self.assertEqual(tier, "pro")
        redis_client.setex.assert_awaited_once_with(
            "user_tier:user-123",
            TIER_CACHE_TTL,
            "pro",
        )

    async def test_get_cached_user_tier_negative_result_is_cached(self) -> None:
        redis_client = AsyncMock()
        redis_client.get.return_value = None

        with patch("app.usage_tracker.fetch_clerk_user_tier", return_value=None):
            tier = await get_cached_user_tier("user-123", redis_client)

        self.assertIsNone(tier)
        redis_client.setex.assert_awaited_once_with(
            "user_tier:user-123",
            NEGATIVE_TIER_CACHE_TTL,
            "none",
        )

    async def test_check_usage_fails_open_when_redis_is_unavailable(self) -> None:
        request = _build_request()

        with patch(
            "app.usage_tracker.extract_user_info_from_token",
            return_value=(None, None),
        ):
            usage_status = await check_usage(request, None)

        self.assertTrue(usage_status.allowed)
        self.assertEqual(usage_status.remaining, -1)
        self.assertFalse(usage_status.is_authenticated)

    async def test_check_usage_handles_redis_errors_for_anonymous_requests(self) -> None:
        request = _build_request(headers={"cf-connecting-ip": "203.0.113.10"})
        redis_client = AsyncMock()
        redis_client.get.side_effect = redis.RedisError("boom")

        with (
            patch(
                "app.usage_tracker.extract_user_info_from_token",
                return_value=(None, None),
            ),
            patch(
                "app.usage_tracker.get_or_create_tracking_id",
                return_value=("track-123", False),
            ),
        ):
            usage_status = await check_usage(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertEqual(usage_status.tracking_id, "track-123")
        self.assertEqual(usage_status.remaining, -1)


if __name__ == "__main__":
    unittest.main()
