import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import redis.asyncio as redis

from app.clerk_auth import ClerkAuthenticationError
from app.usage_tracker import (
    FREE_USER_LIMIT,
    NEGATIVE_TIER_CACHE_TTL,
    TIER_CACHE_TTL,
    USAGE_TTL,
    UsageStatus,
    check_usage,
    get_cached_user_tier,
    get_client_ip,
    get_or_create_tracking_id,
    hash_ip,
    increment_usage,
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

    async def test_stale_jwt_pro_hint_is_not_treated_as_unlimited(self) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client = AsyncMock()
        redis_client.get.return_value = "5"

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token",
                new=AsyncMock(return_value=("user-123", "pro")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(return_value="free"),
            ),
        ):
            usage_status = await check_usage(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertFalse(usage_status.is_pro)
        self.assertEqual(usage_status.remaining, FREE_USER_LIMIT - 5)

    async def test_invalid_session_falls_back_to_anonymous_quota(self) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client = AsyncMock()
        redis_client.get.side_effect = ["0", "0"]

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token",
                new=AsyncMock(side_effect=ClerkAuthenticationError("Invalid session")),
            ),
            patch(
                "app.usage_tracker.get_or_create_tracking_id",
                return_value=("track-123", False),
            ),
        ):
            usage_status = await check_usage(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertFalse(usage_status.is_authenticated)
        self.assertEqual(usage_status.tracking_id, "track-123")

    async def test_valid_active_session_with_pro_tier_is_unlimited(self) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token",
                new=AsyncMock(return_value=("user-123", "pro")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(return_value="pro"),
            ),
        ):
            usage_status = await check_usage(request, AsyncMock())

        self.assertTrue(usage_status.allowed)
        self.assertTrue(usage_status.is_authenticated)
        self.assertTrue(usage_status.is_pro)
        self.assertEqual(usage_status.remaining, -1)

    async def test_valid_active_session_with_free_tier_uses_free_quota(self) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client = AsyncMock()
        redis_client.get.return_value = "2"

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token",
                new=AsyncMock(return_value=("user-123", "free")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(return_value="free"),
            ),
        ):
            usage_status = await check_usage(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertTrue(usage_status.is_authenticated)
        self.assertFalse(usage_status.is_pro)
        self.assertEqual(usage_status.remaining, FREE_USER_LIMIT - 2)

    async def test_invalid_session_cookie_is_treated_as_anonymous(self) -> None:
        request = _build_request(cookies={"__session": "session-token"})
        redis_client = AsyncMock()
        redis_client.get.side_effect = ["0", "0"]

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token",
                new=AsyncMock(side_effect=ClerkAuthenticationError("Invalid session")),
            ),
            patch(
                "app.usage_tracker.get_or_create_tracking_id",
                return_value=("track-456", False),
            ),
        ):
            usage_status = await check_usage(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertFalse(usage_status.is_authenticated)
        self.assertEqual(usage_status.tracking_id, "track-456")

    async def test_redis_unavailable_fails_open_by_default(self) -> None:
        request = _build_request()

        with patch("app.usage_tracker.QUOTA_FAIL_OPEN", True):
            usage_status = await check_usage(request, None)

        self.assertTrue(usage_status.allowed)
        self.assertEqual(usage_status.remaining, -1)
        self.assertFalse(usage_status.is_authenticated)

    async def test_redis_unavailable_is_denied_when_fail_open_disabled(self) -> None:
        request = _build_request()

        with (
            patch("app.usage_tracker.QUOTA_FAIL_OPEN", False),
            patch(
                "app.usage_tracker.get_or_create_tracking_id",
                return_value=("track-123", False),
            ),
        ):
            usage_status = await check_usage(request, None)

        self.assertFalse(usage_status.allowed)
        self.assertEqual(usage_status.remaining, 0)
        self.assertEqual(usage_status.tracking_id, "track-123")

    async def test_redis_unavailable_can_fail_open_when_enabled(self) -> None:
        request = _build_request()

        with patch("app.usage_tracker.QUOTA_FAIL_OPEN", True):
            usage_status = await check_usage(request, None)

        self.assertTrue(usage_status.allowed)
        self.assertEqual(usage_status.remaining, -1)

    async def test_checkout_grace_allows_verified_user(self) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token",
                new=AsyncMock(return_value=("user-123", "free")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=True),
            ),
        ):
            usage_status = await check_usage(request, AsyncMock())

        self.assertTrue(usage_status.allowed)
        self.assertTrue(usage_status.is_pro)
        self.assertEqual(usage_status.tracking_id, "user-123")

    async def test_checkout_grace_does_not_help_invalid_identity(self) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client = AsyncMock()
        redis_client.get.side_effect = ["0", "0"]

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token",
                new=AsyncMock(side_effect=ClerkAuthenticationError("Invalid session")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "app.usage_tracker.get_or_create_tracking_id",
                return_value=("track-999", False),
            ),
        ):
            usage_status = await check_usage(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertFalse(usage_status.is_authenticated)
        self.assertFalse(usage_status.is_pro)
        self.assertEqual(usage_status.tracking_id, "track-999")

    async def test_check_usage_handles_redis_errors_for_anonymous_requests(
        self,
    ) -> None:
        request = _build_request(headers={"cf-connecting-ip": "203.0.113.10"})
        redis_client = AsyncMock()
        redis_client.get.side_effect = redis.RedisError("boom")

        with (
            patch("app.usage_tracker.QUOTA_FAIL_OPEN", False),
            patch(
                "app.usage_tracker.get_or_create_tracking_id",
                return_value=("track-123", False),
            ),
        ):
            usage_status = await check_usage(request, redis_client)

        self.assertFalse(usage_status.allowed)
        self.assertEqual(usage_status.tracking_id, "track-123")
        self.assertEqual(usage_status.remaining, 0)

    async def test_check_usage_handles_redis_errors_for_authenticated_users(
        self,
    ) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client = AsyncMock()
        redis_client.get.side_effect = redis.RedisError("boom")

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token",
                new=AsyncMock(return_value=("user-123", "free")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(return_value="free"),
            ),
            patch("app.usage_tracker.QUOTA_FAIL_OPEN", False),
        ):
            usage_status = await check_usage(request, redis_client)

        self.assertFalse(usage_status.allowed)
        self.assertTrue(usage_status.is_authenticated)
        self.assertEqual(usage_status.tracking_id, "user-123")

    async def test_increment_usage_uses_resolved_user_id_for_authenticated_user(
        self,
    ) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client = AsyncMock()
        usage_status = UsageStatus(
            allowed=True,
            remaining=10,
            limit=FREE_USER_LIMIT,
            is_authenticated=True,
            is_pro=False,
            tracking_id="user-123",
        )

        await increment_usage(request, redis_client, usage_status)

        redis_client.incr.assert_awaited_once_with("user:user-123:usage_count")
        redis_client.expire.assert_awaited_once_with(
            "user:user-123:usage_count",
            USAGE_TTL,
        )

    async def test_increment_usage_does_not_reauthenticate_authenticated_user(
        self,
    ) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client = AsyncMock()
        usage_status = UsageStatus(
            allowed=True,
            remaining=10,
            limit=FREE_USER_LIMIT,
            is_authenticated=True,
            is_pro=False,
            tracking_id="user-123",
        )

        with patch(
            "app.usage_tracker.extract_user_info_from_token",
            new=AsyncMock(side_effect=AssertionError("should not re-authenticate")),
        ) as extract_mock:
            await increment_usage(request, redis_client, usage_status)

        extract_mock.assert_not_called()
        redis_client.incr.assert_awaited_once_with("user:user-123:usage_count")

    async def test_increment_usage_updates_both_anonymous_counters(self) -> None:
        request = _build_request(
            headers={"cf-connecting-ip": "203.0.113.10"},
            client_host="198.51.100.8",
        )
        redis_client = AsyncMock()
        usage_status = UsageStatus(
            allowed=True,
            remaining=5,
            limit=10,
            is_authenticated=False,
            is_pro=False,
            tracking_id="track-123",
        )

        await increment_usage(request, redis_client, usage_status)

        ip_key = f"anon:ip:{hash_ip('203.0.113.10')}:usage_count"
        self.assertEqual(redis_client.incr.await_count, 2)
        redis_client.incr.assert_any_await("anon:track-123:usage_count")
        redis_client.incr.assert_any_await(ip_key)
        self.assertEqual(redis_client.expire.await_count, 2)
        redis_client.expire.assert_any_await("anon:track-123:usage_count", USAGE_TTL)
        redis_client.expire.assert_any_await(ip_key, USAGE_TTL)

    async def test_increment_usage_skips_pro_user(self) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client = AsyncMock()
        usage_status = UsageStatus(
            allowed=True,
            remaining=-1,
            limit=-1,
            is_authenticated=True,
            is_pro=True,
            tracking_id="user-123",
        )

        await increment_usage(request, redis_client, usage_status)

        redis_client.incr.assert_not_called()
        redis_client.expire.assert_not_called()

    async def test_increment_usage_skips_missing_tracking_id_for_authenticated_user(
        self,
    ) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client = AsyncMock()
        usage_status = UsageStatus(
            allowed=True,
            remaining=10,
            limit=FREE_USER_LIMIT,
            is_authenticated=True,
            is_pro=False,
            tracking_id=None,
        )

        with self.assertLogs("app.usage_tracker", level="WARNING") as logs:
            await increment_usage(request, redis_client, usage_status)

        redis_client.incr.assert_not_called()
        redis_client.expire.assert_not_called()
        self.assertTrue(
            any("tracking_id is missing" in message for message in logs.output)
        )


if __name__ == "__main__":
    unittest.main()
