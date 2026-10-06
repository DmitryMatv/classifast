import asyncio
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import redis.asyncio as redis

from app import usage_tracker
from app.clerk_auth import ClerkAuthenticationError, ClerkInfrastructureError
from app.usage_tracker import (
    ANON_LIMIT,
    ANON_USAGE_TTL,
    FREE_USER_LIMIT,
    NEGATIVE_TIER_CACHE_TTL,
    TIER_CACHE_SENTINEL_EXPLICIT_NEGATIVE,
    TIER_CACHE_SENTINEL_TRANSIENT_UNAVAILABLE,
    TIER_CACHE_TTL,
    USAGE_TTL,
    QuotaUnavailableError,
    SignedInCaller,
    TierResolution,
    UsageStatus,
    check_usage,
    get_cached_user_tier,
    get_client_ip,
    get_or_create_tracking_id,
    hash_ip,
    reserve_usage,
    resolve_signed_in_caller,
    set_cached_user_tier,
)


def _build_request(
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    client_host: str = "127.0.0.1",
) -> Mock:
    request = Mock()
    request.headers = headers or {}
    request.cookies = cookies or {}
    request.client = SimpleNamespace(host=client_host)
    return request


async def _resolve_and_reserve(request: Mock, redis_client) -> UsageStatus:
    caller = await resolve_signed_in_caller(request, redis_client)
    return await reserve_usage(request, redis_client, caller)


def _build_redis_client_with_pipeline(
    execute_result: list[object],
) -> tuple[AsyncMock, Mock]:
    redis_client = AsyncMock()
    pipeline = Mock()
    pipeline.execute = AsyncMock(return_value=execute_result)
    redis_client.pipeline = Mock(return_value=pipeline)
    return redis_client, pipeline


class _TierCacheRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    async def get(self, key: str) -> bytes | None:
        value = self.values.get(key)
        return value.encode() if value is not None else None

    async def setex(self, key: str, ttl: int, value: str) -> bool:
        self.values[key] = value
        self.expirations[key] = ttl
        return True

    async def set(
        self, key: str, value: str, *, ex: int, nx: bool = False
    ) -> bool | None:
        if nx and key in self.values:
            return None
        self.values[key] = value
        self.expirations[key] = ex
        return True

    async def exists(self, key: str) -> bool:
        return key in self.values

    async def ttl(self, key: str) -> int:
        return self.expirations.get(key, -2)


class UsageTrackerHelperTests(unittest.TestCase):
    def test_quota_fail_open_is_not_available(self) -> None:
        self.assertFalse(hasattr(usage_tracker, "QUOTA_FAIL_OPEN"))
        self.assertFalse(hasattr(usage_tracker, "quota_fail_open_enabled"))

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
    async def _lookup_during_authoritative_tier_update(
        self,
        redis_client: _TierCacheRedis,
        stale_resolution: TierResolution,
        authoritative_tier: str,
    ) -> TierResolution:
        lookup_started = asyncio.Event()
        finish_lookup = asyncio.Event()

        async def fetch_stale_tier(user_id: str) -> TierResolution:
            lookup_started.set()
            await finish_lookup.wait()
            return stale_resolution

        with patch(
            "app.usage_tracker.fetch_clerk_user_tier",
            new=AsyncMock(side_effect=fetch_stale_tier),
        ):
            lookup = asyncio.create_task(get_cached_user_tier("user-123", redis_client))
            try:
                await asyncio.wait_for(lookup_started.wait(), timeout=5)
                await set_cached_user_tier("user-123", authoritative_tier, redis_client)
                finish_lookup.set()
                return await asyncio.wait_for(lookup, timeout=5)
            finally:
                finish_lookup.set()
                if not lookup.done():
                    lookup.cancel()

    async def test_cache_miss_lookup_preserves_authoritative_pro_update(self) -> None:
        stale_resolutions = (
            TierResolution(status="confirmed_non_pro", tier="free"),
            TierResolution(status="confirmed_non_pro"),
            TierResolution(status="explicit_negative"),
            TierResolution(status="transient_unavailable"),
        )

        for stale_resolution in stale_resolutions:
            with self.subTest(stale_resolution=stale_resolution):
                redis_client = _TierCacheRedis()
                resolution = await self._lookup_during_authoritative_tier_update(
                    redis_client, stale_resolution, "pro"
                )

                self.assertEqual(
                    resolution, TierResolution(status="confirmed_pro", tier="pro")
                )
                self.assertEqual(await redis_client.get("user_tier:user-123"), b"pro")
                self.assertEqual(
                    await redis_client.ttl("user_tier:user-123"), TIER_CACHE_TTL
                )

                with patch(
                    "app.usage_tracker.authenticate_clerk_token_local",
                    new=AsyncMock(return_value=("user-123", "free")),
                ):
                    usage_status = await _resolve_and_reserve(
                        _build_request(headers={"authorization": "Bearer token"}),
                        redis_client,
                    )

                self.assertTrue(usage_status.allowed)
                self.assertTrue(usage_status.is_pro)
                self.assertEqual(usage_status.remaining, -1)

    async def test_cache_miss_lookup_preserves_authoritative_free_update(self) -> None:
        redis_client = _TierCacheRedis()
        pipeline = Mock()
        pipeline.execute = AsyncMock(return_value=[FREE_USER_LIMIT + 1, True])
        redis_client.pipeline = Mock(return_value=pipeline)

        resolution = await self._lookup_during_authoritative_tier_update(
            redis_client,
            TierResolution(status="confirmed_pro", tier="pro"),
            "free",
        )

        self.assertEqual(
            resolution, TierResolution(status="confirmed_non_pro", tier="free")
        )
        self.assertEqual(await redis_client.get("user_tier:user-123"), b"free")
        self.assertEqual(await redis_client.ttl("user_tier:user-123"), TIER_CACHE_TTL)

        with patch(
            "app.usage_tracker.authenticate_clerk_token_local",
            new=AsyncMock(return_value=("user-123", "pro")),
        ):
            usage_status = await _resolve_and_reserve(
                _build_request(headers={"authorization": "Bearer token"}),
                redis_client,
            )

        self.assertFalse(usage_status.allowed)
        self.assertFalse(usage_status.is_pro)
        self.assertEqual(usage_status.remaining, 0)

    async def test_get_cached_user_tier_uses_negative_cache_sentinel(self) -> None:
        redis_client = AsyncMock()
        redis_client.get.return_value = TIER_CACHE_SENTINEL_EXPLICIT_NEGATIVE

        resolution = await get_cached_user_tier("user-123", redis_client)

        self.assertEqual(resolution.status, "explicit_negative")

    async def test_get_cached_user_tier_fetches_and_caches_on_miss(self) -> None:
        redis_client = AsyncMock()
        redis_client.get.return_value = None

        with patch(
            "app.usage_tracker.fetch_clerk_user_tier",
            return_value=TierResolution(status="confirmed_pro", tier="pro"),
        ):
            resolution = await get_cached_user_tier("user-123", redis_client)

        self.assertEqual(resolution.status, "confirmed_pro")
        self.assertEqual(resolution.tier, "pro")
        redis_client.set.assert_awaited_once_with(
            "user_tier:user-123",
            "pro",
            ex=TIER_CACHE_TTL,
            nx=True,
        )

    async def test_get_cached_user_tier_negative_result_is_cached(self) -> None:
        redis_client = AsyncMock()
        redis_client.get.return_value = None

        with patch(
            "app.usage_tracker.fetch_clerk_user_tier",
            return_value=TierResolution(status="explicit_negative"),
        ):
            resolution = await get_cached_user_tier("user-123", redis_client)

        self.assertEqual(resolution.status, "explicit_negative")
        redis_client.set.assert_awaited_once_with(
            "user_tier:user-123",
            TIER_CACHE_SENTINEL_EXPLICIT_NEGATIVE,
            ex=NEGATIVE_TIER_CACHE_TTL,
            nx=True,
        )

    async def test_get_cached_user_tier_transient_result_is_cached(self) -> None:
        redis_client = AsyncMock()
        redis_client.get.return_value = None

        with patch(
            "app.usage_tracker.fetch_clerk_user_tier",
            return_value=TierResolution(status="transient_unavailable"),
        ):
            resolution = await get_cached_user_tier("user-123", redis_client)

        self.assertEqual(resolution.status, "transient_unavailable")
        redis_client.set.assert_awaited_once_with(
            "user_tier:user-123",
            TIER_CACHE_SENTINEL_TRANSIENT_UNAVAILABLE,
            ex=NEGATIVE_TIER_CACHE_TTL,
            nx=True,
        )

    async def test_get_cached_user_tier_returns_clerk_resolution_when_fill_fails(
        self,
    ) -> None:
        redis_client = AsyncMock()
        redis_client.get.return_value = None
        redis_client.set.side_effect = redis.RedisError("cache fill unavailable")
        clerk_resolution = TierResolution(status="confirmed_pro", tier="pro")

        with patch(
            "app.usage_tracker.fetch_clerk_user_tier",
            return_value=clerk_resolution,
        ):
            resolution = await get_cached_user_tier("user-123", redis_client)

        self.assertEqual(resolution, clerk_resolution)

    async def test_get_cached_user_tier_returns_clerk_resolution_when_winner_read_fails(
        self,
    ) -> None:
        for failure in (redis.RedisError("cache read unavailable"), b"\xff"):
            with self.subTest(failure=failure):
                redis_client = AsyncMock()
                redis_client.get.side_effect = [None, failure]
                clerk_resolution = TierResolution(status="confirmed_pro", tier="pro")

                with patch(
                    "app.usage_tracker.fetch_clerk_user_tier",
                    return_value=clerk_resolution,
                ):
                    resolution = await get_cached_user_tier("user-123", redis_client)

                self.assertEqual(resolution, clerk_resolution)

    async def test_set_cached_user_tier_stores_pro_tier(self) -> None:
        redis_client = AsyncMock()

        await set_cached_user_tier("user-123", "pro", redis_client)

        redis_client.setex.assert_awaited_once_with(
            "user_tier:user-123",
            TIER_CACHE_TTL,
            "pro",
        )

    async def test_set_cached_user_tier_stores_non_pro_as_free(self) -> None:
        redis_client = AsyncMock()

        await set_cached_user_tier("user-123", "starter", redis_client)

        redis_client.setex.assert_awaited_once_with(
            "user_tier:user-123",
            TIER_CACHE_TTL,
            "free",
        )

    async def test_set_cached_user_tier_is_best_effort_on_redis_error(self) -> None:
        redis_client = AsyncMock()
        redis_client.setex.side_effect = redis.RedisError("boom")

        with self.assertLogs("app.usage_tracker", level="WARNING") as logs:
            await set_cached_user_tier("user-123", "pro", redis_client)

        self.assertTrue(
            any("Failed to sync tier cache" in line for line in logs.output)
        )

    async def test_stale_jwt_pro_hint_is_not_treated_as_unlimited(self) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client, _ = _build_redis_client_with_pipeline([6, True])

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(return_value=("user-123", "pro")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(
                    return_value=TierResolution(
                        status="confirmed_non_pro",
                        tier="free",
                    )
                ),
            ),
        ):
            usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertFalse(usage_status.is_pro)
        self.assertEqual(usage_status.remaining, FREE_USER_LIMIT - 6)

    async def test_jwt_pro_hint_stays_unlimited_when_clerk_tier_is_unknown(
        self,
    ) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(return_value=("user-123", "pro")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(
                    return_value=TierResolution(status="transient_unavailable")
                ),
            ),
        ):
            usage_status = await _resolve_and_reserve(request, AsyncMock())

        self.assertTrue(usage_status.allowed)
        self.assertTrue(usage_status.is_authenticated)
        self.assertTrue(usage_status.is_pro)
        self.assertEqual(usage_status.remaining, -1)

    async def test_jwt_pro_hint_is_not_unlimited_when_clerk_tier_is_explicit_negative(
        self,
    ) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client, _ = _build_redis_client_with_pipeline([5, True])

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(return_value=("user-123", "pro")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(return_value=TierResolution(status="explicit_negative")),
            ),
        ):
            usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertTrue(usage_status.is_authenticated)
        self.assertFalse(usage_status.is_pro)
        self.assertEqual(usage_status.remaining, FREE_USER_LIMIT - 5)

    async def test_missing_jwt_pro_hint_and_unknown_clerk_tier_uses_free_quota(
        self,
    ) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client, _ = _build_redis_client_with_pipeline([5, True])

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(return_value=("user-123", None)),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(
                    return_value=TierResolution(status="transient_unavailable")
                ),
            ),
        ):
            usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertFalse(usage_status.is_pro)
        self.assertEqual(usage_status.remaining, FREE_USER_LIMIT - 5)

    async def test_invalid_session_falls_back_to_anonymous_quota(self) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client, _ = _build_redis_client_with_pipeline([1, True, 1, True])

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(side_effect=ClerkAuthenticationError("Invalid session")),
            ),
            patch(
                "app.usage_tracker.get_or_create_tracking_id",
                return_value=("track-123", False),
            ),
        ):
            usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertFalse(usage_status.is_authenticated)
        self.assertEqual(usage_status.tracking_id, "track-123")

    async def test_bearer_infrastructure_failure_falls_back_to_anonymous_quota(
        self,
    ) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client, _ = _build_redis_client_with_pipeline([1, True, 1, True])

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(side_effect=ClerkInfrastructureError()),
            ),
            patch(
                "app.usage_tracker.get_or_create_tracking_id",
                return_value=("track-infra", False),
            ),
        ):
            usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertFalse(usage_status.is_authenticated)
        self.assertFalse(usage_status.is_pro)
        self.assertEqual(usage_status.tracking_id, "track-infra")

    async def test_session_cookie_infrastructure_failure_falls_back_to_anonymous_quota(
        self,
    ) -> None:
        request = _build_request(cookies={"__session": "session-token"})
        redis_client, _ = _build_redis_client_with_pipeline([1, True, 1, True])

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(side_effect=ClerkInfrastructureError()),
            ),
            patch(
                "app.usage_tracker.get_or_create_tracking_id",
                return_value=("track-cookie", False),
            ),
        ):
            usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertFalse(usage_status.is_authenticated)
        self.assertFalse(usage_status.is_pro)
        self.assertEqual(usage_status.tracking_id, "track-cookie")

    async def test_valid_active_session_with_pro_tier_is_unlimited(self) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(return_value=("user-123", "pro")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(
                    return_value=TierResolution(status="confirmed_pro", tier="pro")
                ),
            ),
        ):
            usage_status = await _resolve_and_reserve(request, AsyncMock())

        self.assertTrue(usage_status.allowed)
        self.assertTrue(usage_status.is_authenticated)
        self.assertTrue(usage_status.is_pro)
        self.assertEqual(usage_status.remaining, -1)

    async def test_valid_active_session_with_free_tier_uses_free_quota(self) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client, _ = _build_redis_client_with_pipeline([3, True])

        with (
            patch("app.clerk_auth.CLERK_PERMITTED_ORIGINS", "https://classifast.com"),
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(return_value=("user-123", "free")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(
                    return_value=TierResolution(
                        status="confirmed_non_pro",
                        tier="free",
                    )
                ),
            ),
        ):
            usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertTrue(usage_status.is_authenticated)
        self.assertFalse(usage_status.is_pro)
        self.assertEqual(usage_status.remaining, FREE_USER_LIMIT - 3)

    async def test_invalid_session_cookie_is_treated_as_anonymous(self) -> None:
        request = _build_request(cookies={"__session": "session-token"})
        redis_client, _ = _build_redis_client_with_pipeline([1, True, 1, True])

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(side_effect=ClerkAuthenticationError("Invalid session")),
            ),
            patch(
                "app.usage_tracker.get_or_create_tracking_id",
                return_value=("track-456", False),
            ),
        ):
            usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertFalse(usage_status.is_authenticated)
        self.assertEqual(usage_status.tracking_id, "track-456")

    async def test_quota_auth_does_not_verify_live_session_for_bearer_requests(
        self,
    ) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client, _ = _build_redis_client_with_pipeline([1, True])

        with (
            patch("app.clerk_auth.CLERK_PERMITTED_ORIGINS", "https://classifast.com"),
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(return_value=("user-123", "free")),
            ) as auth_mock,
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(
                    return_value=TierResolution(
                        status="confirmed_non_pro",
                        tier="free",
                    )
                ),
            ),
            patch(
                "app.usage_tracker.verify_clerk_session_active",
                new=AsyncMock(side_effect=AssertionError("should not verify session")),
                create=True,
            ) as verify_mock,
        ):
            usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertTrue(usage_status.is_authenticated)
        auth_mock.assert_awaited_once_with("token", validate_azp=True)
        verify_mock.assert_not_called()

    async def test_quota_auth_skips_azp_when_permitted_origins_not_configured(
        self,
    ) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client, _ = _build_redis_client_with_pipeline([1, True])

        with (
            patch("app.clerk_auth.CLERK_PERMITTED_ORIGINS", ""),
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(return_value=("user-123", "free")),
            ) as auth_mock,
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(
                    return_value=TierResolution(
                        status="confirmed_non_pro",
                        tier="free",
                    )
                ),
            ),
        ):
            usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertTrue(usage_status.allowed)
        auth_mock.assert_awaited_once_with("token", validate_azp=False)

    async def test_quota_auth_does_not_verify_live_session_for_session_cookie(
        self,
    ) -> None:
        request = _build_request(cookies={"__session": "session-token"})
        redis_client, _ = _build_redis_client_with_pipeline([1, True])

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(return_value=("user-123", "free")),
            ) as auth_mock,
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(
                    return_value=TierResolution(
                        status="confirmed_non_pro",
                        tier="free",
                    )
                ),
            ),
            patch(
                "app.usage_tracker.verify_clerk_session_active",
                new=AsyncMock(side_effect=AssertionError("should not verify session")),
                create=True,
            ) as verify_mock,
        ):
            usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertTrue(usage_status.is_authenticated)
        self.assertEqual(auth_mock.await_count, 1)
        auth_mock.assert_awaited_once_with("session-token", validate_azp=False)
        verify_mock.assert_not_called()

    async def test_redis_unavailable_raises_quota_unavailable(self) -> None:
        request = _build_request()

        with self.assertRaises(QuotaUnavailableError):
            await _resolve_and_reserve(request, None)

    async def test_redis_unavailable_short_circuits_before_tier_or_grace_checks(
        self,
    ) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(return_value=("user-123", "pro")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(side_effect=AssertionError("should not check grace")),
            ) as grace_mock,
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(side_effect=AssertionError("should not resolve tier")),
            ) as tier_mock,
        ):
            with self.assertRaises(QuotaUnavailableError):
                await _resolve_and_reserve(request, None)

        grace_mock.assert_not_called()
        tier_mock.assert_not_called()

    async def test_checkout_grace_allows_verified_user(self) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(return_value=("user-123", "free")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=True),
            ),
        ):
            usage_status = await _resolve_and_reserve(request, AsyncMock())

        self.assertTrue(usage_status.allowed)
        self.assertTrue(usage_status.is_pro)
        self.assertEqual(usage_status.tracking_id, "user-123")

    async def test_checkout_grace_does_not_help_invalid_identity(self) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client, _ = _build_redis_client_with_pipeline([1, True, 1, True])

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
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
            usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertFalse(usage_status.is_authenticated)
        self.assertFalse(usage_status.is_pro)
        self.assertEqual(usage_status.tracking_id, "track-999")

    async def test_reserve_usage_handles_redis_errors_for_anonymous_requests(
        self,
    ) -> None:
        request = _build_request(headers={"cf-connecting-ip": "203.0.113.10"})
        redis_client, pipeline = _build_redis_client_with_pipeline([])
        pipeline.execute.side_effect = redis.RedisError("boom")

        with self.assertRaises(QuotaUnavailableError):
            await _resolve_and_reserve(request, redis_client)

    async def test_reserve_usage_handles_pipeline_creation_error(self) -> None:
        request = _build_request()
        redis_client = AsyncMock()
        redis_client.pipeline = Mock(side_effect=redis.RedisError("boom"))

        with self.assertRaises(QuotaUnavailableError):
            await _resolve_and_reserve(request, redis_client)

    async def test_reserve_usage_handles_redis_errors_for_authenticated_users(
        self,
    ) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client, pipeline = _build_redis_client_with_pipeline([])
        pipeline.execute.side_effect = redis.RedisError("boom")

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(return_value=("user-123", "free")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(
                    return_value=TierResolution(
                        status="confirmed_non_pro",
                        tier="free",
                    )
                ),
            ),
        ):
            with self.assertRaises(QuotaUnavailableError):
                await _resolve_and_reserve(request, redis_client)

    async def test_authenticated_reservation_at_limit_is_allowed(self) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client, pipeline = _build_redis_client_with_pipeline(
            [FREE_USER_LIMIT, True]
        )

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(return_value=("user-123", "free")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(
                    return_value=TierResolution(
                        status="confirmed_non_pro",
                        tier="free",
                    )
                ),
            ),
        ):
            usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertEqual(usage_status.remaining, 0)
        self.assertEqual(usage_status.tracking_id, "user-123")
        redis_client.pipeline.assert_called_once_with(transaction=True)
        self.assertEqual(
            pipeline.method_calls,
            [
                unittest.mock.call.incr("user:user-123:usage_count"),
                unittest.mock.call.expire(
                    "user:user-123:usage_count",
                    USAGE_TTL,
                ),
                unittest.mock.call.execute(),
            ],
        )

    async def test_authenticated_reservation_above_limit_is_denied(self) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client, _ = _build_redis_client_with_pipeline([FREE_USER_LIMIT + 1, True])

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(return_value=("user-123", "free")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(
                    return_value=TierResolution(
                        status="confirmed_non_pro",
                        tier="free",
                    )
                ),
            ),
        ):
            usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertFalse(usage_status.allowed)
        self.assertEqual(usage_status.remaining, 0)

    async def test_anonymous_reservation_updates_both_counters_atomically(
        self,
    ) -> None:
        request = _build_request(
            headers={"cf-connecting-ip": "203.0.113.10"},
            client_host="198.51.100.8",
        )
        redis_client, pipeline = _build_redis_client_with_pipeline([4, True, 7, True])

        with patch(
            "app.usage_tracker.get_or_create_tracking_id",
            return_value=("track-123", False),
        ):
            usage_status = await _resolve_and_reserve(request, redis_client)

        ip_key = f"anon:ip:{hash_ip('203.0.113.10')}:usage_count"
        self.assertTrue(usage_status.allowed)
        self.assertEqual(usage_status.remaining, ANON_LIMIT - 7)
        redis_client.pipeline.assert_called_once_with(transaction=True)
        redis_client.get.assert_not_awaited()
        self.assertEqual(
            pipeline.method_calls,
            [
                unittest.mock.call.incr("anon:track-123:usage_count"),
                unittest.mock.call.expire("anon:track-123:usage_count", ANON_USAGE_TTL),
                unittest.mock.call.incr(ip_key),
                unittest.mock.call.expire(ip_key, ANON_USAGE_TTL),
                unittest.mock.call.execute(),
            ],
        )

    async def test_anonymous_reservation_at_limit_is_allowed(self) -> None:
        request = _build_request()
        redis_client, _ = _build_redis_client_with_pipeline(
            [ANON_LIMIT, True, ANON_LIMIT - 2, True]
        )

        usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertEqual(usage_status.remaining, 0)

    async def test_anonymous_reservation_above_limit_is_denied(self) -> None:
        request = _build_request()
        redis_client, _ = _build_redis_client_with_pipeline(
            [ANON_LIMIT - 2, True, ANON_LIMIT + 1, True]
        )

        usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertFalse(usage_status.allowed)
        self.assertEqual(usage_status.remaining, 0)

    async def test_pro_reservation_does_not_create_usage_pipeline(self) -> None:
        request = _build_request(headers={"authorization": "Bearer token"})
        redis_client = AsyncMock()

        with (
            patch(
                "app.usage_tracker.authenticate_clerk_token_local",
                new=AsyncMock(return_value=("user-123", "pro")),
            ),
            patch(
                "app.usage_tracker.has_active_grace",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.usage_tracker.get_cached_user_tier",
                new=AsyncMock(
                    return_value=TierResolution(status="confirmed_pro", tier="pro")
                ),
            ),
        ):
            usage_status = await _resolve_and_reserve(request, redis_client)

        self.assertTrue(usage_status.allowed)
        self.assertTrue(usage_status.is_pro)
        redis_client.pipeline.assert_not_called()

    async def test_check_usage_reads_quota_without_writing(self) -> None:
        tracking_id = "1b4e28ba-2fa1-41d2-883f-0016d3cca427"
        request = _build_request(
            headers={"cf-connecting-ip": "203.0.113.10"},
            cookies={"cf_track": tracking_id},
        )
        anonymous_keys = [
            f"anon:{tracking_id}:usage_count",
            f"anon:ip:{hash_ip('203.0.113.10')}:usage_count",
        ]
        free_user = SignedInCaller(user_id="user-123", is_pro=False)
        free_keys = ["user:user-123:usage_count"]
        cases = (
            (None, [str(ANON_LIMIT - 4), str(ANON_LIMIT - 1)], True, 1, anonymous_keys),
            (None, [str(ANON_LIMIT), None], False, 0, anonymous_keys),
            (None, [None, str(ANON_LIMIT + 3)], False, 0, anonymous_keys),
            (free_user, [str(FREE_USER_LIMIT - 1)], True, 1, free_keys),
            (free_user, [str(FREE_USER_LIMIT)], False, 0, free_keys),
        )
        for caller, stored_counts, allowed, remaining, keys in cases:
            with self.subTest(caller=caller, stored_counts=stored_counts):
                redis_client = AsyncMock()
                redis_client.mget.return_value = stored_counts

                usage_status = await check_usage(request, redis_client, caller)

                self.assertEqual(usage_status.allowed, allowed)
                self.assertEqual(usage_status.remaining, remaining)
                self.assertEqual(
                    [call[0] for call in redis_client.method_calls], ["mget"]
                )
                self.assertEqual(list(redis_client.mget.await_args.args[0]), keys)

    async def test_check_usage_for_pro_caller_does_not_touch_redis(self) -> None:
        redis_client = AsyncMock()

        usage_status = await check_usage(
            _build_request(),
            redis_client,
            SignedInCaller(user_id="user-123", is_pro=True),
        )

        self.assertEqual(
            usage_status, UsageStatus(True, -1, -1, True, True, "user-123")
        )
        self.assertEqual(redis_client.method_calls, [])

    async def test_check_usage_fails_closed_when_usage_tracking_is_unavailable(
        self,
    ) -> None:
        failure = redis.RedisError("boom")
        failing_client = AsyncMock()
        failing_client.mget.side_effect = failure
        for redis_client, cause in ((None, None), (failing_client, failure)):
            with self.subTest(cause=cause):
                with self.assertRaises(QuotaUnavailableError) as caught:
                    await check_usage(_build_request(), redis_client, None)

                self.assertEqual(
                    str(caught.exception), "Usage tracking is temporarily unavailable"
                )
                self.assertIs(caught.exception.__cause__, cause)


if __name__ == "__main__":
    unittest.main()
