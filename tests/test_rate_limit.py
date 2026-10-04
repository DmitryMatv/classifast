import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, HTTPException, Request

from app import rate_limit


class _CounterRedis:
    def __init__(self, count: int | None = None, expires_at: int | None = None):
        self.count = count
        self.expires_at = expires_at
        self.now = 0

    async def incr(self, key: str) -> int:
        if self.expires_at is not None and self.now >= self.expires_at:
            self.count = None
            self.expires_at = None
        self.count = (self.count or 0) + 1
        return self.count

    async def expire(self, key: str, seconds: int, *, nx: bool = False) -> bool:
        if self.count is None or (nx and self.expires_at is not None):
            return False
        self.expires_at = self.now + seconds
        return True

    def pipeline(self, *, transaction: bool = True):
        commands = []
        pipeline = MagicMock()
        pipeline.__aenter__.return_value = pipeline
        pipeline.incr.side_effect = lambda key: commands.append(("incr", (key,), {}))
        pipeline.expire.side_effect = lambda key, seconds, **kwargs: commands.append(
            ("expire", (key, seconds), kwargs)
        )

        async def execute():
            return [
                await getattr(self, name)(*args, **kwargs)
                for name, args, kwargs in commands
            ]

        pipeline.execute = AsyncMock(side_effect=execute)
        return pipeline


class CheckoutCounterExpiryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        self.request = Request(
            {
                "type": "http",
                "app": self.app,
                "headers": [],
                "client": ("203.0.113.10", 1234),
            }
        )
        self.limit_patch = patch("app.rate_limit.CHECKOUT_RATE_LIMIT", 10)
        self.window_patch = patch(
            "app.rate_limit.CHECKOUT_RATE_LIMIT_WINDOW_SECONDS", 60
        )
        self.limit_patch.start()
        self.window_patch.start()
        self.addCleanup(self.limit_patch.stop)
        self.addCleanup(self.window_patch.stop)

    async def test_new_counter_expires_after_one_window(self) -> None:
        counter = _CounterRedis()
        self.app.state.redis_client = counter

        await rate_limit.enforce_checkout_rate_limit(self.request)

        self.assertEqual(counter.count, 1)
        self.assertEqual(counter.expires_at, 60)

    async def test_existing_counter_without_ttl_is_repaired(self) -> None:
        counter = _CounterRedis(count=1)
        self.app.state.redis_client = counter

        await rate_limit.enforce_checkout_rate_limit(self.request)

        self.assertEqual(counter.count, 2)
        self.assertEqual(counter.expires_at, 60)

    async def test_over_limit_counter_without_ttl_is_repaired_before_denial(
        self,
    ) -> None:
        counter = _CounterRedis(count=10)
        self.app.state.redis_client = counter

        with self.assertRaises(HTTPException) as ctx:
            await rate_limit.enforce_checkout_rate_limit(self.request)

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(counter.count, 11)
        self.assertEqual(counter.expires_at, 60)

    async def test_existing_ttl_is_preserved(self) -> None:
        counter = _CounterRedis(count=1, expires_at=30)
        counter.now = 5
        self.app.state.redis_client = counter

        await rate_limit.enforce_checkout_rate_limit(self.request)

        self.assertEqual(counter.count, 2)
        self.assertEqual(counter.expires_at, 30)

    async def test_expired_window_resets_counter_and_allows_checkout(self) -> None:
        counter = _CounterRedis(count=10, expires_at=30)
        self.app.state.redis_client = counter

        with self.assertRaises(HTTPException) as ctx:
            await rate_limit.enforce_checkout_rate_limit(self.request)
        self.assertEqual(ctx.exception.status_code, 429)

        counter.now = 30
        await rate_limit.enforce_checkout_rate_limit(self.request)

        self.assertEqual(counter.count, 1)
        self.assertEqual(counter.expires_at, 90)
