import asyncio
import gc
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import NAMESPACE_DNS, uuid5

import httpx
import redis.exceptions
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import api, main
from app.classification_executor import ClassificationExecutor, ClassificationQueueFull
from app.query_enhancer import EnhancementOutcome, EnhancementStatus
from app.usage_tracker import (
    ANON_LIMIT,
    FREE_USER_LIMIT,
    SignedInCaller,
    TierResolution,
    UsageStatus,
    hash_ip,
)
from app.web import router
from tests.helpers import (
    EmptyUsageRedis,
    build_classification_service,
    event_loop_turn,
)


def classification_result(query: str) -> dict:
    return {
        "results": [],
        "version_config": {},
        "version_name": "v1",
        "collection_name": "test",
        "query": query,
    }


def quota_cookie_key(query: str) -> str:
    return f"anon:{uuid5(NAMESPACE_DNS, query)}:usage_count"


class LocalQuotaLedger:
    def __init__(self, *, hold: str | None = None) -> None:
        self.counts: dict[str, int] = {}
        self.executed: list[list[str]] = []
        self.fail_tracking_ids: set[str] = set()
        self.hold = hold
        self.hold_started = asyncio.Event()
        self.release_hold = asyncio.Event()

    @property
    def charged(self) -> list[str]:
        return [keys[0] for keys in self.executed]

    async def mget(self, keys: tuple[str, ...]) -> list[int | None]:
        return [self.counts.get(key) for key in keys]

    def pipeline(self, *, transaction: bool) -> Mock:
        commands: list[tuple[str, str]] = []
        pipeline = Mock()
        pipeline.incr.side_effect = lambda key: commands.append(("incr", key))
        pipeline.expire.side_effect = lambda key, ttl: commands.append(("expire", key))

        async def execute() -> list[int | bool]:
            keys = [key for command, key in commands if command == "incr"]
            self.executed.append(keys)
            if self.hold is not None and quota_cookie_key(self.hold) in keys:
                self.hold_started.set()
                await self.release_hold.wait()
            if any(quota_cookie_key(name) in keys for name in self.fail_tracking_ids):
                raise redis.exceptions.ConnectionError("local quota storage failed")
            results: list[int | bool] = []
            for command, key in commands:
                if command == "incr":
                    self.counts[key] = self.counts.get(key, 0) + 1
                    results.append(self.counts[key])
                else:
                    results.append(True)
            return results

        pipeline.execute = AsyncMock(side_effect=execute)
        return pipeline


class ClassificationQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.executor = ClassificationExecutor()
        self.service = build_classification_service(executor=self.executor)
        self.tasks: list[asyncio.Task] = []
        self.releases: list[threading.Event | asyncio.Event] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.releases.append(self.release)
        self.ran: list[str] = []

        def pipeline(*, query: str, **kwargs) -> dict:
            self.ran.append(query)
            if query == "active":
                self.started.set()
                if not self.release.wait(5):
                    raise AssertionError("test did not release active worker")
            return classification_result(query)

        self.pipeline = patch(
            "app.classification_service.perform_classification", side_effect=pipeline
        ).start()
        self.addCleanup(patch.stopall)

    async def asyncTearDown(self) -> None:
        for release in self.releases:
            release.set()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.executor.close()

    async def submit(self, query: str, **kwargs) -> asyncio.Task:
        task = asyncio.create_task(self.service.classify(query, "UNSPSC", **kwargs))
        self.tasks.append(task)
        await event_loop_turn()
        return task

    async def fill_queue(self) -> list[asyncio.Task]:
        active = await self.submit("active")
        self.assertTrue(await asyncio.to_thread(self.started.wait, 1))
        waiting = [await self.submit(f"waiting-{index}") for index in range(4)]
        return [active, *waiting]

    async def assert_overflow(self) -> None:
        overflow = await self.submit("overflow")
        self.assertTrue(
            overflow.done(), "sixth classification must reject before worker release"
        )
        with self.assertRaises(ClassificationQueueFull):
            await overflow

    def page_app(self) -> FastAPI:
        app = FastAPI()
        app.mount(
            "/static",
            StaticFiles(
                directory=Path(__file__).resolve().parents[1] / "app" / "static"
            ),
            name="static",
        )
        app.include_router(router)
        app.state.classification_service = self.service
        return app

    def metered_app(self, ledger: LocalQuotaLedger) -> FastAPI:
        self.releases.append(ledger.release_hold)
        app = FastAPI()
        app.include_router(router)
        app.state.classification_service = self.service
        app.state.redis_client = ledger
        return app

    async def submit_fragment(
        self,
        client: httpx.AsyncClient,
        query: str,
        *,
        enhancement_enabled: bool = False,
    ) -> asyncio.Task:
        params = {"product_description": query}
        if enhancement_enabled:
            params["enhance_query"] = "1"
        task = asyncio.create_task(
            client.get(
                "/UNSPSC/fragment",
                params=params,
                headers={"Cookie": f"cf_track={uuid5(NAMESPACE_DNS, query)}"},
            )
        )
        self.tasks.append(task)
        await event_loop_turn()
        return task

    async def test_pending_quota_owns_capacity_and_waiting_gates_are_fifo(self) -> None:
        ledger = LocalQuotaLedger(hold="gate-active")
        with patch(
            "app.web.is_verified_google_search_crawler_request",
            new=AsyncMock(return_value=False),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.metered_app(ledger)),
                base_url="http://testserver",
            ) as client:
                active = await self.submit_fragment(client, "gate-active")
                await asyncio.wait_for(ledger.hold_started.wait(), 1)
                waiting = [
                    await self.submit_fragment(client, f"gate-{i}") for i in range(4)
                ]
                overflow = await self.submit_fragment(client, "overflow")
                response = await asyncio.wait_for(overflow, 1)
                self.assertEqual(response.status_code, 503)
                self.assertTrue(
                    response.headers["Content-Type"].startswith("text/html")
                )
                self.assertEqual(ledger.charged, [quota_cookie_key("gate-active")])
                self.assertEqual(ledger.counts, {})
                self.assertEqual(self.ran, [])

                ledger.release_hold.set()
                responses = await asyncio.wait_for(asyncio.gather(active, *waiting), 1)

        expected = ["gate-active", *[f"gate-{i}" for i in range(4)]]
        self.assertEqual([response.status_code for response in responses], [200] * 5)
        self.assertEqual(self.ran, expected)
        self.assertEqual(
            ledger.charged, [quota_cookie_key(query) for query in expected]
        )
        self.assertNotIn(quota_cookie_key("overflow"), ledger.counts)

    async def test_exhausted_caller_gets_paywall_without_queue_slot_or_charge(
        self,
    ) -> None:
        exhausted_ip_key = f"anon:ip:{hash_ip('203.0.113.9')}:usage_count"
        cases = (
            (
                None,
                {quota_cookie_key("exhausted"): ANON_LIMIT},
                "Sign in to continue",
            ),
            (
                SignedInCaller(user_id="user-1", is_pro=False),
                {"user:user-1:usage_count": FREE_USER_LIMIT},
                f"used your {FREE_USER_LIMIT} free trial searches",
            ),
        )
        for caller, exhausted_counts, paywall_text in cases:
            with self.subTest(caller=caller):
                self.ran.clear()
                ledger = LocalQuotaLedger(hold="gate-active")
                ledger.counts.update(exhausted_counts)

                async def resolve_caller(request, redis_client):
                    if request.query_params["product_description"] == "exhausted":
                        return caller
                    return None

                with (
                    patch(
                        "app.web.resolve_signed_in_caller",
                        side_effect=resolve_caller,
                    ),
                    patch(
                        "app.web.is_verified_google_search_crawler_request",
                        new=AsyncMock(return_value=False),
                    ),
                ):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=self.metered_app(ledger)),
                        base_url="http://testserver",
                    ) as client:
                        active = await self.submit_fragment(client, "gate-active")
                        await asyncio.wait_for(ledger.hold_started.wait(), 1)
                        waiting = [
                            await self.submit_fragment(client, f"gate-{i}")
                            for i in range(4)
                        ]
                        paywall = await asyncio.wait_for(
                            client.get(
                                "/UNSPSC/fragment",
                                params={"product_description": "exhausted"},
                                headers={
                                    "Cookie": "cf_track="
                                    f"{uuid5(NAMESPACE_DNS, 'exhausted')}",
                                    "CF-Connecting-IP": "203.0.113.9",
                                },
                            ),
                            1,
                        )
                        ledger.release_hold.set()
                        responses = await asyncio.wait_for(
                            asyncio.gather(active, *waiting), 1
                        )

                self.assertEqual(paywall.status_code, 200)
                self.assertIn(paywall_text, paywall.text)
                self.assertEqual(
                    paywall.headers["Cache-Control"], "no-store, max-age=0"
                )
                expected = ["gate-active", *[f"gate-{i}" for i in range(4)]]
                self.assertEqual(
                    [response.status_code for response in responses], [200] * 5
                )
                self.assertEqual(self.ran, expected)
                self.assertEqual(
                    ledger.charged, [quota_cookie_key(query) for query in expected]
                )
                for key, count in exhausted_counts.items():
                    self.assertEqual(ledger.counts[key], count)
                self.assertNotIn(exhausted_ip_key, ledger.counts)

    async def test_slow_crawler_verification_does_not_hold_the_queue_turn(
        self,
    ) -> None:
        verification_started = asyncio.Event()
        release_verification = asyncio.Event()
        self.releases.append(release_verification)

        async def verify(request):
            if request.query_params["product_description"] != "crawler":
                return False
            verification_started.set()
            await release_verification.wait()
            return True

        app = FastAPI()
        app.include_router(router)
        app.state.classification_service = self.service
        app.state.redis_client = EmptyUsageRedis()
        usage = UsageStatus(True, 9, 10, False, False, "test-track")
        with (
            patch("app.web.reserve_usage", new=AsyncMock(return_value=usage)),
            patch(
                "app.web.is_verified_google_search_crawler_request",
                side_effect=verify,
            ),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                crawler = await self.submit_fragment(client, "crawler")
                await asyncio.wait_for(verification_started.wait(), 1)
                human = await asyncio.wait_for(
                    await self.submit_fragment(client, "human"), 1
                )
                self.assertEqual(human.status_code, 200)
                self.assertEqual(self.ran, ["human"])

                release_verification.set()
                crawled = await asyncio.wait_for(crawler, 1)

        self.assertEqual(crawled.status_code, 200)
        self.assertEqual(self.ran, ["human", "crawler"])

    async def test_slow_clerk_lookups_do_not_hold_the_queue_turn(self) -> None:
        for slow_lookup in ("token", "tier"):
            with self.subTest(slow_lookup=slow_lookup):
                self.ran.clear()
                ledger = LocalQuotaLedger()
                lookup_started = asyncio.Event()
                release_lookup = asyncio.Event()
                self.releases.append(release_lookup)

                async def hold(lookup: str) -> None:
                    if lookup == slow_lookup:
                        lookup_started.set()
                        await release_lookup.wait()

                async def identify(request):
                    if request.query_params["product_description"] != "signed-in":
                        return None, None
                    await hold("token")
                    return "user-1", None

                async def resolve_tier(user_id, redis_client):
                    await hold("tier")
                    return TierResolution("confirmed_non_pro")

                with (
                    patch(
                        "app.usage_tracker.extract_user_info_from_token",
                        side_effect=identify,
                    ),
                    patch(
                        "app.usage_tracker.has_active_grace",
                        new=AsyncMock(return_value=False),
                    ),
                    patch(
                        "app.usage_tracker.get_cached_user_tier",
                        side_effect=resolve_tier,
                    ),
                    patch(
                        "app.web.is_verified_google_search_crawler_request",
                        new=AsyncMock(return_value=False),
                    ),
                ):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=self.metered_app(ledger)),
                        base_url="http://testserver",
                    ) as client:
                        signed_in = await self.submit_fragment(client, "signed-in")
                        await asyncio.wait_for(lookup_started.wait(), 1)
                        visitor = await asyncio.wait_for(
                            await self.submit_fragment(client, "visitor"), 1
                        )
                        self.assertEqual(visitor.status_code, 200)
                        self.assertEqual(self.ran, ["visitor"])

                        release_lookup.set()
                        signed_in_response = await asyncio.wait_for(signed_in, 1)

                self.assertEqual(signed_in_response.status_code, 200)
                self.assertEqual(self.ran, ["visitor", "signed-in"])
                self.assertEqual(ledger.counts["user:user-1:usage_count"], 1)

    async def test_cancelled_waiting_quota_gate_never_charges_and_slot_is_reusable(
        self,
    ) -> None:
        ledger = LocalQuotaLedger(hold="gate-active")
        with patch(
            "app.web.is_verified_google_search_crawler_request",
            new=AsyncMock(return_value=False),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.metered_app(ledger)),
                base_url="http://testserver",
            ) as client:
                active = await self.submit_fragment(client, "gate-active")
                await asyncio.wait_for(ledger.hold_started.wait(), 1)
                waiting = [
                    await self.submit_fragment(client, f"gate-{i}") for i in range(4)
                ]
                waiting[1].cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await waiting[1]
                replacement = await self.submit_fragment(client, "replacement")
                overflow = await self.submit_fragment(client, "overflow")
                self.assertEqual((await asyncio.wait_for(overflow, 1)).status_code, 503)
                self.assertEqual(ledger.charged, [quota_cookie_key("gate-active")])
                self.assertEqual(self.ran, [])
                ledger.release_hold.set()
                responses = await asyncio.wait_for(
                    asyncio.gather(
                        active, waiting[0], waiting[2], waiting[3], replacement
                    ),
                    1,
                )

        expected = ["gate-active", "gate-0", "gate-2", "gate-3", "replacement"]
        self.assertEqual([response.status_code for response in responses], [200] * 5)
        self.assertEqual(
            ledger.charged, [quota_cookie_key(query) for query in expected]
        )
        self.assertEqual(self.ran, expected)
        self.assertNotIn(quota_cookie_key("gate-1"), ledger.counts)
        self.assertEqual(ledger.counts[quota_cookie_key("replacement")], 1)

    async def test_denied_or_failed_quota_gate_releases_capacity_to_waiters(
        self,
    ) -> None:
        for denial in (True, False):
            with self.subTest(denial=denial):
                ledger = LocalQuotaLedger(hold="gate-active")
                if not denial:
                    ledger.fail_tracking_ids.add("gate-active")

                with patch(
                    "app.web.is_verified_google_search_crawler_request",
                    new=AsyncMock(return_value=False),
                ):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=self.metered_app(ledger)),
                        base_url="http://testserver",
                    ) as client:
                        active = await self.submit_fragment(client, "gate-active")
                        await asyncio.wait_for(ledger.hold_started.wait(), 1)
                        if denial:
                            # Exhausted after the pre-check passed, so only the
                            # in-turn charge can deny it.
                            ledger.counts[quota_cookie_key("gate-active")] = ANON_LIMIT
                        waiting = [
                            await self.submit_fragment(client, f"gate-{i}")
                            for i in range(4)
                        ]
                        overflow = await self.submit_fragment(client, "overflow")
                        self.assertEqual(
                            (await asyncio.wait_for(overflow, 1)).status_code, 503
                        )
                        ledger.release_hold.set()
                        rejected, *responses = await asyncio.wait_for(
                            asyncio.gather(active, *waiting), 1
                        )
                        next_response = await client.get(
                            "/UNSPSC/fragment", params={"product_description": "next"}
                        )

                self.assertEqual(rejected.status_code, 200 if denial else 503)
                self.assertIn(
                    "Sign in to continue" if denial else "temporarily unavailable",
                    rejected.text,
                )
                self.assertEqual(
                    rejected.headers["Cache-Control"], "no-store, max-age=0"
                )
                self.assertEqual(
                    [response.status_code for response in responses], [200] * 4
                )
                self.assertEqual(next_response.status_code, 200)
                self.assertNotIn("gate-active", self.ran)

    async def test_shutdown_during_quota_drains_allowed_active_request_only(
        self,
    ) -> None:
        ledger = LocalQuotaLedger(hold="gate-active")
        complete_started = threading.Event()
        release_complete = threading.Event()
        self.releases.append(release_complete)
        events: list[str] = []

        prepared = SimpleNamespace(
            context=SimpleNamespace(normalized_query="gate-active"),
            exact_outcome=lambda top_k: None,
        )

        def prepare(**kwargs):
            events.append("prepare")
            return prepared

        async def enhance(query, classifier_type):
            events.append("enhance")
            return EnhancementOutcome("enhanced query", EnhancementStatus.APPLIED)

        def complete(**kwargs):
            complete_started.set()
            if not release_complete.wait(5):
                raise AssertionError("test did not release completion after quota")
            events.append("complete")
            return classification_result("gate-active")

        async def shutdown() -> None:
            await self.executor.close()
            events.append("clients")

        self.service = build_classification_service(
            executor=self.executor, enhancer=SimpleNamespace(enhance=enhance)
        )
        with (
            patch(
                "app.web.is_verified_google_search_crawler_request",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.classification_service.prepare_classification", side_effect=prepare
            ),
            patch(
                "app.classification_service.complete_classification",
                side_effect=complete,
            ),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.metered_app(ledger)),
                base_url="http://testserver",
            ) as client:
                active = await self.submit_fragment(
                    client, "gate-active", enhancement_enabled=True
                )
                await asyncio.wait_for(ledger.hold_started.wait(), 1)
                waiting = [
                    await self.submit_fragment(client, f"gate-{i}") for i in range(4)
                ]
                closing = asyncio.create_task(shutdown())
                self.tasks.append(closing)
                await event_loop_turn()
                for task in waiting:
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, 1)
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    await self.service.classify("new", "UNSPSC")
                self.assertEqual(ledger.charged, [quota_cookie_key("gate-active")])
                self.assertEqual(events, [])
                self.assertFalse(closing.done())
                ledger.release_hold.set()
                self.assertTrue(await asyncio.to_thread(complete_started.wait, 1))
                self.assertEqual(events, ["prepare", "enhance"])
                self.assertFalse(closing.done())
                release_complete.set()
                response = await asyncio.wait_for(active, 1)
                await asyncio.wait_for(closing, 1)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(events, ["prepare", "enhance", "complete", "clients"])
        self.assertEqual(len(ledger.executed), 1)

    async def assert_shutdown_rejected_gate(self, *, quota_error: bool) -> None:
        ledger = LocalQuotaLedger(hold="gate-active")
        if quota_error:
            ledger.fail_tracking_ids.add("gate-active")

        with (
            patch(
                "app.web.is_verified_google_search_crawler_request",
                new=AsyncMock(return_value=False),
            ),
            patch("app.classification_service.prepare_classification") as prepare,
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.metered_app(ledger)),
                base_url="http://testserver",
            ) as client:
                active = await self.submit_fragment(
                    client, "gate-active", enhancement_enabled=True
                )
                await asyncio.wait_for(ledger.hold_started.wait(), 1)
                if not quota_error:
                    # Exhausted after the pre-check passed, so only the in-turn
                    # charge can deny it.
                    ledger.counts[quota_cookie_key("gate-active")] = ANON_LIMIT
                waiting = await self.submit_fragment(client, "waiting")
                closing = asyncio.create_task(self.executor.close())
                self.tasks.append(closing)
                await event_loop_turn()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(waiting, 1)
                self.assertFalse(closing.done())
                ledger.release_hold.set()
                response = await asyncio.wait_for(active, 1)
                await asyncio.wait_for(closing, 1)

        self.assertEqual(response.status_code, 503 if quota_error else 200)
        self.assertIn(
            "temporarily unavailable" if quota_error else "Sign in to continue",
            response.text,
        )
        self.assertEqual(ledger.charged, [quota_cookie_key("gate-active")])
        prepare.assert_not_called()
        self.pipeline.assert_not_called()

    async def test_shutdown_during_denied_quota_does_not_start_pipeline(self) -> None:
        await self.assert_shutdown_rejected_gate(quota_error=False)

    async def test_shutdown_during_failed_quota_does_not_start_pipeline(self) -> None:
        await self.assert_shutdown_rejected_gate(quota_error=True)

    async def test_one_active_four_waiting_reject_sixth_and_run_fifo(self) -> None:
        accepted = await self.fill_queue()
        await self.assert_overflow()
        self.assertEqual(self.ran, ["active"])

        self.release.set()
        outcomes = await asyncio.wait_for(asyncio.gather(*accepted), 1)
        expected = ["active", *[f"waiting-{index}" for index in range(4)]]
        self.assertEqual(self.ran, expected)
        self.assertEqual([outcome.query for outcome in outcomes], expected)
        self.assertEqual((await self.service.classify("next", "UNSPSC")).query, "next")

    async def test_cancelled_waiting_job_frees_capacity_and_never_runs(self) -> None:
        accepted = await self.fill_queue()
        accepted[2].cancel()
        with self.assertRaises(asyncio.CancelledError):
            await accepted[2]
        replacement = await self.submit("replacement")
        await self.assert_overflow()

        self.release.set()
        outcomes = await asyncio.wait_for(
            asyncio.gather(
                accepted[0], accepted[1], accepted[3], accepted[4], replacement
            ),
            1,
        )
        expected = ["active", "waiting-0", "waiting-2", "waiting-3", "replacement"]
        self.assertEqual(self.ran, expected)
        self.assertEqual([outcome.query for outcome in outcomes], expected)

    async def test_cancelled_active_response_keeps_capacity_until_thread_finishes(
        self,
    ) -> None:
        accepted = await self.fill_queue()
        accepted[0].cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(accepted[0], 1)
        await self.assert_overflow()
        self.assertEqual(self.ran, ["active"])

        self.release.set()
        await asyncio.wait_for(asyncio.gather(*accepted[1:]), 1)
        self.assertEqual((await self.service.classify("next", "UNSPSC")).query, "next")

    async def test_error_and_pipeline_cancellation_allow_waiting_jobs_to_continue(
        self,
    ) -> None:
        for error in (ValueError("worker failed"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                self.started.clear()
                self.release.clear()

                def fail_active(*, query: str, **kwargs) -> dict:
                    if query == "active":
                        self.started.set()
                        if not self.release.wait(5):
                            raise AssertionError("test did not release active worker")
                        raise error
                    return classification_result(query)

                self.pipeline.side_effect = fail_active
                accepted = await self.fill_queue()
                await self.assert_overflow()
                self.release.set()
                with self.assertRaises(type(error)) as caught:
                    await accepted[0]
                if not isinstance(error, asyncio.CancelledError):
                    self.assertIs(caught.exception, error)
                outcomes = await asyncio.wait_for(asyncio.gather(*accepted[1:]), 1)
                self.assertEqual(
                    [outcome.query for outcome in outcomes],
                    [f"waiting-{i}" for i in range(4)],
                )

    async def enhanced_service(self):
        enhancing = asyncio.Event()
        release_enhancement = asyncio.Event()
        enhancement_cancelled = asyncio.Event()
        self.releases.append(release_enhancement)
        prepared = SimpleNamespace(
            context=SimpleNamespace(normalized_query="active"),
            exact_outcome=lambda top_k: None,
        )

        async def enhance(query: str, classifier_type: str) -> EnhancementOutcome:
            enhancing.set()
            try:
                await release_enhancement.wait()
            except asyncio.CancelledError:
                enhancement_cancelled.set()
                raise
            return EnhancementOutcome("enhanced active", EnhancementStatus.APPLIED)

        enhancer = SimpleNamespace(enhance=enhance)
        self.service = build_classification_service(
            executor=self.executor, enhancer=enhancer
        )
        prepare = patch(
            "app.classification_service.prepare_classification", return_value=prepared
        ).start()
        complete = patch(
            "app.classification_service.complete_classification",
            return_value=classification_result("active"),
        ).start()
        active = await self.submit("active", enhancement_enabled=True)
        await asyncio.wait_for(enhancing.wait(), 1)
        return active, release_enhancement, enhancement_cancelled, prepare, complete

    async def test_enhancement_holds_whole_turn_and_completes_before_next_pipeline(
        self,
    ) -> None:
        (
            active,
            release_enhancement,
            _,
            prepare,
            complete,
        ) = await self.enhanced_service()
        waiting = [await self.submit(f"waiting-{index}") for index in range(4)]
        await self.assert_overflow()
        await event_loop_turn()
        self.assertEqual(self.ran, [])
        complete.assert_not_called()

        release_enhancement.set()
        outcomes = await asyncio.wait_for(asyncio.gather(active, *waiting), 1)
        self.assertIs(outcomes[0].enhancement_status, EnhancementStatus.APPLIED)
        prepare.assert_called_once()
        complete.assert_called_once()
        self.assertEqual(self.ran, [f"waiting-{index}" for index in range(4)])

    async def test_cancelled_enhancement_stops_completion_and_releases_turn(
        self,
    ) -> None:
        active, _, enhancement_cancelled, _, complete = await self.enhanced_service()
        waiting = await self.submit("waiting")
        active.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await active
        await asyncio.wait_for(enhancement_cancelled.wait(), 1)
        self.assertEqual((await asyncio.wait_for(waiting, 1)).query, "waiting")
        complete.assert_not_called()

    async def test_cancelled_thread_stages_keep_capacity_and_observe_raced_errors(
        self,
    ) -> None:
        loop = asyncio.get_running_loop()
        errors = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        try:
            for blocked_stage in ("prepare", "complete"):
                with self.subTest(stage=blocked_stage):
                    started = threading.Event()
                    release = threading.Event()
                    self.releases.append(release)
                    prepared = SimpleNamespace(
                        context=SimpleNamespace(normalized_query="active"),
                        exact_outcome=lambda top_k: None,
                    )
                    owners = []

                    async def enhance(query, classifier_type):
                        owners.append(asyncio.current_task())
                        return EnhancementOutcome(
                            "enhanced active", EnhancementStatus.APPLIED
                        )

                    enhancer = SimpleNamespace(enhance=AsyncMock(side_effect=enhance))
                    self.service = build_classification_service(
                        executor=self.executor, enhancer=enhancer
                    )

                    def block_then_fail() -> None:
                        started.set()
                        if not release.wait(5):
                            raise AssertionError("test did not release thread stage")
                        raise ValueError("stage failed after caller cancellation")

                    def prepare(**kwargs):
                        if blocked_stage == "prepare":
                            block_then_fail()
                        return prepared

                    def complete(**kwargs):
                        block_then_fail()

                    with (
                        patch(
                            "app.classification_service.prepare_classification",
                            side_effect=prepare,
                        ),
                        patch(
                            "app.classification_service.complete_classification",
                            side_effect=complete,
                        ) as finish,
                    ):
                        active = await self.submit("active", enhancement_enabled=True)
                        self.assertTrue(await asyncio.to_thread(started.wait, 1))
                        waiting = [await self.submit(f"waiting-{i}") for i in range(4)]
                        active.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await asyncio.wait_for(active, 1)
                        await self.assert_overflow()
                        if blocked_stage == "complete":
                            for _ in range(2):
                                owners[0].cancel()
                                await event_loop_turn()
                                await self.assert_overflow()
                        release.set()
                        await asyncio.wait_for(asyncio.gather(*waiting), 1)
                        if blocked_stage == "prepare":
                            enhancer.enhance.assert_not_awaited()
                            finish.assert_not_called()
                        else:
                            finish.assert_called_once()
                    gc.collect()
                    await event_loop_turn()
            self.assertEqual(errors, [])
        finally:
            loop.set_exception_handler(previous_handler)

    async def test_close_drains_enhancement_and_completion_before_clients(self) -> None:
        active, release_enhancement, _, _, complete = await self.enhanced_service()
        waiting = await self.submit("waiting")
        complete_started = threading.Event()
        release_complete = threading.Event()
        self.releases.append(release_complete)
        events: list[str] = []

        def finish(**kwargs) -> dict:
            complete_started.set()
            if not release_complete.wait(5):
                raise AssertionError("test did not release completion")
            events.append("complete")
            return classification_result("active")

        complete.side_effect = finish

        async def close_clients(clients) -> None:
            events.append("clients")

        clients = main.StartupClients()
        with (
            patch.object(main, "ClassificationExecutor", return_value=self.executor),
            patch.object(
                main, "initialize_startup_clients", new=AsyncMock(return_value=clients)
            ),
            patch.object(main, "assign_startup_clients"),
            patch.object(main, "close_startup_clients", side_effect=close_clients),
        ):
            lifespan = main.lifespan(FastAPI())
            await lifespan.__aenter__()
            closing = asyncio.create_task(lifespan.__aexit__(None, None, None))
            self.tasks.append(closing)
            await event_loop_turn()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(waiting, 1)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                await self.service.classify("new", "UNSPSC")
            self.assertEqual(events, [])
            self.assertFalse(closing.done())

            concurrent_close = asyncio.create_task(self.executor.close())
            self.tasks.append(concurrent_close)
            closing.cancel()
            await event_loop_turn()
            closing.cancel()
            await event_loop_turn()
            self.assertFalse(closing.done())
            self.assertFalse(concurrent_close.done())
            release_enhancement.set()
            self.assertTrue(await asyncio.to_thread(complete_started.wait, 1))
            self.assertEqual(events, [])
            release_complete.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(closing, 1)
            await asyncio.wait_for(concurrent_close, 1)
            self.assertEqual((await active).query, "active")
            self.assertEqual(events, ["complete", "clients"])
            self.assertEqual(self.ran, [])
            await self.executor.close()

    async def test_fragment_overflow_is_503_and_no_store(self) -> None:
        accepted = await self.fill_queue()
        app = FastAPI()
        app.include_router(router)
        app.state.classification_service = self.service
        app.state.redis_client = EmptyUsageRedis()
        usage = UsageStatus(True, 9, 10, False, False, "test-track")
        with (
            patch("app.web.reserve_usage", new=AsyncMock(return_value=usage)),
            patch(
                "app.web.is_verified_google_search_crawler_request",
                new=AsyncMock(return_value=False),
            ),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                with self.assertLogs("app.web", "WARNING") as logs:
                    response = await asyncio.wait_for(
                        client.get(
                            "/UNSPSC/fragment",
                            params={"product_description": "overflow"},
                        ),
                        1,
                    )
        self.assertTrue(any("queue full" in line for line in logs.output))
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["Cache-Control"], "no-store, max-age=0")
        self.assertEqual(response.headers["Cloudflare-CDN-Cache-Control"], "no-store")
        self.assertTrue(response.headers["Content-Type"].startswith("text/html"))
        self.assertIn(
            "Classification queue is full. Please try again later.", response.text
        )
        self.assertNotIn('{"detail":', response.text)
        self.release.set()
        await asyncio.wait_for(asyncio.gather(*accepted), 1)

    async def test_api_overflow_is_json_503_and_no_store(self) -> None:
        accepted = await self.fill_queue()
        app = FastAPI()
        app.include_router(api.router, prefix="/api/v1/rapid")
        app.state.classification_service = self.service
        with patch.object(api, "RAPIDAPI_SECRET", "local-test-secret"):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                with self.assertLogs("app.api", "WARNING") as logs:
                    response = await asyncio.wait_for(
                        client.get(
                            "/api/v1/rapid/classify",
                            params={"query": "overflow", "standard": "UNSPSC"},
                            headers={"X-RapidAPI-Proxy-Secret": "local-test-secret"},
                        ),
                        1,
                    )
        self.assertTrue(any("queue full" in line for line in logs.output))
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["Content-Type"], "application/json")
        self.assertEqual(
            response.json(),
            {"detail": "Classification queue is full. Please try again later."},
        )
        self.assertEqual(response.headers["Cache-Control"], "no-store, max-age=0")
        self.assertEqual(response.headers["Cloudflare-CDN-Cache-Control"], "no-store")
        self.release.set()
        await asyncio.wait_for(asyncio.gather(*accepted), 1)

    async def test_ssr_overflow_keeps_indexable_page_fallback_for_visitors(
        self,
    ) -> None:
        accepted = await self.fill_queue()
        with patch(
            "app.web.is_verified_google_search_crawler_request",
            new=AsyncMock(return_value=False),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.page_app()),
                base_url="http://testserver",
            ) as client:
                response = await asyncio.wait_for(client.get("/UNSPSC/"), 1)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["Content-Type"].startswith("text/html"))
        self.assertTrue('data-autoload-enabled="true"' in response.text)
        self.assertEqual(response.headers["Cache-Control"], "no-store, max-age=0")
        self.assertEqual(response.headers["Cloudflare-CDN-Cache-Control"], "no-store")
        self.assertEqual(response.headers["X-Robots-Tag"], "index, follow")
        self.assertNotIn("Retry-After", response.headers)
        self.release.set()
        await asyncio.wait_for(asyncio.gather(*accepted), 1)

    async def test_ssr_overflow_asks_verified_google_crawler_to_retry(self) -> None:
        accepted = await self.fill_queue()
        with patch(
            "app.web.is_verified_google_search_crawler_request",
            new=AsyncMock(return_value=True),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.page_app()),
                base_url="http://testserver",
            ) as client:
                for path in ("/UNSPSC/", "/UNSPSC/laptop_computer/"):
                    with self.subTest(path=path):
                        response = await asyncio.wait_for(client.get(path), 1)
                        self.assertEqual(response.status_code, 503)
                        self.assertEqual(response.headers["Retry-After"], "120")
                        self.assertEqual(
                            response.headers["Cache-Control"], "no-store, max-age=0"
                        )
                        self.assertEqual(
                            response.headers["Cloudflare-CDN-Cache-Control"],
                            "no-store",
                        )
        self.assertEqual(self.ran, ["active"])
        self.release.set()
        await asyncio.wait_for(asyncio.gather(*accepted), 1)

    async def test_fragment_overflow_does_not_charge_anonymous_or_free_user_quota(
        self,
    ) -> None:
        accepted = await self.fill_queue()
        app = FastAPI()
        app.include_router(router)
        app.state.classification_service = self.service
        pipeline = Mock()
        pipeline.execute = AsyncMock(return_value=[1, True, 1, True])
        redis_client = Mock()
        redis_client.mget = AsyncMock(side_effect=EmptyUsageRedis().mget)
        redis_client.pipeline.return_value = pipeline
        app.state.redis_client = redis_client

        for user in ((None, None), ("free-user", "free")):
            with self.subTest(user=user[0]):
                pipeline.execute.reset_mock()
                with (
                    patch(
                        "app.usage_tracker.extract_user_info_from_token",
                        new=AsyncMock(return_value=user),
                    ),
                    patch(
                        "app.usage_tracker._resolve_authenticated_pro_status",
                        new=AsyncMock(return_value=False),
                    ),
                    patch(
                        "app.web.is_verified_google_search_crawler_request",
                        new=AsyncMock(return_value=False),
                    ),
                ):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app),
                        base_url="http://testserver",
                    ) as client:
                        response = await asyncio.wait_for(
                            client.get(
                                "/UNSPSC/fragment",
                                params={"product_description": "overflow"},
                            ),
                            1,
                        )
                self.assertEqual(response.status_code, 503)
                pipeline.execute.assert_not_awaited()

        self.release.set()
        await asyncio.wait_for(asyncio.gather(*accepted), 1)
