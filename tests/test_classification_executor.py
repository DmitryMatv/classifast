import asyncio
import gc
import threading
import unittest
from collections.abc import Callable
from typing import TypeVar

from fastapi import HTTPException

from app.classification_executor import (
    ClassificationExecutor,
    ClassificationQueueFull,
    StageRunner,
)
from tests.helpers import event_loop_turn

ResultT = TypeVar("ResultT")


class ClassificationExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.executor = ClassificationExecutor()

    async def asyncTearDown(self) -> None:
        await self.executor.close()

    async def submit_stage(self, operation: Callable[[], ResultT]) -> ResultT:
        async def admitted(run_stage: StageRunner) -> ResultT:
            return await run_stage(operation)

        return await self.executor.schedule(admitted)

    async def test_runs_sync_work_outside_event_loop_thread(self) -> None:
        event_loop_thread = threading.get_ident()

        worker_thread = await self.submit_stage(threading.get_ident)

        self.assertNotEqual(worker_thread, event_loop_thread)

    async def test_event_loop_remains_responsive_while_work_is_blocked(self) -> None:
        release = threading.Event()
        started = threading.Event()

        def blocking() -> str:
            started.set()
            release.wait(timeout=2)
            return "done"

        task = asyncio.create_task(self.submit_stage(blocking))
        await asyncio.to_thread(started.wait, 1)
        heartbeat = asyncio.create_task(asyncio.sleep(0, result="alive"))

        self.assertEqual(await heartbeat, "alive")
        release.set()
        self.assertEqual(await task, "done")

    async def test_two_classifications_never_overlap(self) -> None:
        release_first = threading.Event()
        first_started = threading.Event()
        second_started = threading.Event()

        def first() -> None:
            first_started.set()
            release_first.wait(timeout=2)

        def second() -> None:
            second_started.set()

        first_task = asyncio.create_task(self.submit_stage(first))
        await asyncio.to_thread(first_started.wait, 1)
        second_task = asyncio.create_task(self.submit_stage(second))
        await event_loop_turn()
        self.assertFalse(second_started.is_set())

        release_first.set()
        await first_task
        await second_task
        self.assertTrue(second_started.is_set())

    async def test_http_exception_propagates_unchanged(self) -> None:
        error = HTTPException(status_code=503, detail="unavailable")

        with self.assertRaises(HTTPException) as ctx:
            await self.submit_stage(lambda: (_ for _ in ()).throw(error))

        self.assertIs(ctx.exception, error)

    async def test_generic_exception_propagates_unchanged(self) -> None:
        error = ValueError("bad classification")

        with self.assertRaises(ValueError) as ctx:
            await self.submit_stage(lambda: (_ for _ in ()).throw(error))

        self.assertIs(ctx.exception, error)

    async def test_cancelling_active_waiter_does_not_release_worker(self) -> None:
        release_first = threading.Event()
        first_started = threading.Event()
        second_started = threading.Event()

        def first() -> None:
            first_started.set()
            release_first.wait(timeout=2)

        first_task = asyncio.create_task(self.submit_stage(first))
        await asyncio.to_thread(first_started.wait, 1)
        first_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first_task

        second_task = asyncio.create_task(
            self.submit_stage(lambda: second_started.set())
        )
        await event_loop_turn()
        self.assertFalse(second_started.is_set())

        release_first.set()
        await second_task
        self.assertTrue(second_started.is_set())

    async def test_rejects_new_work_after_close(self) -> None:
        await self.executor.close()

        with self.assertRaisesRegex(RuntimeError, "closed"):
            await self.submit_stage(lambda: None)

    async def test_close_waits_for_active_work_and_cancels_queued_work(self) -> None:
        active_started = threading.Event()
        release_active = threading.Event()
        queued_ran = threading.Event()

        def active() -> None:
            active_started.set()
            release_active.wait(timeout=2)

        active_task = asyncio.create_task(self.submit_stage(active))
        self.assertTrue(await asyncio.to_thread(active_started.wait, 1))
        queued_task = asyncio.create_task(self.submit_stage(queued_ran.set))
        close_task = asyncio.create_task(self.executor.close())

        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(close_task), timeout=0.05)
        self.assertFalse(queued_ran.is_set())

        release_active.set()
        await asyncio.wait_for(close_task, timeout=1)
        await asyncio.wait_for(active_task, timeout=1)
        with self.assertRaises(asyncio.CancelledError):
            await queued_task
        self.assertFalse(queued_ran.is_set())

    async def test_close_is_idempotent(self) -> None:
        await self.executor.close()
        await self.executor.close()

    async def test_concurrent_close_callers_wait_for_the_same_shutdown(self) -> None:
        active_started = threading.Event()
        release_active = threading.Event()

        def active() -> None:
            active_started.set()
            release_active.wait(timeout=2)

        active_task = asyncio.create_task(self.submit_stage(active))
        self.assertTrue(await asyncio.to_thread(active_started.wait, 1))
        first_close = asyncio.create_task(self.executor.close())
        second_close = asyncio.create_task(self.executor.close())

        await event_loop_turn()
        self.assertFalse(first_close.done())
        self.assertFalse(second_close.done())

        release_active.set()
        await asyncio.wait_for(asyncio.gather(first_close, second_close), timeout=1)
        await active_task

    async def test_cancelled_close_waits_for_shutdown_before_reraising(self) -> None:
        active_started = threading.Event()
        release_active = threading.Event()

        def active() -> None:
            active_started.set()
            release_active.wait(timeout=2)

        active_task = asyncio.create_task(self.submit_stage(active))
        self.assertTrue(await asyncio.to_thread(active_started.wait, 1))
        cancelled_close = asyncio.create_task(self.executor.close())
        await event_loop_turn()
        cancelled_close.cancel()

        await event_loop_turn()
        self.assertFalse(cancelled_close.done())

        second_close = asyncio.create_task(self.executor.close())
        await event_loop_turn()
        self.assertFalse(second_close.done())

        release_active.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(cancelled_close, timeout=1)
        await asyncio.wait_for(second_close, timeout=1)
        await active_task

    async def test_rejects_work_once_shutdown_begins(self) -> None:
        close_task = asyncio.create_task(self.executor.close())
        await event_loop_turn()

        with self.assertRaisesRegex(RuntimeError, "closed"):
            await self.submit_stage(lambda: None)

        await close_task

    async def test_awaited_child_stages_run_before_and_during_shutdown(self) -> None:
        between_stages = asyncio.Event()
        release_second = asyncio.Event()
        second_started = threading.Event()
        release_thread = threading.Event()
        events: list[str] = []

        def first() -> None:
            events.append("first")

        def second() -> None:
            second_started.set()
            if not release_thread.wait(5):
                raise AssertionError("test did not release second child stage")
            events.append("second")

        async def operation(run_stage: StageRunner) -> None:
            await asyncio.create_task(run_stage(first))
            between_stages.set()
            await release_second.wait()
            await asyncio.create_task(run_stage(second))

        async def cleanup() -> None:
            await self.executor.close()
            events.append("clients")

        active = asyncio.create_task(self.executor.schedule(operation))
        closing = None
        try:
            await asyncio.wait_for(between_stages.wait(), 1)
            closing = asyncio.create_task(cleanup())
            await event_loop_turn()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                await self.submit_stage(lambda: None)
            release_second.set()
            self.assertTrue(await asyncio.to_thread(second_started.wait, 1))
            self.assertEqual(events, ["first"])
            self.assertFalse(closing.done())
            release_thread.set()
            await asyncio.wait_for(asyncio.gather(active, closing), 1)
            self.assertEqual(events, ["first", "second", "clients"])
        finally:
            release_second.set()
            release_thread.set()
            await asyncio.gather(
                active, *([closing] if closing else []), return_exceptions=True
            )

    async def test_cancelled_awaited_child_drains_thread_across_repeated_cancellation(
        self,
    ) -> None:
        started = threading.Event()
        release = threading.Event()
        events: list[str] = []
        owners: list[asyncio.Task] = []
        children: list[asyncio.Task] = []
        errors: list[dict] = []
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, context: errors.append(context))

        def fail_after_release() -> None:
            started.set()
            if not release.wait(5):
                raise AssertionError("test did not release cancelled child stage")
            events.append("thread-finished")
            raise ValueError("child stage failed after cancellation")

        async def operation(run_stage: StageRunner) -> None:
            owners.append(asyncio.current_task())
            child = asyncio.create_task(run_stage(fail_after_release))
            children.append(child)
            await child

        active = asyncio.create_task(self.executor.schedule(operation))
        waiting: list[asyncio.Task] = []
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            for _ in range(4):
                waiting.append(
                    asyncio.create_task(
                        self.submit_stage(lambda: events.append("waiting"))
                    )
                )
                await event_loop_turn()
            active.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(active, 1)
            for _ in range(2):
                owners[0].cancel()
                await event_loop_turn()
                with self.assertRaises(ClassificationQueueFull):
                    await self.submit_stage(lambda: None)
                self.assertFalse(children[0].done())
                self.assertEqual(events, [])
            release.set()
            await asyncio.wait_for(asyncio.gather(*waiting), 1)
            self.assertEqual(events, ["thread-finished", *["waiting"] * 4])
            await self.submit_stage(lambda: events.append("next"))
            self.assertEqual(events[-1], "next")
            gc.collect()
            await event_loop_turn()
            self.assertEqual(errors, [])
        finally:
            release.set()
            await asyncio.gather(active, *waiting, return_exceptions=True)
            loop.set_exception_handler(previous_handler)
