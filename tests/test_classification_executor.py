import asyncio
import threading
import unittest

from fastapi import HTTPException

from app.classification_executor import ClassificationExecutor


class ClassificationExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.executor = ClassificationExecutor()

    async def asyncTearDown(self) -> None:
        await self.executor.close()

    async def test_runs_sync_work_outside_event_loop_thread(self) -> None:
        event_loop_thread = threading.get_ident()

        worker_thread = await self.executor.run(threading.get_ident)

        self.assertNotEqual(worker_thread, event_loop_thread)

    async def test_event_loop_remains_responsive_while_work_is_blocked(self) -> None:
        release = threading.Event()
        started = threading.Event()

        def blocking() -> str:
            started.set()
            release.wait(timeout=2)
            return "done"

        task = asyncio.create_task(self.executor.run(blocking))
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

        first_task = asyncio.create_task(self.executor.run(first))
        await asyncio.to_thread(first_started.wait, 1)
        second_task = asyncio.create_task(self.executor.run(second))
        await asyncio.sleep(0.05)
        self.assertFalse(second_started.is_set())

        release_first.set()
        await first_task
        await second_task
        self.assertTrue(second_started.is_set())

    async def test_http_exception_propagates_unchanged(self) -> None:
        error = HTTPException(status_code=503, detail="unavailable")

        with self.assertRaises(HTTPException) as ctx:
            await self.executor.run(lambda: (_ for _ in ()).throw(error))

        self.assertIs(ctx.exception, error)

    async def test_generic_exception_propagates_unchanged(self) -> None:
        error = ValueError("bad classification")

        with self.assertRaises(ValueError) as ctx:
            await self.executor.run(lambda: (_ for _ in ()).throw(error))

        self.assertIs(ctx.exception, error)

    async def test_cancelling_active_waiter_does_not_release_worker(self) -> None:
        release_first = threading.Event()
        first_started = threading.Event()
        second_started = threading.Event()

        def first() -> None:
            first_started.set()
            release_first.wait(timeout=2)

        first_task = asyncio.create_task(self.executor.run(first))
        await asyncio.to_thread(first_started.wait, 1)
        first_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first_task

        second_task = asyncio.create_task(
            self.executor.run(lambda: second_started.set())
        )
        await asyncio.sleep(0.05)
        self.assertFalse(second_started.is_set())

        release_first.set()
        await second_task
        self.assertTrue(second_started.is_set())

    async def test_rejects_new_work_after_close(self) -> None:
        await self.executor.close()

        with self.assertRaisesRegex(RuntimeError, "closed"):
            await self.executor.run(lambda: None)

    async def test_close_waits_for_active_work_and_cancels_queued_work(self) -> None:
        active_started = threading.Event()
        release_active = threading.Event()
        queued_ran = threading.Event()

        def active() -> None:
            active_started.set()
            release_active.wait(timeout=2)

        active_task = asyncio.create_task(self.executor.run(active))
        self.assertTrue(await asyncio.to_thread(active_started.wait, 1))
        queued_task = asyncio.create_task(self.executor.run(queued_ran.set))
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

        active_task = asyncio.create_task(self.executor.run(active))
        self.assertTrue(await asyncio.to_thread(active_started.wait, 1))
        first_close = asyncio.create_task(self.executor.close())
        second_close = asyncio.create_task(self.executor.close())

        await asyncio.sleep(0.05)
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

        active_task = asyncio.create_task(self.executor.run(active))
        self.assertTrue(await asyncio.to_thread(active_started.wait, 1))
        cancelled_close = asyncio.create_task(self.executor.close())
        await asyncio.sleep(0)
        cancelled_close.cancel()

        await asyncio.sleep(0.05)
        self.assertFalse(cancelled_close.done())

        second_close = asyncio.create_task(self.executor.close())
        await asyncio.sleep(0.05)
        self.assertFalse(second_close.done())

        release_active.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(cancelled_close, timeout=1)
        await asyncio.wait_for(second_close, timeout=1)
        await active_task

    async def test_rejects_work_once_shutdown_begins(self) -> None:
        close_task = asyncio.create_task(self.executor.close())
        await asyncio.sleep(0)

        with self.assertRaisesRegex(RuntimeError, "closed"):
            await self.executor.run(lambda: None)

        await close_task
