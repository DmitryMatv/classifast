import asyncio
import functools
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ParamSpec, Protocol, TypeVar

ResultT = TypeVar("ResultT")
Params = ParamSpec("Params")

QUEUE_CAPACITY = 5  # one active classification plus four waiting


class ClassificationQueueFull(Exception):
    def __init__(self) -> None:
        super().__init__("Classification queue is full. Please try again later.")


class StageRunner(Protocol):
    async def __call__(
        self,
        operation: Callable[Params, ResultT],
        /,
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> ResultT: ...


async def _wait_through_cancellation(
    future: asyncio.Future[Any],
) -> asyncio.CancelledError | None:
    """Wait for ``future`` to finish and return any cancellation received meanwhile.

    ``asyncio.shield`` would log a late failure of ``future`` after its waiter is
    cancelled, even when the caller retrieves that failure.
    """
    cancellation = None
    while not future.done():
        try:
            await asyncio.wait((future,))
        except asyncio.CancelledError as exc:
            cancellation = exc
    return cancellation


class ClassificationExecutor:
    """Serialize complete classifications and their dedicated thread stages.

    Each job runs in its own task so a cancelled caller returns immediately,
    while the job keeps its queue slot until any running thread stage finishes.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="classification",
        )
        self._closed = False
        self._shutdown_task: asyncio.Task[None] | None = None
        self._jobs: set[asyncio.Task[object]] = set()
        self._job_lock = asyncio.Lock()
        self._active_owner: asyncio.Task[object] | None = None

    async def schedule(
        self,
        operation: Callable[[StageRunner], Awaitable[ResultT]],
    ) -> ResultT:
        if self._closed:
            raise RuntimeError("Classification executor is closed")
        if len(self._jobs) >= QUEUE_CAPACITY:
            raise ClassificationQueueFull()

        owner = asyncio.create_task(self._run_job(operation))
        self._jobs.add(owner)
        owner.add_done_callback(self._job_done)
        try:
            await asyncio.wait((owner,))
            return owner.result()
        except asyncio.CancelledError:
            owner.cancel()
            if owner is not self._active_owner:
                self._jobs.discard(owner)
            raise

    async def _run_job(
        self, operation: Callable[[StageRunner], Awaitable[ResultT]]
    ) -> ResultT:
        async with self._job_lock:
            self._active_owner = asyncio.current_task()
            try:
                return await operation(self._run_stage)
            finally:
                self._active_owner = None

    def _job_done(self, owner: asyncio.Task[object]) -> None:
        self._jobs.discard(owner)
        if not owner.cancelled():
            owner.exception()

    async def _run_stage(
        self,
        callable_: Callable[Params, ResultT],
        /,
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> ResultT:
        loop = asyncio.get_running_loop()
        operation = functools.partial(callable_, *args, **kwargs)
        stage = loop.run_in_executor(self._executor, operation)
        cancellation = await _wait_through_cancellation(stage)
        if cancellation is not None:
            if not stage.cancelled():
                stage.exception()
            raise cancellation
        return stage.result()

    async def close(self) -> None:
        """Stop accepting work and wait for the dedicated worker to terminate.

        Shutdown is shared by concurrent callers and cannot be abandoned by
        cancelling one waiter. A cancelled caller receives its cancellation
        only after active classification has finished and queued work has been
        cancelled, keeping client cleanup ordered after executor shutdown.
        """
        if self._shutdown_task is None:
            self._closed = True
            for owner in tuple(self._jobs):
                if owner is not self._active_owner:
                    owner.cancel()
                    self._jobs.discard(owner)
            self._shutdown_task = asyncio.create_task(self._finish_shutdown())

        cancellation = await _wait_through_cancellation(self._shutdown_task)
        self._shutdown_task.result()
        if cancellation is not None:
            raise cancellation

    async def _finish_shutdown(self) -> None:
        await asyncio.gather(*self._jobs, return_exceptions=True)
        await asyncio.to_thread(
            self._executor.shutdown,
            wait=True,
            cancel_futures=True,
        )
