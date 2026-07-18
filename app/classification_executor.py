import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

ResultT = TypeVar("ResultT")


class ClassificationExecutor:
    """Run the synchronous classification pipeline on one dedicated worker."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="classification",
        )
        self._closed = False
        self._shutdown_task: asyncio.Task[None] | None = None

    async def run(
        self,
        callable_: Callable[..., ResultT],
        *args: Any,
        **kwargs: Any,
    ) -> ResultT:
        if self._closed:
            raise RuntimeError("Classification executor is closed")
        loop = asyncio.get_running_loop()
        operation = functools.partial(callable_, *args, **kwargs)
        return await loop.run_in_executor(self._executor, operation)

    async def close(self) -> None:
        """Stop accepting work and wait for the dedicated worker to terminate.

        Shutdown is shared by concurrent callers and cannot be abandoned by
        cancelling one waiter. A cancelled caller receives its cancellation
        only after active classification has finished and queued work has been
        cancelled, keeping client cleanup ordered after executor shutdown.
        """
        if self._shutdown_task is None:
            self._closed = True
            self._shutdown_task = asyncio.create_task(
                asyncio.to_thread(
                    self._executor.shutdown,
                    wait=True,
                    cancel_futures=True,
                )
            )

        shutdown_task = self._shutdown_task
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(shutdown_task)
                break
            except asyncio.CancelledError as exc:
                if shutdown_task.cancelled():
                    raise
                cancellation = exc

        if cancellation is not None:
            raise cancellation
