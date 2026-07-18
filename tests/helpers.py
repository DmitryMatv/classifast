from typing import Any, Callable, TypeVar

ResultT = TypeVar("ResultT")


class InlineClassificationExecutor:
    """Test executor that preserves the production executor's async interface."""

    async def run(
        self,
        callable_: Callable[..., ResultT],
        *args: Any,
        **kwargs: Any,
    ) -> ResultT:
        return callable_(*args, **kwargs)
