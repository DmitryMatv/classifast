import asyncio
from collections.abc import Callable, Iterable
from typing import Any, ParamSpec, TypeVar

from app.classification_executor import ClassificationExecutor
from app.classification_service import ClassificationOutcome, ClassificationService

ResultT = TypeVar("ResultT")
Params = ParamSpec("Params")


async def event_loop_turn() -> None:
    ready = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(ready.set_result, None)
    await ready


class EmptyUsageRedis:
    """Redis with no stored usage for tests that patch the quota charge.

    Only the read-only quota pre-check is supported; any write fails.
    """

    async def mget(self, keys: Iterable[str]) -> list[None]:
        return [None for _ in keys]


class InlineClassificationExecutor(ClassificationExecutor):
    """Test executor that preserves the production executor's async interface."""

    async def _run_stage(
        self,
        callable_: Callable[Params, ResultT],
        /,
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> ResultT:
        return callable_(*args, **kwargs)


def build_classification_service(
    *,
    embed_client: Any = None,
    qdrant_client: Any = None,
    quantization_cache: dict[str, bool] | None = None,
    reranker: Any = None,
    executor: Any = None,
    enhancer: Any = None,
) -> ClassificationService:
    """Build a classification module with infrastructure bound at construction."""
    return ClassificationService(
        embed_client=embed_client if embed_client is not None else object(),
        qdrant_client=qdrant_client if qdrant_client is not None else object(),
        quantization_cache=quantization_cache if quantization_cache is not None else {},
        reranker=reranker,
        executor=executor if executor is not None else InlineClassificationExecutor(),
        enhancer=enhancer,
    )


def build_classification_outcome(
    *,
    results: list[dict[str, Any]] | None = None,
    version_config: dict[str, Any] | None = None,
    version_name: str = "v1",
    collection_name: str = "test_collection",
    query: str = "test query",
    elapsed_seconds: float = 0.0,
) -> ClassificationOutcome:
    return ClassificationOutcome(
        results=results if results is not None else [],
        version_config=version_config if version_config is not None else {},
        version_name=version_name,
        collection_name=collection_name,
        query=query,
        elapsed_seconds=elapsed_seconds,
    )
