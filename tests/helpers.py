from collections.abc import Callable
from typing import Any, TypeVar

from app.classification_executor import ClassificationExecutor
from app.classification_service import ClassificationOutcome, ClassificationService

ResultT = TypeVar("ResultT")


class InlineClassificationExecutor(ClassificationExecutor):
    """Test executor that preserves the production executor's async interface."""

    async def run(
        self,
        callable_: Callable[..., ResultT],
        *args: Any,
        **kwargs: Any,
    ) -> ResultT:
        return callable_(*args, **kwargs)


def build_classification_service(
    *,
    embed_client: Any = None,
    qdrant_client: Any = None,
    quantization_cache: dict[str, bool] | None = None,
    reranker: Any = None,
    executor: Any = None,
) -> ClassificationService:
    """Build a classification module with infrastructure bound at construction."""
    return ClassificationService(
        embed_client=embed_client if embed_client is not None else object(),
        qdrant_client=qdrant_client if qdrant_client is not None else object(),
        quantization_cache=quantization_cache if quantization_cache is not None else {},
        reranker=reranker,
        executor=executor if executor is not None else InlineClassificationExecutor(),
    )


def build_classification_outcome(
    *,
    results: list[dict[str, Any]] | None = None,
    version_config: dict[str, Any] | None = None,
    version_name: str = "v1",
    collection_name: str = "test_collection",
    query: str = "test query",
) -> ClassificationOutcome:
    return ClassificationOutcome(
        results=results if results is not None else [],
        version_config=version_config if version_config is not None else {},
        version_name=version_name,
        collection_name=collection_name,
        query=query,
    )
