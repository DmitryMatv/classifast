from dataclasses import dataclass
from typing import Any

from .classification_executor import ClassificationExecutor
from .classifier import perform_classification


@dataclass(frozen=True)
class ClassificationOutcome:
    """Result of a classification request, with named fields only."""

    results: list[dict[str, Any]]
    version_config: dict[str, Any]
    version_name: str
    collection_name: str
    query: str


class ClassificationService:
    """Deep classification module: bind infrastructure once, classify anywhere.

    Callers cross one small interface (``classify``) and never learn about
    embedding clients, Qdrant clients, quantization caches, or rerankers.
    The synchronous pipeline is serialized through the executor internally.
    """

    def __init__(
        self,
        embed_client: Any,
        qdrant_client: Any,
        quantization_cache: dict[str, bool] | None,
        reranker: Any,
        executor: ClassificationExecutor,
    ) -> None:
        self._embed_client = embed_client
        self._qdrant_client = qdrant_client
        self._quantization_cache = quantization_cache
        self._reranker = reranker
        self._executor = executor

    async def classify(
        self,
        query: str,
        classifier_type: str,
        version: str | None = None,
        top_k: int = 3,
    ) -> ClassificationOutcome:
        """Classify ``query`` against ``classifier_type`` and return an outcome."""
        result = await self._executor.run(
            perform_classification,
            embed_client=self._embed_client,
            qdrant_client=self._qdrant_client,
            query=query,
            classifier_type=classifier_type,
            version=version,
            top_k=top_k,
            quantization_cache=self._quantization_cache,
            reranker=self._reranker,
        )
        return ClassificationOutcome(
            results=result["results"],
            version_config=result["version_config"],
            version_name=result["version_name"],
            collection_name=result["collection_name"],
            query=result["query"],
        )
