from dataclasses import dataclass
from typing import Any

from .classification_executor import ClassificationExecutor
from .classifier import (
    QueryFormat,
    complete_classification,
    perform_classification,
    prepare_classification,
)
from .query_enhancer import EnhancementOutcome, EnhancementStatus, QueryEnhancer


@dataclass(frozen=True)
class ClassificationOutcome:
    """Result of a classification request, with named fields only."""

    results: list[dict[str, Any]]
    version_config: dict[str, Any]
    version_name: str
    collection_name: str
    query: str
    enhancement_status: EnhancementStatus | None = None


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
        enhancer: QueryEnhancer | None = None,
    ) -> None:
        self._embed_client = embed_client
        self._qdrant_client = qdrant_client
        self._quantization_cache = quantization_cache
        self._reranker = reranker
        self._executor = executor
        self._enhancer = enhancer

    async def classify(
        self,
        query: str,
        classifier_type: str,
        version: str | None = None,
        top_k: int = 3,
        semantic_query: str | None = None,
        enhancement_enabled: bool = False,
    ) -> ClassificationOutcome:
        """Classify ``query`` against ``classifier_type`` and return an outcome."""
        if enhancement_enabled and semantic_query is not None:
            raise ValueError(
                "semantic_query and enhancement_enabled cannot be combined"
            )

        enhancement_status = None
        if enhancement_enabled:
            prepared = await self._executor.run(
                prepare_classification,
                embed_client=self._embed_client,
                qdrant_client=self._qdrant_client,
                query=query,
                classifier_type=classifier_type,
                version=version,
            )
            exact_outcome = prepared.exact_outcome(top_k)
            if exact_outcome is not None:
                result = exact_outcome
                enhancement_status = EnhancementStatus.SKIPPED
            else:
                enhancement = (
                    await self._enhancer.enhance(
                        prepared.context.normalized_query, classifier_type
                    )
                    if self._enhancer is not None
                    else EnhancementOutcome(query, EnhancementStatus.FAILED)
                )
                enhancement_status = enhancement.status
                applied = enhancement.status is EnhancementStatus.APPLIED
                result = await self._executor.run(
                    complete_classification,
                    prepared=prepared,
                    embed_client=self._embed_client,
                    qdrant_client=self._qdrant_client,
                    top_k=top_k,
                    quantization_cache=self._quantization_cache,
                    reranker=self._reranker,
                    semantic_query=enhancement.text if applied else None,
                    query_format=(
                        QueryFormat.INPUT_FIRST if applied else QueryFormat.LEGACY
                    ),
                )
        else:
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
                semantic_query=semantic_query,
            )
        return ClassificationOutcome(
            results=result["results"],
            version_config=result["version_config"],
            version_name=result["version_name"],
            collection_name=result["collection_name"],
            query=result["query"],
            enhancement_status=enhancement_status,
        )
