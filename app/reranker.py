"""OpenRouter reranking API adapter for cross-encoder reranking."""

from __future__ import annotations

from typing import Any, Sequence

import httpx
import tenacity

OPENROUTER_RERANK_URL = "https://openrouter.ai/api/v1/rerank"
TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class RerankerResponseError(RuntimeError):
    """Raised when OpenRouter returns an unusable reranking response."""


def _is_transient_rerank_error(error: BaseException) -> bool:
    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in TRANSIENT_STATUS_CODES
    return False


class OpenRouterReranker:
    """Batch query-document pairs through OpenRouter's /rerank endpoint."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._client = client or httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_seconds,
        )

    def close(self) -> None:
        self._client.close()

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        timeout_seconds: float | None = None,
    ) -> list[float]:
        """Return relevance scores in [0, 1] matching ``documents`` order.

        ``top_n`` is set to ``len(documents)`` so every submitted document
        receives a score and the caller can compare ordering directly.

        ``timeout_seconds`` caps this call's total wall-clock budget
        (including retries); it never raises the client's configured timeout.
        """
        if not documents:
            return []

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": list(documents),
            "top_n": len(documents),
        }
        response_payload = self._post_with_retry(payload, timeout_seconds)
        return self._parse_scores(response_payload, n_documents=len(documents))

    def _request_timeout(self, timeout_seconds: float | None) -> float:
        if timeout_seconds is None:
            return self._timeout_seconds
        return max(0.1, min(timeout_seconds, self._timeout_seconds))

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        retry=tenacity.retry_if_exception(_is_transient_rerank_error),
        reraise=True,
    )
    def _post_with_retry_unbounded(self, payload: dict[str, Any]) -> Any:
        response = self._client.post(OPENROUTER_RERANK_URL, json=payload)
        response.raise_for_status()
        return response.json()

    def _post_with_retry(
        self, payload: dict[str, Any], timeout_seconds: float | None
    ) -> Any:
        if timeout_seconds is None:
            return self._post_with_retry_unbounded(payload)

        request_timeout = self._request_timeout(timeout_seconds)
        retryer = tenacity.Retrying(
            stop=(
                tenacity.stop_after_attempt(3)
                | tenacity.stop_after_delay(request_timeout)
            ),
            wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
            retry=tenacity.retry_if_exception(_is_transient_rerank_error),
            reraise=True,
        )

        def _post() -> Any:
            response = self._client.post(
                OPENROUTER_RERANK_URL, json=payload, timeout=request_timeout
            )
            response.raise_for_status()
            return response.json()

        return retryer(_post)

    @staticmethod
    def _parse_scores(response_payload: Any, *, n_documents: int) -> list[float]:
        if not isinstance(response_payload, dict) or "results" not in response_payload:
            raise RerankerResponseError(
                "Reranking response is missing a 'results' array"
            )
        results = response_payload["results"]
        if not isinstance(results, list) or len(results) != n_documents:
            raise RerankerResponseError(
                "Reranking response count does not match requested document count"
            )

        scores_by_index: dict[int, float] = {}
        for entry in results:
            if not isinstance(entry, dict):
                raise RerankerResponseError("Reranking response item is malformed")

            index = entry.get("index")
            score = entry.get("relevance_score")
            if isinstance(index, bool) or not isinstance(index, int):
                raise RerankerResponseError(
                    "Reranking response index is not an integer"
                )
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RerankerResponseError("Reranking response score is not numeric")

            numeric_score = float(score)
            if not 0.0 <= numeric_score <= 1.0:
                raise RerankerResponseError(
                    "Reranking response score is outside [0, 1]"
                )
            scores_by_index[index] = numeric_score

        if len(scores_by_index) != n_documents:
            raise RerankerResponseError(
                "Reranking response indices are not unique or do not cover every document"
            )

        return [scores_by_index[i] for i in range(n_documents)]
