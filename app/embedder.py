"""OpenRouter embeddings API adapter for query embeddings."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
import tenacity

logger = logging.getLogger(__name__)

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
AUTO_PROVIDER = "auto"
TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class EmbeddingResponseError(RuntimeError):
    """Raised when OpenRouter returns an unusable embedding response."""


def _is_transient_embedding_error(error: BaseException) -> bool:
    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in TRANSIENT_STATUS_CODES
    return False


def _provider_routing(provider: str) -> dict[str, Any]:
    if provider == AUTO_PROVIDER:
        return {"sort": "latency"}
    return {"only": [provider], "allow_fallbacks": False}


class OpenRouterEmbedder:
    """Embed one text through OpenRouter's /embeddings endpoint.

    ``provider`` is ``"auto"`` for OpenRouter's latency-sorted routing, or one
    provider name pinned without fallbacks.
    """

    def __init__(
        self,
        api_key: str,
        provider: str = AUTO_PROVIDER,
        timeout_seconds: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.provider = provider
        self._provider_routing = _provider_routing(provider)
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

    def embed(
        self,
        text: str,
        *,
        model: str,
        dimensions: int | None = None,
        max_seconds: float | None = None,
    ) -> list[float]:
        """Return the pooled embedding vector for ``text``.

        ``max_seconds`` caps this call's total wall-clock budget (including
        retries); it never raises the client's configured timeout.
        """
        payload: dict[str, Any] = {
            "model": model,
            "input": text,
            "encoding_format": "float",
            "provider": self._provider_routing,
        }
        if dimensions is not None:
            payload["dimensions"] = dimensions

        start = time.perf_counter()
        response_payload = self._post_with_retry(payload, max_seconds)
        vector = self._parse_embedding(response_payload)
        logger.info(
            "EMBEDDING: provider=%s duration=%.3fs",
            response_payload.get("provider"),
            time.perf_counter() - start,
        )
        return vector

    def _post_with_retry(
        self, payload: dict[str, Any], max_seconds: float | None
    ) -> Any:
        stop: tenacity.stop.stop_base = tenacity.stop_after_attempt(3)
        request_timeout = self._timeout_seconds
        if max_seconds is not None:
            request_timeout = max(0.1, min(max_seconds, self._timeout_seconds))
            stop = stop | tenacity.stop_after_delay(max_seconds)
        retryer = tenacity.Retrying(
            stop=stop,
            wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
            retry=tenacity.retry_if_exception(_is_transient_embedding_error),
            reraise=True,
        )

        def _post() -> Any:
            response = self._client.post(
                OPENROUTER_EMBEDDINGS_URL, json=payload, timeout=request_timeout
            )
            response.raise_for_status()
            return response.json()

        return retryer(_post)

    @staticmethod
    def _parse_embedding(response_payload: Any) -> list[float]:
        if not isinstance(response_payload, dict):
            raise EmbeddingResponseError("Embedding response is not a JSON object")
        # OpenRouter can report an upstream failure as HTTP 200 with an error body.
        error = response_payload.get("error")
        if error is not None:
            message = error.get("message") if isinstance(error, dict) else error
            raise EmbeddingResponseError(f"OpenRouter embedding error: {message}")

        data = response_payload.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise EmbeddingResponseError("Embedding response is missing 'data'")
        embedding = data[0].get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise EmbeddingResponseError("Embedding response vector is empty")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in embedding
        ):
            raise EmbeddingResponseError(
                "Embedding response contains non-numeric values"
            )
        return [float(value) for value in embedding]
