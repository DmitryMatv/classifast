"""Hugging Face Inference API adapter for cross-encoder reranking."""

from __future__ import annotations

from typing import Any, Sequence

import httpx
import tenacity

HF_INFERENCE_MODELS_URL = "https://router.huggingface.co/hf-inference/models"
TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class RerankerResponseError(RuntimeError):
    """Raised when Hugging Face returns an unusable reranking response."""


def _is_transient_rerank_error(error: BaseException) -> bool:
    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in TRANSIENT_STATUS_CODES
    return False


class HuggingFaceReranker:
    """Batch query-document pairs through Hugging Face's hf-inference provider."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model_name = model_name
        self._client = client or httpx.Client(
            base_url=HF_INFERENCE_MODELS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_seconds,
        )

    def close(self) -> None:
        self._client.close()

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return sigmoid-normalized relevance scores matching ``documents`` order."""
        if not documents:
            return []

        payload = {
            "inputs": [
                {"text": query, "text_pair": document} for document in documents
            ],
            "parameters": {"function_to_apply": "sigmoid", "top_k": 1},
        }
        response_payload = self._post_with_retry(payload)
        return self._parse_scores(response_payload, expected_count=len(documents))

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        retry=tenacity.retry_if_exception(_is_transient_rerank_error),
        reraise=True,
    )
    def _post_with_retry(self, payload: dict[str, Any]) -> Any:
        response = self._client.post(self.model_name, json=payload)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _parse_scores(response_payload: Any, expected_count: int) -> list[float]:
        if (
            not isinstance(response_payload, list)
            or len(response_payload) != expected_count
        ):
            raise RerankerResponseError(
                "Reranking response count does not match requested document count"
            )

        scores: list[float] = []
        for result in response_payload:
            prediction: Any
            if isinstance(result, dict):
                prediction = result
            elif (
                isinstance(result, list)
                and len(result) == 1
                and isinstance(result[0], dict)
            ):
                prediction = result[0]
            else:
                raise RerankerResponseError("Reranking response item is malformed")

            score = prediction.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RerankerResponseError("Reranking response score is not numeric")
            numeric_score = float(score)
            if not 0.0 <= numeric_score <= 1.0:
                raise RerankerResponseError(
                    "Reranking response score is outside [0, 1]"
                )
            scores.append(numeric_score)

        return scores
