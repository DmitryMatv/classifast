import unittest
from unittest.mock import Mock, patch

import httpx

from app.reranker import (
    OPENROUTER_RERANK_URL,
    OpenRouterReranker,
    RerankerResponseError,
)


class OpenRouterRerankerTests(unittest.TestCase):
    def _client_with_response(self, payload):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        client = Mock()
        client.post.return_value = response
        return client

    def test_rerank_posts_batched_documents_and_preserves_document_order(self) -> None:
        client = self._client_with_response(
            {
                "model": "voyageai/rerank-2.5",
                "results": [
                    {
                        "index": 0,
                        "relevance_score": 0.2,
                        "document": {"text": "first document"},
                    },
                    {
                        "index": 1,
                        "relevance_score": 0.9,
                        "document": {"text": "second document"},
                    },
                ],
            }
        )
        reranker = OpenRouterReranker(
            api_key="test-token",
            model_name="voyageai/rerank-2.5",
            client=client,
        )

        scores = reranker.rerank("pump query", ["first document", "second document"])

        self.assertEqual(scores, [0.2, 0.9])
        client.post.assert_called_once_with(
            OPENROUTER_RERANK_URL,
            json={
                "model": "voyageai/rerank-2.5",
                "query": "pump query",
                "documents": ["first document", "second document"],
                "top_n": 2,
            },
        )

    @patch("app.reranker.httpx.Client")
    def test_default_client_uses_bearer_token_and_timeout(self, client_class) -> None:
        client = Mock()
        client_class.return_value = client

        reranker = OpenRouterReranker(
            api_key="secret-token",
            model_name="voyageai/rerank-2.5",
            timeout_seconds=12.5,
        )
        reranker.close()

        client_class.assert_called_once_with(
            headers={
                "Authorization": "Bearer secret-token",
                "Content-Type": "application/json",
            },
            timeout=12.5,
        )
        client.close.assert_called_once_with()

    def test_rerank_reorders_results_to_match_document_order(self) -> None:
        client = self._client_with_response(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.4},
                ]
            }
        )
        reranker = OpenRouterReranker(
            api_key="test-token",
            model_name="model",
            client=client,
        )

        scores = reranker.rerank("query", ["first", "second"])

        self.assertEqual(scores, [0.4, 0.9])

    def test_rerank_rejects_incomplete_or_invalid_scores(self) -> None:
        payloads = (
            {"results": [{"index": 0, "relevance_score": 0.4}]},
            {
                "results": [
                    {"index": 0, "relevance_score": "0.4"},
                    {"index": 1, "relevance_score": 0.5},
                ]
            },
            {
                "results": [
                    {"index": 0, "relevance_score": 1.1},
                    {"index": 1, "relevance_score": 0.5},
                ]
            },
            {
                "results": [
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 0, "relevance_score": 0.5},
                ]
            },
            {"results": "not-a-list"},
            "not-a-dict",
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                reranker = OpenRouterReranker(
                    api_key="test-token",
                    model_name="model",
                    client=self._client_with_response(payload),
                )

                with self.assertRaises(RerankerResponseError):
                    reranker.rerank("query", ["first", "second"])

    def test_transient_http_failure_is_retried(self) -> None:
        request = httpx.Request("POST", OPENROUTER_RERANK_URL)
        unavailable = httpx.Response(503, request=request)
        recovered = Mock()
        recovered.raise_for_status.return_value = None
        recovered.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.6}]
        }
        client = Mock()
        client.post.side_effect = [
            httpx.HTTPStatusError("unavailable", request=request, response=unavailable),
            recovered,
        ]
        reranker = OpenRouterReranker(
            api_key="test-token", model_name="model", client=client
        )

        with patch("app.reranker.tenacity.nap.sleep"):
            self.assertEqual(reranker.rerank("query", ["document"]), [0.6])

        self.assertEqual(client.post.call_count, 2)

    def test_non_transient_http_failure_is_not_retried(self) -> None:
        request = httpx.Request("POST", OPENROUTER_RERANK_URL)
        forbidden = httpx.Response(403, request=request)
        client = Mock()
        client.post.side_effect = httpx.HTTPStatusError(
            "forbidden", request=request, response=forbidden
        )
        reranker = OpenRouterReranker(
            api_key="test-token", model_name="model", client=client
        )

        with self.assertRaises(httpx.HTTPStatusError):
            reranker.rerank("query", ["document"])

        client.post.assert_called_once()

    def test_empty_documents_returns_empty_scores_without_request(self) -> None:
        client = Mock()
        reranker = OpenRouterReranker(
            api_key="test-token", model_name="model", client=client
        )

        self.assertEqual(reranker.rerank("query", []), [])
        client.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
