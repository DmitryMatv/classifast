import unittest
from unittest.mock import Mock, patch

import httpx

from app.reranker import HuggingFaceReranker, RerankerResponseError


class HuggingFaceRerankerTests(unittest.TestCase):
    def _client_with_response(self, payload):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        client = Mock()
        client.post.return_value = response
        return client

    def test_rerank_posts_batched_pairs_and_preserves_document_order(self) -> None:
        client = self._client_with_response(
            [
                [{"label": "LABEL_0", "score": 0.2}],
                [{"label": "LABEL_0", "score": 0.9}],
            ]
        )
        reranker = HuggingFaceReranker(
            api_key="test-token",
            model_name="BAAI/bge-reranker-v2-m3",
            client=client,
        )

        scores = reranker.rerank("pump query", ["first document", "second document"])

        self.assertEqual(scores, [0.2, 0.9])
        client.post.assert_called_once_with(
            "BAAI/bge-reranker-v2-m3",
            json={
                "inputs": [
                    {"text": "pump query", "text_pair": "first document"},
                    {"text": "pump query", "text_pair": "second document"},
                ],
                "parameters": {"function_to_apply": "sigmoid", "top_k": 1},
            },
        )

    @patch("app.reranker.httpx.Client")
    def test_default_client_uses_bearer_token_and_timeout(self, client_class) -> None:
        client = Mock()
        client_class.return_value = client

        reranker = HuggingFaceReranker(
            api_key="secret-token",
            model_name="BAAI/bge-reranker-v2-m3",
            timeout_seconds=12.5,
        )
        reranker.close()

        client_class.assert_called_once_with(
            base_url="https://router.huggingface.co/hf-inference/models",
            headers={
                "Authorization": "Bearer secret-token",
                "Content-Type": "application/json",
            },
            timeout=12.5,
        )
        client.close.assert_called_once_with()

    def test_rerank_rejects_incomplete_or_invalid_scores(self) -> None:
        for payload in (
            [[{"label": "LABEL_0", "score": 0.4}]],
            [[{"label": "LABEL_0", "score": "0.4"}], [{"score": 0.5}]],
            [[{"label": "LABEL_0", "score": 1.1}], [{"score": 0.5}]],
        ):
            with self.subTest(payload=payload):
                reranker = HuggingFaceReranker(
                    api_key="test-token",
                    model_name="model",
                    client=self._client_with_response(payload),
                )

                with self.assertRaises(RerankerResponseError):
                    reranker.rerank("query", ["first", "second"])

    def test_transient_http_failure_is_retried(self) -> None:
        request = httpx.Request("POST", "https://example.test/model")
        unavailable = httpx.Response(503, request=request)
        recovered = Mock()
        recovered.raise_for_status.return_value = None
        recovered.json.return_value = [[{"label": "LABEL_0", "score": 0.6}]]
        client = Mock()
        client.post.side_effect = [
            httpx.HTTPStatusError("unavailable", request=request, response=unavailable),
            recovered,
        ]
        reranker = HuggingFaceReranker(
            api_key="test-token", model_name="model", client=client
        )

        with patch("app.reranker.tenacity.nap.sleep"):
            self.assertEqual(reranker.rerank("query", ["document"]), [0.6])

        self.assertEqual(client.post.call_count, 2)

    def test_non_transient_http_failure_is_not_retried(self) -> None:
        request = httpx.Request("POST", "https://example.test/model")
        forbidden = httpx.Response(403, request=request)
        client = Mock()
        client.post.side_effect = httpx.HTTPStatusError(
            "forbidden", request=request, response=forbidden
        )
        reranker = HuggingFaceReranker(
            api_key="test-token", model_name="model", client=client
        )

        with self.assertRaises(httpx.HTTPStatusError):
            reranker.rerank("query", ["document"])

        client.post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
