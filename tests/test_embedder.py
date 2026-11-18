import json
import time
import unittest
from collections.abc import Callable
from functools import partial
from unittest.mock import patch

import httpx

from app.embedder import (
    OPENROUTER_EMBEDDINGS_URL,
    EmbeddingResponseError,
    OpenRouterEmbedder,
)

MODEL = "qwen/qwen3-embedding-8b"
REAL_SLEEP = time.sleep
Handler = Callable[[httpx.Request], httpx.Response]


def ok(vector: list[float], provider: str = "Nebius") -> httpx.Response:
    return httpx.Response(
        200, json={"provider": provider, "data": [{"embedding": vector}]}
    )


def embedder_for(
    handler: Handler, provider: str = "auto", timeout_seconds: float = 20.0
) -> OpenRouterEmbedder:
    return OpenRouterEmbedder(
        api_key="test-key",
        provider=provider,
        timeout_seconds=timeout_seconds,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


class RecordingHandler:
    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responses.pop(0)

    def body(self, index: int = 0) -> dict:
        return json.loads(self.requests[index].content)


class OpenRouterEmbedderTests(unittest.TestCase):
    def test_auto_provider_requests_latency_sorted_routing(self) -> None:
        handler = RecordingHandler(ok([0.1, 0.2]))

        vector = embedder_for(handler, provider="auto").embed(
            "pump", model=MODEL, dimensions=2
        )

        self.assertEqual(vector, [0.1, 0.2])
        self.assertEqual(str(handler.requests[0].url), OPENROUTER_EMBEDDINGS_URL)
        self.assertEqual(
            handler.body(),
            {
                "model": MODEL,
                "input": "pump",
                "dimensions": 2,
                "encoding_format": "float",
                "provider": {"sort": "latency"},
            },
        )

    def test_named_provider_is_pinned_without_fallbacks(self) -> None:
        handler = RecordingHandler(ok([0.1]))

        embedder_for(handler, provider="deepinfra").embed("pump", model=MODEL)

        body = handler.body()
        self.assertEqual(
            body["provider"], {"only": ["deepinfra"], "allow_fallbacks": False}
        )
        self.assertNotIn("dimensions", body)

    def test_default_client_sends_api_key(self) -> None:
        handler = RecordingHandler(ok([0.1]))
        mocked_client = partial(httpx.Client, transport=httpx.MockTransport(handler))

        with patch("app.embedder.httpx.Client", mocked_client):
            embedder = OpenRouterEmbedder(api_key="secret-key")
        embedder.embed("pump", model=MODEL)
        embedder.close()

        self.assertEqual(
            handler.requests[0].headers["Authorization"], "Bearer secret-key"
        )

    def test_transient_statuses_are_retried_until_success(self) -> None:
        handler = RecordingHandler(
            httpx.Response(429), httpx.Response(503), ok([1, 2.5])
        )

        with patch("tenacity.nap.time.sleep"):
            vector = embedder_for(handler).embed("pump", model=MODEL)

        self.assertEqual(vector, [1.0, 2.5])
        self.assertEqual(len(handler.requests), 3)

    def test_transport_error_is_retried(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ConnectError("reset", request=request)
            return ok([0.5])

        with patch("tenacity.nap.time.sleep"):
            vector = embedder_for(handler).embed("pump", model=MODEL)

        self.assertEqual(vector, [0.5])
        self.assertEqual(calls, 2)

    def test_client_error_is_not_retried(self) -> None:
        handler = RecordingHandler(httpx.Response(400), ok([0.1]))

        with self.assertRaises(httpx.HTTPStatusError) as ctx:
            embedder_for(handler).embed("pump", model=MODEL)

        self.assertEqual(ctx.exception.response.status_code, 400)
        self.assertEqual(len(handler.requests), 1)

    def test_error_payload_with_http_200_raises_without_retry(self) -> None:
        handler = RecordingHandler(
            httpx.Response(
                200, json={"error": {"code": 502, "message": "upstream failed"}}
            ),
            ok([0.1]),
        )

        with self.assertRaisesRegex(EmbeddingResponseError, "upstream failed"):
            embedder_for(handler).embed("pump", model=MODEL)

        self.assertEqual(len(handler.requests), 1)

    def test_malformed_payloads_raise(self) -> None:
        payloads = [
            [0.1, 0.2],
            {"data": []},
            {"data": [{"embedding": []}]},
            {"data": [{"embedding": [0.1, "x"]}]},
            {"data": [{"embedding": [True, 0.2]}]},
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                handler = RecordingHandler(httpx.Response(200, json=payload))

                with self.assertRaises(EmbeddingResponseError):
                    embedder_for(handler).embed("pump", model=MODEL)

    def test_budget_caps_request_timeout(self) -> None:
        handler = RecordingHandler(ok([0.1]))

        embedder_for(handler, timeout_seconds=20.0).embed(
            "pump", model=MODEL, max_seconds=3.0
        )

        self.assertEqual(handler.requests[0].extensions["timeout"]["read"], 3.0)

    def test_budget_never_raises_configured_timeout(self) -> None:
        handler = RecordingHandler(ok([0.1]))

        embedder_for(handler, timeout_seconds=5.0).embed(
            "pump", model=MODEL, max_seconds=999.0
        )

        self.assertEqual(handler.requests[0].extensions["timeout"]["read"], 5.0)

    def test_retries_continue_past_request_timeout_within_budget(self) -> None:
        responses = [httpx.Response(503), httpx.Response(503), ok([1.0])]
        calls = 0

        def slow_then_ok(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            REAL_SLEEP(0.06)
            return responses.pop(0)

        with patch("tenacity.nap.time.sleep"):
            vector = embedder_for(slow_then_ok, timeout_seconds=0.05).embed(
                "pump", model=MODEL, max_seconds=10.0
            )

        self.assertEqual(vector, [1.0])
        self.assertEqual(calls, 3)

    def test_spent_budget_stops_retries(self) -> None:
        calls = 0

        def slow_unavailable(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            time.sleep(0.15)
            return httpx.Response(503)

        with self.assertRaises(httpx.HTTPStatusError):
            embedder_for(slow_unavailable).embed("pump", model=MODEL, max_seconds=0.1)

        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
