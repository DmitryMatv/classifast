import unittest
import json

import httpx

from app.query_enhancer import (
    QUERY_ENHANCEMENT_MODEL,
    EnhancementStatus,
    QueryEnhancer,
)


class QueryEnhancerTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_term_becomes_bounded_semantic_text(self) -> None:
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "  Threaded metal fastener  "}}
                    ]
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            enhancer = QueryEnhancer("test-key", client=client)
            result = await enhancer.enhance("bolt", "UNSPSC")

        self.assertEqual(result.text, "bolt\n\nThreaded metal fastener")
        self.assertIs(result.status, EnhancementStatus.APPLIED)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/api/v1/chat/completions")
        self.assertEqual(requests[0].read().decode().count(QUERY_ENHANCEMENT_MODEL), 1)
        messages = json.loads(requests[0].read())["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertTrue(messages[0]["content"].startswith("bolt\n\nFor a UNSPSC"))

    async def test_failures_and_invalid_output_keep_original(self) -> None:
        for response, status in (
            (httpx.Response(503), EnhancementStatus.FAILED),
            (httpx.Response(200, json={"choices": []}), EnhancementStatus.FAILED),
            (
                httpx.Response(
                    200, json={"choices": [{"message": {"content": "x" * 241}}]}
                ),
                EnhancementStatus.FAILED,
            ),
            (
                httpx.Response(200, json={"choices": [{"message": {"content": " "}}]}),
                EnhancementStatus.SKIPPED,
            ),
        ):
            with self.subTest(response=response):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(lambda request: response)
                ) as client:
                    enhancer = QueryEnhancer("test-key", client=client)
                    outcome = await enhancer.enhance("bolt", "UNSPSC")
                    self.assertEqual(outcome.text, "bolt")
                    self.assertIs(outcome.status, status)

    async def test_numeric_code_makes_no_network_request(self) -> None:
        def unexpected_request(request: httpx.Request) -> httpx.Response:
            raise AssertionError("Code lookup should skip the LLM")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(unexpected_request)
        ) as client:
            enhancer = QueryEnhancer("test-key", client=client)
            for code in ("8471", "8471.50", "SH203-C20", "SH203"):
                outcome = await enhancer.enhance(code, "HS")
                self.assertEqual(outcome.text, code)
                self.assertIs(outcome.status, EnhancementStatus.SKIPPED)

    async def test_measurement_text_still_reaches_model(self) -> None:
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "A threaded metal fastener"}}]},
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        ) as client:
            enhancer = QueryEnhancer("test-key", client=client)
            for query in ("bolt 123 mm", "M8 bolt"):
                self.assertIs(
                    (await enhancer.enhance(query, "UNSPSC")).status,
                    EnhancementStatus.APPLIED,
                )
        self.assertEqual(len(requests), 2)

    async def test_model_text_is_sanitized_before_semantic_search(self) -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"message": {"content": "Threaded fastener …"}}
                        ]
                    },
                )
            )
        ) as client:
            enhancer = QueryEnhancer("test-key", client=client)
            self.assertEqual(
                (await enhancer.enhance("bolt", "UNSPSC")).text,
                "bolt\n\nThreaded fastener",
            )

    async def test_invalid_original_never_reaches_model(self) -> None:
        def unexpected_request(request: httpx.Request) -> httpx.Response:
            raise AssertionError("Invalid input should not call the LLM")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(unexpected_request)
        ) as client:
            enhancer = QueryEnhancer("test-key", client=client)
            outcome = await enhancer.enhance("<bad>", "UNSPSC")
            self.assertEqual(outcome.text, "<bad>")
            self.assertIs(outcome.status, EnhancementStatus.FAILED)
