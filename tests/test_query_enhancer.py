import unittest

import httpx

from app.query_enhancer import QUERY_ENHANCEMENT_MODEL, QueryEnhancer


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

        self.assertEqual(result, "bolt. Threaded metal fastener")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/api/v1/chat/completions")
        self.assertEqual(requests[0].read().decode().count(QUERY_ENHANCEMENT_MODEL), 1)

    async def test_failures_and_invalid_output_keep_original(self) -> None:
        for response in (
            httpx.Response(503),
            httpx.Response(200, json={"choices": []}),
            httpx.Response(
                200, json={"choices": [{"message": {"content": "x" * 241}}]}
            ),
            httpx.Response(200, json={"choices": [{"message": {"content": " "}}]}),
        ):
            with self.subTest(response=response):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(lambda request: response)
                ) as client:
                    enhancer = QueryEnhancer("test-key", client=client)
                    self.assertEqual(await enhancer.enhance("bolt", "UNSPSC"), "bolt")

    async def test_code_shaped_query_makes_no_network_request(self) -> None:
        def unexpected_request(request: httpx.Request) -> httpx.Response:
            raise AssertionError("Code lookup should skip the LLM")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(unexpected_request)
        ) as client:
            enhancer = QueryEnhancer("test-key", client=client)
            self.assertEqual(await enhancer.enhance("8471", "HS"), "8471")
            self.assertEqual(await enhancer.enhance("AB-12345", "UNSPSC"), "AB-12345")

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
                await enhancer.enhance("bolt", "UNSPSC"),
                "bolt. Threaded fastener",
            )

    async def test_invalid_original_never_reaches_model(self) -> None:
        def unexpected_request(request: httpx.Request) -> httpx.Response:
            raise AssertionError("Invalid input should not call the LLM")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(unexpected_request)
        ) as client:
            enhancer = QueryEnhancer("test-key", client=client)
            self.assertEqual(await enhancer.enhance("<bad>", "UNSPSC"), "<bad>")
