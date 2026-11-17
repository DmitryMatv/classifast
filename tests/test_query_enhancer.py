import json
import unittest

import httpx

from app.classifier_config import CLASSIFIER_CONFIG
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
                    "choices": [{"message": {"content": "  Threaded metal fastener  "}}]
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            enhancer = QueryEnhancer("test-key", client=client)
            result = await enhancer.enhance("bolt", "UNSPSC")

        self.assertEqual(result.text, "bolt\n\nThreaded metal fastener")
        self.assertIs(result.status, EnhancementStatus.APPLIED)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/api/v1/chat/completions")
        body = json.loads(requests[0].read())
        self.assertEqual(body["model"], QUERY_ENHANCEMENT_MODEL)
        self.assertEqual(body["reasoning"], {"effort": "none"})
        self.assertNotIn("temperature", body)
        system, user = body["messages"]
        self.assertEqual(system["role"], "system")
        self.assertIn("UNSPSC classification", system["content"])
        self.assertIn("a purchased product or service", system["content"])
        self.assertEqual(user, {"role": "user", "content": "bolt"})

    async def test_system_prompt_uses_the_classifier_subject(self) -> None:
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "Pet grooming"}}]}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            enhancer = QueryEnhancer("test-key", client=client)
            await enhancer.enhance("dog grooming", "naics")

        system = json.loads(requests[0].read())["messages"][0]["content"]
        self.assertIn("NAICS classification", system)
        self.assertIn("a business or the economic activity it performs", system)

    def test_every_classifier_has_a_query_subject(self) -> None:
        for classifier_type, config in CLASSIFIER_CONFIG.items():
            with self.subTest(classifier_type=classifier_type):
                self.assertTrue(config["query_subject"].strip())

    async def test_no_description_replies_are_skipped(self) -> None:
        for reply in ("NONE", "None.", '""', "''"):
            with self.subTest(reply=reply):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(
                            200, json={"choices": [{"message": {"content": reply}}]}
                        )
                    )
                ) as client:
                    enhancer = QueryEnhancer("test-key", client=client)
                    outcome = await enhancer.enhance("bolt", "UNSPSC")
                self.assertEqual(outcome.text, "bolt")
                self.assertIs(outcome.status, EnhancementStatus.SKIPPED)

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
                json={
                    "choices": [{"message": {"content": "A threaded metal fastener"}}]
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            enhancer = QueryEnhancer("test-key", client=client)
            for query in ("bolt 123 mm", "M8 bolt"):
                self.assertIs(
                    (await enhancer.enhance(query, "UNSPSC")).status,
                    EnhancementStatus.APPLIED,
                )
        self.assertEqual(len(requests), 2)

    async def test_short_alphanumeric_terms_still_reach_model(self) -> None:
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "A common product term"}}]},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            enhancer = QueryEnhancer("test-key", client=client)
            for query in ("3D", "B2B"):
                outcome = await enhancer.enhance(query, "UNSPSC")
                self.assertIs(outcome.status, EnhancementStatus.APPLIED)
        self.assertEqual(len(requests), 2)

    async def test_model_text_is_sanitized_before_semantic_search(self) -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": "Threaded fastener …"}}]},
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
