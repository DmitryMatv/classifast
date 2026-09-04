import re
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI

from app.classifier_config import CLASSIFIER_CONFIG
from app.dependencies import group_original_id_tokens
from app.usage_tracker import UsageStatus
from app.web import router
from tests.helpers import build_classification_service


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.classification_service = build_classification_service()
    app.state.redis_client = object()
    return app


class GroupOriginalIdTokensTests(unittest.TestCase):
    @staticmethod
    def gap_indexes(tokens: list[dict[str, object]]) -> list[int]:
        return [index for index, token in enumerate(tokens) if token["gap_after"]]

    def test_even_numeric_string_keeps_two_digit_groups(self) -> None:
        tokens = group_original_id_tokens("12345678")

        self.assertEqual(len(tokens), 8)
        self.assertEqual(
            [token["char"] for token in tokens],
            list("12345678"),
        )
        self.assertEqual(
            [token["char"] for token in tokens if token["gap_after"]],
            ["2", "4", "6"],
        )
        self.assertEqual(self.gap_indexes(tokens), [1, 3, 5])

    def test_odd_numeric_string_groups_from_the_right(self) -> None:
        tokens = group_original_id_tokens("12345")

        self.assertEqual(self.gap_indexes(tokens), [0, 2])

    def test_six_digit_string_keeps_two_digit_groups(self) -> None:
        tokens = group_original_id_tokens("123456")

        self.assertEqual(self.gap_indexes(tokens), [1, 3])

    def test_letter_prefix_is_separate_and_odd_digit_run_groups_from_right(
        self,
    ) -> None:
        tokens = group_original_id_tokens("EC00123")

        self.assertEqual(self.gap_indexes(tokens), [1, 2, 4])

    def test_hyphen_does_not_create_a_gap(self) -> None:
        tokens = group_original_id_tokens("12-34")

        self.assertEqual(self.gap_indexes(tokens), [])

    def test_punctuation_separated_runs_are_grouped_independently(self) -> None:
        tokens = group_original_id_tokens("123.4567")

        self.assertEqual(self.gap_indexes(tokens), [0, 5])

    def test_punctuation_breaks_digit_runs(self) -> None:
        tokens = group_original_id_tokens("12.34")

        self.assertEqual(
            [token["char"] for token in tokens if token["gap_after"]],
            [],
        )

    def test_empty_and_none_are_handled_without_errors(self) -> None:
        self.assertEqual(group_original_id_tokens(""), [])
        self.assertEqual(group_original_id_tokens(None), [])


class ResultsOriginalIdFormattingFragmentTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_test_app()

    async def _request_fragment(
        self,
        classifier_type: str,
        version: str,
        original_id: str,
        base_url: str = "",
        append_code_to_url: bool = True,
        code_url_suffix: str = "",
    ) -> httpx.Response:
        result = {
            "results": [
                {
                    "score": 0.91,
                    "payload": {
                        "original_id": original_id,
                        "class_name": "Sample class",
                        "definition": "Sample definition.",
                    },
                }
            ],
            "version_config": {
                "base_url": base_url,
                "append_code_to_url": append_code_to_url,
                "code_url_suffix": code_url_suffix,
                "tooltip": "",
            },
            "version_name": version,
            "collection_name": "test_collection",
            "query": "sample query",
        }

        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            with (
                patch(
                    "app.classification_service.perform_classification",
                    return_value=result,
                ),
                patch(
                    "app.web.reserve_usage",
                    new=AsyncMock(
                        return_value=UsageStatus(
                            allowed=True,
                            remaining=9,
                            limit=10,
                            is_authenticated=False,
                            is_pro=False,
                        )
                    ),
                ),
            ):
                return await client.get(
                    f"/{classifier_type}/fragment",
                    params={
                        "product_description": "sample query",
                        "version": version,
                        "top_k": 10,
                        "push_url": "false",
                    },
                    follow_redirects=True,
                )

    async def test_unspsc_fragment_renders_three_spacing_markers(self) -> None:
        version = next(iter(CLASSIFIER_CONFIG["UNSPSC"]["versions"]))
        response = await self._request_fragment("UNSPSC", version, "12345678")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.history, [])
        self.assertEqual(response.text.count("code-spacer-halves"), 3)

    async def test_odd_code_fragment_renders_right_aligned_groups(self) -> None:
        version = next(iter(CLASSIFIER_CONFIG["NAICS"]["versions"]))
        response = await self._request_fragment("NAICS", version, "12345")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text.count("code-spacer-halves"), 2)
        code_start = response.text.index(
            '<span class="text-gray-600 text-2xl font-bold font-mono">'
        )
        code_end = response.text.index(
            '<span class="ml-auto text-gray-600">',
            code_start,
        )
        code_html = response.text[code_start:code_end]
        events = [
            match.group(1) or "|"
            for match in re.finditer(
                r">([0-9])</span>|code-spacer-halves",
                code_html,
            )
        ]
        self.assertEqual(events, ["1", "|", "2", "3", "|", "4", "5"])

    async def test_cpv_code_renders_with_spacing_markers(self) -> None:
        version = next(iter(CLASSIFIER_CONFIG["CPV"]["versions"]))
        response = await self._request_fragment("CPV", version, "72212910-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text.count("code-spacer-halves"), 3)
        self.assertIn(
            'data-copy-original-id="72212910-1"',
            response.text,
        )

    async def test_emdn_separates_letter_prefix_and_groups_numeric_hierarchy(
        self,
    ) -> None:
        version = next(iter(CLASSIFIER_CONFIG["EMDN"]["versions"]))
        response = await self._request_fragment("EMDN", version, "A0101010101")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text.count("code-spacer-halves"), 5)
        code_start = response.text.index(
            '<span class="text-gray-600 text-2xl font-bold font-mono">'
        )
        code_end = response.text.index(
            '<span class="ml-auto text-gray-600">',
            code_start,
        )
        code_html = response.text[code_start:code_end]
        events = [
            match.group(1) or "|"
            for match in re.finditer(
                r">([A-Z0-9])</span>|code-spacer-halves",
                code_html,
            )
        ]
        self.assertEqual(
            events,
            [
                "A",
                "|",
                "0",
                "1",
                "|",
                "0",
                "1",
                "|",
                "0",
                "1",
                "|",
                "0",
                "1",
                "|",
                "0",
                "1",
            ],
        )

    async def test_copy_button_keeps_raw_original_id(self) -> None:
        version = next(iter(CLASSIFIER_CONFIG["UNSPSC"]["versions"]))
        response = await self._request_fragment("UNSPSC", version, "12345678")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.history, [])
        self.assertIn(
            'data-copy-original-id="12345678"',
            response.text,
        )

    async def test_gpc_code_link_opens_browser_homepage_without_code(self) -> None:
        version = next(iter(CLASSIFIER_CONFIG["GPC"]["versions"]))
        response = await self._request_fragment(
            "GPC",
            version,
            "10000123",
            base_url="https://gpc-browser.gs1.org/",
            append_code_to_url=False,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn('href="https://gpc-browser.gs1.org/"', response.text)
        self.assertNotIn("https://gpc-browser.gs1.org/10000123", response.text)

    async def test_emdn_code_link_appends_title_fragment_after_code(self) -> None:
        version = next(iter(CLASSIFIER_CONFIG["EMDN"]["versions"]))
        response = await self._request_fragment(
            "EMDN",
            version,
            "F02020201",
            base_url="https://webgate.ec.europa.eu/dyna2/emdn/",
            code_url_suffix="#title",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            'href="https://webgate.ec.europa.eu/dyna2/emdn/F02020201#title"',
            response.text,
        )


if __name__ == "__main__":
    unittest.main()
