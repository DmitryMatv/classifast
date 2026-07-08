import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI

from app.classifier_config import CLASSIFIER_CONFIG
from app.dependencies import group_original_id_tokens
from app.usage_tracker import UsageStatus
from app.web import router


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.embed_client = object()
    app.state.qdrant_client = object()
    app.state.collection_quantization_cache = {}
    app.state.zclient = None
    app.state.redis_client = object()
    return app


class GroupOriginalIdTokensTests(unittest.TestCase):
    def test_numeric_string_adds_gap_after_every_second_digit(self) -> None:
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

    def test_alphanumeric_string_groups_only_consecutive_digit_run(self) -> None:
        tokens = group_original_id_tokens("EC000123")

        self.assertEqual(
            [token["char"] for token in tokens if token["gap_after"]],
            ["0", "1"],
        )

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
                "base_url": "",
                "tooltip": "",
            },
        }

        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            with (
                patch("app.web.perform_classification", return_value=result),
                patch(
                    "app.web.check_usage",
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
                patch("app.web.increment_usage", new=AsyncMock()),
            ):
                return await client.get(
                    f"/{classifier_type}/fragment",
                    params={
                        "product_description": "sample query",
                        "version": version,
                        "top_k": 10,
                        "push_url": "false",
                        "track_usage": "true",
                    },
                    follow_redirects=True,
                )

    async def test_unspsc_fragment_renders_three_spacing_markers(self) -> None:
        version = next(iter(CLASSIFIER_CONFIG["UNSPSC"]["versions"]))
        response = await self._request_fragment("UNSPSC", version, "12345678")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.history[0].status_code, 303)
        self.assertEqual(response.text.count("code-spacer-halves"), 3)

    async def test_copy_button_keeps_raw_original_id(self) -> None:
        version = next(iter(CLASSIFIER_CONFIG["UNSPSC"]["versions"]))
        response = await self._request_fragment("UNSPSC", version, "12345678")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.history[0].status_code, 303)
        self.assertIn(
            "onclick=\"window.copyOriginalId('12345678', this)\"",
            response.text,
        )


if __name__ == "__main__":
    unittest.main()
