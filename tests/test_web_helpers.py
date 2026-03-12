import unittest
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI

from app.classifier_config import CLASSIFIER_CONFIG
from app.usage_tracker import UsageStatus
from app.web import router, slugify


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.embed_client = object()
    app.state.qdrant_client = object()
    app.state.collection_quantization_cache = {}
    app.state.zclient = None
    app.state.redis_client = object()
    return app


class SlugifyTests(unittest.TestCase):
    def test_preserves_supported_punctuation(self) -> None:
        self.assertEqual(
            slugify("Pump, valve (industrial)'s"), "Pump,_valve_(industrial)'s"
        )

    def test_normalizes_internal_whitespace(self) -> None:
        self.assertEqual(slugify("  multi \n spaced\tquery  "), "multi_spaced_query")

    def test_strips_unsupported_characters(self) -> None:
        self.assertEqual(
            slugify("gearbox <script>alert(1)</script>"),
            "gearbox_scriptalert(1)script",
        )

    def test_keeps_non_latin_characters(self) -> None:
        self.assertEqual(slugify("насос промышленный"), "насос_промышленный")


class FragmentRouteContractTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_test_app()
        cls.classifier_type = "NAICS"
        versions = list(CLASSIFIER_CONFIG["NAICS"]["versions"])
        cls.default_version = versions[0]
        cls.non_default_version = versions[1]

    def _classification_result(self) -> dict:
        return {
            "results": [
                {
                    "score": 0.93,
                    "payload": {
                        "original_id": "123456",
                        "class_name": "Pump manufacturing",
                        "definition": "Industrial pump manufacturing.",
                    },
                }
            ],
            "version_config": {
                "base_url": "",
                "tooltip": "",
            },
        }

    async def _request_fragment(self, **extra_params: str) -> httpx.Response:
        params = {
            "product_description": "industrial pump",
            "version": self.default_version,
            "top_k": 10,
        }
        params.update(extra_params)
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get(f"/{self.classifier_type}/fragment", params=params)

    @patch("app.web.perform_classification")
    async def test_default_version_omits_version_query_from_hx_push_url(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(push_url="true", track_usage="false")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("HX-Push-Url"),
            "/NAICS/industrial_pump",
        )

    @patch("app.web.perform_classification")
    async def test_non_default_version_is_appended_to_hx_push_url(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(
            push_url="true",
            track_usage="false",
            version=self.non_default_version,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("HX-Push-Url"),
            f"/NAICS/industrial_pump?{urlencode({'version': self.non_default_version})}",
        )

    @patch("app.web.increment_usage", new_callable=AsyncMock)
    @patch("app.web.check_usage", new_callable=AsyncMock)
    async def test_paywall_response_uses_no_store_cache_headers(
        self,
        check_usage_mock: AsyncMock,
        increment_usage_mock: AsyncMock,
    ) -> None:
        check_usage_mock.return_value = UsageStatus(
            allowed=False,
            remaining=0,
            limit=10,
            is_authenticated=False,
            is_pro=False,
            tracking_id="track-123",
        )

        response = await self._request_fragment(push_url="true", track_usage="true")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store, max-age=0")
        self.assertEqual(
            response.headers["Cloudflare-CDN-Cache-Control"],
            "no-store",
        )
        self.assertNotIn("stale-while-revalidate", response.headers["Cache-Control"])
        self.assertEqual(
            response.headers.get("HX-Push-Url"),
            "/NAICS/industrial_pump",
        )
        increment_usage_mock.assert_not_awaited()

    @patch("app.web.perform_classification")
    async def test_empty_query_fragment_returns_cacheable_empty_results(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        response = await self._request_fragment(
            product_description="   ",
            push_url="false",
            track_usage="false",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Cache-Control"],
            "public, max-age=60, stale-while-revalidate=600",
        )
        self.assertEqual(
            response.headers["Cloudflare-CDN-Cache-Control"],
            "max-age=14400, stale-while-revalidate=86400",
        )
        perform_classification_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
