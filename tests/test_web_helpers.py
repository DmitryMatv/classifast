import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.classifier_config import CLASSIFIER_CONFIG
from app.usage_tracker import UsageStatus
from app.web import router, slugify

BASE_DIR = Path(__file__).resolve().parents[1]


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.mount(
        "/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static"
    )
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

    def test_preserves_colons_and_semicolons(self) -> None:
        self.assertEqual(
            slugify("pump: 10 bar; stainless steel"),
            "pump:_10_bar;_stainless_steel",
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
        cls.classifier_type, config = next(
            (name, config)
            for name, config in CLASSIFIER_CONFIG.items()
            if len(config["versions"]) >= 2
        )
        versions = list(config["versions"])
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
        }
        params.update(extra_params)
        if "top_k" in params and params["top_k"] is None:
            params.pop("top_k")
        if "version" in params and params["version"] is None:
            params.pop("version")
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
            f"/{self.classifier_type}/industrial_pump",
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
            f"/{self.classifier_type}/industrial_pump?"
            f"{urlencode({'version': self.non_default_version})}",
        )

    @patch("app.web.perform_classification")
    async def test_non_default_top_k_is_appended_to_hx_push_url(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(
            push_url="true",
            track_usage="false",
            top_k="30",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("HX-Push-Url"),
            f"/{self.classifier_type}/industrial_pump?{urlencode({'top_k': 30})}",
        )

    @patch("app.web.perform_classification")
    async def test_non_default_version_and_top_k_are_appended_to_hx_push_url(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(
            push_url="true",
            track_usage="false",
            version=self.non_default_version,
            top_k="30",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("HX-Push-Url"),
            f"/{self.classifier_type}/industrial_pump?"
            + urlencode({"version": self.non_default_version, "top_k": 30}),
        )

    @patch("app.web.increment_usage", new_callable=AsyncMock)
    @patch("app.web.check_usage", new_callable=AsyncMock)
    @patch("app.web.perform_classification")
    async def test_paywall_response_uses_no_store_cache_headers(
        self,
        perform_classification_mock: Mock,
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
            f"/{self.classifier_type}/industrial_pump",
        )
        perform_classification_mock.assert_not_called()
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
            "public, max-age=86400, stale-while-revalidate=604800",
        )
        self.assertEqual(
            response.headers["Cloudflare-CDN-Cache-Control"],
            "max-age=604800, stale-while-revalidate=604800",
        )
        perform_classification_mock.assert_not_called()

    @patch("app.web.perform_classification")
    async def test_fragment_uses_default_top_k_when_omitted(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(top_k=None, track_usage="false")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(perform_classification_mock.call_args.kwargs["top_k"], 10)

    @patch("app.web.perform_classification")
    async def test_fragment_uses_default_version_when_omitted(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(version=None, track_usage="false")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            perform_classification_mock.call_args.kwargs["version"],
            self.default_version,
        )


class PageRouteDefaultTopKTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_test_app()

    async def _request(self, path: str) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get(path)

    async def test_unspsc_page_defaults_to_top_10(self) -> None:
        response = await self._request("/UNSPSC/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('option value="10" selected', response.text)
        self.assertIn('id="classifier-form"', response.text)

    async def test_naics_page_defaults_to_top_10(self) -> None:
        response = await self._request("/NAICS/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('option value="10" selected', response.text)
        self.assertIn('id="classifier-form"', response.text)

    async def test_unspsc_invalid_top_k_falls_back_to_10(self) -> None:
        response = await self._request("/UNSPSC/?top_k=999")

        self.assertEqual(response.status_code, 200)
        self.assertIn('option value="10" selected', response.text)
        self.assertIn('id="classifier-form"', response.text)

    async def test_naics_invalid_top_k_falls_back_to_10(self) -> None:
        response = await self._request("/NAICS/?top_k=999")

        self.assertEqual(response.status_code, 200)
        self.assertIn('option value="10" selected', response.text)
        self.assertIn('id="classifier-form"', response.text)


class BaseClassifierPageSSRTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_test_app()
        cls.unspsc_version = next(iter(CLASSIFIER_CONFIG["UNSPSC"]["versions"]))
        cls.naics_version = next(iter(CLASSIFIER_CONFIG["NAICS"]["versions"]))

    def _classification_result(self, original_id: str, class_name: str) -> dict:
        return {
            "results": [
                {
                    "score": 0.97,
                    "payload": {
                        "original_id": original_id,
                        "class_name": class_name,
                        "definition": f"{class_name} definition.",
                    },
                }
            ],
            "version_config": {
                "base_url": "",
                "tooltip": "",
            },
        }

    async def _request(self, path: str) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get(path)

    @patch("app.web.increment_usage", new_callable=AsyncMock)
    @patch("app.web.check_usage", new_callable=AsyncMock)
    @patch("app.web.perform_classification")
    async def test_unspsc_base_page_inlines_ssr_results(
        self,
        perform_classification_mock: Mock,
        check_usage_mock: AsyncMock,
        increment_usage_mock: AsyncMock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result(
            "43211503", "Laptop computers"
        )

        response = await self._request("/UNSPSC/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Laptop computers", response.text)
        self.assertNotIn("Loading...</p>", response.text)
        self.assertIn('data-autoload-enabled="false"', response.text)
        self.assertNotIn('id="initial-results-loader"', response.text)
        self.assertNotIn("Set-Cookie", response.headers)
        self.assertEqual(
            perform_classification_mock.call_args.kwargs["version"], self.unspsc_version
        )
        self.assertEqual(perform_classification_mock.call_args.kwargs["top_k"], 10)
        check_usage_mock.assert_not_awaited()
        increment_usage_mock.assert_not_awaited()

    @patch("app.web.perform_classification")
    async def test_naics_base_page_inlines_ssr_results(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result(
            "541511", "Custom computer programming services"
        )

        response = await self._request("/NAICS/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Custom computer programming services", response.text)
        self.assertIn('data-autoload-enabled="false"', response.text)
        self.assertNotIn('id="initial-results-loader"', response.text)
        self.assertEqual(
            perform_classification_mock.call_args.kwargs["version"], self.naics_version
        )
        self.assertEqual(perform_classification_mock.call_args.kwargs["top_k"], 10)

    @patch("app.web.perform_classification")
    async def test_base_page_primes_score_bar_animation_before_page_scripts(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result(
            "43211503", "Laptop computers"
        )

        response = await self._request("/UNSPSC/")

        self.assertEqual(response.status_code, 200)
        animation_bootstrap = (
            "<script>\n"
            '        document.documentElement.classList.add("js-score-animations");\n'
            "    </script>"
        )
        self.assertIn("/js/classifier.js", response.text)
        self.assertLess(
            response.text.index(animation_bootstrap),
            response.text.index("/js/classifier.js"),
        )

    @patch("app.web.perform_classification")
    async def test_base_page_normalizes_invalid_version_before_ssr(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result(
            "43211503", "Laptop computers"
        )

        response = await self._request("/UNSPSC/?version=missing-version")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            perform_classification_mock.call_args.kwargs["version"], self.unspsc_version
        )
        self.assertIn('data-autoload-enabled="false"', response.text)
        self.assertNotIn('id="initial-results-loader"', response.text)

    @patch("app.web.perform_classification")
    async def test_search_page_keeps_client_loaded_initial_results(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        response = await self._request("/UNSPSC/industrial_pump")

        self.assertEqual(response.status_code, 200)
        self.assertIn('data-autoload-enabled="true"', response.text)
        self.assertIn('data-initial-query-present="true"', response.text)
        self.assertIn('data-initial-track-usage="true"', response.text)
        self.assertNotIn('id="initial-results-loader"', response.text)
        self.assertNotIn("Laptop computers", response.text)
        perform_classification_mock.assert_not_called()

    @patch("app.web.increment_usage", new_callable=AsyncMock)
    @patch("app.web.check_usage", new_callable=AsyncMock)
    @patch("app.web.perform_classification", side_effect=RuntimeError("qdrant down"))
    async def test_base_page_ssr_failure_falls_back_to_loader(
        self,
        perform_classification_mock: Mock,
        check_usage_mock: AsyncMock,
        increment_usage_mock: AsyncMock,
    ) -> None:
        response = await self._request("/UNSPSC/")
        expected_example = CLASSIFIER_CONFIG["UNSPSC"]["example"].strip()

        self.assertEqual(response.status_code, 200)
        self.assertIn('data-autoload-enabled="true"', response.text)
        self.assertNotIn('id="initial-results-loader"', response.text)
        self.assertIn(expected_example, response.text)
        self.assertNotIn("Set-Cookie", response.headers)
        perform_classification_mock.assert_called_once()
        check_usage_mock.assert_not_awaited()
        increment_usage_mock.assert_not_awaited()


class UnspscFragmentDefaultTopKTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_test_app()
        cls.version = next(iter(CLASSIFIER_CONFIG["UNSPSC"]["versions"]))

    def _classification_result(self) -> dict:
        return {
            "results": [],
            "version_config": {
                "base_url": "",
                "tooltip": "",
            },
        }

    @patch("app.web.perform_classification")
    async def test_unspsc_fragment_uses_default_top_k_when_omitted(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()
        transport = httpx.ASGITransport(app=self.app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get(
                "/UNSPSC/fragment",
                params={
                    "product_description": "industrial pump",
                    "version": self.version,
                    "track_usage": "false",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(perform_classification_mock.call_args.kwargs["top_k"], 10)


if __name__ == "__main__":
    unittest.main()
