import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.classifier_config import CLASSIFIER_CONFIG
from app.web import router

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


class ClassifierPageSeoTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_test_app()
        cls.classifier_type = "UNSPSC"
        cls.primary_version_label = next(iter(CLASSIFIER_CONFIG["UNSPSC"]["versions"]))

    def _classification_result(self) -> dict:
        return {
            "results": [
                {
                    "score": 0.97,
                    "payload": {
                        "original_id": "43211503",
                        "class_name": "Laptop computers",
                        "definition": "Portable laptop computers for business use.",
                    },
                }
            ],
            "version_config": {
                "base_url": "https://example.com/code/",
                "tooltip": "Mock tooltip",
            },
        }

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    def _assert_landing_page_fallback_shell(self, response: httpx.Response) -> None:
        example_query = (
            CLASSIFIER_CONFIG[self.classifier_type]["example"]
            .replace("Example:", "")
            .strip()
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Robots-Tag"], "index, follow")
        self.assertIn('<meta name="robots" content="index, follow">', response.text)
        self.assertIn('data-initial-results-loader="true"', response.text)
        self.assertIn(example_query, response.text)
        self.assertNotIn("Laptop computers", response.text)
        self.assertNotIn("Backend services not available", response.text)
        self.assertNotIn("boom", response.text)

    def _assert_query_param_shell(self, response: httpx.Response, robots: str) -> None:
        example_query = (
            CLASSIFIER_CONFIG[self.classifier_type]["example"]
            .replace("Example:", "")
            .strip()
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Robots-Tag"], robots)
        self.assertIn(
            f'<meta name="robots" content="{robots}">',
            response.text,
        )
        self.assertIn(
            '<link rel="canonical" href="https://classifast.com/UNSPSC/">',
            response.text,
        )
        self.assertIn('data-initial-results-loader="true"', response.text)
        self.assertIn(example_query, response.text)
        self.assertNotIn("Laptop computers", response.text)

    @patch("app.web.perform_classification")
    async def test_base_landing_page_is_indexable_and_server_renders_results(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request("GET", f"/{self.classifier_type}/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Robots-Tag"], "index, follow")
        self.assertIn('<meta name="robots" content="index, follow">', response.text)
        self.assertIn("Laptop computers", response.text)
        self.assertNotIn("Loading...", response.text)
        self.assertNotIn('data-initial-results-loader="true"', response.text)

    async def test_generated_search_page_is_noindexed_but_keeps_search_flow(
        self,
    ) -> None:
        response = await self._request(
            "GET", f"/{self.classifier_type}/laptop-computer/"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Robots-Tag"], "noindex, follow")
        self.assertIn('<meta name="robots" content="noindex, follow">', response.text)
        self.assertIn(
            '<link rel="canonical" href="https://classifast.com/UNSPSC/laptop-computer/">',
            response.text,
        )
        self.assertIn('data-initial-results-loader="true"', response.text)
        self.assertIn("laptop-computer", response.text)

    async def test_variant_query_param_page_is_noindexed_and_keeps_base_canonical(
        self,
    ) -> None:
        response = await self._request(
            "GET",
            f"/{self.classifier_type}/",
            params={"version": self.primary_version_label},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Robots-Tag"], "noindex, follow")
        self.assertIn('<meta name="robots" content="noindex, follow">', response.text)
        self.assertIn(
            '<link rel="canonical" href="https://classifast.com/UNSPSC/">',
            response.text,
        )

    @patch("app.web.perform_classification")
    async def test_tracking_param_landing_page_skips_ssr_but_stays_indexable(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        response = await self._request(
            "GET",
            f"/{self.classifier_type}/",
            params={"utm_source": "google"},
        )

        self._assert_query_param_shell(response, "index, follow")
        perform_classification_mock.assert_not_called()

    @patch("app.web.verify_checkout_token", new_callable=AsyncMock)
    @patch("app.web.perform_classification")
    async def test_checkout_return_params_skip_ssr_but_still_verify_checkout_token(
        self,
        perform_classification_mock: Mock,
        verify_checkout_token_mock: AsyncMock,
    ) -> None:
        response = await self._request(
            "GET",
            f"/{self.classifier_type}/",
            params={"checkout": "success", "checkout_token": "test-token"},
        )

        self._assert_query_param_shell(response, "index, follow")
        perform_classification_mock.assert_not_called()
        verify_checkout_token_mock.assert_awaited_once()

    @patch("app.web.perform_classification")
    async def test_top_k_variant_page_is_noindexed_and_skips_ssr(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        response = await self._request(
            "GET",
            f"/{self.classifier_type}/",
            params={"top_k": 30},
        )

        self._assert_query_param_shell(response, "noindex, follow")
        perform_classification_mock.assert_not_called()

    @patch("app.web.perform_classification")
    async def test_version_variant_page_is_noindexed_and_skips_ssr(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        response = await self._request(
            "GET",
            f"/{self.classifier_type}/",
            params={"version": self.primary_version_label},
        )

        self._assert_query_param_shell(response, "noindex, follow")
        perform_classification_mock.assert_not_called()

    @patch("app.web.increment_usage", new_callable=AsyncMock)
    @patch("app.web.check_usage", new_callable=AsyncMock)
    @patch("app.web.perform_classification")
    async def test_landing_page_ssr_does_not_track_usage(
        self,
        perform_classification_mock: Mock,
        check_usage_mock: AsyncMock,
        increment_usage_mock: AsyncMock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request("GET", f"/{self.classifier_type}/")

        self.assertEqual(response.status_code, 200)
        check_usage_mock.assert_not_awaited()
        increment_usage_mock.assert_not_awaited()

    @patch("app.web.perform_classification")
    async def test_landing_page_falls_back_to_shell_when_ssr_raises_http_exception(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.side_effect = HTTPException(
            status_code=503, detail="Backend services not available"
        )

        response = await self._request("GET", f"/{self.classifier_type}/")

        self._assert_landing_page_fallback_shell(response)

    @patch("app.web.perform_classification")
    async def test_landing_page_falls_back_to_shell_when_ssr_raises_runtime_error(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.side_effect = RuntimeError("boom")

        response = await self._request("GET", f"/{self.classifier_type}/")

        self._assert_landing_page_fallback_shell(response)

    @patch("app.web.increment_usage", new_callable=AsyncMock)
    @patch("app.web.check_usage", new_callable=AsyncMock)
    @patch("app.web.perform_classification")
    async def test_landing_page_fallback_still_does_not_track_usage(
        self,
        perform_classification_mock: Mock,
        check_usage_mock: AsyncMock,
        increment_usage_mock: AsyncMock,
    ) -> None:
        perform_classification_mock.side_effect = HTTPException(
            status_code=503, detail="Backend services not available"
        )

        response = await self._request("GET", f"/{self.classifier_type}/")

        self._assert_landing_page_fallback_shell(response)
        check_usage_mock.assert_not_awaited()
        increment_usage_mock.assert_not_awaited()

    @patch("app.web.perform_classification")
    async def test_unspsc_version_text_comes_from_config(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request("GET", f"/{self.classifier_type}/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(self.primary_version_label, response.text)
        self.assertNotIn("August 2023", response.text)

    async def test_head_tracking_param_keeps_existing_indexable_policy(self) -> None:
        response = await self._request(
            "HEAD",
            f"/{self.classifier_type}/",
            params={"utm_source": "google"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Robots-Tag"], "index, follow")
        self.assertEqual(
            response.headers["Link"],
            '<https://classifast.com/UNSPSC/>; rel="canonical"',
        )


if __name__ == "__main__":
    unittest.main()
