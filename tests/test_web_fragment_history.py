import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import FastAPI

from app.classifier_config import CLASSIFIER_CONFIG
from app.usage_tracker import UsageStatus
from app.web import router
from tests.helpers import build_classification_service


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.classification_service = build_classification_service()
    app.state.redis_client = object()
    return app


class FragmentHistoryContractTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_test_app()
        classifier_type, config = next(
            (
                (name, cfg)
                for name, cfg in CLASSIFIER_CONFIG.items()
                if cfg.get("versions")
            )
        )
        cls.classifier_type = classifier_type
        cls.version = next(iter(config["versions"]))

    def _classification_result(self) -> dict:
        return {
            "results": [
                {
                    "score": 0.91,
                    "payload": {
                        "original_id": "12345678",
                        "class_name": "Trash removal services",
                        "definition": "Collection and disposal of waste.",
                    },
                }
            ],
            "version_config": {
                "base_url": "",
                "tooltip": "",
            },
            "version_name": self.version,
            "collection_name": "test_collection",
            "query": "trash removal",
        }

    async def _request_fragment(self, **extra_params):
        params = {
            "product_description": "trash removal",
            "version": self.version,
            "top_k": 10,
        }
        params.update(extra_params)
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get(f"/{self.classifier_type}/fragment", params=params)

    def _allowed_usage(self) -> UsageStatus:
        return UsageStatus(
            allowed=True,
            remaining=9,
            limit=10,
            is_authenticated=False,
            is_pro=False,
            tracking_id="track-123",
        )

    @patch("app.web.is_verified_google_search_crawler_request", new_callable=AsyncMock)
    @patch("app.web.reserve_usage", new_callable=AsyncMock)
    @patch("app.classification_service.perform_classification")
    async def test_push_url_true_returns_one_cacheable_metered_response(
        self,
        perform_classification_mock: Mock,
        reserve_usage_mock: AsyncMock,
        crawler_check_mock: AsyncMock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()
        reserve_usage_mock.return_value = self._allowed_usage()
        crawler_check_mock.return_value = False

        response = await self._request_fragment(push_url="true")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Cache-Control"],
            "public, max-age=86400, stale-while-revalidate=604800",
        )
        self.assertEqual(
            response.headers["Cloudflare-CDN-Cache-Control"],
            "public, max-age=604800, stale-while-revalidate=604800",
        )
        self.assertEqual(
            response.headers["HX-Push-Url"],
            f"/{self.classifier_type}/trash_removal/",
        )
        self.assertNotIn("X-RateLimit-Remaining", response.headers)
        self.assertNotIn("X-RateLimit-Limit", response.headers)
        reserve_usage_mock.assert_awaited_once()
        perform_classification_mock.assert_called_once()

    @patch("app.web.is_verified_google_search_crawler_request", new_callable=AsyncMock)
    @patch("app.web.reserve_usage", new_callable=AsyncMock)
    @patch("app.classification_service.perform_classification")
    async def test_push_url_false_suppresses_history_but_still_tracks(
        self,
        perform_classification_mock: Mock,
        reserve_usage_mock: AsyncMock,
        crawler_check_mock: AsyncMock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()
        reserve_usage_mock.return_value = self._allowed_usage()
        crawler_check_mock.return_value = False

        response = await self._request_fragment(push_url="false")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("HX-Push-Url", response.headers)
        reserve_usage_mock.assert_awaited_once()
        perform_classification_mock.assert_called_once()

    @patch("app.web.is_verified_google_search_crawler_request", new_callable=AsyncMock)
    @patch("app.web.reserve_usage", new_callable=AsyncMock)
    @patch("app.classification_service.perform_classification")
    async def test_legacy_track_usage_false_no_longer_bypasses_quota(
        self,
        perform_classification_mock: Mock,
        reserve_usage_mock: AsyncMock,
        crawler_check_mock: AsyncMock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()
        reserve_usage_mock.return_value = self._allowed_usage()
        crawler_check_mock.return_value = False

        response = await self._request_fragment(push_url="false", track_usage="false")

        self.assertEqual(response.status_code, 200)
        reserve_usage_mock.assert_awaited_once()
        perform_classification_mock.assert_called_once()

    @patch("app.web.is_verified_google_search_crawler_request", new_callable=AsyncMock)
    @patch("app.web.reserve_usage", new_callable=AsyncMock)
    @patch("app.classification_service.perform_classification")
    async def test_legacy_url_change_false_only_suppresses_history(
        self,
        perform_classification_mock: Mock,
        reserve_usage_mock: AsyncMock,
        crawler_check_mock: AsyncMock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()
        reserve_usage_mock.return_value = self._allowed_usage()
        crawler_check_mock.return_value = False

        response = await self._request_fragment(url_change="false")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("HX-Push-Url", response.headers)
        reserve_usage_mock.assert_awaited_once()
        perform_classification_mock.assert_called_once()

    @patch("app.web.is_verified_google_search_crawler_request", new_callable=AsyncMock)
    @patch("app.web.reserve_usage", new_callable=AsyncMock)
    @patch("app.classification_service.perform_classification")
    async def test_explicit_push_url_wins_over_legacy_url_change(
        self,
        perform_classification_mock: Mock,
        reserve_usage_mock: AsyncMock,
        crawler_check_mock: AsyncMock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()
        reserve_usage_mock.return_value = self._allowed_usage()
        crawler_check_mock.return_value = False

        response = await self._request_fragment(
            push_url="true",
            url_change="false",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("HX-Push-Url", response.headers)
        reserve_usage_mock.assert_awaited_once()
        perform_classification_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
