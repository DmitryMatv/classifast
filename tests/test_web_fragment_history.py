import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import FastAPI

from app.classifier_config import CLASSIFIER_CONFIG
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

    @patch("app.web.increment_usage", new_callable=AsyncMock)
    @patch("app.web.check_usage", new_callable=AsyncMock)
    @patch("app.web.perform_classification")
    async def test_push_url_true_returns_hx_push_url(
        self,
        perform_classification_mock: Mock,
        check_usage_mock: AsyncMock,
        increment_usage_mock: AsyncMock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()
        check_usage_mock.return_value = UsageStatus(
            allowed=True,
            remaining=9,
            limit=10,
            is_authenticated=False,
            is_pro=False,
        )

        response = await self._request_fragment(push_url="true", track_usage="true")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("HX-Push-Url"),
            f"/{self.classifier_type}/trash_removal",
        )
        check_usage_mock.assert_awaited_once()
        increment_usage_mock.assert_awaited_once()

    @patch("app.web.increment_usage", new_callable=AsyncMock)
    @patch("app.web.check_usage", new_callable=AsyncMock)
    @patch("app.web.perform_classification")
    async def test_push_url_false_omits_hx_push_url(
        self,
        perform_classification_mock: Mock,
        check_usage_mock: AsyncMock,
        increment_usage_mock: AsyncMock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()
        check_usage_mock.return_value = UsageStatus(
            allowed=True,
            remaining=9,
            limit=10,
            is_authenticated=False,
            is_pro=False,
        )

        response = await self._request_fragment(push_url="false", track_usage="true")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("HX-Push-Url", response.headers)
        check_usage_mock.assert_awaited_once()
        increment_usage_mock.assert_awaited_once()

    @patch("app.web.increment_usage", new_callable=AsyncMock)
    @patch("app.web.check_usage", new_callable=AsyncMock)
    @patch("app.web.perform_classification")
    async def test_track_usage_false_bypasses_quota_logic(
        self,
        perform_classification_mock: Mock,
        check_usage_mock: AsyncMock,
        increment_usage_mock: AsyncMock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(push_url="false", track_usage="false")

        self.assertEqual(response.status_code, 200)
        check_usage_mock.assert_not_awaited()
        increment_usage_mock.assert_not_awaited()

    @patch("app.web.increment_usage", new_callable=AsyncMock)
    @patch("app.web.check_usage", new_callable=AsyncMock)
    @patch("app.web.perform_classification")
    async def test_track_usage_true_still_tracks_when_push_url_false(
        self,
        perform_classification_mock: Mock,
        check_usage_mock: AsyncMock,
        increment_usage_mock: AsyncMock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()
        check_usage_mock.return_value = UsageStatus(
            allowed=True,
            remaining=4,
            limit=10,
            is_authenticated=False,
            is_pro=False,
        )

        response = await self._request_fragment(push_url="false", track_usage="true")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("HX-Push-Url", response.headers)
        check_usage_mock.assert_awaited_once()
        increment_usage_mock.assert_awaited_once()

    @patch("app.web.increment_usage", new_callable=AsyncMock)
    @patch("app.web.check_usage", new_callable=AsyncMock)
    @patch("app.web.perform_classification")
    async def test_legacy_url_change_false_preserves_old_no_tracking_behavior(
        self,
        perform_classification_mock: Mock,
        check_usage_mock: AsyncMock,
        increment_usage_mock: AsyncMock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(url_change="false")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("HX-Push-Url", response.headers)
        check_usage_mock.assert_not_awaited()
        increment_usage_mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
