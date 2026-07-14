import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, Request

from app import api, web
from app.classifier_config import CLASSIFIER_CONFIG
from app.usage_tracker import UsageStatus


def build_classification_result(version_name: str) -> dict:
    return {
        "results": [
            {
                "score": 0.91,
                "payload": {
                    "original_id": "12345",
                    "class_name": "Test classification",
                },
            }
        ],
        "version_name": version_name,
        "version_config": {
            "base_url": "https://example.com/",
            "tooltip": "",
        },
    }


def build_request(
    test_app: FastAPI,
    path: str,
    *,
    headers: dict[str, str] | None = None,
) -> Request:
    raw_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": raw_headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "app": test_app,
    }
    return Request(scope)


class RapidApiQuotaBypassTests(unittest.IsolatedAsyncioTestCase):
    def build_app(self) -> FastAPI:
        test_app = FastAPI()
        test_app.state.embed_client = object()
        test_app.state.qdrant_client = object()
        test_app.state.collection_quantization_cache = {}
        test_app.state.zclient = None
        test_app.state.redis_client = object()
        return test_app

    async def test_verified_request_succeeds_without_quota_headers_or_usage_calls(self):
        rapid_app = self.build_app()
        request = build_request(
            rapid_app,
            "/api/v1/rapid/classify",
            headers={"X-RapidAPI-Proxy-Secret": "secret"},
        )

        with (
            patch.object(api, "RAPIDAPI_SECRET", "secret"),
            patch(
                "app.api.perform_classification",
                return_value=build_classification_result("2025"),
            ),
            patch("app.usage_tracker.check_usage", new=AsyncMock()) as check_usage,
            patch(
                "app.usage_tracker.increment_usage", new=AsyncMock()
            ) as increment_usage,
        ):
            self.assertTrue(api.verify_rapidapi_auth(request))
            response = await api.rapid_classify(
                request,
                query=" test widget ",
                standard=" unspsc ",
                top_k=1,
                version=None,
            )

        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["query"], "test widget")
        self.assertEqual(payload["standard"], "unspsc")
        self.assertNotIn("X-RateLimit-Remaining", response.headers)
        self.assertNotIn("X-RateLimit-Limit", response.headers)
        check_usage.assert_not_awaited()
        increment_usage.assert_not_awaited()


class RapidApiHealthTests(unittest.IsolatedAsyncioTestCase):
    def build_app(self) -> tuple[FastAPI, MagicMock, MagicMock]:
        test_app = FastAPI()
        embed_client = MagicMock()
        qdrant_client = MagicMock()
        test_app.state.embed_client = embed_client
        test_app.state.qdrant_client = qdrant_client
        return test_app, embed_client, qdrant_client

    async def test_ping_reports_embedding_available_when_client_exists_without_inference(
        self,
    ):
        rapid_app, embed_client, qdrant_client = self.build_app()
        request = build_request(rapid_app, "/api/v1/rapid/ping")

        response = await api.rapid_health_public(request)

        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["services"]["embedding"], "configured")
        self.assertEqual(payload["services"]["database"], "healthy")
        embed_client.feature_extraction.assert_not_called()
        qdrant_client.get_collections.assert_called_once_with()

    async def test_ping_returns_503_when_embedding_client_missing(self):
        rapid_app = FastAPI()
        qdrant_client = MagicMock()
        rapid_app.state.qdrant_client = qdrant_client
        request = build_request(rapid_app, "/api/v1/rapid/ping")

        response = await api.rapid_health_public(request)

        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(payload["services"]["embedding"], "unavailable")
        self.assertEqual(payload["services"]["database"], "healthy")

    async def test_ping_returns_503_when_database_unhealthy(self):
        rapid_app, _, qdrant_client = self.build_app()
        qdrant_client.get_collections.side_effect = RuntimeError("down")
        request = build_request(rapid_app, "/api/v1/rapid/ping")

        response = await api.rapid_health_public(request)

        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(payload["services"]["embedding"], "configured")
        self.assertEqual(payload["services"]["database"], "unhealthy")


class WebsiteQuotaRegressionTests(unittest.IsolatedAsyncioTestCase):
    def build_app(self) -> FastAPI:
        test_app = FastAPI()
        test_app.state.embed_client = object()
        test_app.state.qdrant_client = object()
        test_app.state.collection_quantization_cache = {}
        test_app.state.zclient = None
        test_app.state.redis_client = object()
        return test_app

    async def test_fragment_requests_still_use_quota_enforcement(self):
        classifier_type = "UNSPSC"
        version_name = next(iter(CLASSIFIER_CONFIG[classifier_type]["versions"]))
        website_app = self.build_app()
        request = build_request(website_app, f"/{classifier_type}/fragment")
        usage_status = UsageStatus(
            allowed=True,
            remaining=9,
            limit=10,
            is_authenticated=False,
            is_pro=False,
            tracking_id=None,
        )

        with (
            patch(
                "app.web.is_verified_google_search_crawler_request",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "app.web.check_usage", new=AsyncMock(return_value=usage_status)
            ) as check_usage,
            patch("app.web.increment_usage", new=AsyncMock()) as increment_usage,
            patch(
                "app.web.perform_classification",
                return_value=build_classification_result(version_name),
            ),
            patch("app.web.add_quota_headers") as add_quota_headers,
        ):

            def quota_header_side_effect(response, current_usage_status) -> None:
                response.headers["X-RateLimit-Remaining"] = str(
                    current_usage_status.remaining
                )
                response.headers["X-RateLimit-Limit"] = str(current_usage_status.limit)

            add_quota_headers.side_effect = quota_header_side_effect

            response = await web.get_classification_fragment(
                request,
                classifier_type=classifier_type,
                product_description="test widget",
                version=version_name,
                top_k=10,
                push_url=None,
                url_change=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Cache-Control"],
            "public, max-age=86400, stale-while-revalidate=604800",
        )
        self.assertEqual(
            response.headers["Cloudflare-CDN-Cache-Control"],
            "public, max-age=604800, stale-while-revalidate=604800",
        )
        self.assertNotIn("X-RateLimit-Remaining", response.headers)
        self.assertNotIn("X-RateLimit-Limit", response.headers)
        check_usage.assert_awaited_once()
        increment_usage.assert_awaited_once()
        add_quota_headers.assert_not_called()


if __name__ == "__main__":
    unittest.main()
