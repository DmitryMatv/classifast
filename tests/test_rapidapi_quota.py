import json
import unittest
from unittest.mock import AsyncMock, patch

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

        with patch.object(api, "RAPIDAPI_SECRET", "secret"), patch(
            "app.api.perform_classification",
            return_value=build_classification_result("2025"),
        ), patch("app.usage_tracker.check_usage", new=AsyncMock()) as check_usage, patch(
            "app.usage_tracker.increment_usage", new=AsyncMock()
        ) as increment_usage:
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

        with patch("app.web.check_usage", new=AsyncMock(return_value=usage_status)) as check_usage, patch(
            "app.web.increment_usage", new=AsyncMock()
        ) as increment_usage, patch(
            "app.web.perform_classification",
            return_value=build_classification_result(version_name),
        ), patch("app.web.add_quota_headers") as add_quota_headers:
            def quota_header_side_effect(response, current_usage_status):
                response.headers["X-RateLimit-Remaining"] = str(current_usage_status.remaining)
                response.headers["X-RateLimit-Limit"] = str(current_usage_status.limit)

            add_quota_headers.side_effect = quota_header_side_effect

            response = await web.get_classification_fragment(
                request,
                classifier_type=classifier_type,
                product_description="test widget",
                version=version_name,
                url_change=True,
            )

        self.assertEqual(response.status_code, 200)
        check_usage.assert_awaited_once()
        increment_usage.assert_awaited_once()
        self.assertEqual(response.headers["X-RateLimit-Remaining"], "9")
        self.assertEqual(response.headers["X-RateLimit-Limit"], "10")


if __name__ == "__main__":
    unittest.main()
