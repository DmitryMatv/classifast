import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.classifier_config import CLASSIFIER_CONFIG
from app.classifier_page_delivery import (
    SITEMAP_QUERY_PATHS,
    build_classification_results_context,
    build_fragment_push_url,
    get_homepage_popular_lookup_links,
    get_popular_lookup_links,
    slugify,
)
from app.usage_tracker import QuotaUnavailableError, UsageStatus
from app.web import router
from tests.helpers import build_classification_service

BASE_DIR = Path(__file__).resolve().parents[1]


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.mount(
        "/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static"
    )
    app.include_router(router)
    app.state.classification_service = build_classification_service()
    app.state.redis_client = object()
    return app


class ClassificationResultsContextTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.classification_service.perform_classification")
    async def test_timing_measures_classification_call_duration(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = {
            "results": [],
            "version_config": {},
            "version_name": "v1",
            "collection_name": "test_collection",
            "query": "industrial pump",
        }
        request = MagicMock()
        request.app.state.classification_service = build_classification_service()

        with patch(
            "app.classifier_page_delivery.perf_counter", side_effect=[10.0, 13.5]
        ):
            context = await build_classification_results_context(
                request=request,
                classifier_type="UNSPSC",
                query="industrial pump",
                version="v1",
                top_k=10,
            )

        self.assertEqual(context["total_request_time"], 3.5)


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

    def test_fragment_push_url_uses_canonical_separator_for_hyphenated_query(
        self,
    ) -> None:
        self.assertEqual(
            build_fragment_push_url(
                "NAICS", "property-management", "2022", "2022", 10, 10
            ),
            "/NAICS/property_management/",
        )

    def test_fragment_push_url_preserves_legitimate_hyphen(self) -> None:
        self.assertEqual(
            build_fragment_push_url("NAICS", "semi-automatic", "2022", "2022", 10, 10),
            "/NAICS/semi-automatic/",
        )


class PopularLookupTests(unittest.TestCase):
    def test_curated_lookup_links_are_sitemap_backed_and_canonical(self) -> None:
        links = get_popular_lookup_links("UNSPSC")

        self.assertGreaterEqual(len(links), 15)
        self.assertEqual(links[0]["url"], "/UNSPSC/laptop_computer/")
        self.assertTrue(all(link["url"] in SITEMAP_QUERY_PATHS for link in links))
        self.assertTrue(all(link["url"].endswith("/") for link in links))

    def test_homepage_lookup_links_are_small_and_cross_standard(self) -> None:
        links = get_homepage_popular_lookup_links()

        self.assertEqual(len(links), 8)
        self.assertEqual(sum(link["classifier_type"] == "UNSPSC" for link in links), 6)
        self.assertTrue(all(link["url"] in SITEMAP_QUERY_PATHS for link in links))


class QueryPageSsrTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_test_app()

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
            "version_name": "v1",
            "collection_name": "test_collection",
            "query": "industrial pump",
        }

    async def _request(
        self,
        path: str,
        follow_redirects: bool = True,
        method: str = "GET",
    ) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            follow_redirects=follow_redirects,
        ) as client:
            return await client.request(method, path)

    async def test_noncanonical_search_path_redirects_to_trailing_slash(self) -> None:
        response = await self._request(
            "/UNSPSC/laptop_computer",
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/UNSPSC/laptop_computer/")

    async def test_repeated_search_path_trailing_slashes_redirect(self) -> None:
        response = await self._request(
            "/UNSPSC/laptop_computer//",
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/UNSPSC/laptop_computer/")

    async def test_repeated_classifier_path_trailing_slashes_redirect(self) -> None:
        response = await self._request(
            "/UNSPSC//",
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/UNSPSC/")

    async def test_search_redirect_preserves_query_parameters(self) -> None:
        response = await self._request(
            "/UNSPSC/laptop_computer?top_k=30&version=v1",
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 301)
        self.assertEqual(
            response.headers["location"],
            "/UNSPSC/laptop_computer/?top_k=30&version=v1",
        )

    async def test_lowercase_search_path_redirects_to_canonical_url(self) -> None:
        response = await self._request(
            "/unspsc/laptop_computer",
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/UNSPSC/laptop_computer/")

    async def test_head_noncanonical_search_path_redirects(self) -> None:
        response = await self._request(
            "/UNSPSC/laptop_computer",
            follow_redirects=False,
            method="HEAD",
        )

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/UNSPSC/laptop_computer/")

    async def test_percent_encoded_slug_serves_page_instead_of_redirect_loop(
        self,
    ) -> None:
        response = await self._request(
            "/UNSPSC/https%3A_classifast.com_UNSPSC/",
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["link"],
            '<https://classifast.com/UNSPSC/https%3A_classifast.com_UNSPSC/>; rel="canonical"',
        )

    async def test_encoded_punctuation_slug_serves_canonical_page(self) -> None:
        response = await self._request(
            "/UNSPSC/pump%3A_10_bar%3B_stainless_steel/",
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["link"],
            '<https://classifast.com/UNSPSC/pump%3A_10_bar%3B_stainless_steel/>; rel="canonical"',
        )

    async def test_head_percent_encoded_slug_serves_page(self) -> None:
        response = await self._request(
            "/UNSPSC/https%3A_classifast.com_UNSPSC/",
            follow_redirects=False,
            method="HEAD",
        )

        self.assertEqual(response.status_code, 200)

    async def test_hyphenated_search_path_redirects_to_underscore_canonical(
        self,
    ) -> None:
        response = await self._request(
            "/NAICS/property-management",
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/NAICS/property_management/")

    async def test_slash_terminated_hyphenated_path_redirects_to_underscore_canonical(
        self,
    ) -> None:
        response = await self._request(
            "/NAICS/property-management/",
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/NAICS/property_management/")

    async def test_legitimate_hyphenated_path_remains_canonical(self) -> None:
        response = await self._request(
            "/NAICS/semi-automatic/",
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["link"],
            '<https://classifast.com/NAICS/semi-automatic/>; rel="canonical"',
        )

    async def test_sitemap_slug_preserves_legitimate_hyphen(self) -> None:
        response = await self._request(
            "/HS/cotton_t-shirt/",
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["link"],
            '<https://classifast.com/HS/cotton_t-shirt/>; rel="canonical"',
        )

    @patch("app.classification_service.perform_classification")
    async def test_query_breadcrumb_uses_canonical_url(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request("/NAICS/property_management/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            '"item": "https://classifast.com/NAICS/property_management/"',
            response.text,
        )
        self.assertNotIn(
            '"item": "https://classifast.com/NAICS/property management"',
            response.text,
        )

    @patch("app.classification_service.perform_classification")
    async def test_sitemap_query_is_not_server_rendered_for_regular_visitors(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        response = await self._request("/UNSPSC/laptop_computer/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('data-autoload-enabled="true"', response.text)
        self.assertIn("Loading...", response.text)
        perform_classification_mock.assert_not_called()

    @patch(
        "app.web.is_verified_google_search_crawler_request",
        new=AsyncMock(return_value=True),
    )
    @patch("app.classification_service.perform_classification")
    async def test_sitemap_query_is_server_rendered_for_verified_google_crawler(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request("/UNSPSC/laptop_computer/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Pump manufacturing", response.text)
        self.assertIn('data-autoload-enabled="false"', response.text)
        self.assertNotIn("Popular UNSPSC code lookups", response.text)
        perform_classification_mock.assert_called_once()

    @patch("app.classification_service.perform_classification")
    async def test_classifier_base_page_renders_popular_lookup_links(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request("/UNSPSC/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Popular UNSPSC code lookups", response.text)
        self.assertIn('<a href="/UNSPSC/laptop_computer/"', response.text)

    async def test_homepage_renders_popular_lookup_links(self) -> None:
        response = await self._request("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Popular code lookups", response.text)
        self.assertIn('<a href="/UNSPSC/network_switch/"', response.text)

    @patch(
        "app.web.is_verified_google_search_crawler_request",
        new=AsyncMock(return_value=True),
    )
    @patch("app.classification_service.perform_classification")
    async def test_successful_empty_sitemap_query_is_not_left_loading(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = {
            "results": [],
            "version_config": {
                "base_url": "",
                "tooltip": "",
            },
            "version_name": "v1",
            "collection_name": "test_collection",
            "query": "industrial pump",
        }

        response = await self._request("/UNSPSC/laptop_computer/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("No matching classification results found.", response.text)
        self.assertNotIn("Loading...", response.text)
        self.assertIn('data-autoload-enabled="false"', response.text)
        perform_classification_mock.assert_called_once()

    @patch("app.classification_service.perform_classification")
    async def test_non_sitemap_query_keeps_htmx_autoload(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        response = await self._request("/UNSPSC/custom_internal_query/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('data-autoload-enabled="true"', response.text)
        self.assertIn("Loading...", response.text)
        perform_classification_mock.assert_not_called()

    @patch("app.classification_service.perform_classification")
    async def test_sitemap_query_with_parameters_keeps_htmx_autoload(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        response = await self._request("/UNSPSC/laptop_computer/?top_k=30")

        self.assertEqual(response.status_code, 200)
        self.assertIn('data-autoload-enabled="true"', response.text)
        perform_classification_mock.assert_not_called()

    @patch(
        "app.web.is_verified_google_search_crawler_request",
        new=AsyncMock(return_value=True),
    )
    @patch(
        "app.classification_service.perform_classification",
        side_effect=RuntimeError("classification unavailable"),
    )
    async def test_failed_ssr_falls_back_to_htmx_autoload(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        response = await self._request("/UNSPSC/laptop_computer/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('data-autoload-enabled="true"', response.text)
        self.assertEqual(
            response.headers["Cache-Control"],
            "no-store, max-age=0",
        )
        self.assertEqual(
            response.headers["Cloudflare-CDN-Cache-Control"],
            "no-store",
        )
        self.assertEqual(response.headers["X-Robots-Tag"], "noindex, nofollow")
        perform_classification_mock.assert_called_once()


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

    def setUp(self) -> None:
        usage_status = UsageStatus(
            allowed=True,
            remaining=9,
            limit=10,
            is_authenticated=False,
            is_pro=False,
            tracking_id="track-default",
        )
        self.default_reserve_usage = patch(
            "app.web.reserve_usage",
            new=AsyncMock(return_value=usage_status),
        ).start()
        self.default_crawler_check = patch(
            "app.web.is_verified_google_search_crawler_request",
            new=AsyncMock(return_value=False),
        ).start()
        self.addCleanup(patch.stopall)

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
            "version_name": "v1",
            "collection_name": "test_collection",
            "query": "industrial pump",
        }

    async def _request_fragment(
        self, follow_redirects: bool = False, **extra_params: str
    ) -> httpx.Response:
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
            return await client.get(
                f"/{self.classifier_type}/fragment",
                params=params,
                follow_redirects=follow_redirects,
            )

    @patch("app.classification_service.perform_classification")
    async def test_default_parameters_are_omitted_from_hx_push_url(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(push_url="true")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("HX-Push-Url"),
            f"/{self.classifier_type}/industrial_pump/",
        )

    @patch("app.classification_service.perform_classification")
    async def test_non_default_version_is_appended_to_hx_push_url(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(
            push_url="true",
            version=self.non_default_version,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("HX-Push-Url"),
            f"/{self.classifier_type}/industrial_pump/?"
            f"{urlencode({'version': self.non_default_version})}",
        )

    @patch("app.classification_service.perform_classification")
    async def test_non_default_top_k_is_appended_to_hx_push_url(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(
            push_url="true",
            top_k="30",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("HX-Push-Url"),
            f"/{self.classifier_type}/industrial_pump/?{urlencode({'top_k': 30})}",
        )

    @patch("app.classification_service.perform_classification")
    async def test_non_default_version_and_top_k_are_appended_to_hx_push_url(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(
            push_url="true",
            version=self.non_default_version,
            top_k="30",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("HX-Push-Url"),
            f"/{self.classifier_type}/industrial_pump/?"
            + urlencode({"version": self.non_default_version, "top_k": 30}),
        )

    @patch("app.web.reserve_usage", new_callable=AsyncMock)
    @patch("app.classification_service.perform_classification")
    async def test_paywall_response_uses_no_store_cache_headers(
        self,
        perform_classification_mock: Mock,
        reserve_usage_mock: AsyncMock,
    ) -> None:
        reserve_usage_mock.return_value = UsageStatus(
            allowed=False,
            remaining=0,
            limit=10,
            is_authenticated=False,
            is_pro=False,
            tracking_id="track-123",
        )

        response = await self._request_fragment(push_url="true")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store, max-age=0")
        self.assertEqual(
            response.headers["Cloudflare-CDN-Cache-Control"],
            "no-store",
        )
        self.assertNotIn("stale-while-revalidate", response.headers["Cache-Control"])
        self.assertEqual(
            response.headers.get("HX-Push-Url"),
            f"/{self.classifier_type}/industrial_pump/",
        )
        self.assertIn("Sign in to continue", response.text)
        self.assertIn('id="signin-button"', response.text)
        self.assertIn("Upgrade to Pro", response.text)
        self.assertNotIn("more free searches", response.text)
        perform_classification_mock.assert_not_called()
        reserve_usage_mock.assert_awaited_once()

    @patch("app.web.reserve_usage", new_callable=AsyncMock)
    @patch("app.classification_service.perform_classification")
    async def test_authenticated_paywall_hides_sign_in_button(
        self,
        perform_classification_mock: Mock,
        reserve_usage_mock: AsyncMock,
    ) -> None:
        reserve_usage_mock.return_value = UsageStatus(
            allowed=False,
            remaining=0,
            limit=30,
            is_authenticated=True,
            is_pro=False,
            tracking_id="user-123",
        )

        response = await self._request_fragment(push_url="true")

        self.assertEqual(response.status_code, 200)
        self.assertIn("You've used your 30 free trial searches", response.text)
        self.assertIn("Upgrade to Pro", response.text)
        self.assertNotIn('id="signin-button"', response.text)
        perform_classification_mock.assert_not_called()
        reserve_usage_mock.assert_awaited_once()

    @patch("app.web.reserve_usage", new_callable=AsyncMock)
    @patch("app.classification_service.perform_classification")
    async def test_allowed_cold_fragment_returns_cacheable_results(
        self,
        perform_classification_mock: Mock,
        reserve_usage_mock: AsyncMock,
    ) -> None:
        reserve_usage_mock.return_value = UsageStatus(
            allowed=True,
            remaining=9,
            limit=10,
            is_authenticated=False,
            is_pro=False,
            tracking_id="track-123",
        )
        perform_classification_mock.return_value = self._classification_result()

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
        self.assertEqual(response.headers["Cache-Tag"], "classification-results")
        self.assertNotIn("Location", response.headers)
        self.assertNotIn("X-RateLimit-Remaining", response.headers)
        self.assertNotIn("X-RateLimit-Limit", response.headers)
        perform_classification_mock.assert_called_once()
        self.assertEqual(
            perform_classification_mock.call_args.kwargs["classifier_type"],
            self.classifier_type,
        )
        self.assertEqual(
            perform_classification_mock.call_args.kwargs["query"], "industrial pump"
        )
        self.assertEqual(
            perform_classification_mock.call_args.kwargs["version"],
            self.default_version,
        )
        self.assertEqual(perform_classification_mock.call_args.kwargs["top_k"], 10)
        reserve_usage_mock.assert_awaited_once()

    @patch("app.classification_service.perform_classification")
    async def test_fragment_renders_queue_inclusive_timing(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        with patch(
            "app.classifier_page_delivery.perf_counter", side_effect=[10.0, 13.5]
        ):
            response = await self._request_fragment(push_url="true")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Finished in 3.50 seconds.", response.text)

    @patch("app.web.is_verified_google_search_crawler_request", new_callable=AsyncMock)
    @patch("app.web.reserve_usage", new_callable=AsyncMock)
    @patch("app.classification_service.perform_classification")
    async def test_verified_google_crawler_returns_cacheable_results(
        self,
        perform_classification_mock: Mock,
        reserve_usage_mock: AsyncMock,
        crawler_check_mock: AsyncMock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()
        crawler_check_mock.return_value = True

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
        self.assertNotIn("X-RateLimit-Remaining", response.headers)
        self.assertNotIn("X-RateLimit-Limit", response.headers)
        crawler_check_mock.assert_awaited_once()
        reserve_usage_mock.assert_not_awaited()
        perform_classification_mock.assert_called_once()

    @patch("app.web.reserve_usage", new_callable=AsyncMock)
    @patch("app.classification_service.perform_classification")
    async def test_quota_unavailable_during_reservation_returns_no_store_503(
        self,
        perform_classification_mock: Mock,
        reserve_usage_mock: AsyncMock,
    ) -> None:
        reserve_usage_mock.side_effect = QuotaUnavailableError(
            "Usage tracking is temporarily unavailable"
        )

        response = await self._request_fragment(push_url="true")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["Cache-Control"], "no-store, max-age=0")
        self.assertEqual(
            response.headers["Cloudflare-CDN-Cache-Control"],
            "no-store",
        )
        self.assertIn("Usage tracking is temporarily unavailable", response.text)
        perform_classification_mock.assert_not_called()
        reserve_usage_mock.assert_awaited_once()

    @patch("app.web.reserve_usage", new_callable=AsyncMock)
    @patch("app.classification_service.perform_classification")
    async def test_quota_unavailable_message_is_html_escaped(
        self,
        perform_classification_mock: Mock,
        reserve_usage_mock: AsyncMock,
    ) -> None:
        reserve_usage_mock.side_effect = QuotaUnavailableError(
            'Unavailable <script>alert("x")</script>'
        )

        response = await self._request_fragment(push_url="true")

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("<script>", response.text)
        self.assertIn("&lt;script&gt;", response.text)

    @patch(
        "app.classification_service.perform_classification",
        side_effect=RuntimeError("qdrant down"),
    )
    async def test_classification_error_returns_no_store_500(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        response = await self._request_fragment(push_url="true")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.headers["Cache-Control"], "no-store, max-age=0")
        self.assertEqual(
            response.headers["Cloudflare-CDN-Cache-Control"],
            "no-store",
        )
        perform_classification_mock.assert_called_once()

    @patch("app.classification_service.perform_classification")
    async def test_empty_query_fragment_returns_cacheable_empty_results(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        response = await self._request_fragment(
            product_description="   ",
            push_url="false",
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
        perform_classification_mock.assert_not_called()

    @patch("app.web.reserve_usage", new_callable=AsyncMock)
    @patch("app.classification_service.perform_classification")
    async def test_empty_metered_query_does_not_consume_quota(
        self,
        perform_classification_mock: Mock,
        reserve_usage_mock: AsyncMock,
    ) -> None:
        response = await self._request_fragment(
            product_description="   ",
            push_url="true",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Cache-Control"],
            "public, max-age=86400, stale-while-revalidate=604800",
        )
        self.assertNotIn("X-RateLimit-Remaining", response.headers)
        self.assertNotIn("X-RateLimit-Limit", response.headers)
        perform_classification_mock.assert_not_called()
        reserve_usage_mock.assert_not_awaited()

    @patch("app.classification_service.perform_classification")
    async def test_fragment_uses_default_top_k_when_omitted(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(top_k=None)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(perform_classification_mock.call_args.kwargs["top_k"], 10)

    @patch("app.classification_service.perform_classification")
    async def test_fragment_uses_default_version_when_omitted(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(version=None)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            perform_classification_mock.call_args.kwargs["version"],
            self.default_version,
        )

    async def test_fragment_rejects_top_k_above_100(self) -> None:
        response = await self._request_fragment(top_k="999")

        self.assertEqual(response.status_code, 422)

    async def test_fragment_rejects_top_k_below_1(self) -> None:
        response = await self._request_fragment(top_k="0")

        self.assertEqual(response.status_code, 422)


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

    async def test_gpc_page_displays_gs1_logo(self) -> None:
        response = await self._request("/GPC/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('src="http://testserver/static/images/GS1.png"', response.text)
        self.assertIn('alt="GS1 GPC classification standard logo"', response.text)

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
            "version_name": "v1",
            "collection_name": "test_collection",
            "query": "industrial pump",
        }

    async def _request(self, path: str) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get(path)

    @patch("app.web.reserve_usage", new_callable=AsyncMock)
    @patch("app.classification_service.perform_classification")
    async def test_unspsc_base_page_inlines_ssr_results(
        self,
        perform_classification_mock: Mock,
        reserve_usage_mock: AsyncMock,
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
        self.assertEqual(
            perform_classification_mock.call_args.kwargs["classifier_type"], "UNSPSC"
        )
        self.assertEqual(
            perform_classification_mock.call_args.kwargs["query"],
            CLASSIFIER_CONFIG["UNSPSC"]["example"].strip(),
        )
        reserve_usage_mock.assert_not_awaited()

    @patch("app.classification_service.perform_classification")
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

    @patch("app.classification_service.perform_classification")
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
            '<script data-cfasync="false">\n'
            '        document.documentElement.classList.add("js-score-animations");\n'
            "    </script>"
        )
        self.assertIn("/js/classifier.js", response.text)
        self.assertLess(
            response.text.index(animation_bootstrap),
            response.text.index("/js/classifier.js"),
        )

    @patch("app.classification_service.perform_classification")
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

    @patch("app.classification_service.perform_classification")
    async def test_search_page_keeps_client_loaded_initial_results(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        response = await self._request("/UNSPSC/custom_internal_query/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('data-autoload-enabled="true"', response.text)
        self.assertIn('data-initial-query-present="true"', response.text)
        self.assertNotIn("data-initial-track-usage", response.text)
        self.assertIn('hx-sync="this:drop"', response.text)
        self.assertNotIn('id="initial-results-loader"', response.text)
        self.assertNotIn("Laptop computers", response.text)
        perform_classification_mock.assert_not_called()

    @patch("app.web.reserve_usage", new_callable=AsyncMock)
    @patch(
        "app.classification_service.perform_classification",
        side_effect=RuntimeError("qdrant down"),
    )
    async def test_base_page_ssr_failure_falls_back_to_loader(
        self,
        perform_classification_mock: Mock,
        reserve_usage_mock: AsyncMock,
    ) -> None:
        response = await self._request("/UNSPSC/")
        expected_example = CLASSIFIER_CONFIG["UNSPSC"]["example"].strip()

        self.assertEqual(response.status_code, 200)
        self.assertIn('data-autoload-enabled="true"', response.text)
        self.assertNotIn('id="initial-results-loader"', response.text)
        self.assertIn(expected_example, response.text)
        self.assertNotIn("Set-Cookie", response.headers)
        self.assertEqual(
            response.headers["Cache-Control"],
            "no-store, max-age=0",
        )
        self.assertEqual(
            response.headers["Cloudflare-CDN-Cache-Control"],
            "no-store",
        )
        self.assertEqual(response.headers["X-Robots-Tag"], "noindex, nofollow")
        perform_classification_mock.assert_called_once()
        reserve_usage_mock.assert_not_awaited()


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
            "version_name": "v1",
            "collection_name": "test_collection",
            "query": "industrial pump",
        }

    @patch("app.web.is_verified_google_search_crawler_request", new_callable=AsyncMock)
    @patch("app.web.reserve_usage", new_callable=AsyncMock)
    @patch("app.classification_service.perform_classification")
    async def test_unspsc_fragment_uses_default_top_k_when_omitted(
        self,
        perform_classification_mock: Mock,
        reserve_usage_mock: AsyncMock,
        crawler_check_mock: AsyncMock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()
        reserve_usage_mock.return_value = UsageStatus(
            allowed=True,
            remaining=9,
            limit=10,
            is_authenticated=False,
            is_pro=False,
            tracking_id="track-123",
        )
        crawler_check_mock.return_value = False
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
        reserve_usage_mock.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
