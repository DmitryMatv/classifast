import json
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.classifier_config import CLASSIFIER_CONFIG
from app.usage_tracker import UsageStatus
from app.web import normalize_query_text, router, slugify

BASE_DIR = Path(__file__).resolve().parents[1]


class StartTagFinder(HTMLParser):
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher
        self.matched_attrs: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.matched_attrs is not None:
            return

        normalized_attrs = {key: value or "" for key, value in attrs}
        if self.matcher(tag, normalized_attrs):
            self.matched_attrs = normalized_attrs


def find_tag_attrs(
    markup: str,
    matcher,
    *,
    error_message: str,
) -> dict[str, str]:
    parser = StartTagFinder(matcher)
    parser.feed(markup)
    if parser.matched_attrs is None:
        raise AssertionError(error_message)
    return parser.matched_attrs


def extract_initial_loader_vals(markup: str) -> dict:
    attrs = find_tag_attrs(
        markup,
        lambda tag, attrs: (
            tag == "div" and attrs.get("data-initial-results-loader") == "true"
        ),
        error_message="Initial results loader hx-vals not found",
    )
    hx_vals = attrs.get("hx-vals")
    if hx_vals is None:
        raise AssertionError("Initial results loader hx-vals not found")
    return json.loads(hx_vals)


def extract_classifier_form_attrs(markup: str) -> dict[str, str]:
    return find_tag_attrs(
        markup,
        lambda tag, attrs: tag == "form" and "hx-get" in attrs,
        error_message="Classifier form tag not found",
    )


def extract_initial_loader_attrs(markup: str) -> dict[str, str]:
    return find_tag_attrs(
        markup,
        lambda tag, attrs: (
            tag == "div" and attrs.get("data-initial-results-loader") == "true"
        ),
        error_message="Initial results loader tag not found",
    )


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

    def test_normalizes_internal_whitespace(self) -> None:
        self.assertEqual(slugify("  multi \n spaced\tquery  "), "multi_spaced_query")

    def test_strips_unsupported_characters(self) -> None:
        self.assertEqual(
            slugify("gearbox <script>alert(1)</script>"),
            "gearbox_scriptalert(1)script",
        )

    def test_keeps_non_latin_characters(self) -> None:
        self.assertEqual(slugify("насос промышленный"), "насос_промышленный")

    def test_keeps_arabic_characters(self) -> None:
        self.assertEqual(slugify("مضخة صناعية"), "مضخة_صناعية")

    def test_normalizes_decomposed_unicode_before_slugifying(self) -> None:
        self.assertEqual(slugify("Cafe\u0301"), "Café")

    def test_keeps_up_to_1000_characters_in_slug(self) -> None:
        self.assertEqual(len(slugify("x" * 1200)), 1000)


class NormalizeQueryTextTests(unittest.TestCase):
    def test_trims_and_collapses_whitespace(self) -> None:
        self.assertEqual(
            normalize_query_text("  industrial \n pump\t "), "industrial pump"
        )

    def test_normalizes_to_nfc(self) -> None:
        self.assertEqual(normalize_query_text("Cafe\u0301"), "Café")

    def test_preserves_non_latin_text(self) -> None:
        self.assertEqual(
            normalize_query_text("насос промышленный"), "насос промышленный"
        )

    def test_preserves_user_visible_punctuation(self) -> None:
        self.assertEqual(
            normalize_query_text("Pump, valve (industrial)'s"),
            "Pump, valve (industrial)'s",
        )


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
        }
        params.update(extra_params)
        if "top_k" in params and params["top_k"] is None:
            params.pop("top_k")
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

    @patch("app.web.perform_classification")
    async def test_fragment_normalizes_whitespace_and_nfc_before_classifying(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(
            product_description="  Cafe\u0301   pump  ",
            push_url="true",
            track_usage="false",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            perform_classification_mock.call_args.kwargs["query"],
            "Café pump",
        )
        self.assertEqual(
            response.headers.get("HX-Push-Url"),
            "/NAICS/Caf%C3%A9_pump",
        )

    @patch("app.web.perform_classification")
    async def test_fragment_push_url_preserves_lossy_query_text_with_q_param(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(
            product_description="bolt & nut / washer",
            push_url="true",
            track_usage="false",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("HX-Push-Url"),
            "/NAICS/bolt_nut_washer?q=bolt+%26+nut+%2F+washer",
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
            "max-age=86400, stale-while-revalidate=86400",
        )
        perform_classification_mock.assert_not_called()

    @patch("app.web.perform_classification")
    async def test_fragment_uses_default_top_k_when_omitted_for_non_unspsc(
        self,
        perform_classification_mock: Mock,
    ) -> None:
        perform_classification_mock.return_value = self._classification_result()

        response = await self._request_fragment(top_k=None, track_usage="false")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(perform_classification_mock.call_args.kwargs["top_k"], 10)


class SearchRedirectTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_test_app()
        versions = list(CLASSIFIER_CONFIG["NAICS"]["versions"])
        cls.default_version = versions[0]
        cls.non_default_version = versions[1]

    async def _request(self, path: str, params: dict | None = None) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get(path, params=params)

    async def test_search_redirects_to_canonical_slug_path(self) -> None:
        response = await self._request(
            "/NAICS/search",
            params={"product_description": "industrial  pump"},
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/NAICS/industrial_pump")

    async def test_search_redirect_keeps_non_default_version(self) -> None:
        response = await self._request(
            "/NAICS/search",
            params={
                "product_description": "industrial pump",
                "version": self.non_default_version,
            },
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(
            response.headers["location"],
            f"/NAICS/industrial_pump?{urlencode({'version': self.non_default_version})}",
        )

    async def test_search_redirect_keeps_non_default_top_k(self) -> None:
        response = await self._request(
            "/NAICS/search",
            params={
                "product_description": "industrial pump",
                "top_k": 30,
            },
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(
            response.headers["location"], "/NAICS/industrial_pump?top_k=30"
        )

    async def test_search_redirect_normalizes_decomposed_unicode(self) -> None:
        response = await self._request(
            "/NAICS/search",
            params={"product_description": "Cafe\u0301 pump"},
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/NAICS/Caf%C3%A9_pump")

    async def test_search_redirect_preserves_lossy_query_text_with_q_param(
        self,
    ) -> None:
        response = await self._request(
            "/NAICS/search",
            params={"product_description": "bolt & nut / washer"},
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(
            response.headers["location"],
            "/NAICS/bolt_nut_washer?q=bolt+%26+nut+%2F+washer",
        )

    async def test_search_redirect_preserves_long_query_beyond_slug_limit(self) -> None:
        long_query = "x" * 1200
        response = await self._request(
            "/NAICS/search",
            params={"product_description": long_query},
        )

        self.assertEqual(response.status_code, 303)
        location = response.headers["location"]
        self.assertTrue(location.startswith("/NAICS/" + ("x" * 1000)))
        self.assertIn(f"?q={long_query}", location)

    async def test_empty_search_redirects_to_base_classifier_page(self) -> None:
        response = await self._request(
            "/NAICS/search",
            params={"product_description": "   "},
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/NAICS/")


class PageRouteCanonicalizationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_test_app()
        versions = list(CLASSIFIER_CONFIG["NAICS"]["versions"])
        cls.default_version = versions[0]

    async def _request(self, path: str) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get(path)

    async def test_lowercase_classifier_redirects_to_uppercase_canonical_path(
        self,
    ) -> None:
        response = await self._request("/naics/industrial_pump")

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/NAICS/industrial_pump")

    async def test_non_canonical_slug_redirects_to_canonical_slug(self) -> None:
        response = await self._request("/NAICS/industrial__pump")

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/NAICS/industrial_pump")

    async def test_default_version_query_is_stripped_from_page_url(self) -> None:
        response = await self._request(
            f"/NAICS/industrial_pump?version={self.default_version}"
        )

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/NAICS/industrial_pump")

    async def test_default_top_k_query_is_stripped_from_page_url(self) -> None:
        response = await self._request("/NAICS/industrial_pump?top_k=10")

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/NAICS/industrial_pump")

    async def test_invalid_top_k_query_is_normalized_away(self) -> None:
        response = await self._request("/NAICS/industrial_pump?top_k=999")

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/NAICS/industrial_pump")

    async def test_decomposed_unicode_path_redirects_to_nfc_canonical_slug(
        self,
    ) -> None:
        response = await self._request("/NAICS/Cafe%CC%81_pump")

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/NAICS/Caf%C3%A9_pump")

    async def test_lossy_query_param_is_retained_in_canonical_page_url(self) -> None:
        response = await self._request(
            "/NAICS/bolt_nut_washer?q=bolt+%26+nut+%2F+washer"
        )

        self.assertEqual(response.status_code, 200)
        initial_loader_vals = extract_initial_loader_vals(response.text)
        self.assertEqual(
            initial_loader_vals["product_description"], "bolt & nut / washer"
        )
        self.assertIn(">bolt &amp; nut / washer</textarea>", response.text)

    async def test_long_lossy_query_param_is_used_for_initial_loader(self) -> None:
        long_query = "x" * 1200
        response = await self._request(f"/NAICS/{'x' * 1000}?q={long_query}")

        self.assertEqual(response.status_code, 200)
        initial_loader_vals = extract_initial_loader_vals(response.text)
        self.assertEqual(initial_loader_vals["product_description"], long_query)


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

    async def test_unspsc_page_defaults_to_top_30(self) -> None:
        response = await self._request("/UNSPSC/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('option value="30" selected', response.text)
        self.assertIn('"top_k": 30', response.text)

    async def test_naics_page_defaults_to_top_10(self) -> None:
        response = await self._request("/NAICS/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('option value="10" selected', response.text)
        self.assertIn('"top_k": 10', response.text)

    async def test_unspsc_invalid_top_k_redirects_to_canonical_page(self) -> None:
        response = await self._request("/UNSPSC/?top_k=999")

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/UNSPSC/")

    async def test_naics_invalid_top_k_redirects_to_canonical_page(self) -> None:
        response = await self._request("/NAICS/?top_k=999")

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "/NAICS/")


class ClassifierPageRenderingContractTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_unspsc_base_page_keeps_track_usage_explicit_without_push_url(
        self,
    ) -> None:
        response = await self._request("/UNSPSC/")

        self.assertEqual(response.status_code, 200)
        form_attrs = extract_classifier_form_attrs(response.text)
        loader_attrs = extract_initial_loader_attrs(response.text)

        self.assertIn('name="track_usage" value="true"', response.text)
        self.assertEqual(form_attrs.get("action"), "http://testserver/UNSPSC/search")
        self.assertEqual(form_attrs.get("method"), "get")
        self.assertEqual(form_attrs.get("hx-get"), "http://testserver/UNSPSC/fragment")
        self.assertEqual(form_attrs.get("hx-push-url"), "false")
        self.assertNotIn("data-classifier-type", form_attrs)
        self.assertNotIn("data-default-version", form_attrs)
        self.assertNotIn('name="push_url"', response.text)
        self.assertEqual(loader_attrs.get("hx-push-url"), "false")

        initial_loader_vals = extract_initial_loader_vals(response.text)
        self.assertEqual(initial_loader_vals["track_usage"], False)
        self.assertNotIn("push_url", initial_loader_vals)

    async def test_unspsc_search_page_loader_uses_canonical_first_party_params(
        self,
    ) -> None:
        response = await self._request("/UNSPSC/industrial_pump")

        self.assertEqual(response.status_code, 200)
        form_attrs = extract_classifier_form_attrs(response.text)
        loader_attrs = extract_initial_loader_attrs(response.text)

        self.assertIn('name="track_usage" value="true"', response.text)
        self.assertEqual(form_attrs.get("action"), "http://testserver/UNSPSC/search")
        self.assertEqual(form_attrs.get("method"), "get")
        self.assertEqual(form_attrs.get("hx-get"), "http://testserver/UNSPSC/fragment")
        self.assertEqual(form_attrs.get("hx-push-url"), "false")
        self.assertNotIn("data-classifier-type", form_attrs)
        self.assertNotIn("data-default-version", form_attrs)
        self.assertNotIn('name="push_url"', response.text)
        self.assertEqual(loader_attrs.get("hx-push-url"), "false")

        initial_loader_vals = extract_initial_loader_vals(response.text)
        self.assertEqual(initial_loader_vals["product_description"], "industrial pump")
        self.assertEqual(initial_loader_vals["track_usage"], True)
        self.assertNotIn("push_url", initial_loader_vals)


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
        self.assertEqual(perform_classification_mock.call_args.kwargs["top_k"], 30)


if __name__ == "__main__":
    unittest.main()
