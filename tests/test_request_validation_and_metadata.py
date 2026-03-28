import unittest
from html import escape
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.classifier_config import CLASSIFIER_CONFIG
from app.main import QueryNormalizationMiddleware, URLEncodingValidationMiddleware
from app.web import get_default_top_k, router

BASE_DIR = Path(__file__).resolve().parents[1]


def _build_middleware_test_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(URLEncodingValidationMiddleware)

    @app.get("/echo")
    async def echo(request: Request):
        return JSONResponse({"q": request.query_params.get("q", "")})

    return app


def _build_query_normalization_test_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(QueryNormalizationMiddleware)

    @app.get("/echo")
    async def echo(request: Request):
        return JSONResponse({"q": request.query_params.get("q", "")})

    @app.get("/{classifier_type}/fragment")
    async def fragment_echo(request: Request, classifier_type: str):
        return JSONResponse(
            {
                "classifier_type": classifier_type,
                "query": request.url.query,
            }
        )

    return app


def _build_web_test_app() -> FastAPI:
    app = FastAPI()
    app.mount(
        "/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static"
    )
    app.include_router(router)
    return app


class RequestValidationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_middleware_test_app()

    async def test_long_benign_query_is_not_rejected_by_double_counted_length(self):
        transport = httpx.ASGITransport(app=self.app)
        long_query = ("industrial-pump-" * 240).rstrip("-")

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get("/echo", params={"q": long_query})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["q"], long_query)

    async def test_oversized_query_key_is_rejected(self):
        transport = httpx.ASGITransport(app=self.app)
        oversized_key = "k" * 5001

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get(f"/echo?{oversized_key}=1")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "INVALID_ENCODING")

    async def test_suspicious_encoding_pattern_is_still_rejected(self):
        transport = httpx.ASGITransport(app=self.app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get("/echo?q=%25%25%25")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "INVALID_ENCODING")


class QueryNormalizationMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_query_normalization_test_app()
        cls.classifier_type = "NAICS"

    async def test_redirects_only_for_whitespace_normalization(self):
        transport = httpx.ASGITransport(app=self.app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get("/echo?q=%20industrial%20%20pump%20")

        self.assertEqual(response.status_code, 308)
        self.assertEqual(
            response.headers["Location"],
            "http://testserver/echo?q=industrial%20pump",
        )

    async def test_fragment_keeps_existing_semantic_params_without_redirect(self):
        transport = httpx.ASGITransport(app=self.app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get(
                f"/{self.classifier_type}/fragment",
                params={
                    "product_description": "pump",
                    "push_url": "false",
                    "track_usage": "false",
                    "top_k": "30",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["query"],
            "product_description=pump&push_url=false&track_usage=false&top_k=30",
        )


class ClassifierPageMetadataTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_web_test_app()
        if "UNSPSC" in CLASSIFIER_CONFIG:
            cls.classifier_type = "UNSPSC"
            cls.config = CLASSIFIER_CONFIG["UNSPSC"]
        else:
            cls.classifier_type, cls.config = next(
                (
                    (name, cfg)
                    for name, cfg in CLASSIFIER_CONFIG.items()
                    if cfg.get("versions")
                )
            )

    async def test_search_page_renders_query_specific_json_ld(self):
        transport = httpx.ASGITransport(app=self.app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get(f"/{self.classifier_type}/industrial_pump")

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            f'"name": "{self.classifier_type} Code for industrial pump | Classifast"',
            response.text,
        )
        self.assertIn(
            "Find accurate "
            f"{self.classifier_type} codes for industrial pump with fastest classification tool.",
            response.text,
        )

    async def test_base_page_keeps_generic_json_ld(self):
        transport = httpx.ASGITransport(app=self.app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get(f"/{self.classifier_type}/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            f'"name": "{self._expected_generic_name()}"',
            response.text,
        )
        self.assertNotIn(
            f'"name": "{self.classifier_type} Code for industrial pump | Classifast"',
            response.text,
        )

    async def test_homepage_renders_stable_desktop_auth_slot(self):
        transport = httpx.ASGITransport(app=self.app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('class="auth-slot"', response.text)
        self.assertIn('data-auth-slot="desktop"', response.text)
        self.assertIn('id="desktop-auth-container"', response.text)

    async def test_classifier_page_renders_stable_desktop_auth_slot(self):
        transport = httpx.ASGITransport(app=self.app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get(f"/{self.classifier_type}/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('class="auth-slot"', response.text)
        self.assertIn('data-auth-slot="desktop"', response.text)
        self.assertIn('id="desktop-auth-container"', response.text)

    async def test_search_page_gates_initial_loader_on_auth_ready(self):
        transport = httpx.ASGITransport(app=self.app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get(f"/{self.classifier_type}/industrial_pump")

        self.assertEqual(response.status_code, 200)
        self.assertIn('id="classifier-form"', response.text)
        self.assertIn('data-autoload-enabled="true"', response.text)
        self.assertIn('data-initial-query-present="true"', response.text)
        self.assertIn('data-initial-track-usage="true"', response.text)
        self.assertIn('data-default-example-prefill="false"', response.text)
        self.assertIn(
            f'data-default-top-k="{get_default_top_k(self.classifier_type)}"',
            response.text,
        )
        self.assertIn(
            f'data-default-version="{next(iter(self.config["versions"]))}"',
            response.text,
        )
        self.assertIn(">industrial pump</textarea>", response.text)
        self.assertNotIn('id="initial-results-loader"', response.text)
        self.assertNotIn('name="push_url"', response.text)
        self.assertNotIn('name="track_usage"', response.text)

    async def test_base_page_keeps_immediate_unmetered_initial_loader(self):
        transport = httpx.ASGITransport(app=self.app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get(f"/{self.classifier_type}/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('id="classifier-form"', response.text)
        self.assertIn('data-autoload-enabled="true"', response.text)
        self.assertIn('data-initial-query-present="false"', response.text)
        self.assertIn('data-initial-track-usage="false"', response.text)
        self.assertIn('data-default-example-prefill="true"', response.text)
        self.assertIn(
            f'data-default-top-k="{get_default_top_k(self.classifier_type)}"',
            response.text,
        )
        self.assertIn(
            f'data-default-version="{next(iter(self.config["versions"]))}"',
            response.text,
        )
        example_query = self.config["example"]
        self.assertIn(f">{escape(example_query)}</textarea>", response.text)
        self.assertIn(
            f'placeholder="{escape(example_query)}"',
            response.text,
        )
        self.assertNotIn('id="initial-results-loader"', response.text)
        self.assertNotIn('name="push_url"', response.text)
        self.assertNotIn('name="track_usage"', response.text)

    async def test_base_page_uses_textarea_value_as_initial_autoload_source(self):
        transport = httpx.ASGITransport(app=self.app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get(f"/{self.classifier_type}/")

        self.assertEqual(response.status_code, 200)
        example_query = self.config["example"]
        self.assertIn(
            f'id="product_description_area" placeholder="{escape(example_query)}"',
            response.text,
        )
        self.assertIn(f">{escape(example_query)}</textarea>", response.text)

    def _expected_generic_name(self) -> str:
        if self.classifier_type == "UNSPSC":
            return "UNSPSC Code Lookup & Classification | Classifast"
        if self.classifier_type == "ETIM":
            return "ETIM Product Classification & Lookup | Classifast"
        return (
            f"Fast {self.classifier_type} Classification, Code Lookup, Search | "
            "Classifast"
        )


if __name__ == "__main__":
    unittest.main()
