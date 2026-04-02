import unittest
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.mapping_store import MAPPING_PRODUCTS
from app.web import download_mapping_sample, router

BASE_DIR = Path(__file__).resolve().parents[1]


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.mount(
        "/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static"
    )
    app.include_router(router)
    return app


class MappingStorefrontRouteTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_test_app()
        cls.product = next(iter(MAPPING_PRODUCTS.values()))

    async def _request(self, method: str, path: str) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            follow_redirects=False,
        ) as client:
            return await client.request(method, path)

    async def test_mapping_index_sets_cache_and_canonical_headers(self) -> None:
        response = await self._request("GET", "/mapping/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Link"],
            '<https://classifast.com/mapping/>; rel="canonical"',
        )
        self.assertEqual(response.headers["X-Robots-Tag"], "index, follow")
        self.assertIn(self.product.title, response.text)
        self.assertIn(
            f'href="http://testserver/mapping/{self.product.slug}/"',
            response.text,
        )
        self.assertIn("Learn more", response.text)
        self.assertNotIn('data-auth-slot="desktop"', response.text)
        self.assertNotIn('id="desktop-auth-container"', response.text)
        self.assertIn('data-auth-ui="disabled"', response.text)
        self.assertIn(
            "background-image: linear-gradient(180deg, #d6f0ff 0%, #f8fafc 100%);",
            response.text,
        )
        self.assertNotIn(
            "background-attachment: scroll, scroll, scroll;", response.text
        )

    async def test_mapping_product_head_uses_product_canonical_link(self) -> None:
        response = await self._request("HEAD", f"/mapping/{self.product.slug}/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Link"],
            f'<https://classifast.com/mapping/{self.product.slug}/>; rel="canonical"',
        )

    async def test_mapping_product_page_does_not_render_auth_controls(self) -> None:
        response = await self._request("GET", f"/mapping/{self.product.slug}/")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn('data-auth-slot="desktop"', response.text)
        self.assertNotIn('id="desktop-auth-container"', response.text)
        self.assertIn('data-auth-ui="disabled"', response.text)
        self.assertIn(
            "background-image: linear-gradient(180deg, #d6f0ff 0%, #f8fafc 100%);",
            response.text,
        )
        self.assertNotIn(
            "background-attachment: scroll, scroll, scroll;", response.text
        )

    async def test_mapping_routes_redirect_to_trailing_slash(self) -> None:
        index_response = await self._request("GET", "/mapping")
        product_response = await self._request("GET", f"/mapping/{self.product.slug}")

        self.assertEqual(index_response.status_code, 301)
        self.assertEqual(index_response.headers["location"], "/mapping/")
        self.assertEqual(product_response.status_code, 301)
        self.assertEqual(
            product_response.headers["location"],
            f"/mapping/{self.product.slug}/",
        )

    async def test_unknown_mapping_product_returns_404(self) -> None:
        response = await self._request("GET", "/mapping/not-a-product/")

        self.assertEqual(response.status_code, 404)

    async def test_legacy_mapping_routes_redirect_to_singular_paths(self) -> None:
        index_response = await self._request("GET", "/mappings/")
        product_response = await self._request("GET", f"/mappings/{self.product.slug}/")
        sample_response = await self._request(
            "GET", f"/mappings/{self.product.slug}/sample"
        )

        self.assertEqual(index_response.status_code, 301)
        self.assertEqual(index_response.headers["location"], "/mapping/")
        self.assertEqual(product_response.status_code, 301)
        self.assertEqual(
            product_response.headers["location"],
            f"/mapping/{self.product.slug}/",
        )
        self.assertEqual(sample_response.status_code, 301)
        self.assertEqual(
            sample_response.headers["location"],
            f"/mapping/{self.product.slug}/sample",
        )

    async def test_mapping_sample_download_is_public_and_cacheable(self) -> None:
        response = await download_mapping_sample(self.product.slug)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Cache-Control"],
            "public, max-age=600, stale-while-revalidate=3600",
        )
        self.assertIn("attachment;", response.headers["content-disposition"])
        self.assertIn(
            Path(self.product.sample_file_path).name,
            response.headers["content-disposition"],
        )
        self.assertNotIn("set-cookie", response.headers)


class MappingSitemapTests(unittest.TestCase):
    def test_sitemap_includes_mapping_storefront_urls(self) -> None:
        sitemap = (BASE_DIR / "app" / "static" / "sitemap.xml").read_text()

        self.assertIn("https://classifast.com/mapping/", sitemap)
        for slug in MAPPING_PRODUCTS:
            self.assertIn(f"https://classifast.com/mapping/{slug}/", sitemap)


if __name__ == "__main__":
    unittest.main()
