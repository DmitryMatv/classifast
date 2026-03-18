import unittest
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.web import router

BASE_DIR = Path(__file__).resolve().parents[1]


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.mount(
        "/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static"
    )
    app.include_router(router)
    return app


class HomepageHeaderTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_test_app()

    async def _request(self, method: str, path: str) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path)

    async def test_homepage_get_sets_cache_and_canonical_headers(self) -> None:
        response = await self._request("GET", "/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Cache-Control"],
            "public, max-age=60, stale-while-revalidate=600",
        )
        self.assertEqual(
            response.headers["Cloudflare-CDN-Cache-Control"],
            "max-age=7200, stale-while-revalidate=86400",
        )
        self.assertEqual(
            response.headers["Link"],
            '<https://classifast.com/>; rel="canonical"',
        )
        self.assertEqual(response.headers["X-Robots-Tag"], "index, follow")

    async def test_homepage_head_returns_same_cache_contract(self) -> None:
        response = await self._request("HEAD", "/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Cache-Control"],
            "public, max-age=60, stale-while-revalidate=600",
        )
        self.assertEqual(
            response.headers["Cloudflare-CDN-Cache-Control"],
            "max-age=7200, stale-while-revalidate=86400",
        )
        self.assertEqual(
            response.headers["Link"],
            '<https://classifast.com/>; rel="canonical"',
        )

    async def test_classifier_head_uses_query_specific_canonical_link(self) -> None:
        response = await self._request("HEAD", "/NAICS/industrial_pump")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["Link"],
            '<https://classifast.com/NAICS/industrial_pump>; rel="canonical"',
        )


if __name__ == "__main__":
    unittest.main()
