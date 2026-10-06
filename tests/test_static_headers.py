import unittest
from pathlib import Path

import httpx
import pytest

from app.cache_profiles import (
    STATIC_CODE,
    STATIC_MEDIA,
    STATIC_TEXT,
    CacheProfile,
    build_cache_headers,
)
from app.main import app, get_static_cache_profile, static_file_response


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mounted_static_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    static_files = next(route.app for route in app.routes if route.name == "static")
    monkeypatch.setattr(static_files, "directory", tmp_path)
    monkeypatch.setattr(static_files, "all_directories", [str(tmp_path)])
    return tmp_path


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("asset_path", "cache_profile"),
    [
        ("app.js", STATIC_CODE),
        ("favicon.png", STATIC_MEDIA),
        ("robots.txt", STATIC_TEXT),
    ],
)
async def test_mounted_static_etag_revalidation(
    mounted_static_dir: Path, asset_path: str, cache_profile: CacheProfile
) -> None:
    contents = b"static asset contents"
    (mounted_static_dir / asset_path).write_bytes(contents)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        original = await client.get(f"/static/{asset_path}")
        assert original.status_code == 200
        assert original.content == contents

        revalidated = await client.get(
            f"/static/{asset_path}",
            headers={"If-None-Match": original.headers["ETag"]},
        )

    assert revalidated.status_code == 304
    assert revalidated.content == b""
    assert revalidated.headers["ETag"] == original.headers["ETag"]
    for response in (original, revalidated):
        for name, value in build_cache_headers(cache_profile).items():
            assert response.headers[name] == value
        assert {
            token.strip().lower() for token in response.headers["Vary"].split(",")
        } == {"accept-encoding"}
        assert response.headers["Cache-Tag"] == "static-files"
        assert "Set-Cookie" not in response.headers


@pytest.mark.anyio
async def test_mounted_static_changed_file_returns_new_etag(
    mounted_static_dir: Path,
) -> None:
    asset = mounted_static_dir / "app.js"
    asset.write_bytes(b"original contents")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        original = await client.get("/static/app.js")
        assert original.status_code == 200
        original_etag = original.headers["ETag"]

        updated_contents = b"updated contents with a different size"
        asset.write_bytes(updated_contents)
        changed = await client.get(
            "/static/app.js", headers={"If-None-Match": original_etag}
        )
        assert changed.status_code == 200
        assert changed.content == updated_contents
        assert changed.headers["ETag"] != original_etag

        revalidated = await client.get(
            "/static/app.js", headers={"If-None-Match": changed.headers["ETag"]}
        )

    assert revalidated.status_code == 304
    assert revalidated.content == b""
    assert revalidated.headers["ETag"] == changed.headers["ETag"]


class StaticHeaderTests(unittest.TestCase):
    def test_js_assets_use_static_code_profile(self) -> None:
        response_headers = build_cache_headers(get_static_cache_profile("htmx.min.js"))

        self.assertEqual(get_static_cache_profile("htmx.min.js"), STATIC_CODE)
        self.assertEqual(
            response_headers["Cache-Control"],
            "public, max-age=300, stale-while-revalidate=3600",
        )
        self.assertEqual(
            response_headers["Cloudflare-CDN-Cache-Control"],
            "max-age=43200, stale-while-revalidate=86400",
        )
        self.assertNotIn("Expires", response_headers)

    def test_media_assets_use_static_media_profile_without_immutable(self) -> None:
        response_headers = build_cache_headers(
            get_static_cache_profile("images/favicon-32x32.png")
        )

        self.assertEqual(
            get_static_cache_profile("images/favicon-32x32.png"), STATIC_MEDIA
        )
        self.assertEqual(
            response_headers["Cache-Control"],
            "public, max-age=3600, stale-while-revalidate=86400",
        )
        self.assertEqual(
            response_headers["Cloudflare-CDN-Cache-Control"],
            "max-age=604800, stale-while-revalidate=86400",
        )
        self.assertNotIn("Expires", response_headers)
        self.assertNotIn("immutable", response_headers["Cache-Control"])

    def test_favicon_uses_static_media_profile(self) -> None:
        response = static_file_response(
            "images/favicon.ico", cache_profile=STATIC_MEDIA
        )

        self.assertEqual(
            response.headers["Cache-Control"],
            "public, max-age=3600, stale-while-revalidate=86400",
        )
        self.assertEqual(
            response.headers["Cloudflare-CDN-Cache-Control"],
            "max-age=604800, stale-while-revalidate=86400",
        )
        self.assertNotIn("Expires", response.headers)
        self.assertEqual(response.headers["Cache-Tag"], "static-files")

    def test_root_static_text_files_use_static_text_profile(self) -> None:
        for path in ("robots.txt", "sitemap.xml"):
            response = static_file_response(path, cache_profile=STATIC_TEXT)

            self.assertEqual(
                response.headers["Cache-Control"],
                "public, max-age=600, stale-while-revalidate=3600",
            )
            self.assertEqual(
                response.headers["Cloudflare-CDN-Cache-Control"],
                "max-age=7200, stale-while-revalidate=86400",
            )
            self.assertNotIn("Expires", response.headers)
            self.assertEqual(response.headers["Cache-Tag"], "static-files")

    def test_csv_downloads_use_static_text_profile(self) -> None:
        self.assertEqual(
            get_static_cache_profile("mapping_samples/example.csv"), STATIC_TEXT
        )

    def test_binary_downloads_use_static_media_profile(self) -> None:
        for path in (
            "exports/example.xlsx",
            "exports/example.pdf",
            "exports/example.zip",
        ):
            self.assertEqual(get_static_cache_profile(path), STATIC_MEDIA)


if __name__ == "__main__":
    unittest.main()
