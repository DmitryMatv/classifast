import unittest

from app.cache_profiles import (
    STATIC_CODE,
    STATIC_MEDIA,
    STATIC_TEXT,
    build_cache_headers,
)
from app.main import get_static_cache_profile, static_file_response


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
