import unittest
from pathlib import Path

from app.cache_profiles import (
    STATIC_CODE,
    STATIC_MEDIA,
    STATIC_TEXT,
    build_cache_headers,
)
from app.main import get_static_cache_profile, static_file_response

BASE_DIR = Path(__file__).resolve().parents[1]


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
            self.assertEqual(response.headers["Cache-Tag"], "static-files")

    def test_robots_txt_disallows_fragment_and_noisy_query_variants(self) -> None:
        robots_text = (BASE_DIR / "app" / "static" / "robots.txt").read_text()

        self.assertIn("Disallow: /*/fragment", robots_text)
        self.assertIn("Disallow: /*?*top_k=", robots_text)
        self.assertIn("Disallow: /*?*version=", robots_text)

    def test_sitemap_excludes_generated_search_urls(self) -> None:
        sitemap_text = (BASE_DIR / "app" / "static" / "sitemap.xml").read_text()

        self.assertIn("<loc>https://classifast.com/UNSPSC/</loc>", sitemap_text)
        self.assertIn("<loc>https://blog.classifast.com/</loc>", sitemap_text)
        self.assertNotIn(
            "<loc>https://classifast.com/UNSPSC/laptop-computer/</loc>",
            sitemap_text,
        )
        self.assertNotIn(
            "<loc>https://classifast.com/NAICS/short-term-rentals/</loc>",
            sitemap_text,
        )
        self.assertNotIn(
            "<loc>https://classifast.com/NACE/pharmacy/</loc>",
            sitemap_text,
        )


if __name__ == "__main__":
    unittest.main()
