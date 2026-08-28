import unittest

from app.dependencies import _asset_version_cache, asset_url


class AssetUrlTests(unittest.TestCase):
    def setUp(self) -> None:
        _asset_version_cache.clear()

    def test_existing_asset_gets_content_hash_version(self) -> None:
        url = asset_url("/js/common.js")

        self.assertTrue(url.startswith("/static/js/common.js?v="))
        version = url.split("?v=")[1]
        self.assertEqual(len(version), 10)
        self.assertTrue(all(c in "0123456789abcdef" for c in version))

    def test_version_is_stable_across_calls(self) -> None:
        self.assertEqual(asset_url("/js/common.js"), asset_url("/js/common.js"))

    def test_missing_asset_falls_back_to_unversioned_url(self) -> None:
        self.assertEqual(
            asset_url("/js/does-not-exist.js"), "/static/js/does-not-exist.js"
        )

    def test_css_assets_are_versioned(self) -> None:
        url = asset_url("/css/styles.css")

        self.assertIn("?v=", url)


if __name__ == "__main__":
    unittest.main()
