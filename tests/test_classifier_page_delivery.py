import unittest

from app.classifier_config import CLASSIFIER_CONFIG
from app.classifier_page_delivery import (
    SITEMAP_QUERY_PATHS,
    build_classifier_canonical_url,
    build_classifier_redirect_url,
    build_fragment_page_title,
    build_fragment_push_url,
    decode_search_query,
    resolve_classifier_options,
    should_ssr,
)


class ResolveClassifierOptionsMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        config = CLASSIFIER_CONFIG["ISIC"]
        self.first_version = next(iter(config["versions"]))
        self.second_version = list(config["versions"])[1]
        self.config = config

    def test_defaults_when_nothing_provided(self) -> None:
        version, top_k, first_version = resolve_classifier_options(
            self.config, None, None, default_top_k=10
        )

        self.assertEqual(version, self.first_version)
        self.assertEqual(top_k, 10)
        self.assertEqual(first_version, self.first_version)

    def test_page_semantics_fall_back_on_unknown_version(self) -> None:
        version, _, first_version = resolve_classifier_options(
            self.config, "missing-version", 5, default_top_k=10
        )

        self.assertEqual(version, first_version)

    def test_fragment_semantics_pass_unknown_version_through(self) -> None:
        version, _, _ = resolve_classifier_options(
            self.config,
            "missing-version",
            5,
            default_top_k=10,
            allow_invalid_version=True,
        )

        self.assertEqual(version, "missing-version")

    def test_page_semantics_fall_back_on_out_of_range_top_k(self) -> None:
        _, top_k, _ = resolve_classifier_options(
            self.config, None, 999, default_top_k=10
        )

        self.assertEqual(top_k, 10)

    def test_zero_top_k_falls_back_to_default(self) -> None:
        _, top_k, _ = resolve_classifier_options(self.config, None, 0, default_top_k=10)

        self.assertEqual(top_k, 10)

    def test_explicit_valid_values_pass_through(self) -> None:
        version, top_k, _ = resolve_classifier_options(
            self.config, self.second_version, 30, default_top_k=10
        )

        self.assertEqual(version, self.second_version)
        self.assertEqual(top_k, 30)


class SsrEligibilityMatrixTests(unittest.TestCase):
    def test_sitemap_query_without_params_is_server_rendered(self) -> None:
        canonical = build_classifier_canonical_url("UNSPSC", "laptop computer")

        self.assertTrue(should_ssr("laptop computer", False, canonical))

    def test_empty_query_is_not_server_rendered(self) -> None:
        canonical = build_classifier_canonical_url("UNSPSC", "")

        self.assertFalse(should_ssr("", False, canonical))

    def test_non_sitemap_query_is_not_server_rendered(self) -> None:
        canonical = build_classifier_canonical_url("UNSPSC", "custom query")

        self.assertFalse(should_ssr("custom query", False, canonical))

    def test_query_params_disable_server_rendering(self) -> None:
        canonical = build_classifier_canonical_url("UNSPSC", "laptop computer")

        self.assertFalse(should_ssr("laptop computer", True, canonical))

    def test_sitemap_canonicals_are_eligible(self) -> None:
        self.assertTrue(SITEMAP_QUERY_PATHS)
        for path in SITEMAP_QUERY_PATHS:
            parts = path.strip("/").split("/")
            canonical = build_classifier_canonical_url(
                parts[0], parts[1].replace("_", " ")
            )
            self.assertTrue(should_ssr(parts[1].replace("_", " "), False, canonical))


class UrlPolicyMatrixTests(unittest.TestCase):
    def test_redirect_url_preserves_query_params(self) -> None:
        url = build_classifier_redirect_url("UNSPSC", "laptop_computer", "top_k=30")

        self.assertEqual(url, "/UNSPSC/laptop_computer/?top_k=30")

    def test_redirect_url_normalizes_hyphenated_slug(self) -> None:
        url = build_classifier_redirect_url("NAICS", "property-management", "")

        self.assertEqual(url, "/NAICS/property_management/")

    def test_fragment_push_url_omits_defaults(self) -> None:
        url = build_fragment_push_url(
            "NAICS", "property management", "2022", "2022", 10, 10
        )

        self.assertEqual(url, "/NAICS/property_management/")

    def test_fragment_push_url_includes_non_defaults(self) -> None:
        url = build_fragment_push_url("ISIC", "gas station", "Rev 5", "Rev 4", 30, 10)

        self.assertEqual(url, "/ISIC/gas_station/?version=Rev+5&top_k=30")

    def test_fragment_page_title_uses_query(self) -> None:
        self.assertEqual(
            build_fragment_page_title("UNSPSC", "industrial pump"),
            "UNSPSC codes for 'Industrial Pump'",
        )

    def test_decode_search_query_replaces_separators(self) -> None:
        self.assertEqual(
            decode_search_query("property_management"), "property management"
        )


if __name__ == "__main__":
    unittest.main()
