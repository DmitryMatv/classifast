import ipaddress
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from app import google_crawlers
from app.google_crawlers import (
    _get_google_common_crawler_networks,
    _parse_google_crawler_networks,
    is_verified_google_search_crawler_request,
)


def _build_request(
    *,
    user_agent: str,
    client_ip: str | None = "66.249.64.1",
    peer_ip: str = "127.0.0.1",
    forwarded_for: str | None = None,
) -> Mock:
    request = Mock()
    request.headers = {"user-agent": user_agent}
    if client_ip is not None:
        request.headers["cf-connecting-ip"] = client_ip
    if forwarded_for is not None:
        request.headers["x-forwarded-for"] = forwarded_for
    request.cookies = {}
    request.client = SimpleNamespace(host=peer_ip)
    return request


class GoogleCrawlerParserTests(unittest.TestCase):
    def test_malformed_and_empty_google_range_payloads_parse_as_empty(self) -> None:
        self.assertEqual(_parse_google_crawler_networks({}), [])
        self.assertEqual(_parse_google_crawler_networks({"prefixes": []}), [])
        self.assertEqual(
            _parse_google_crawler_networks(
                {
                    "prefixes": [
                        {"ipv4Prefix": "not-a-cidr"},
                        {"ipv6Prefix": "also-not-a-cidr"},
                        {"irrelevant": "66.249.64.0/27"},
                        "not-a-dict",
                    ]
                }
            ),
            [],
        )


class GoogleCrawlerVerificationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        google_crawlers._cached_networks = None
        google_crawlers._cached_at = 0.0
        env_patcher = patch.dict(
            os.environ, {"GOOGLE_CRAWLER_TRUST_CF_CONNECTING_IP": "true"}
        )
        env_patcher.start()
        self.addCleanup(env_patcher.stop)

    async def test_supported_googlebot_user_agent_and_ip_returns_true(self) -> None:
        request = _build_request(
            user_agent=(
                "Mozilla/5.0 AppleWebKit/537.36 "
                "(KHTML, like Gecko; compatible; Googlebot/2.1; "
                "+http://www.google.com/bot.html) Chrome/120.0 Safari/537.36"
            ),
            client_ip="66.249.64.1",
        )

        with (
            patch.dict(os.environ, {"GOOGLE_CRAWLER_TRUST_CF_CONNECTING_IP": "true"}),
            patch(
                "app.google_crawlers._get_google_common_crawler_networks",
                new=AsyncMock(return_value=[ipaddress.ip_network("66.249.64.0/27")]),
            ) as ranges_mock,
        ):
            verified = await is_verified_google_search_crawler_request(request)

        self.assertTrue(verified)
        ranges_mock.assert_awaited_once()

    async def test_supported_inspection_tool_user_agent_and_ip_returns_true(
        self,
    ) -> None:
        request = _build_request(
            user_agent="Mozilla/5.0 (compatible; Google-InspectionTool/1.0;)",
            client_ip="66.249.64.1",
        )

        with patch(
            "app.google_crawlers._get_google_common_crawler_networks",
            new=AsyncMock(return_value=[ipaddress.ip_network("66.249.64.0/27")]),
        ) as ranges_mock:
            verified = await is_verified_google_search_crawler_request(request)

        self.assertTrue(verified)
        ranges_mock.assert_awaited_once()

    async def test_spoofed_googlebot_user_agent_from_non_google_ip_returns_false(
        self,
    ) -> None:
        request = _build_request(
            user_agent="Googlebot/2.1 (+http://www.google.com/bot.html)",
            client_ip="203.0.113.10",
        )

        with patch(
            "app.google_crawlers._get_google_common_crawler_networks",
            new=AsyncMock(return_value=[ipaddress.ip_network("66.249.64.0/27")]),
        ) as ranges_mock:
            verified = await is_verified_google_search_crawler_request(request)

        self.assertFalse(verified)
        ranges_mock.assert_awaited_once()

    async def test_x_forwarded_for_google_ip_is_not_trusted_without_cloudflare_header(
        self,
    ) -> None:
        request = _build_request(
            user_agent="Googlebot/2.1 (+http://www.google.com/bot.html)",
            client_ip=None,
            peer_ip="203.0.113.10",
            forwarded_for="66.249.64.1",
        )

        with patch(
            "app.google_crawlers._get_google_common_crawler_networks",
            new=AsyncMock(return_value=[ipaddress.ip_network("66.249.64.0/27")]),
        ) as ranges_mock:
            verified = await is_verified_google_search_crawler_request(request)

        self.assertFalse(verified)
        ranges_mock.assert_awaited_once()

    async def test_cloudflare_header_is_ignored_without_explicit_trust(self) -> None:
        request = _build_request(
            user_agent="Googlebot/2.1 (+http://www.google.com/bot.html)",
            client_ip="66.249.64.1",
            peer_ip="203.0.113.10",
        )

        with (
            patch.dict(os.environ, {"GOOGLE_CRAWLER_TRUST_CF_CONNECTING_IP": "false"}),
            patch(
                "app.google_crawlers._get_google_common_crawler_networks",
                new=AsyncMock(return_value=[ipaddress.ip_network("66.249.64.0/27")]),
            ) as ranges_mock,
        ):
            verified = await is_verified_google_search_crawler_request(request)

        self.assertFalse(verified)
        ranges_mock.assert_awaited_once()

    async def test_peer_ip_is_used_when_cloudflare_header_is_missing(self) -> None:
        request = _build_request(
            user_agent="Googlebot/2.1 (+http://www.google.com/bot.html)",
            client_ip=None,
            peer_ip="66.249.64.1",
        )

        with patch(
            "app.google_crawlers._get_google_common_crawler_networks",
            new=AsyncMock(return_value=[ipaddress.ip_network("66.249.64.0/27")]),
        ) as ranges_mock:
            verified = await is_verified_google_search_crawler_request(request)

        self.assertTrue(verified)
        ranges_mock.assert_awaited_once()

    async def test_normal_browser_user_agent_does_not_fetch_ranges(self) -> None:
        request = _build_request(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
            ),
            client_ip="66.249.64.1",
        )

        with patch(
            "app.google_crawlers._get_google_common_crawler_networks",
            new=AsyncMock(side_effect=AssertionError("should not fetch ranges")),
        ) as ranges_mock:
            verified = await is_verified_google_search_crawler_request(request)

        self.assertFalse(verified)
        ranges_mock.assert_not_awaited()

    async def test_invalid_client_ip_returns_false(self) -> None:
        request = _build_request(
            user_agent="Googlebot/2.1 (+http://www.google.com/bot.html)",
            client_ip="not-an-ip",
        )

        with patch(
            "app.google_crawlers._get_google_common_crawler_networks",
            new=AsyncMock(side_effect=AssertionError("should not fetch ranges")),
        ) as ranges_mock:
            verified = await is_verified_google_search_crawler_request(request)

        self.assertFalse(verified)
        ranges_mock.assert_not_awaited()

    async def test_fetch_failure_with_no_cache_returns_empty_ranges(self) -> None:
        with (
            patch(
                "app.google_crawlers._fetch_google_common_crawler_networks",
                new=AsyncMock(side_effect=httpx.ConnectError("boom")),
            ),
            patch("app.google_crawlers._range_cache_ttl_seconds", return_value=0),
        ):
            networks = await _get_google_common_crawler_networks()

        self.assertEqual(networks, [])
        self.assertEqual(google_crawlers._cached_networks, [])

    async def test_unexpired_existing_cache_is_used_without_fetching(self) -> None:
        cached_network = ipaddress.ip_network("66.249.64.0/27")
        google_crawlers._cached_networks = [cached_network]
        google_crawlers._cached_at = 100.0

        with (
            patch(
                "app.google_crawlers._fetch_google_common_crawler_networks",
                new=AsyncMock(side_effect=AssertionError("should use cache")),
            ) as fetch_mock,
            patch("app.google_crawlers._range_cache_ttl_seconds", return_value=60),
            patch("app.google_crawlers.time.monotonic", return_value=110.0),
        ):
            networks = await _get_google_common_crawler_networks()

        self.assertEqual(networks, [cached_network])
        fetch_mock.assert_not_awaited()

    async def test_fetch_failure_with_expired_cache_fails_closed(
        self,
    ) -> None:
        cached_network = ipaddress.ip_network("66.249.64.0/27")
        google_crawlers._cached_networks = [cached_network]
        google_crawlers._cached_at = 100.0
        fetch_mock = AsyncMock(side_effect=httpx.ConnectError("boom"))

        with (
            patch(
                "app.google_crawlers._fetch_google_common_crawler_networks",
                new=fetch_mock,
            ),
            patch("app.google_crawlers._range_cache_ttl_seconds", return_value=60),
            patch("app.google_crawlers._range_negative_ttl_seconds", return_value=300),
            patch("app.google_crawlers.time.monotonic", return_value=200.0),
        ):
            networks = await _get_google_common_crawler_networks()

        self.assertEqual(networks, [])
        self.assertEqual(google_crawlers._cached_networks, [])
        self.assertEqual(google_crawlers._cached_at, 200.0)
        fetch_mock.assert_awaited_once()

    async def test_fetch_failure_is_negative_cached_and_throttles_retries(
        self,
    ) -> None:
        cached_network = ipaddress.ip_network("66.249.64.0/27")
        google_crawlers._cached_networks = [cached_network]
        google_crawlers._cached_at = 100.0
        fetch_mock = AsyncMock(side_effect=httpx.ConnectError("boom"))

        with (
            patch(
                "app.google_crawlers._fetch_google_common_crawler_networks",
                new=fetch_mock,
            ),
            patch("app.google_crawlers._range_cache_ttl_seconds", return_value=60),
            patch("app.google_crawlers._range_negative_ttl_seconds", return_value=300),
            patch("app.google_crawlers.time.monotonic", return_value=200.0),
        ):
            first_result = await _get_google_common_crawler_networks()

        with (
            patch(
                "app.google_crawlers._fetch_google_common_crawler_networks",
                new=fetch_mock,
            ),
            patch("app.google_crawlers._range_cache_ttl_seconds", return_value=60),
            patch("app.google_crawlers._range_negative_ttl_seconds", return_value=300),
            patch("app.google_crawlers.time.monotonic", return_value=201.0),
        ):
            second_result = await _get_google_common_crawler_networks()

        self.assertEqual(first_result, [])
        self.assertEqual(second_result, [])
        self.assertEqual(google_crawlers._cached_networks, [])
        self.assertEqual(google_crawlers._cached_at, 200.0)
        fetch_mock.assert_awaited_once()

    async def test_empty_refresh_is_negative_cached(self) -> None:
        fetch_mock = AsyncMock(return_value=[])

        with (
            patch(
                "app.google_crawlers._fetch_google_common_crawler_networks",
                new=fetch_mock,
            ),
            patch("app.google_crawlers._range_cache_ttl_seconds", return_value=60),
            patch("app.google_crawlers.time.monotonic", return_value=100.0),
        ):
            networks = await _get_google_common_crawler_networks()

        self.assertEqual(networks, [])
        fetch_mock.assert_awaited_once()
        self.assertEqual(google_crawlers._cached_networks, [])
        self.assertEqual(google_crawlers._cached_at, 100.0)

        with (
            patch(
                "app.google_crawlers._fetch_google_common_crawler_networks",
                new=AsyncMock(side_effect=AssertionError("should use negative cache")),
            ) as second_fetch_mock,
            patch("app.google_crawlers._range_cache_ttl_seconds", return_value=60),
            patch("app.google_crawlers.time.monotonic", return_value=110.0),
        ):
            networks = await _get_google_common_crawler_networks()

        self.assertEqual(networks, [])
        second_fetch_mock.assert_not_awaited()

    async def test_disabled_bypass_returns_false(self) -> None:
        request = _build_request(
            user_agent="Googlebot/2.1 (+http://www.google.com/bot.html)",
            client_ip="66.249.64.1",
        )

        with (
            patch.dict(os.environ, {"GOOGLE_CRAWLER_BYPASS_ENABLED": "false"}),
            patch(
                "app.google_crawlers._get_google_common_crawler_networks",
                new=AsyncMock(side_effect=AssertionError("should not fetch ranges")),
            ) as ranges_mock,
        ):
            verified = await is_verified_google_search_crawler_request(request)

        self.assertFalse(verified)
        ranges_mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
