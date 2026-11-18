import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from app.classifier_page_delivery import get_classifier_or_404
from app.usage_tracker import SignedInCaller
from tests.test_web_helpers import _build_test_app

RANKED_NAMES = ["Rank one pumps", "Rank two valves", "Rank three hoses", "Rank four"]


def _classification_result(scores=(0.9, 0.8, 0.7, 0.6)) -> dict:
    return {
        "results": [
            {
                "score": score,
                "payload": {
                    "original_id": f"10{rank}0",
                    "class_name": name,
                    "definition": f"{name} definition.",
                },
            }
            for rank, (name, score) in enumerate(zip(RANKED_NAMES, scores), start=1)
        ],
        "version_config": {"base_url": "", "tooltip": ""},
        "version_name": "v1",
        "collection_name": "test_collection",
        "query": "industrial pump",
    }


def _enable_lock(enabled: bool) -> None:
    patch("app.web.RESULT_LOCK_ENABLED", enabled).start()
    patch("app.classifier_page_delivery.RESULT_LOCK_ENABLED", enabled).start()


class ResultLockFragmentTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _build_test_app()

    def setUp(self) -> None:
        self.addCleanup(patch.stopall)
        _enable_lock(True)
        patch(
            "app.web.is_verified_google_search_crawler_request",
            new=AsyncMock(return_value=False),
        ).start()
        self.check_usage = patch("app.web.check_usage", new=AsyncMock()).start()
        self.reserve_usage = patch("app.web.reserve_usage", new=AsyncMock()).start()
        self.resolve_caller = patch(
            "app.web.resolve_signed_in_caller", new=AsyncMock(return_value=None)
        ).start()
        self.classify = patch(
            "app.classification_service.perform_classification"
        ).start()
        self.rate_limited = patch(
            "app.web.lookup_rate_limit_exceeded", new=AsyncMock(return_value=False)
        ).start()
        self.classify.return_value = _classification_result()

    async def _request_fragment(
        self, query: str = "industrial pump", **extra_params: str
    ) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.get(
                "/ISIC/fragment",
                params={"product_description": query, **extra_params},
            )

    def _assert_locked(self, html: str) -> None:
        self.assertIn(RANKED_NAMES[0], html)
        self.assertIn(RANKED_NAMES[3], html)
        for hidden in RANKED_NAMES[1:3]:
            self.assertNotIn(hidden, html)
        self.assertNotIn("1020", html)
        self.assertNotIn("1030", html)
        self.assertEqual(html.count("data-result-lock"), 2)
        self.assertIn("80.0%", html)
        self.assertIn("70.0%", html)

    async def test_anonymous_fragment_hides_ranks_two_and_three_without_quota(
        self,
    ) -> None:
        response = await self._request_fragment()

        self.assertEqual(response.status_code, 200)
        self._assert_locked(response.text)
        self.assertIn("public", response.headers["Cloudflare-CDN-Cache-Control"])
        self.check_usage.assert_not_awaited()
        self.reserve_usage.assert_not_awaited()
        self.resolve_caller.assert_not_awaited()

    async def test_lock_cards_show_the_gap_to_the_top_match_when_it_is_close(
        self,
    ) -> None:
        self.classify.return_value = _classification_result((0.741, 0.676, 0.5, 0.4))

        response = await self._request_fragment()

        self.assertIn("Only 6.5% behind #1.", response.text)
        self.assertEqual(response.text.count("% behind"), 1)
        self.assertIn(
            "Check it with Pro before you rely on the top match.", response.text
        )
        self.assertIn('aria-label="Unlock result #2 with Pro"', response.text)
        self.assertIn('aria-label="Unlock result #3 with Pro"', response.text)

    async def test_pro_unlock_shows_every_rank_and_is_not_cached(self) -> None:
        self.resolve_caller.return_value = SignedInCaller(user_id="u1", is_pro=True)

        response = await self._request_fragment(unlock="1")

        self.assertEqual(response.status_code, 200)
        for name in RANKED_NAMES:
            self.assertIn(name, response.text)
        self.assertNotIn("data-result-lock", response.text)
        self.assertEqual(response.headers["Cloudflare-CDN-Cache-Control"], "no-store")
        self.assertIn("no-store", response.headers["Cache-Control"])
        self.check_usage.assert_not_awaited()

    async def test_non_pro_unlock_stays_locked_and_is_not_cached(self) -> None:
        self.resolve_caller.return_value = SignedInCaller(user_id="u1", is_pro=False)

        response = await self._request_fragment(unlock="1")

        self._assert_locked(response.text)
        self.assertEqual(response.headers["Cloudflare-CDN-Cache-Control"], "no-store")

    async def test_the_classifier_example_is_never_locked(self) -> None:
        example = get_classifier_or_404("ISIC")[1]["example"]

        response = await self._request_fragment(query=f"  {example} ")

        for name in RANKED_NAMES:
            self.assertIn(name, response.text)
        self.assertNotIn("data-result-lock", response.text)
        self.assertIn("public", response.headers["Cloudflare-CDN-Cache-Control"])

    async def test_rate_limited_lookup_returns_429_without_classifying(self) -> None:
        self.rate_limited.return_value = True

        response = await self._request_fragment()

        self.assertEqual(response.status_code, 429)
        self.assertIn("Too many lookups", response.text)
        self.assertEqual(response.headers["Cloudflare-CDN-Cache-Control"], "no-store")
        self.classify.assert_not_called()

    async def test_pro_lookups_are_not_rate_limited(self) -> None:
        self.rate_limited.return_value = True
        self.resolve_caller.return_value = SignedInCaller(user_id="u1", is_pro=True)

        response = await self._request_fragment(unlock="1")

        self.assertEqual(response.status_code, 200)
        self.rate_limited.assert_not_awaited()

    async def test_unlock_request_is_not_cached_while_lock_is_disabled(self) -> None:
        _enable_lock(False)
        self.check_usage.return_value = Mock(allowed=True)
        self.reserve_usage.return_value = Mock(allowed=True)

        response = await self._request_fragment(unlock="1")

        for name in RANKED_NAMES:
            self.assertIn(name, response.text)
        self.assertEqual(response.headers["Cloudflare-CDN-Cache-Control"], "no-store")
        self.reserve_usage.assert_awaited_once()

    async def test_disabled_lock_keeps_metered_full_results(self) -> None:
        _enable_lock(False)
        self.check_usage.return_value = Mock(allowed=True)
        self.reserve_usage.return_value = Mock(allowed=True)

        response = await self._request_fragment()

        for name in RANKED_NAMES:
            self.assertIn(name, response.text)
        self.assertIn("public", response.headers["Cloudflare-CDN-Cache-Control"])
        self.check_usage.assert_awaited_once()
        self.rate_limited.assert_not_awaited()


class ResultLockSsrTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.addCleanup(patch.stopall)
        _enable_lock(True)
        patch(
            "app.web.is_verified_google_search_crawler_request",
            new=AsyncMock(return_value=True),
        ).start()
        classify = patch("app.classification_service.perform_classification").start()
        classify.return_value = _classification_result()

    async def test_crawler_page_gets_the_same_locked_results(self) -> None:
        transport = httpx.ASGITransport(app=_build_test_app())
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            response = await client.get("/UNSPSC/laptop_computer/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('data-result-lock="true"', response.text)
        self.assertIn(RANKED_NAMES[0], response.text)
        self.assertNotIn(RANKED_NAMES[1], response.text)
        self.assertNotIn(RANKED_NAMES[2], response.text)

    async def test_example_results_on_the_base_page_stay_unlocked(self) -> None:
        transport = httpx.ASGITransport(app=_build_test_app())
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            response = await client.get("/UNSPSC/")

        self.assertEqual(response.status_code, 200)
        for name in RANKED_NAMES:
            self.assertIn(name, response.text)
        self.assertNotIn("data-result-lock>", response.text)


if __name__ == "__main__":
    unittest.main()
