import time
import unittest
from unittest.mock import Mock, patch

from app.classifier import (
    DEFAULT_OUTBOUND_BUDGET_SECONDS,
    MIN_RERANK_BUDGET_SECONDS,
    _rank_semantic_results,
    outbound_budget_seconds,
)


def _semantic_candidate(point_id: int, score: float) -> dict:
    return {
        "id": point_id,
        "score": score,
        "payload": {"original_id": f"id-{point_id}"},
    }


class OutboundBudgetSecondsTests(unittest.TestCase):
    def test_default_budget_when_env_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("CLASSIFICATION_OUTBOUND_BUDGET_SECONDS", None)
            self.assertEqual(outbound_budget_seconds(), DEFAULT_OUTBOUND_BUDGET_SECONDS)

    def test_env_override_is_honored(self) -> None:
        with patch.dict(
            "os.environ", {"CLASSIFICATION_OUTBOUND_BUDGET_SECONDS": "12.5"}
        ):
            self.assertEqual(outbound_budget_seconds(), 12.5)

    def test_invalid_or_non_positive_values_fall_back_to_default(self) -> None:
        for raw in ("not-a-number", "0", "-5"):
            with self.subTest(raw=raw):
                with patch.dict(
                    "os.environ", {"CLASSIFICATION_OUTBOUND_BUDGET_SECONDS": raw}
                ):
                    self.assertEqual(
                        outbound_budget_seconds(), DEFAULT_OUTBOUND_BUDGET_SECONDS
                    )


class RankSemanticResultsBudgetTests(unittest.TestCase):
    def test_rerank_is_skipped_when_outbound_budget_is_exhausted(self) -> None:
        reranker = Mock()
        candidates = [_semantic_candidate(1, 0.8), _semantic_candidate(2, 0.7)]
        expired_deadline = time.monotonic() - 1

        results = _rank_semantic_results(
            reranker,
            "query",
            candidates,
            id_match_results=[],
            top_k=2,
            rerank_top_n=15,
            rerank_instruction=None,
            deadline=expired_deadline,
        )

        reranker.rerank.assert_not_called()
        self.assertEqual([r["id"] for r in results], [1, 2])
        self.assertEqual(results[0]["rerank_relevance_score"], 0.0)
        self.assertEqual(results[0]["score"], 0.8)

    def test_remaining_budget_is_passed_to_reranker(self) -> None:
        reranker = Mock()
        reranker.rerank.return_value = [0.9, 0.1]
        candidates = [_semantic_candidate(1, 0.8), _semantic_candidate(2, 0.7)]
        deadline = time.monotonic() + 7.5

        results = _rank_semantic_results(
            reranker,
            "query",
            candidates,
            id_match_results=[],
            top_k=2,
            rerank_top_n=15,
            rerank_instruction=None,
            deadline=deadline,
        )

        passed_timeout = reranker.rerank.call_args.kwargs["timeout_seconds"]
        self.assertAlmostEqual(passed_timeout, 7.5, delta=0.5)
        self.assertEqual(results[0]["rerank_relevance_score"], 0.9)

    def test_tiny_remaining_budget_skips_rerank(self) -> None:
        reranker = Mock()
        candidates = [_semantic_candidate(1, 0.8)]
        deadline = time.monotonic() + MIN_RERANK_BUDGET_SECONDS / 2

        results = _rank_semantic_results(
            reranker,
            "query",
            candidates,
            id_match_results=[],
            top_k=1,
            rerank_top_n=15,
            rerank_instruction=None,
            deadline=deadline,
        )

        reranker.rerank.assert_not_called()
        self.assertEqual(len(results), 1)

    def test_no_deadline_preserves_unbounded_behavior(self) -> None:
        reranker = Mock()
        reranker.rerank.return_value = [0.5]
        candidates = [_semantic_candidate(1, 0.8)]

        _rank_semantic_results(
            reranker,
            "query",
            candidates,
            id_match_results=[],
            top_k=1,
            rerank_top_n=15,
            rerank_instruction=None,
            deadline=None,
        )

        passed_timeout = reranker.rerank.call_args.kwargs["timeout_seconds"]
        self.assertIsNone(passed_timeout)


if __name__ == "__main__":
    unittest.main()
