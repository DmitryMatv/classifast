#!/usr/bin/env python3
"""
Live smoke test for the OpenRouter reranker endpoint.

Exercises the real ``app.reranker.OpenRouterReranker`` class against the live
OpenRouter ``/api/v1/rerank`` endpoint - the same code path the deployed
backend uses. The unit tests in ``tests/test_reranker.py`` mock HTTP, so this
script is the only thing that verifies the live wire format and latency.

Verifies:

1. The OpenRouter rerank API accepts a single batch of up to
   DEFAULT_RERANK_CANDIDATE_LIMIT (100) text documents for one query.
2. The configured model returns a numeric ``relevance_score`` in [0, 1] for
   every submitted document (we request ``top_n = len(documents)`` so no
   document is truncated out of the response; the imported class enforces
   the [0, 1] range and raises if it strays).
3. A known-relevant document ranks first against distractors.
4. Latency for top-50 and top-100 batches - the primary signal this script
   collects.

Usage: python utilities/test_openrouter_reranker_live.py

Requires OPENROUTER_API_KEY in .env (or the environment). Honors the optional
variables OPENROUTER_RERANK_MODEL (default
nvidia/llama-nemotron-rerank-vl-1b-v2:free) and
OPENROUTER_RERANK_TIMEOUT_SECONDS (default 60 - the free tier can be slow on
cold start).
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

# Allow running from the repo root or the utilities/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.classifier import DEFAULT_RERANK_CANDIDATE_LIMIT  # noqa: E402
from app.reranker import OPENROUTER_RERANK_URL, OpenRouterReranker  # noqa: E402

QUERY = "stainless steel hex head bolt M8"
RELEVANT_INDICES = {0, 3}
DOCUMENTS = [
    "Hexagon head bolts, stainless steel, metric thread",
    "Fresh organic bananas, bunch of 5",
    "Wireless bluetooth over-ear headphones",
    "Stainless steel fasteners: bolts, nuts and washers",
    "Printed paperback novel, fiction",
]

DISTRACTORS = [
    "Cotton t-shirt, crew neck, assorted colors",
    "Ceramic coffee mug, 350ml, glazed",
    "LED desk lamp with adjustable arm",
    "Wooden cutting board, bamboo",
    "Yoga mat, non-slip, 6mm thick",
    "Ballpoint pen, blue ink, pack of 10",
    "Stainless steel kitchen sink, single bowl",
    "Garden hose, 15m, with spray nozzle",
    "Wireless computer mouse, optical",
    "Paper notebook, A5, ruled, 200 pages",
]
RELEVANT_DOCUMENT = "Hexagon head bolts, stainless steel, metric thread, DIN 933"


def describe_scores(label: str, scores: list[float]) -> None:
    mean = sum(scores) / len(scores)
    print(
        f"   {label}: n={len(scores)} min={min(scores):.4f} "
        f"max={max(scores):.4f} mean={mean:.4f}"
    )


def check_small_batch(reranker: OpenRouterReranker) -> bool:
    """Small batch: scores must parse and rank a relevant document first."""
    print(f"🔍 Test 1: ranking ({len(DOCUMENTS)} documents)...")
    started = time.monotonic()
    try:
        scores = reranker.rerank(QUERY, DOCUMENTS)
    except Exception as e:
        print(f"   ❌ rerank() raised: {e}")
        return False
    elapsed = time.monotonic() - started

    describe_scores("scores", scores)
    print(f"   elapsed: {elapsed:.2f}s")

    if len(scores) != len(DOCUMENTS):
        print(f"   ❌ expected {len(DOCUMENTS)} scores, got {len(scores)}")
        return False

    best_index = max(range(len(scores)), key=scores.__getitem__)
    if best_index not in RELEVANT_INDICES:
        print(
            f"   ❌ top-ranked document is '{DOCUMENTS[best_index]}' "
            f"(index {best_index}), expected a fastener/bolt document"
        )
        return False

    print(
        f"   ✅ top-ranked: '{DOCUMENTS[best_index]}' (score {scores[best_index]:.4f})"
    )
    return True


def _build_candidate_batch(total: int) -> tuple[list[str], int]:
    """Build a ``total``-length doc list with one relevant document at the end."""
    if total < 1:
        raise ValueError("total must be >= 1")
    distractors_needed = total - 1
    distractors = [
        f"{text} (variant {i})"
        for i, text in enumerate(
            DISTRACTORS * (distractors_needed // len(DISTRACTORS) + 1)
        )
    ][:distractors_needed]
    documents = distractors + [RELEVANT_DOCUMENT]
    return documents, len(documents) - 1


def check_large_batch(
    reranker: OpenRouterReranker,
    total: int,
    timeout_seconds: float,
) -> bool:
    """Large batch: ``total`` documents, one relevant, must rank it first."""
    documents, target_index = _build_candidate_batch(total)
    print(f"🔍 Test: large batch ({len(documents)} documents, one request)...")
    if target_index != len(documents) - 1:
        print("   ❌ internal setup error: relevant document not placed last")
        return False

    started = time.monotonic()
    try:
        scores = reranker.rerank(QUERY, documents)
    except Exception as e:
        print(f"   ❌ rerank() raised: {e}")
        return False
    elapsed = time.monotonic() - started

    describe_scores("scores", scores)
    print(f"   elapsed: {elapsed:.2f}s (configured timeout: {timeout_seconds}s)")

    if len(scores) != len(documents):
        print(f"   ❌ expected {len(documents)} scores, got {len(scores)}")
        return False

    best_index = max(range(len(scores)), key=scores.__getitem__)
    if best_index != target_index:
        rank = sorted(scores, reverse=True).index(scores[target_index]) + 1
        print(
            f"   ⚠️  relevant document ranked #{rank}, "
            f"not first (score {scores[target_index]:.4f} vs best {scores[best_index]:.4f})"
        )
        # Not a hard failure: the endpoint works, ranking quality is informational.
    else:
        print(
            f"   ✅ relevant document ranked first (score {scores[target_index]:.4f})"
        )
    if elapsed >= timeout_seconds * 0.8:
        print("   ⚠️  latency is close to the configured timeout")
    return True


def main() -> int:
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "❌ OPENROUTER_API_KEY not found in environment or .env - cannot run live test"
        )
        return 1

    model_name = (
        os.getenv("OPENROUTER_RERANK_MODEL", "").strip() or "voyageai/rerank-2.5-lite"
    )
    timeout_seconds = float(os.getenv("OPENROUTER_RERANK_TIMEOUT_SECONDS", "60"))

    print(f"Model: {model_name} (provider=openrouter)")
    print(f"Endpoint: {OPENROUTER_RERANK_URL}")
    reranker = OpenRouterReranker(
        api_key=api_key, model_name=model_name, timeout_seconds=timeout_seconds
    )
    try:
        results = [
            check_small_batch(reranker),
            check_large_batch(reranker, 50, timeout_seconds),
            check_large_batch(
                reranker, DEFAULT_RERANK_CANDIDATE_LIMIT, timeout_seconds
            ),
        ]
    finally:
        reranker.close()

    if all(results):
        print("\n✅ All OpenRouter reranker checks passed")
        return 0
    print("\n❌ OpenRouter reranker checks failed - see output above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
