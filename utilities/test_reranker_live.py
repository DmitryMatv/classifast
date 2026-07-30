#!/usr/bin/env python3
"""
Live smoke test for the Hugging Face reranker endpoint.

Verifies the two runtime assumptions the unit tests cannot cover (they mock HTTP):

1. Sigmoid normalization: scores come back in [0, 1], i.e. the router honors
   ``parameters.function_to_apply: "sigmoid"`` for the configured model.
2. Batch size: the hf-inference router accepts a single batch of
   DEFAULT_RERANK_CANDIDATE_LIMIT (100) text pairs without rejecting the
   request or exceeding the configured timeout.

Usage: python utilities/test_reranker_live.py

Requires HF_TOKEN in .env (or the environment). Honors the same optional
variables as app startup: HF_RERANK_MODEL (default BAAI/bge-reranker-v2-m3)
and HF_RERANK_TIMEOUT_SECONDS (default 30).
"""

import os
import sys
import time

from dotenv import load_dotenv

# Allow running from the repo root or the utilities/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.classifier import DEFAULT_RERANK_CANDIDATE_LIMIT
from app.reranker import HuggingFaceReranker

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


def describe_scores(label: str, scores: list[float]) -> None:
    mean = sum(scores) / len(scores)
    print(
        f"   {label}: n={len(scores)} min={min(scores):.4f} "
        f"max={max(scores):.4f} mean={mean:.4f}"
    )


def check_sigmoid_and_ranking(reranker: HuggingFaceReranker) -> bool:
    """Small batch: scores must parse and rank a relevant document first."""
    print(f"🔍 Test 1: sigmoid + ranking ({len(DOCUMENTS)} documents)...")
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
    if not all(0.0 <= score <= 1.0 for score in scores):
        print("   ❌ scores outside [0, 1] - sigmoid not applied?")
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


def check_full_batch(reranker: HuggingFaceReranker, timeout_seconds: float) -> bool:
    """Full candidate batch: the router must accept 100 pairs in one request."""
    distractors = [f"{text} (variant {i})" for i, text in enumerate(DISTRACTORS * 10)]
    documents = distractors[: DEFAULT_RERANK_CANDIDATE_LIMIT - 1] + [
        "Hexagon head bolts, stainless steel, metric thread, DIN 933"
    ]
    target_index = len(documents) - 1
    print(f"🔍 Test 2: full batch ({len(documents)} documents, one request)...")

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
    if not all(0.0 <= score <= 1.0 for score in scores):
        print("   ❌ scores outside [0, 1] - sigmoid not applied?")
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

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found in environment or .env - cannot run live test")
        return 1

    model_name = os.getenv("HF_RERANK_MODEL", "").strip() or "BAAI/bge-reranker-v2-m3"
    timeout_seconds = float(os.getenv("HF_RERANK_TIMEOUT_SECONDS", "30"))

    print(f"Model: {model_name} (provider=hf-inference)")
    reranker = HuggingFaceReranker(
        api_key=hf_token, model_name=model_name, timeout_seconds=timeout_seconds
    )
    try:
        results = [
            check_sigmoid_and_ranking(reranker),
            check_full_batch(reranker, timeout_seconds),
        ]
    finally:
        reranker.close()

    if all(results):
        print("\n✅ All live reranker checks passed")
        return 0
    print(
        "\n❌ Live reranker checks failed - reranking would silently fall back to semantic scores"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
