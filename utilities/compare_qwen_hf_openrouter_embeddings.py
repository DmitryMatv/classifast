#!/usr/bin/env python3
"""Compare Qwen3 embedding vectors returned by Hugging Face and OpenRouter."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

DEFAULT_TEXT = "Industrial pump for chemical garbage processing (or, to put it simply, garbage in -> pump -> garbage out).GARBAGE IN = GARBAGE OUT !!!"
# DEFAULT_TEXT = "trash"

DEFAULT_HF_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3-embedding-8b"

DEFAULT_DIMENSIONS = 2048
# DEFAULT_DIMENSIONS = 4096

DEFAULT_HF_PROVIDER = "auto"

EXIT_EXACT_MATCH = 0
EXIT_DIFFERENT = 1
EXIT_FAILURE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed the exact same text with Hugging Face Inference and "
            "OpenRouter, then compare the returned vectors."
        )
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help=f"Text to embed exactly as provided. Default: {DEFAULT_TEXT!r}",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=int(os.getenv("HF_EMBEDDING_DIMS", str(DEFAULT_DIMENSIONS))),
        help="Embedding dimensions to request from both providers.",
    )
    parser.add_argument(
        "--hf-model",
        default=os.getenv("HF_EMBEDDING_MODEL", "").strip() or DEFAULT_HF_MODEL,
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--openrouter-model",
        default=(
            os.getenv("OPENROUTER_EMBEDDING_MODEL", "").strip()
            or DEFAULT_OPENROUTER_MODEL
        ),
        help="OpenRouter embedding model id.",
    )
    parser.add_argument(
        "--openrouter-provider",
        default=os.getenv("OPENROUTER_PROVIDER", "").strip() or None,
        help=(
            "Optional OpenRouter provider name to force via provider.only. "
            "Defaults to OPENROUTER_PROVIDER. Leave unset to use OpenRouter routing."
        ),
    )
    parser.add_argument(
        "--hf-provider",
        default=os.getenv("HF_INFERENCE_PROVIDER", "").strip() or DEFAULT_HF_PROVIDER,
        help="Hugging Face Inference provider.",
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required in .env or the shell environment")
    return value


def normalize_single_embedding(response: Any, source_name: str) -> list[float]:
    """Coerce common embedding response shapes into one flat float vector."""
    if hasattr(response, "tolist"):
        response = response.tolist()

    if not isinstance(response, list):
        raise RuntimeError(f"{source_name} embedding response is not a list")
    if not response:
        raise RuntimeError(f"{source_name} embedding response is empty")

    if len(response) == 1 and isinstance(response[0], list):
        response = response[0]

    if any(isinstance(item, list) for item in response):
        raise RuntimeError(
            f"{source_name} embedding response appears token-level or batched; "
            "expected one pooled vector"
        )

    try:
        return [float(value) for value in response]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{source_name} embedding response contains non-numeric values"
        ) from exc


def get_huggingface_embedding(
    *,
    hf_token: str,
    hf_provider: str,
    model: str,
    text: str,
    dimensions: int,
) -> tuple[list[float], float]:
    client = InferenceClient(provider=hf_provider, api_key=hf_token)
    start = time.perf_counter()
    response = client.feature_extraction(
        [text],
        model=model,
        dimensions=dimensions,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return normalize_single_embedding(response, "Hugging Face"), elapsed_ms


def get_openrouter_embedding(
    *,
    openrouter_api_key: str,
    model: str,
    text: str,
    dimensions: int,
    provider: str | None,
) -> tuple[list[float], float]:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The openai package is not installed. Run: "
            "source .venv/bin/activate && pip install -r requirements.txt"
        ) from exc

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "input": text,
        "dimensions": dimensions,
        "encoding_format": "float",
    }
    if provider:
        kwargs["extra_body"] = {"provider": {"only": [provider]}}

    start = time.perf_counter()
    response = client.embeddings.create(**kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if not response.data:
        raise RuntimeError("OpenRouter embedding response contained no data")

    return normalize_single_embedding(
        response.data[0].embedding, "OpenRouter"
    ), elapsed_ms


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return math.nan
    return dot_product / (left_norm * right_norm)


def first_difference(
    left: list[float],
    right: list[float],
) -> tuple[int, float | None, float | None] | None:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index, left_value, right_value

    if len(left) != len(right):
        index = min(len(left), len(right))
        left_value = left[index] if index < len(left) else None
        right_value = right[index] if index < len(right) else None
        return index, left_value, right_value

    return None


def print_comparison(
    *,
    text: str,
    hf_model: str,
    hf_provider: str,
    openrouter_model: str,
    openrouter_provider: str | None,
    dimensions: int,
    hf_vector: list[float],
    hf_elapsed_ms: float,
    openrouter_vector: list[float],
    openrouter_elapsed_ms: float,
) -> bool:
    exact_equal = hf_vector == openrouter_vector
    same_dimensions = len(hf_vector) == len(openrouter_vector)
    paired_count = min(len(hf_vector), len(openrouter_vector))
    abs_diffs = [
        abs(hf_vector[index] - openrouter_vector[index])
        for index in range(paired_count)
    ]

    print(f"Text: {text}")
    print(f"HF model: {hf_model}")
    print(f"HF provider: {hf_provider}")
    print(f"OpenRouter model: {openrouter_model}")
    if openrouter_provider:
        print(f"OpenRouter provider: {openrouter_provider}")
    print(f"Dimensions requested: {dimensions}")
    print()
    print(f"HF response time: {hf_elapsed_ms:.2f} ms")
    print(f"OpenRouter response time: {openrouter_elapsed_ms:.2f} ms")
    if hf_elapsed_ms < openrouter_elapsed_ms:
        print(
            "Faster response: Hugging Face "
            f"by {openrouter_elapsed_ms - hf_elapsed_ms:.2f} ms"
        )
    elif openrouter_elapsed_ms < hf_elapsed_ms:
        print(
            "Faster response: OpenRouter "
            f"by {hf_elapsed_ms - openrouter_elapsed_ms:.2f} ms"
        )
    else:
        print("Faster response: tie")
    print()
    print(f"HF vector length: {len(hf_vector)}")
    print(f"OpenRouter vector length: {len(openrouter_vector)}")
    print()
    print(f"Exact equal: {str(exact_equal).lower()}")

    if same_dimensions:
        print(
            f"Cosine similarity: {cosine_similarity(hf_vector, openrouter_vector):.8f}"
        )
    else:
        print("Cosine similarity: unavailable (dimension mismatch)")

    print(f"Max abs diff: {max(abs_diffs) if abs_diffs else math.nan:.8f}")
    print(
        "Mean abs diff: "
        f"{sum(abs_diffs) / len(abs_diffs) if abs_diffs else math.nan:.8f}"
    )

    difference = first_difference(hf_vector, openrouter_vector)
    if difference is None:
        print("First differing index: none")
    else:
        index, hf_value, openrouter_value = difference
        print(f"First differing index: {index}")
        print(f"HF[{index}]: {hf_value}")
        print(f"OR[{index}]: {openrouter_value}")

    return exact_equal


def find_dimension_errors(
    *,
    requested_dimensions: int,
    hf_vector: list[float],
    openrouter_vector: list[float],
) -> list[str]:
    errors = []
    if len(hf_vector) != requested_dimensions:
        errors.append(
            "Hugging Face returned "
            f"{len(hf_vector)} dimensions, expected {requested_dimensions}"
        )
    if len(openrouter_vector) != requested_dimensions:
        errors.append(
            "OpenRouter returned "
            f"{len(openrouter_vector)} dimensions, expected {requested_dimensions}"
        )
    if len(hf_vector) != len(openrouter_vector):
        errors.append(
            "Provider dimension mismatch: "
            f"Hugging Face={len(hf_vector)}, OpenRouter={len(openrouter_vector)}"
        )
    return errors


def main() -> int:
    load_dotenv()

    try:
        args = parse_args()
        if args.dimensions <= 0:
            raise RuntimeError("--dimensions must be a positive integer")

        hf_token = require_env("HF_TOKEN")
        openrouter_api_key = require_env("OPENROUTER_API_KEY")

        hf_vector, hf_elapsed_ms = get_huggingface_embedding(
            hf_token=hf_token,
            hf_provider=args.hf_provider,
            model=args.hf_model,
            text=args.text,
            dimensions=args.dimensions,
        )
        openrouter_vector, openrouter_elapsed_ms = get_openrouter_embedding(
            openrouter_api_key=openrouter_api_key,
            model=args.openrouter_model,
            text=args.text,
            dimensions=args.dimensions,
            provider=args.openrouter_provider,
        )

        exact_equal = print_comparison(
            text=args.text,
            hf_model=args.hf_model,
            hf_provider=args.hf_provider,
            openrouter_model=args.openrouter_model,
            openrouter_provider=args.openrouter_provider,
            dimensions=args.dimensions,
            hf_vector=hf_vector,
            hf_elapsed_ms=hf_elapsed_ms,
            openrouter_vector=openrouter_vector,
            openrouter_elapsed_ms=openrouter_elapsed_ms,
        )

        dimension_errors = find_dimension_errors(
            requested_dimensions=args.dimensions,
            hf_vector=hf_vector,
            openrouter_vector=openrouter_vector,
        )
        if dimension_errors:
            print(file=sys.stderr)
            for error in dimension_errors:
                print(f"Error: {error}", file=sys.stderr)
            return EXIT_FAILURE

        return EXIT_EXACT_MATCH if exact_equal else EXIT_DIFFERENT
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
