import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from google.genai import errors as genai_errors

from app.classifier import (
    get_classification_cache_headers,
    get_embedding,
    perform_classification,
    sanitize_query_text,
)
from app.classifier_config import CLASSIFIER_CONFIG


def _first_classifier_with_version() -> tuple[str, str]:
    classifier_type, config = next(
        (name, cfg) for name, cfg in CLASSIFIER_CONFIG.items() if cfg.get("versions")
    )
    version = next(iter(config["versions"]))
    return classifier_type, version


class SanitizeQueryTextTests(unittest.TestCase):
    def test_accepts_long_benign_text(self) -> None:
        query = ("industrial pump assembly " * 100).strip()

        sanitized = sanitize_query_text(query)

        self.assertEqual(sanitized, query)

    def test_accepts_numeric_classification_codes(self) -> None:
        self.assertEqual(sanitize_query_text("8471.30-0000"), "8471.30-0000")

    def test_rejects_suspicious_percent_encoding_pattern(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            sanitize_query_text("25252525")

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("suspicious URL encoding patterns", ctx.exception.detail)

    def test_rejects_too_short_query(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            sanitize_query_text("a")

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("too short", ctx.exception.detail)

    def test_rejects_too_long_query(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            sanitize_query_text("x" * 4001)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("too long", ctx.exception.detail)


class ClassificationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.classifier_type, cls.version = _first_classifier_with_version()

    def test_invalid_classifier_returns_not_found(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            perform_classification(
                embed_client=object(),
                qdrant_client=object(),
                query="industrial pump",
                classifier_type="DOES_NOT_EXIST",
                version=None,
                quantization_cache={},
                zclient=None,
            )

        self.assertEqual(ctx.exception.status_code, 404)

    def test_invalid_version_returns_not_found(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            perform_classification(
                embed_client=object(),
                qdrant_client=object(),
                query="industrial pump",
                classifier_type=self.classifier_type,
                version="missing-version",
                quantization_cache={},
                zclient=None,
            )

        self.assertEqual(ctx.exception.status_code, 404)

    def test_reranking_is_used_when_no_id_matches_exist(self) -> None:
        semantic_results = [
            {
                "id": "semantic-1",
                "score": 0.41,
                "payload": {"original_id": "1234", "class_name": "Pump body"},
            },
            {
                "id": "semantic-2",
                "score": 0.39,
                "payload": {"original_id": "5678", "class_name": "Pump casing"},
            },
        ]
        reranked_results = [
            {
                "id": "semantic-2",
                "score": 0.39,
                "zeroentropy_relevance_score": 0.91,
                "payload": {"original_id": "5678", "class_name": "Pump casing"},
            },
            {
                "id": "semantic-1",
                "score": 0.41,
                "zeroentropy_relevance_score": 0.52,
                "payload": {"original_id": "1234", "class_name": "Pump body"},
            },
        ]

        with (
            patch("app.classifier.perform_exact_id_search", return_value=[]),
            patch("app.classifier.perform_partial_id_search", return_value=[]),
            patch("app.classifier.get_embedding", return_value=[0.1, 0.2, 0.3]),
            patch(
                "app.classifier.perform_semantic_search",
                return_value=semantic_results,
            ) as semantic_mock,
            patch(
                "app.classifier.rerank_with_zeroentropy",
                return_value=reranked_results,
            ) as rerank_mock,
        ):
            result = perform_classification(
                embed_client=object(),
                qdrant_client=object(),
                query="industrial pump",
                classifier_type=self.classifier_type,
                version=self.version,
                top_k=2,
                quantization_cache={},
                zclient=object(),
            )

        self.assertEqual(
            [item["id"] for item in result["results"]],
            ["semantic-2", "semantic-1"],
        )
        self.assertEqual(result["results"][0]["score"], 0.91)
        self.assertEqual(semantic_mock.call_args.kwargs["top_k"], 2)
        rerank_mock.assert_called_once()
        self.assertEqual(rerank_mock.call_args.kwargs["rerank_top_n"], 2)


class EmbeddingAndCacheHeaderTests(unittest.TestCase):
    def test_embedding_failure_returns_stable_http_exception(self) -> None:
        client = Mock()
        client.models.embed_content.side_effect = RuntimeError("boom")

        with self.assertRaises(HTTPException) as ctx:
            get_embedding(client, "gemini-embedding-001", "industrial pump")

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(
            ctx.exception.detail,
            "Failed to generate embedding for classification",
        )

    def test_server_error_is_re_raised_for_retry_handling(self) -> None:
        client = Mock()
        client.models.embed_content.side_effect = genai_errors.ServerError(
            503,
            {"error": "temporary failure"},
        )

        with self.assertRaises(genai_errors.ServerError):
            get_embedding(client, "gemini-embedding-001", "industrial pump")

    def test_cache_headers_match_cloudflare_policy(self) -> None:
        headers = get_classification_cache_headers()

        self.assertEqual(
            headers["Cache-Control"],
            "public, max-age=60, stale-while-revalidate=600",
        )
        self.assertEqual(
            headers["Cloudflare-CDN-Cache-Control"],
            "max-age=86400, stale-while-revalidate=86400",
        )
        self.assertEqual(headers["Vary"], "Accept-Encoding")


if __name__ == "__main__":
    unittest.main()
