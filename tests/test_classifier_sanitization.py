import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException
from httpx import Request, Response
from openai import APIStatusError, APITimeoutError

from app.classifier import (
    build_query_embedding_text,
    get_classification_cache_headers,
    get_embedding,
    perform_classification,
    rerank_with_zeroentropy,
    sanitize_query_text,
)
from app.classifier_config import CLASSIFIER_CONFIG
from app.id_lookup import normalize_original_id_for_lookup

EMBED_MODEL = "qwen/qwen3-embedding-8b"


def _embedding_response(vector):
    return SimpleNamespace(data=[SimpleNamespace(embedding=vector)])


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


class OriginalIdNormalizationTests(unittest.TestCase):
    def test_formatted_and_unformatted_ids_normalize_equally(self) -> None:
        self.assertEqual(
            normalize_original_id_for_lookup("03111000-2"),
            normalize_original_id_for_lookup("031110002"),
        )

    def test_punctuation_and_spaces_are_removed(self) -> None:
        self.assertEqual(
            normalize_original_id_for_lookup(" 03.111-000/2 "),
            "3111002",
        )

    def test_leading_and_trailing_zero_behavior_is_preserved(self) -> None:
        self.assertEqual(normalize_original_id_for_lookup("0008471000"), "8471")
        self.assertEqual(normalize_original_id_for_lookup("000"), "000")

    def test_alphanumeric_ids_casefold(self) -> None:
        self.assertEqual(normalize_original_id_for_lookup("EC000123"), "ec000123")


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
            patch(
                "app.classifier.get_embedding",
                return_value=[0.1, 0.2, 0.3],
            ) as embedding_mock,
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
        self.assertEqual(result["query"], "industrial pump")
        expected_embedding_text = build_query_embedding_text(
            "industrial pump",
            CLASSIFIER_CONFIG[self.classifier_type]["query_instruction"],
        )
        self.assertEqual(
            embedding_mock.call_args.kwargs["text"],
            expected_embedding_text,
        )
        self.assertEqual(result["results"][0]["score"], 0.91)
        self.assertEqual(semantic_mock.call_args.kwargs["top_k"], 2)
        rerank_mock.assert_called_once()
        self.assertEqual(rerank_mock.call_args.kwargs["rerank_top_n"], 2)

    def test_build_query_embedding_text_adds_instruction(self) -> None:
        result = build_query_embedding_text("industrial pump", "Find matching codes.")

        self.assertEqual(
            result,
            "Instruct: Find matching codes.\nQuery: industrial pump",
        )

    def test_build_query_embedding_text_returns_query_without_instruction(self) -> None:
        self.assertEqual(
            build_query_embedding_text("industrial pump", None),
            "industrial pump",
        )
        self.assertEqual(
            build_query_embedding_text("industrial pump", ""),
            "industrial pump",
        )

    def test_rerank_failure_preserves_semantic_scores(self) -> None:
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
        zclient = SimpleNamespace(
            models=SimpleNamespace(rerank=Mock(side_effect=RuntimeError("down")))
        )

        result = rerank_with_zeroentropy(
            zclient=zclient,
            query="industrial pump",
            candidates=semantic_results,
            top_k=2,
            rerank_top_n=2,
        )

        self.assertEqual([item["score"] for item in result], [0.41, 0.39])
        self.assertTrue(all(item["score"] != 0.0 for item in result))
        self.assertTrue(
            all("zeroentropy_relevance_score" not in item for item in result)
        )


class EmbeddingAndCacheHeaderTests(unittest.TestCase):
    def test_embedding_failure_returns_stable_http_exception(self) -> None:
        client = Mock()
        client.embeddings.create.side_effect = RuntimeError("boom")

        with self.assertRaises(HTTPException) as ctx:
            get_embedding(client, EMBED_MODEL, "industrial pump")

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(
            ctx.exception.detail,
            "Failed to generate embedding for classification",
        )

    def test_flat_embedding_response_is_returned_as_floats(self) -> None:
        client = Mock()
        client.embeddings.create.return_value = _embedding_response([1, 2.5, "3.0"])

        result = get_embedding(
            client,
            EMBED_MODEL,
            "industrial pump",
            embed_dims=3,
        )

        self.assertEqual(result, [1.0, 2.5, 3.0])
        client.embeddings.create.assert_called_once_with(
            model=EMBED_MODEL,
            input="industrial pump",
            dimensions=3,
            encoding_format="float",
        )

    def test_embedding_without_dimensions_omits_dimensions_parameter(self) -> None:
        client = Mock()
        client.embeddings.create.return_value = _embedding_response([1, 2.5, "3.0"])

        result = get_embedding(client, EMBED_MODEL, "industrial pump")

        self.assertEqual(result, [1.0, 2.5, 3.0])
        client.embeddings.create.assert_called_once_with(
            model=EMBED_MODEL,
            input="industrial pump",
            encoding_format="float",
        )

    def test_nested_single_embedding_response_is_flattened(self) -> None:
        client = Mock()
        client.embeddings.create.return_value = _embedding_response([[0.1, 0.2, 0.3]])

        result = get_embedding(
            client,
            EMBED_MODEL,
            "industrial pump",
            embed_dims=3,
        )

        self.assertEqual(result, [0.1, 0.2, 0.3])

    def test_array_like_embedding_response_is_converted(self) -> None:
        class ArrayLike:
            def tolist(self):
                return [[0.1, 0.2, 0.3]]

        client = Mock()
        client.embeddings.create.return_value = _embedding_response(ArrayLike())

        result = get_embedding(
            client,
            EMBED_MODEL,
            "industrial pump",
            embed_dims=3,
        )

        self.assertEqual(result, [0.1, 0.2, 0.3])

    def test_empty_embedding_response_returns_stable_http_exception(self) -> None:
        client = Mock()
        client.embeddings.create.return_value = _embedding_response([])

        with self.assertRaises(HTTPException) as ctx:
            get_embedding(client, EMBED_MODEL, "industrial pump")

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(
            ctx.exception.detail,
            "Failed to generate embedding for classification",
        )

    def test_empty_openrouter_data_returns_stable_http_exception(self) -> None:
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(data=[])

        with self.assertRaises(HTTPException) as ctx:
            get_embedding(client, EMBED_MODEL, "industrial pump")

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(
            ctx.exception.detail,
            "Failed to generate embedding for classification",
        )

    def test_token_level_embedding_response_returns_stable_http_exception(self) -> None:
        client = Mock()
        client.embeddings.create.return_value = _embedding_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )

        with self.assertRaises(HTTPException) as ctx:
            get_embedding(client, EMBED_MODEL, "industrial pump")

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(
            ctx.exception.detail,
            "Failed to generate embedding for classification",
        )

    def test_embedding_dimension_mismatch_returns_stable_http_exception(self) -> None:
        client = Mock()
        client.embeddings.create.return_value = _embedding_response([0.1, 0.2])

        with self.assertRaises(HTTPException) as ctx:
            get_embedding(
                client,
                EMBED_MODEL,
                "industrial pump",
                embed_dims=3,
            )

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(
            ctx.exception.detail,
            "Failed to generate embedding for classification",
        )

    def test_transient_openai_timeout_error_is_retried(self) -> None:
        client = Mock()
        client.embeddings.create.side_effect = [
            APITimeoutError(Request("POST", "https://openrouter.ai/api/v1/embeddings")),
            _embedding_response([0.1, 0.2, 0.3]),
        ]

        result = get_embedding(
            client,
            EMBED_MODEL,
            "industrial pump",
            embed_dims=3,
        )

        self.assertEqual(result, [0.1, 0.2, 0.3])
        self.assertEqual(client.embeddings.create.call_count, 2)

    def test_transient_openai_status_error_is_retried(self) -> None:
        request = Request("POST", "https://openrouter.ai/api/v1/embeddings")
        response = Response(503, request=request)
        client = Mock()
        client.embeddings.create.side_effect = [
            APIStatusError("temporary failure", response=response, body=None),
            _embedding_response([0.1, 0.2, 0.3]),
        ]

        result = get_embedding(
            client,
            EMBED_MODEL,
            "industrial pump",
            embed_dims=3,
        )

        self.assertEqual(result, [0.1, 0.2, 0.3])
        self.assertEqual(client.embeddings.create.call_count, 2)

    def test_cache_headers_match_cloudflare_policy(self) -> None:
        headers = get_classification_cache_headers()

        self.assertEqual(
            headers["Cache-Control"],
            "public, max-age=86400, stale-while-revalidate=604800",
        )
        self.assertEqual(
            headers["Cloudflare-CDN-Cache-Control"],
            "max-age=604800, stale-while-revalidate=604800",
        )
        self.assertEqual(headers["Vary"], "Accept-Encoding")


if __name__ == "__main__":
    unittest.main()
