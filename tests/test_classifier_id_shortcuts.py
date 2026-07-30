import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client import models

from app.classifier import perform_classification, perform_partial_id_search
from app.classifier_config import CLASSIFIER_CONFIG
from app.id_lookup import (
    ORIGINAL_ID_FIELD,
    ORIGINAL_ID_NORMALIZED_FIELD,
    ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
    normalize_original_id_for_lookup,
    reverse_normalized_id,
)


def _first_classifier_with_version() -> tuple[str, str]:
    classifier_type, config = next(
        (name, cfg) for name, cfg in CLASSIFIER_CONFIG.items() if cfg.get("versions")
    )
    version = next(iter(config["versions"]))
    return classifier_type, version


class PerformClassificationShortcutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.classifier_type, cls.version = _first_classifier_with_version()

    def test_exact_match_short_circuits_before_semantic_search(self) -> None:
        exact_results = [
            {
                "score": 1.0,
                "id": "exact-1",
                "payload": {
                    "original_id": "8471",
                    "class_name": "Portable computers",
                    "definition": "Automatic data processing machines.",
                },
            }
        ]

        with (
            patch("app.classifier.perform_exact_id_search", return_value=exact_results),
            patch("app.classifier.perform_partial_id_search") as partial_mock,
            patch("app.classifier.get_embedding") as embedding_mock,
            patch("app.classifier.perform_semantic_search") as semantic_mock,
        ):
            result = perform_classification(
                embed_client=object(),
                qdrant_client=object(),
                query="8471",
                classifier_type=self.classifier_type,
                version=self.version,
                top_k=10,
                quantization_cache={},
                reranker=None,
            )

        self.assertEqual(result["results"], exact_results)
        partial_mock.assert_not_called()
        embedding_mock.assert_not_called()
        semantic_mock.assert_not_called()

    def test_partial_matches_still_keep_semantic_path(self) -> None:
        partial_results = [
            {
                "score": 0.9,
                "id": "partial-1",
                "payload": {
                    "original_id": "00084710",
                    "class_name": "Portable computers",
                    "definition": "Automatic data processing machines.",
                },
            }
        ]
        semantic_results = [
            {
                "score": 0.42,
                "id": "semantic-1",
                "payload": {
                    "original_id": "12345678",
                    "class_name": "Task chairs",
                    "definition": "Adjustable office chairs.",
                },
            }
        ]

        with (
            patch("app.classifier.perform_exact_id_search", return_value=[]),
            patch(
                "app.classifier.perform_partial_id_search",
                return_value=partial_results,
            ) as partial_mock,
            patch(
                "app.classifier.get_embedding", return_value=[0.1, 0.2, 0.3]
            ) as embedding_mock,
            patch(
                "app.classifier.perform_semantic_search",
                return_value=semantic_results,
            ) as semantic_mock,
            patch("app.classifier.rerank_with_huggingface") as rerank_mock,
        ):
            result = perform_classification(
                embed_client=object(),
                qdrant_client=object(),
                query="0008471000",
                classifier_type=self.classifier_type,
                version=self.version,
                top_k=10,
                quantization_cache={},
                reranker=object(),
            )

        self.assertEqual(
            [item["id"] for item in result["results"]],
            ["partial-1", "semantic-1"],
        )
        partial_mock.assert_called_once()
        embedding_mock.assert_called_once()
        semantic_mock.assert_called_once()
        self.assertEqual(semantic_mock.call_args.kwargs["top_k"], 10)
        rerank_mock.assert_not_called()


class PartialOriginalIdSearchTests(unittest.TestCase):
    def _point(self, point_id: str, original_id: str) -> SimpleNamespace:
        normalized = normalize_original_id_for_lookup(original_id)
        return SimpleNamespace(
            id=point_id,
            payload={
                ORIGINAL_ID_FIELD: original_id,
                ORIGINAL_ID_NORMALIZED_FIELD: normalized,
                ORIGINAL_ID_NORMALIZED_REVERSED_FIELD: reverse_normalized_id(
                    normalized
                ),
            },
        )

    def test_partial_search_filter_contract(self) -> None:
        captured = {}

        def scroll(**kwargs):
            captured.update(kwargs)
            return [self._point("p1", "03111000-2")], None

        client = SimpleNamespace(scroll=scroll)

        perform_partial_id_search(client, "products", "311")

        conditions = captured["scroll_filter"].should
        keys = {condition.key for condition in conditions}
        self.assertEqual(
            keys,
            {
                ORIGINAL_ID_NORMALIZED_FIELD,
                ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
            },
        )
        self.assertTrue(
            all(
                not (
                    condition.key == ORIGINAL_ID_FIELD
                    and isinstance(condition.match, models.MatchText)
                )
                for condition in conditions
            )
        )

    def test_prefix_match_accepts_formatted_stored_id(self) -> None:
        client = SimpleNamespace(
            scroll=lambda **kwargs: ([self._point("p1", "03111000-2")], None)
        )

        results = perform_partial_id_search(client, "products", "311")

        self.assertEqual([result["id"] for result in results], ["p1"])

    def test_suffix_match_accepts_reversed_field_candidates_and_deduplicates(
        self,
    ) -> None:
        duplicate = self._point("p2", "AA-1002")
        client = SimpleNamespace(
            scroll=lambda **kwargs: (
                [
                    self._point("p1", "03111000-2"),
                    duplicate,
                    duplicate,
                ],
                None,
            )
        )

        results = perform_partial_id_search(client, "products", "1002")

        self.assertEqual([result["id"] for result in results], ["p1", "p2"])

    def test_malformed_normalized_payload_does_not_drop_later_matches(self) -> None:
        malformed = self._point("bad", "BAD-100")
        malformed.payload[ORIGINAL_ID_NORMALIZED_FIELD] = 12345
        client = SimpleNamespace(
            scroll=lambda **kwargs: (
                [
                    malformed,
                    self._point("valid", "03111000-2"),
                ],
                None,
            )
        )

        results = perform_partial_id_search(client, "products", "311")

        self.assertEqual([result["id"] for result in results], ["valid"])


if __name__ == "__main__":
    unittest.main()
