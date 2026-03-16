import unittest
from unittest.mock import patch

from app.classifier import perform_classification
from app.classifier_config import CLASSIFIER_CONFIG


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
                zclient=None,
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
            patch("app.classifier.rerank_with_zeroentropy") as rerank_mock,
        ):
            result = perform_classification(
                embed_client=object(),
                qdrant_client=object(),
                query="0008471000",
                classifier_type=self.classifier_type,
                version=self.version,
                top_k=10,
                quantization_cache={},
                zclient=object(),
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


if __name__ == "__main__":
    unittest.main()
