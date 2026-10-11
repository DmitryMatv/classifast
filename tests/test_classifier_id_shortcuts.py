import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client import models

from app.classifier import (
    perform_classification,
    perform_exact_id_search,
    perform_partial_id_search,
)
from app.classifier_config import CLASSIFIER_CONFIG
from app.id_lookup import (
    ORIGINAL_ID_FIELD,
    ORIGINAL_ID_NORMALIZED_FIELD,
    ORIGINAL_ID_NORMALIZED_KEY_FIELD,
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
            patch(
                "app.classifier.perform_exact_id_search",
                return_value=(exact_results, []),
            ),
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

    def test_semantic_query_changes_embedding_and_reranking_but_not_id_lookup(self) -> None:
        semantic_results = [
            {
                "score": 0.42,
                "id": "semantic-1",
                "payload": {"original_id": "12345", "class_name": "Fasteners"},
            }
        ]
        with (
            patch("app.classifier.perform_exact_id_search", return_value=([], [])) as exact,
            patch("app.classifier.perform_partial_id_search", return_value=[]) as partial,
            patch("app.classifier.get_embedding", return_value=[0.1]) as embedding,
            patch(
                "app.classifier.perform_semantic_search", return_value=semantic_results
            ),
            patch(
                "app.classifier.rerank_candidates", return_value=semantic_results
            ) as rerank,
        ):
            result = perform_classification(
                embed_client=object(),
                qdrant_client=object(),
                query="bolt",
                semantic_query="bolt. Threaded fastener",
                classifier_type=self.classifier_type,
                version=self.version,
                reranker=object(),
            )

        self.assertEqual(result["query"], "bolt")
        self.assertEqual(exact.call_args.kwargs["query_text"], "bolt")
        self.assertEqual(partial.call_args.kwargs["normalized_query"], "bolt")
        self.assertIn("bolt. Threaded fastener", embedding.call_args.kwargs["text"])
        self.assertEqual(rerank.call_args.kwargs["query"], "bolt. Threaded fastener")

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
            patch("app.classifier.perform_exact_id_search", return_value=([], [])),
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
            patch("app.classifier.rerank_candidates") as rerank_mock,
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

        self.assertEqual(sorted(result["id"] for result in results), ["p1", "p2"])

    def test_results_rank_prefix_matches_first_then_shallower_codes(self) -> None:
        client = SimpleNamespace(
            scroll=lambda **kwargs: (
                [
                    self._point("commodity", "43211503"),
                    self._point("suffix", "10004321"),
                    self._point("class", "43211500"),
                    self._point("family", "43210000"),
                    self._point("sibling-class", "43211400"),
                ],
                None,
            )
        )

        results = perform_partial_id_search(client, "products", "4321")

        self.assertEqual(
            [result["id"] for result in results],
            ["family", "sibling-class", "class", "commodity", "suffix"],
        )

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


class ExactOriginalIdSearchTests(unittest.TestCase):
    def _client(self, stored_ids: dict[str, str], captured: dict) -> SimpleNamespace:
        def scroll(**kwargs):
            captured.update(kwargs)
            wanted = set(kwargs["scroll_filter"].must[0].match.any)
            points = [
                SimpleNamespace(id=point_id, payload={ORIGINAL_ID_FIELD: original_id})
                for point_id, original_id in stored_ids.items()
                if normalize_original_id_for_lookup(original_id) in wanted
            ]
            start = kwargs.get("offset") or 0
            end = start + kwargs["limit"]
            return points[start:end], end if end < len(points) else None

        return SimpleNamespace(scroll=scroll)

    def test_filter_matches_normalized_key_field_exactly(self) -> None:
        captured: dict = {}

        perform_exact_id_search(self._client({}, captured), "products", "43.21.15.03")

        condition = captured["scroll_filter"].must[0]
        self.assertEqual(condition.key, ORIGINAL_ID_NORMALIZED_KEY_FIELD)
        self.assertIsInstance(condition.match, models.MatchAny)
        self.assertIn("43211503", condition.match.any)

    def test_formatted_query_is_exact_match(self) -> None:
        client = self._client({"commodity": "43211503"}, {})

        for query in ("43211503", "4321 1503", "43.21.15.03", "43-21-15-03"):
            with self.subTest(query=query):
                exact, same_node = perform_exact_id_search(client, "products", query)
                self.assertEqual([r["id"] for r in exact], ["commodity"])
                self.assertEqual([r["score"] for r in exact], [1.0])
                self.assertEqual(same_node, [])

    def test_exact_match_beyond_first_scroll_page_is_found(self) -> None:
        colliding = [
            "0" * leading + "43" + "0" * trailing
            for leading in range(13)
            for trailing in range(13)
        ]
        stored = {
            f"collision-{i}": code
            for i, code in enumerate(colliding)
            if code != "00430"
        }
        stored["exact"] = "00430"
        client = self._client(stored, {})

        exact, _ = perform_exact_id_search(client, "products", "00430")

        self.assertEqual([r["id"] for r in exact], ["exact"])

    def test_case_insensitive_alphanumeric_code_is_exact_match(self) -> None:
        client = self._client({"etim": "EC000042"}, {})

        exact, _ = perform_exact_id_search(client, "products", "ec000042")

        self.assertEqual([r["id"] for r in exact], ["etim"])

    def test_zero_padding_difference_is_same_node_not_exact(self) -> None:
        client = self._client({"segment": "43000000"}, {})

        exact, same_node = perform_exact_id_search(client, "products", "43")

        self.assertEqual(exact, [])
        self.assertEqual([r["id"] for r in same_node], ["segment"])
        self.assertEqual([r["score"] for r in same_node], [0.9])

    def test_dropped_leading_zero_is_same_node(self) -> None:
        client = self._client({"heading": "0101"}, {})

        exact, same_node = perform_exact_id_search(client, "products", "101")

        self.assertEqual(exact, [])
        self.assertEqual([r["id"] for r in same_node], ["heading"])

    def test_cpv_code_without_leading_zero_is_same_node(self) -> None:
        client = self._client({"cpv": "03111000-2"}, {})

        exact, same_node = perform_exact_id_search(client, "products", "3111000-2")

        self.assertEqual(exact, [])
        self.assertEqual([r["id"] for r in same_node], ["cpv"])

    def test_cpv_code_with_leading_zero_stays_exact(self) -> None:
        client = self._client({"cpv": "03111000-2"}, {})

        exact, _ = perform_exact_id_search(client, "products", "031110002")

        self.assertEqual([r["id"] for r in exact], ["cpv"])

    def test_leading_zero_probe_does_not_pull_unrelated_codes(self) -> None:
        client = self._client({"other": "43211520"}, {})

        self.assertEqual(
            perform_exact_id_search(client, "products", "43211502"), ([], [])
        )

    def test_typed_leading_zero_must_exist_in_stored_code(self) -> None:
        client = self._client(
            {"segment-10": "10000000", "sector-11": "11", "segment-43": "43000000"},
            {},
        )

        for query in ("01", "011", "0043"):
            with self.subTest(query=query):
                self.assertEqual(
                    perform_exact_id_search(client, "products", query), ([], [])
                )

    def test_key_collision_without_right_padding_is_not_same_node(self) -> None:
        client = self._client({"chapter-01": "01", "chapter-10": "10"}, {})

        exact, same_node = perform_exact_id_search(client, "products", "1000")

        self.assertEqual(exact, [])
        self.assertEqual(same_node, [])

    def test_non_latin_description_with_digits_skips_lookup(self) -> None:
        captured: dict = {}
        client = self._client({"isic": "10", "hs": "3101"}, captured)

        for query in ("10 стульев", "椅子 3101", "café 10"):
            with self.subTest(query=query):
                self.assertEqual(
                    perform_exact_id_search(client, "products", query), ([], [])
                )
        self.assertEqual(captured, {})

    def test_query_without_alphanumerics_skips_lookup(self) -> None:
        def scroll(**kwargs):
            raise AssertionError("scroll must not run for an empty key")

        result = perform_exact_id_search(
            SimpleNamespace(scroll=scroll), "products", "--"
        )

        self.assertEqual(result, ([], []))


class SameNodeOrderingTests(unittest.TestCase):
    def _result(self, point_id: str, original_id: str) -> dict:
        return {"score": 0.9, "id": point_id, "payload": {"original_id": original_id}}

    def _classify(self, query: str, same_node: list, prefix: list) -> list[str]:
        classifier_type, version = _first_classifier_with_version()
        with (
            patch(
                "app.classifier.perform_exact_id_search",
                return_value=([], same_node),
            ),
            patch(
                "app.classifier.perform_partial_id_search", return_value=prefix
            ),
            patch("app.classifier.get_embedding", return_value=[0.1, 0.2]),
            patch(
                "app.classifier.perform_semantic_search",
                return_value=[
                    {"score": 0.5, "id": "semantic", "payload": {}},
                ],
            ),
        ):
            result = perform_classification(
                embed_client=object(),
                qdrant_client=object(),
                query=query,
                classifier_type=classifier_type,
                version=version,
                top_k=10,
                quantization_cache={},
                reranker=None,
            )
        return [item["id"] for item in result["results"]]

    def test_same_node_match_leads_prefix_matches_without_duplicates(self) -> None:
        family = self._result("family", "43210000")

        ids = self._classify(
            "4321",
            [family],
            [self._result("class", "43211500"), family],
        )

        self.assertEqual(ids, ["family", "class", "semantic"])

    def test_non_latin_description_reaches_semantic_search(self) -> None:
        classifier_type, version = _first_classifier_with_version()
        client = SimpleNamespace(
            scroll=lambda **kwargs: (
                [SimpleNamespace(id="code-10", payload={ORIGINAL_ID_FIELD: "10"})],
                None,
            )
        )
        with (
            patch("app.classifier.perform_partial_id_search", return_value=[]),
            patch("app.classifier.get_embedding", return_value=[0.1, 0.2]),
            patch(
                "app.classifier.perform_semantic_search",
                return_value=[{"score": 0.5, "id": "semantic", "payload": {}}],
            ),
        ):
            result = perform_classification(
                embed_client=object(),
                qdrant_client=client,
                query="10 стульев",
                classifier_type=classifier_type,
                version=version,
                top_k=3,
                quantization_cache={},
                reranker=None,
            )

        self.assertEqual([item["id"] for item in result["results"]], ["semantic"])

    def test_non_latin_description_does_not_pin_partial_code_matches(self) -> None:
        ids = self._classify("椅子 3101", [], [self._result("hs-3101", "310100")])

        self.assertEqual(ids, ["semantic"])

    def test_short_query_still_surfaces_same_node_match(self) -> None:
        ids = self._classify("43", [self._result("segment", "43000000")], [])

        self.assertEqual(ids, ["segment", "semantic"])


if __name__ == "__main__":
    unittest.main()
