import unittest
from unittest.mock import patch

import pandas as pd

from mapping import generate_cpv_unspsc_map, generate_unspsc_cpv_map


class CPVHierarchyTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.code_to_description = {
            "03000000-1": "Agricultural, farming, fishing, forestry and related products",
            "03100000-2": "Agricultural and horticultural products",
            "03110000-5": "Crops, products of market gardening and horticulture",
            "03111000-2": "Seeds",
            "03111110-7": "Soybean seeds",
        }
        cls.parent_index = generate_cpv_unspsc_map.build_parent_index(
            cls.code_to_description
        )

    def test_division_returns_description_only(self) -> None:
        text = generate_cpv_unspsc_map.build_hierarchy_text(
            "03000000-1",
            self.code_to_description["03000000-1"],
            self.code_to_description,
            self.parent_index,
        )

        self.assertEqual(
            text,
            "Agricultural, farming, fishing, forestry and related products",
        )

    def test_group_includes_division_only(self) -> None:
        text = generate_cpv_unspsc_map.build_hierarchy_text(
            "03100000-2",
            self.code_to_description["03100000-2"],
            self.code_to_description,
            self.parent_index,
        )

        self.assertEqual(
            text,
            "Agricultural and horticultural products (Hierarchy: Agricultural, farming, fishing, forestry and related products)",
        )
        self.assertNotIn("Hierarchy: Agricultural and horticultural products", text)

    def test_class_includes_group_and_division_only(self) -> None:
        text = generate_cpv_unspsc_map.build_hierarchy_text(
            "03110000-5",
            self.code_to_description["03110000-5"],
            self.code_to_description,
            self.parent_index,
        )

        self.assertEqual(
            text,
            "Crops, products of market gardening and horticulture (Hierarchy: Agricultural and horticultural products < Agricultural, farming, fishing, forestry and related products)",
        )
        self.assertNotIn(
            "Hierarchy: Crops, products of market gardening and horticulture", text
        )

    def test_category_includes_class_group_and_division_only(self) -> None:
        text = generate_cpv_unspsc_map.build_hierarchy_text(
            "03111000-2",
            self.code_to_description["03111000-2"],
            self.code_to_description,
            self.parent_index,
        )

        self.assertEqual(
            text,
            "Seeds (Hierarchy: Crops, products of market gardening and horticulture < Agricultural and horticultural products < Agricultural, farming, fishing, forestry and related products)",
        )
        self.assertNotIn("Hierarchy: Seeds", text)

    def test_subcategory_includes_all_true_ancestors(self) -> None:
        text = generate_cpv_unspsc_map.build_hierarchy_text(
            "03111110-7",
            self.code_to_description["03111110-7"],
            self.code_to_description,
            self.parent_index,
        )

        self.assertEqual(
            text,
            "Soybean seeds (Hierarchy: Seeds < Crops, products of market gardening and horticulture < Agricultural and horticultural products < Agricultural, farming, fishing, forestry and related products)",
        )
        self.assertNotIn("Hierarchy: Soybean seeds", text)


class UNSPSCRerankTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.unspsc_lookup = {
            "10000000": {
                "level": "segment",
                "title": "Live Plant and Animal Material and Accessories and Supplies",
                "definition": "Live plant and animal material used for agricultural production.",
                "segment_title": "Live Plant and Animal Material and Accessories and Supplies",
                "family_title": "",
                "class_title": "",
            },
            "10100000": {
                "level": "family",
                "title": "Seeds and bulbs and seedlings and cuttings",
                "definition": "Seeds, bulbs, seedlings and cuttings used for propagation.",
                "segment_title": "Live Plant and Animal Material and Accessories and Supplies",
                "family_title": "Seeds and bulbs and seedlings and cuttings",
                "class_title": "",
            },
            "10110000": {
                "level": "class",
                "title": "Seeds and seedlings and bulbs",
                "definition": "Seeds and seedlings and bulbs used in agriculture.",
                "segment_title": "Live Plant and Animal Material and Accessories and Supplies",
                "family_title": "Seeds and bulbs and seedlings and cuttings",
                "class_title": "Seeds and seedlings and bulbs",
            },
            "10111507": {
                "level": "commodity",
                "title": "Soybean seed",
                "definition": "Soybean seeds for planting.",
                "segment_title": "Live Plant and Animal Material and Accessories and Supplies",
                "family_title": "Seeds and bulbs and seedlings and cuttings",
                "class_title": "Seeds and seedlings and bulbs",
            },
            "10111508": {
                "level": "commodity",
                "title": "Corn seed",
                "definition": "",
                "segment_title": "Live Plant and Animal Material and Accessories and Supplies",
                "family_title": "Seeds and bulbs and seedlings and cuttings",
                "class_title": "Seeds and seedlings and bulbs",
            },
        }

    def test_segment_text_includes_definition(self) -> None:
        text = generate_cpv_unspsc_map.build_unspsc_hierarchy_text(
            "10000000", self.unspsc_lookup
        )

        self.assertEqual(
            text,
            "Live Plant and Animal Material and Accessories and Supplies (Definition: Live plant and animal material used for agricultural production.)",
        )

    def test_family_text_includes_definition_and_parent(self) -> None:
        text = generate_cpv_unspsc_map.build_unspsc_hierarchy_text(
            "10100000", self.unspsc_lookup
        )

        self.assertEqual(
            text,
            "Seeds and bulbs and seedlings and cuttings (Definition: Seeds, bulbs, seedlings and cuttings used for propagation.) (Parent category: Live Plant and Animal Material and Accessories and Supplies)",
        )

    def test_class_text_includes_definition_and_parents_only(self) -> None:
        text = generate_cpv_unspsc_map.build_unspsc_hierarchy_text(
            "10110000", self.unspsc_lookup
        )

        self.assertEqual(
            text,
            "Seeds and seedlings and bulbs (Definition: Seeds and seedlings and bulbs used in agriculture.) (Hierarchy: Seeds and bulbs and seedlings and cuttings < Live Plant and Animal Material and Accessories and Supplies)",
        )
        self.assertNotIn("Hierarchy: Seeds and seedlings and bulbs", text)

    def test_commodity_text_includes_definition_and_full_parent_chain(self) -> None:
        text = generate_cpv_unspsc_map.build_unspsc_hierarchy_text(
            "10111507", self.unspsc_lookup
        )

        self.assertEqual(
            text,
            "Soybean seed (Definition: Soybean seeds for planting.) (Hierarchy: Seeds and seedlings and bulbs < Seeds and bulbs and seedlings and cuttings < Live Plant and Animal Material and Accessories and Supplies)",
        )
        self.assertNotIn("Hierarchy: Soybean seed", text)

    def test_missing_definition_does_not_emit_empty_fragment(self) -> None:
        text = generate_cpv_unspsc_map.build_unspsc_hierarchy_text(
            "10111508", self.unspsc_lookup
        )

        self.assertEqual(
            text,
            "Corn seed (Hierarchy: Seeds and seedlings and bulbs < Seeds and bulbs and seedlings and cuttings < Live Plant and Animal Material and Accessories and Supplies)",
        )
        self.assertNotIn("Definition:", text)

    def test_unknown_code_returns_empty_string(self) -> None:
        self.assertEqual(
            generate_cpv_unspsc_map.build_unspsc_hierarchy_text(
                "99999999", self.unspsc_lookup
            ),
            "",
        )


class UNSPSCRerankDocumentFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.unspsc_lookup = {
            "10111507": {
                "level": "commodity",
                "title": "Soybean seed",
                "definition": "Soybean seeds for planting.",
                "segment_title": "Live Plant and Animal Material and Accessories and Supplies",
                "family_title": "Seeds and bulbs and seedlings and cuttings",
                "class_title": "Seeds and seedlings and bulbs",
            }
        }

    def test_rich_context_enabled_uses_lookup_text(self) -> None:
        candidate = {
            "payload": {
                "original_id": "10111507",
                "class_name": "Fallback title",
                "definition": "Fallback definition",
            }
        }

        text = generate_cpv_unspsc_map.build_unspsc_rerank_document(
            candidate,
            self.unspsc_lookup,
            use_context=True,
        )

        self.assertEqual(
            text,
            "Soybean seed (Definition: Soybean seeds for planting.) (Hierarchy: Seeds and seedlings and bulbs < Seeds and bulbs and seedlings and cuttings < Live Plant and Animal Material and Accessories and Supplies)",
        )

    def test_lookup_miss_falls_back_to_payload_text(self) -> None:
        candidate = {
            "payload": {
                "original_id": "00000000",
                "class_name": "Fallback title",
                "definition": "Fallback definition",
            }
        }

        text = generate_cpv_unspsc_map.build_unspsc_rerank_document(
            candidate,
            self.unspsc_lookup,
            use_context=True,
        )

        self.assertEqual(text, "Fallback title - Definition: Fallback definition")

    def test_context_disabled_uses_payload_text(self) -> None:
        candidate = {
            "payload": {
                "original_id": "10111507",
                "class_name": "Fallback title",
                "definition": "Fallback definition",
            }
        }

        text = generate_cpv_unspsc_map.build_unspsc_rerank_document(
            candidate,
            self.unspsc_lookup,
            use_context=False,
        )

        self.assertEqual(text, "Fallback title - Definition: Fallback definition")

    def test_title_only_payload_returns_title(self) -> None:
        candidate = {
            "payload": {
                "original_id": "10111507",
                "class_name": "Fallback title",
                "definition": "",
            }
        }

        text = generate_cpv_unspsc_map.build_unspsc_rerank_document(
            candidate,
            self.unspsc_lookup,
            use_context=False,
        )

        self.assertEqual(text, "Fallback title")


class MappingWindowUsageTests(unittest.TestCase):
    def test_unspsc_to_cpv_uses_configured_windows(self) -> None:
        row = pd.Series(
            {
                "Segment": 10000000,
                "Family": pd.NA,
                "Segment Title": "Live plants",
                "Segment Definition": "Live plant material.",
                "Family Title": pd.NA,
                "Family Definition": pd.NA,
            }
        )
        matches = [
            {
                "score": 0.9,
                "payload": {"original_id": "03111000-2", "class_name": "Seeds"},
            }
        ]

        with (
            patch(
                "mapping.generate_unspsc_cpv_map.get_embedding",
                return_value=[0.1, 0.2],
            ),
            patch(
                "mapping.generate_unspsc_cpv_map.perform_semantic_search",
                return_value=matches,
            ) as semantic_mock,
            patch(
                "mapping.generate_unspsc_cpv_map.rerank_with_zeroentropy",
                return_value=matches,
            ) as rerank_mock,
        ):
            generate_unspsc_cpv_map.classify_single(
                embed_client=object(),
                qdrant_client=object(),
                zclient=object(),
                row=row,
                cpv_collection="cpv",
                cpv_config={"embed_model_name": "model", "embed_dims": 2},
                quantization_cache={},
            )
        semantic_mock.assert_called_once()
        rerank_mock.assert_called_once()
        self.assertEqual(
            semantic_mock.call_args.kwargs["top_k"],
            generate_unspsc_cpv_map.SEMANTIC_RETRIEVE_LIMIT,
        )
        self.assertEqual(
            rerank_mock.call_args.kwargs["rerank_top_n"],
            generate_unspsc_cpv_map.RERANK_CANDIDATE_LIMIT,
        )

    def test_cpv_to_unspsc_uses_configured_windows(self) -> None:
        row = pd.Series({"CODE": "03111110-7", "EN": "Soybean seeds"})
        code_to_description = {
            "03000000-1": "Agricultural, farming, fishing, forestry and related products",
            "03100000-2": "Agricultural and horticultural products",
            "03110000-5": "Crops, products of market gardening and horticulture",
            "03111000-2": "Seeds",
            "03111110-7": "Soybean seeds",
        }
        parent_index = generate_cpv_unspsc_map.build_parent_index(code_to_description)
        matches = [
            {
                "score": 0.9,
                "payload": {
                    "original_id": "10111507",
                    "class_name": "Soybean seed",
                    "definition": "Soybean seeds for planting.",
                    "id_level": "commodity",
                },
            }
        ]
        unspsc_lookup = {
            "10111507": {
                "level": "commodity",
                "title": "Soybean seed",
                "definition": "Soybean seeds for planting.",
                "segment_title": "Live Plant and Animal Material and Accessories and Supplies",
                "family_title": "Seeds and bulbs and seedlings and cuttings",
                "class_title": "Seeds and seedlings and bulbs",
            }
        }

        with (
            patch(
                "mapping.generate_cpv_unspsc_map.get_embedding",
                return_value=[0.1, 0.2],
            ),
            patch(
                "mapping.generate_cpv_unspsc_map.perform_semantic_search",
                return_value=matches,
            ) as semantic_mock,
            patch(
                "mapping.generate_cpv_unspsc_map.rerank_with_zeroentropy",
                return_value=matches,
            ) as rerank_mock,
        ):
            generate_cpv_unspsc_map.classify_single(
                embed_client=object(),
                qdrant_client=object(),
                zclient=object(),
                row=row,
                unspsc_collection="unspsc",
                quantization_cache={},
                code_to_description=code_to_description,
                parent_index=parent_index,
                unspsc_lookup=unspsc_lookup,
            )

        semantic_mock.assert_called_once()
        rerank_mock.assert_called_once()
        self.assertEqual(
            semantic_mock.call_args.kwargs["top_k"],
            generate_cpv_unspsc_map.SEMANTIC_RETRIEVE_LIMIT,
        )
        self.assertEqual(
            rerank_mock.call_args.kwargs["rerank_top_n"],
            generate_cpv_unspsc_map.RERANK_CANDIDATE_LIMIT,
        )


if __name__ == "__main__":
    unittest.main()
