import unittest
from unittest.mock import MagicMock, patch

from qdrant_client import models

from app import main
from app.id_lookup import (
    ORIGINAL_ID_FIELD,
    ORIGINAL_ID_NORMALIZED_FIELD,
    ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
)
from utilities import sync_payload_indexes as create_text_indexes


class QdrantStartupIndexTests(unittest.TestCase):
    def test_startup_provisions_expected_payload_indexes(self):
        client = MagicMock()

        main.provision_payload_indexes(client, "products")

        self.assertEqual(client.create_payload_index.call_count, 4)

        created_fields = {
            call.kwargs["field_name"]: call.kwargs["field_schema"]
            for call in client.create_payload_index.call_args_list
        }
        self.assertEqual(
            set(created_fields),
            {
                ORIGINAL_ID_FIELD,
                ORIGINAL_ID_NORMALIZED_FIELD,
                ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
                "class_name",
            },
        )

        original_id_schema = created_fields[ORIGINAL_ID_FIELD]
        self.assertIsInstance(original_id_schema, models.KeywordIndexParams)
        self.assertEqual(original_id_schema.type, "keyword")

        for field_name in (
            ORIGINAL_ID_NORMALIZED_FIELD,
            ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
        ):
            normalized_schema = created_fields[field_name]
            self.assertIsInstance(normalized_schema, models.TextIndexParams)
            self.assertEqual(normalized_schema.type, "text")
            self.assertEqual(normalized_schema.tokenizer, models.TokenizerType.PREFIX)
            self.assertEqual(normalized_schema.min_token_len, 1)
            self.assertEqual(normalized_schema.max_token_len, 64)
            self.assertTrue(normalized_schema.lowercase)

        class_name_schema = created_fields["class_name"]
        self.assertIsInstance(class_name_schema, models.TextIndexParams)
        self.assertEqual(class_name_schema.type, "text")
        self.assertEqual(class_name_schema.tokenizer, models.TokenizerType.WORD)
        self.assertEqual(class_name_schema.min_token_len, 1)
        self.assertEqual(class_name_schema.max_token_len, 30)
        self.assertTrue(class_name_schema.lowercase)

    def test_existing_index_warning_mentions_text_remediation(self):
        client = MagicMock()
        client.create_payload_index.side_effect = [
            Exception("payload index already exists"),
            None,
            None,
            None,
        ]

        with self.assertLogs(main.logger, level="WARNING") as logs:
            main.provision_payload_indexes(client, "products")

        self.assertTrue(
            any(
                "utilities/create_text_indexes.py" in message
                or "utilities/sync_payload_indexes.py" in message
                for message in logs.output
            )
        )
        self.assertTrue(any("payload indexes" in message for message in logs.output))


class QdrantIndexContractTests(unittest.TestCase):
    def test_original_id_schema_is_keyword_for_exact_and_partial_contract(self):
        schema = main.get_payload_index_schema(ORIGINAL_ID_FIELD)

        self.assertIsInstance(schema, models.KeywordIndexParams)
        self.assertEqual(schema.type, "keyword")

    def test_normalized_id_schema_is_prefix_text_for_partial_contract(self):
        for field_name in (
            ORIGINAL_ID_NORMALIZED_FIELD,
            ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
        ):
            schema = main.get_payload_index_schema(field_name)

            self.assertIsInstance(schema, models.TextIndexParams)
            self.assertEqual(schema.type, "text")
            self.assertEqual(schema.tokenizer, models.TokenizerType.PREFIX)
            self.assertEqual(schema.min_token_len, 1)
            self.assertEqual(schema.max_token_len, 64)
            self.assertTrue(schema.lowercase)


class QdrantMigrationUtilityTests(unittest.TestCase):
    def make_collection_info(self, payload_schema):
        collection_info = MagicMock()
        collection_info.payload_schema = payload_schema
        return collection_info

    def make_client(self):
        client = MagicMock()
        client.scroll.return_value = ([], None)
        return client

    def make_text_index(self, **overrides):
        params = create_text_indexes.build_text_index_params().model_copy(
            update=overrides
        )
        return models.PayloadIndexInfo(
            data_type=models.PayloadSchemaType.TEXT,
            params=params,
            points=10,
        )

    def make_keyword_index(self):
        return models.PayloadIndexInfo(
            data_type=models.PayloadSchemaType.KEYWORD,
            params=models.KeywordIndexParams(type="keyword"),
            points=10,
        )

    def make_normalized_id_index(self):
        return models.PayloadIndexInfo(
            data_type=models.PayloadSchemaType.TEXT,
            params=create_text_indexes.build_normalized_original_id_text_index_params(),
            points=10,
        )

    def make_expected_payload_schema(self):
        return {
            ORIGINAL_ID_FIELD: self.make_keyword_index(),
            ORIGINAL_ID_NORMALIZED_FIELD: self.make_normalized_id_index(),
            ORIGINAL_ID_NORMALIZED_REVERSED_FIELD: self.make_normalized_id_index(),
            "class_name": self.make_text_index(),
        }

    def test_missing_field_creates_expected_schema_without_delete(self):
        client = self.make_client()
        client.get_collection.return_value = self.make_collection_info({})

        success = create_text_indexes.migrate_collection_payload_indexes(
            client, "products"
        )

        self.assertTrue(success)
        self.assertEqual(client.create_payload_index.call_count, 4)
        client.delete_payload_index.assert_not_called()

        created_fields = {
            call.kwargs["field_name"]: call.kwargs["field_schema"]
            for call in client.create_payload_index.call_args_list
        }
        self.assertIsInstance(
            created_fields[ORIGINAL_ID_FIELD],
            models.KeywordIndexParams,
        )
        self.assertEqual(
            created_fields[ORIGINAL_ID_NORMALIZED_FIELD].tokenizer,
            models.TokenizerType.PREFIX,
        )
        self.assertEqual(
            created_fields[ORIGINAL_ID_NORMALIZED_REVERSED_FIELD].tokenizer,
            models.TokenizerType.PREFIX,
        )
        self.assertIsInstance(
            created_fields["class_name"],
            models.TextIndexParams,
        )

    def test_expected_indexes_are_skipped(self):
        client = self.make_client()
        client.get_collection.return_value = self.make_collection_info(
            self.make_expected_payload_schema()
        )

        success = create_text_indexes.migrate_collection_payload_indexes(
            client, "products"
        )

        self.assertTrue(success)
        client.delete_payload_index.assert_not_called()
        client.create_payload_index.assert_not_called()

    def test_original_id_text_index_is_replaced_with_keyword(self):
        client = self.make_client()
        client.get_collection.return_value = self.make_collection_info(
            {
                ORIGINAL_ID_FIELD: self.make_text_index(),
                "class_name": self.make_text_index(),
            }
        )

        success = create_text_indexes.migrate_collection_payload_indexes(
            client, "products", [ORIGINAL_ID_FIELD]
        )

        self.assertTrue(success)
        self.assertEqual(client.delete_payload_index.call_count, 1)
        self.assertEqual(client.create_payload_index.call_count, 1)
        self.assertEqual(
            client.delete_payload_index.call_args.kwargs["field_name"],
            ORIGINAL_ID_FIELD,
        )
        created_schema = client.create_payload_index.call_args.kwargs["field_schema"]
        self.assertIsInstance(created_schema, models.KeywordIndexParams)

    def test_class_name_keyword_index_is_replaced_with_text(self):
        client = self.make_client()
        client.get_collection.return_value = self.make_collection_info(
            {
                ORIGINAL_ID_FIELD: self.make_keyword_index(),
                "class_name": self.make_keyword_index(),
            }
        )

        success = create_text_indexes.migrate_collection_payload_indexes(
            client, "products", ["class_name"]
        )

        self.assertTrue(success)
        self.assertEqual(client.delete_payload_index.call_count, 1)
        self.assertEqual(client.create_payload_index.call_count, 1)
        self.assertEqual(
            client.delete_payload_index.call_args.kwargs["field_name"],
            "class_name",
        )
        created_schema = client.create_payload_index.call_args.kwargs["field_schema"]
        self.assertIsInstance(created_schema, models.TextIndexParams)
        self.assertEqual(created_schema.tokenizer, models.TokenizerType.WORD)

    def test_noncanonical_class_name_text_index_is_replaced(self):
        client = self.make_client()
        client.get_collection.return_value = self.make_collection_info(
            {
                ORIGINAL_ID_FIELD: self.make_keyword_index(),
                "class_name": self.make_text_index(
                    tokenizer=models.TokenizerType.PREFIX
                ),
            }
        )

        success = create_text_indexes.migrate_collection_payload_indexes(
            client, "products", ["class_name"]
        )

        self.assertTrue(success)
        self.assertEqual(client.delete_payload_index.call_count, 1)
        self.assertEqual(client.create_payload_index.call_count, 1)
        self.assertEqual(
            client.delete_payload_index.call_args.kwargs["field_name"],
            "class_name",
        )

    def test_create_failure_triggers_rollback(self):
        client = self.make_client()
        previous_index = self.make_text_index()
        client.get_collection.return_value = self.make_collection_info(
            {ORIGINAL_ID_FIELD: previous_index}
        )
        client.create_payload_index.side_effect = [
            Exception("timeout"),
            None,
        ]

        success = create_text_indexes.migrate_collection_payload_indexes(
            client,
            "products",
            [ORIGINAL_ID_FIELD],
        )

        self.assertFalse(success)
        self.assertEqual(client.delete_payload_index.call_count, 1)
        self.assertEqual(client.create_payload_index.call_count, 2)
        self.assertIsInstance(
            client.create_payload_index.call_args_list[0].kwargs["field_schema"],
            models.KeywordIndexParams,
        )
        self.assertEqual(
            client.create_payload_index.call_args_list[1].kwargs["field_schema"],
            previous_index.params,
        )

    def test_create_failure_with_failed_rollback_marks_failure(self):
        client = self.make_client()
        previous_index = self.make_text_index()
        client.get_collection.return_value = self.make_collection_info(
            {ORIGINAL_ID_FIELD: previous_index}
        )
        client.create_payload_index.side_effect = [
            Exception("timeout"),
            Exception("rollback failed"),
        ]

        success = create_text_indexes.migrate_collection_payload_indexes(
            client,
            "products",
            [ORIGINAL_ID_FIELD],
        )

        self.assertFalse(success)
        self.assertEqual(client.delete_payload_index.call_count, 1)
        self.assertEqual(client.create_payload_index.call_count, 2)

    def test_normalized_id_keyword_index_is_replaced_with_prefix_text(self):
        client = self.make_client()
        client.get_collection.return_value = self.make_collection_info(
            {
                ORIGINAL_ID_NORMALIZED_FIELD: self.make_keyword_index(),
            }
        )

        success = create_text_indexes.migrate_collection_payload_indexes(
            client,
            "products",
            [ORIGINAL_ID_NORMALIZED_FIELD],
        )

        self.assertTrue(success)
        self.assertEqual(client.delete_payload_index.call_count, 1)
        self.assertEqual(client.create_payload_index.call_count, 1)
        self.assertEqual(
            client.delete_payload_index.call_args.kwargs["field_name"],
            ORIGINAL_ID_NORMALIZED_FIELD,
        )
        created_schema = client.create_payload_index.call_args.kwargs["field_schema"]
        self.assertIsInstance(created_schema, models.TextIndexParams)
        self.assertEqual(created_schema.tokenizer, models.TokenizerType.PREFIX)

    def test_backfill_computes_normalized_and_reversed_payload_values(self):
        client = self.make_client()
        client.scroll.return_value = (
            [
                MagicMock(
                    id="point-1",
                    payload={ORIGINAL_ID_FIELD: "03111000-2"},
                ),
                MagicMock(
                    id="point-2",
                    payload={
                        ORIGINAL_ID_FIELD: "EC000123",
                        ORIGINAL_ID_NORMALIZED_FIELD: "ec000123",
                        ORIGINAL_ID_NORMALIZED_REVERSED_FIELD: "321000ce",
                    },
                ),
                MagicMock(id="point-3", payload={}),
            ],
            None,
        )

        success = create_text_indexes.backfill_normalized_id_payloads(
            client,
            "products",
            batch_size=10,
        )

        self.assertTrue(success)
        client.batch_update_points.assert_called_once()
        operation = client.batch_update_points.call_args.kwargs["update_operations"][0]
        self.assertEqual(operation.set_payload.points, ["point-1"])
        self.assertEqual(
            operation.set_payload.payload,
            {
                ORIGINAL_ID_NORMALIZED_FIELD: "3111002",
                ORIGINAL_ID_NORMALIZED_REVERSED_FIELD: "2001113",
            },
        )

    def test_backfill_fails_on_unexpected_scroll_response_shape(self):
        client = MagicMock()
        client.scroll.return_value = []

        success = create_text_indexes.backfill_normalized_id_payloads(
            client,
            "products",
            batch_size=10,
        )

        self.assertFalse(success)
        client.batch_update_points.assert_not_called()

    def test_backfill_flushes_pending_operations_when_scroll_fails_mid_scan(self):
        client = MagicMock()
        client.scroll.side_effect = [
            (
                [
                    MagicMock(
                        id="point-1",
                        payload={ORIGINAL_ID_FIELD: "03111000-2"},
                    )
                ],
                "next-page",
            ),
            RuntimeError("scroll down"),
        ]

        success = create_text_indexes.backfill_normalized_id_payloads(
            client,
            "products",
            batch_size=10,
        )

        self.assertFalse(success)
        client.batch_update_points.assert_called_once()
        operation = client.batch_update_points.call_args.kwargs["update_operations"][0]
        self.assertEqual(operation.set_payload.points, ["point-1"])
        self.assertEqual(
            operation.set_payload.payload,
            {
                ORIGINAL_ID_NORMALIZED_FIELD: "3111002",
                ORIGINAL_ID_NORMALIZED_REVERSED_FIELD: "2001113",
            },
        )

    def test_empty_classifier_config_returns_no_collections(self):
        self.assertEqual(create_text_indexes.get_all_collection_names({}), [])

    def test_collection_iteration_covers_every_configured_collection(self):
        client = MagicMock()
        config = {
            "A": {
                "versions": {
                    "v1": {"collection_name": "collection_a"},
                    "v2": {"collection_name": "collection_b"},
                }
            },
            "B": {
                "versions": {
                    "v1": {"collection_name": "collection_b"},
                    "v2": {"collection_name": "collection_c"},
                }
            },
        }

        with patch.object(
            create_text_indexes,
            "migrate_collection_payload_indexes",
            return_value=True,
        ) as migrate_collection:
            success_count, error_count = (
                create_text_indexes.migrate_configured_collections(client, config)
            )

        self.assertEqual((success_count, error_count), (3, 0))
        processed = [call.args[1] for call in migrate_collection.call_args_list]
        self.assertEqual(processed, ["collection_a", "collection_b", "collection_c"])

    def test_main_returns_zero_when_no_errors(self):
        with (
            patch.object(
                create_text_indexes, "create_qdrant_client", return_value=MagicMock()
            ),
            patch.object(
                create_text_indexes,
                "migrate_configured_collections",
                return_value=(2, 0),
            ),
        ):
            self.assertEqual(create_text_indexes.main(), 0)

    def test_main_returns_one_when_errors_present(self):
        with (
            patch.object(
                create_text_indexes, "create_qdrant_client", return_value=MagicMock()
            ),
            patch.object(
                create_text_indexes,
                "migrate_configured_collections",
                return_value=(1, 1),
            ),
        ):
            self.assertEqual(create_text_indexes.main(), 1)


if __name__ == "__main__":
    unittest.main()
