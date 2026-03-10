import unittest
from unittest.mock import MagicMock, patch

from qdrant_client import models

from app import main
from utilities import create_text_indexes


class QdrantStartupIndexTests(unittest.TestCase):
    def test_startup_provisions_text_indexes(self):
        client = MagicMock()

        main.provision_text_search_indexes(client, "products")

        self.assertEqual(client.create_payload_index.call_count, 2)

        for call in client.create_payload_index.call_args_list:
            self.assertEqual(call.kwargs["collection_name"], "products")
            self.assertIn(call.kwargs["field_name"], {"original_id", "class_name"})
            self.assertTrue(call.kwargs["wait"])

            field_schema = call.kwargs["field_schema"]
            self.assertIsInstance(field_schema, models.TextIndexParams)
            self.assertEqual(field_schema.type, "text")
            self.assertEqual(field_schema.tokenizer, models.TokenizerType.WORD)
            self.assertEqual(field_schema.min_token_len, 1)
            self.assertEqual(field_schema.max_token_len, 30)
            self.assertTrue(field_schema.lowercase)

    def test_existing_index_warning_mentions_manual_migration(self):
        client = MagicMock()
        client.create_payload_index.side_effect = [
            Exception("payload index already exists"),
            None,
        ]

        with self.assertLogs(main.logger, level="WARNING") as logs:
            main.provision_text_search_indexes(client, "products")

        self.assertTrue(
            any("utilities/create_text_indexes.py" in message for message in logs.output)
        )
        self.assertTrue(any("KEYWORD" in message for message in logs.output))


class QdrantMigrationUtilityTests(unittest.TestCase):
    def test_migration_deletes_then_recreates_indexes(self):
        client = MagicMock()

        success = create_text_indexes.migrate_collection_text_indexes(client, "products")

        self.assertTrue(success)
        self.assertEqual(client.delete_payload_index.call_count, 2)
        self.assertEqual(client.create_payload_index.call_count, 2)

        delete_fields = [
            call.kwargs["field_name"] for call in client.delete_payload_index.call_args_list
        ]
        create_fields = [
            call.kwargs["field_name"] for call in client.create_payload_index.call_args_list
        ]
        self.assertEqual(delete_fields, create_fields)
        self.assertEqual(delete_fields, ["original_id", "class_name"])

        for delete_call in client.delete_payload_index.call_args_list:
            self.assertEqual(delete_call.kwargs["collection_name"], "products")
            self.assertTrue(delete_call.kwargs["wait"])

        for create_call in client.create_payload_index.call_args_list:
            self.assertEqual(create_call.kwargs["collection_name"], "products")
            self.assertTrue(create_call.kwargs["wait"])
            self.assertIsInstance(create_call.kwargs["field_schema"], models.TextIndexParams)

    def test_missing_index_during_delete_is_non_fatal(self):
        client = MagicMock()
        client.delete_payload_index.side_effect = [
            Exception("Index not found"),
            None,
        ]

        success = create_text_indexes.migrate_collection_text_indexes(client, "products")

        self.assertTrue(success)
        self.assertEqual(client.create_payload_index.call_count, 2)

    def test_collection_iteration_covers_every_configured_collection(self):
        client = MagicMock()
        client.get_collection.return_value = MagicMock()
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
            create_text_indexes, "migrate_collection_text_indexes", return_value=True
        ) as migrate_collection:
            success_count, error_count = create_text_indexes.migrate_configured_collections(
                client, config
            )

        self.assertEqual((success_count, error_count), (3, 0))
        processed = [call.args[1] for call in migrate_collection.call_args_list]
        self.assertEqual(processed, ["collection_a", "collection_b", "collection_c"])


if __name__ == "__main__":
    unittest.main()
