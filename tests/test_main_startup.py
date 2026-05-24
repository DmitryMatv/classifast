import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI

from app import main


class MainStartupClientTests(unittest.TestCase):
    @patch.dict(main.os.environ, {}, clear=True)
    def test_initialize_embed_client_returns_none_without_api_key(self):
        with self.assertLogs(main.logger, level="ERROR") as logs:
            client = main.initialize_embed_client()

        self.assertIsNone(client)
        self.assertTrue(
            any("GEMINI_API_KEY not found" in message for message in logs.output)
        )

    @patch.dict(main.os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
    @patch.object(main.genai, "Client")
    def test_initialize_embed_client_builds_client_and_checks_connection(
        self,
        client_class,
    ):
        embed_client = MagicMock()
        client_class.return_value = embed_client

        result = main.initialize_embed_client()

        self.assertIs(result, embed_client)
        client_class.assert_called_once_with(api_key="test-key")
        embed_client.models.list.assert_called_once_with()

    @patch.object(main, "validate_qdrant_collections", side_effect=Exception("boom"))
    @patch.object(main, "get_existing_qdrant_collections", return_value={"products"})
    @patch.object(main, "QdrantClient")
    def test_initialize_qdrant_client_wraps_startup_errors(
        self,
        qdrant_client_class,
        get_existing_collections,
        validate_collections,
    ):
        qdrant_client = MagicMock()
        qdrant_client_class.return_value = qdrant_client

        with self.assertRaisesRegex(
            RuntimeError,
            "Failed to initialize Qdrant client",
        ):
            main.initialize_qdrant_client()

        get_existing_collections.assert_called_once_with(qdrant_client)
        validate_collections.assert_called_once_with(qdrant_client, {"products"})

    @patch.object(main, "provision_payload_indexes")
    @patch.object(
        main,
        "CLASSIFIER_CONFIG",
        {
            "TEST": {
                "embed_dims": 128,
                "versions": {
                    "present": {"collection_name": "products"},
                    "missing": {"collection_name": "missing_products"},
                    "empty": {},
                },
            }
        },
    )
    def test_validate_qdrant_collections_warns_skips_and_provisions(
        self,
        provision_payload_indexes,
    ):
        qdrant_client = MagicMock()
        collection_info = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors={"size": 128}),
                quantization_config=object(),
            )
        )
        qdrant_client.get_collection.return_value = collection_info

        with self.assertLogs(main.logger, level="WARNING") as logs:
            cache = main.validate_qdrant_collections(qdrant_client, {"products"})

        self.assertEqual(cache, {"products": True})
        qdrant_client.get_collection.assert_called_once_with("products")
        provision_payload_indexes.assert_called_once_with(qdrant_client, "products")
        self.assertTrue(any("missing_products" in message for message in logs.output))

    def test_assign_startup_clients_writes_expected_app_state(self):
        app = FastAPI()
        clients = main.StartupClients(
            embed_client=object(),
            qdrant_client=object(),
            collection_quantization_cache={"products": False},
            redis_client=object(),
            zclient=object(),
        )

        main.assign_startup_clients(app, clients)

        self.assertIs(app.state.embed_client, clients.embed_client)
        self.assertIs(app.state.zclient, clients.zclient)
        self.assertIs(app.state.qdrant_client, clients.qdrant_client)
        self.assertEqual(
            app.state.collection_quantization_cache,
            {"products": False},
        )
        self.assertIs(app.state.redis_client, clients.redis_client)


class MainStartupAsyncClientTests(unittest.IsolatedAsyncioTestCase):
    @patch.object(main.redis, "Redis")
    async def test_initialize_redis_client_awaits_awaitable_ping(self, redis_class):
        redis_client = MagicMock()
        redis_client.ping.return_value = asyncio.sleep(0)
        redis_class.return_value = redis_client

        result = await main.initialize_redis_client()

        self.assertIs(result, redis_client)
        redis_client.ping.assert_called_once_with()
        redis_client.close.assert_not_called()

    @patch.object(main.redis, "Redis")
    async def test_initialize_redis_client_accepts_sync_ping(self, redis_class):
        redis_client = MagicMock()
        redis_client.ping.return_value = True
        redis_class.return_value = redis_client

        result = await main.initialize_redis_client()

        self.assertIs(result, redis_client)
        redis_client.ping.assert_called_once_with()
        redis_client.close.assert_not_called()

    @patch.object(main.redis, "Redis")
    async def test_initialize_redis_client_closes_on_failure(self, redis_class):
        redis_client = MagicMock()
        redis_client.ping.side_effect = Exception("redis down")
        redis_client.close = AsyncMock()
        redis_class.return_value = redis_client

        result = await main.initialize_redis_client()

        self.assertIsNone(result)
        redis_client.close.assert_awaited_once_with()

    async def test_close_startup_clients_closes_qdrant_and_redis(self):
        qdrant_client = MagicMock()
        redis_client = MagicMock()
        redis_client.close = AsyncMock()
        clients = main.StartupClients(
            embed_client=None,
            qdrant_client=qdrant_client,
            collection_quantization_cache={},
            redis_client=redis_client,
            zclient=None,
        )

        with self.assertLogs(main.logger, level="INFO") as logs:
            await main.close_startup_clients(clients)

        qdrant_client.close.assert_called_once_with()
        redis_client.close.assert_awaited_once_with()
        self.assertTrue(
            any("Qdrant client closed" in message for message in logs.output)
        )
        self.assertTrue(
            any("Redis client closed" in message for message in logs.output)
        )


if __name__ == "__main__":
    unittest.main()
