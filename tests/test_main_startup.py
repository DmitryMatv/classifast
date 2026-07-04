import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, HTTPException, Request

from app import main


def build_request(test_app: FastAPI) -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/health",
        "raw_path": b"/health",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "app": test_app,
    }
    return Request(scope)


class MainStartupClientTests(unittest.TestCase):
    @patch.dict(main.os.environ, {}, clear=True)
    def test_initialize_embed_client_returns_none_without_api_key(self):
        with self.assertLogs(main.logger, level="ERROR") as logs:
            client = main.initialize_embed_client()

        self.assertIsNone(client)
        self.assertTrue(
            any("OPENROUTER_API_KEY not found" in message for message in logs.output)
        )

    @patch.dict(main.os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    @patch.object(main, "OpenAI")
    def test_initialize_embed_client_builds_client_with_default_base_url(
        self,
        client_class,
    ):
        embed_client = MagicMock()
        client_class.return_value = embed_client

        result = main.initialize_embed_client()

        self.assertIs(result, embed_client)
        client_class.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
            max_retries=0,
            timeout=60,
        )

    @patch.dict(
        main.os.environ,
        {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_BASE_URL": "  "},
        clear=True,
    )
    @patch.object(main, "OpenAI")
    def test_initialize_embed_client_defaults_blank_base_url_to_openrouter(
        self,
        client_class,
    ):
        embed_client = MagicMock()
        client_class.return_value = embed_client

        result = main.initialize_embed_client()

        self.assertIs(result, embed_client)
        client_class.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
            max_retries=0,
            timeout=60,
        )

    @patch.dict(
        main.os.environ,
        {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_BASE_URL": "https://example.test/api/v1",
        },
        clear=True,
    )
    @patch.object(main, "OpenAI")
    def test_initialize_embed_client_uses_configured_base_url(self, client_class):
        embed_client = MagicMock()
        client_class.return_value = embed_client

        result = main.initialize_embed_client()

        self.assertIs(result, embed_client)
        client_class.assert_called_once_with(
            base_url="https://example.test/api/v1",
            api_key="test-key",
            max_retries=0,
            timeout=60,
        )

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

    async def test_health_check_returns_healthy_when_embed_client_exists_and_qdrant_is_healthy(
        self,
    ):
        test_app = FastAPI()
        qdrant_client = MagicMock()
        test_app.state.embed_client = object()
        test_app.state.qdrant_client = qdrant_client
        request = build_request(test_app)

        result = await main.health_check(request)

        self.assertEqual(result, {"status": "healthy"})
        qdrant_client.get_collections.assert_called_once_with()

    async def test_health_check_returns_503_when_embed_client_missing(self):
        test_app = FastAPI()
        qdrant_client = MagicMock()
        test_app.state.qdrant_client = qdrant_client
        request = build_request(test_app)

        with self.assertRaises(HTTPException) as ctx:
            await main.health_check(request)

        self.assertEqual(ctx.exception.status_code, 503)
        qdrant_client.get_collections.assert_not_called()

    async def test_health_check_returns_503_when_qdrant_missing(self):
        test_app = FastAPI()
        test_app.state.embed_client = object()
        request = build_request(test_app)

        with self.assertRaises(HTTPException) as ctx:
            await main.health_check(request)

        self.assertEqual(ctx.exception.status_code, 503)

    async def test_health_check_returns_503_when_qdrant_check_fails(self):
        test_app = FastAPI()
        qdrant_client = MagicMock()
        qdrant_client.get_collections.side_effect = RuntimeError("down")
        test_app.state.embed_client = object()
        test_app.state.qdrant_client = qdrant_client
        request = build_request(test_app)

        with self.assertRaises(HTTPException) as ctx:
            await main.health_check(request)

        self.assertEqual(ctx.exception.status_code, 503)
        qdrant_client.get_collections.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
