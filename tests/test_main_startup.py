import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, HTTPException, Request

from app import main
from app.classification_executor import ClassificationExecutor
from app.qdrant_schema import QdrantSchemaValidationError, QdrantValidationIssue


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
        self.assertTrue(any("HF_TOKEN not found" in message for message in logs.output))

    @patch.dict(main.os.environ, {"HF_TOKEN": "test-key"}, clear=True)
    @patch.object(main, "InferenceClient")
    def test_initialize_embed_client_builds_client_with_default_provider(
        self,
        client_class,
    ):
        embed_client = MagicMock()
        client_class.return_value = embed_client

        result = main.initialize_embed_client()

        self.assertIs(result, embed_client)
        client_class.assert_called_once_with(
            provider="auto",
            api_key="test-key",
        )

    @patch.dict(
        main.os.environ,
        {"HF_TOKEN": "test-key", "HF_INFERENCE_PROVIDER": "  "},
        clear=True,
    )
    @patch.object(main, "InferenceClient")
    def test_initialize_embed_client_defaults_blank_provider_to_auto(
        self,
        client_class,
    ):
        embed_client = MagicMock()
        client_class.return_value = embed_client

        result = main.initialize_embed_client()

        self.assertIs(result, embed_client)
        client_class.assert_called_once_with(
            provider="auto",
            api_key="test-key",
        )

    @patch.dict(
        main.os.environ,
        {"HF_TOKEN": "test-key", "HF_INFERENCE_PROVIDER": "custom-provider"},
        clear=True,
    )
    @patch.object(main, "InferenceClient")
    def test_initialize_embed_client_uses_configured_provider(self, client_class):
        embed_client = MagicMock()
        client_class.return_value = embed_client

        result = main.initialize_embed_client()

        self.assertIs(result, embed_client)
        client_class.assert_called_once_with(
            provider="custom-provider",
            api_key="test-key",
        )

    @patch.dict(main.os.environ, {}, clear=True)
    def test_initialize_openrouter_reranker_returns_none_without_api_key(self):
        with self.assertLogs(main.logger, level="WARNING") as logs:
            reranker = main.initialize_openrouter_reranker()

        self.assertIsNone(reranker)
        self.assertTrue(
            any("OPENROUTER_API_KEY not found" in message for message in logs.output)
        )

    @patch.dict(main.os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    @patch.object(main, "OpenRouterReranker")
    def test_initialize_openrouter_reranker_uses_defaults(self, reranker_class):
        reranker = MagicMock()
        reranker_class.return_value = reranker

        result = main.initialize_openrouter_reranker()

        self.assertIs(result, reranker)
        reranker_class.assert_called_once_with(
            api_key="test-key",
            model_name="nvidia/llama-nemotron-rerank-vl-1b-v2:free",
            timeout_seconds=30.0,
        )

    @patch.dict(
        main.os.environ,
        {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_RERANK_MODEL": "custom/reranker",
            "OPENROUTER_RERANK_TIMEOUT_SECONDS": "12.5",
        },
        clear=True,
    )
    @patch.object(main, "OpenRouterReranker")
    def test_initialize_openrouter_reranker_uses_configuration(self, reranker_class):
        main.initialize_openrouter_reranker()

        reranker_class.assert_called_once_with(
            api_key="test-key",
            model_name="custom/reranker",
            timeout_seconds=12.5,
        )

    @patch.object(
        main, "validate_configured_collections", side_effect=Exception("boom")
    )
    @patch.object(main, "create_qdrant_client")
    def test_initialize_qdrant_client_closes_client_on_startup_errors(
        self, create_client, validate_collections
    ):
        qdrant_client = MagicMock()
        create_client.return_value = qdrant_client

        with self.assertRaisesRegex(RuntimeError, "Failed to initialize Qdrant client"):
            main.initialize_qdrant_client()

        validate_collections.assert_called_once_with(qdrant_client)
        qdrant_client.close.assert_called_once_with()

    @patch.object(main, "validate_configured_collections")
    @patch.object(main, "create_qdrant_client")
    def test_initialize_qdrant_client_returns_read_only_validation_cache(
        self, create_client, validate_collections
    ):
        qdrant_client = MagicMock()
        create_client.return_value = qdrant_client
        validate_collections.return_value = {"products": True}

        client, cache = main.initialize_qdrant_client()

        self.assertIs(client, qdrant_client)
        self.assertEqual(cache, {"products": True})
        create_client.assert_called_once_with(timeout=30)
        qdrant_client.create_payload_index.assert_not_called()
        qdrant_client.delete_payload_index.assert_not_called()
        qdrant_client.set_payload.assert_not_called()
        qdrant_client.batch_update_points.assert_not_called()

    @patch.object(main, "validate_configured_collections")
    @patch.object(main, "create_qdrant_client")
    def test_initialize_qdrant_client_closes_and_fails_on_contract_error(
        self, create_client, validate_collections
    ):
        qdrant_client = MagicMock()
        create_client.return_value = qdrant_client
        validate_collections.side_effect = QdrantSchemaValidationError(
            [QdrantValidationIssue("products", "missing_collection", "missing")]
        )

        with self.assertRaisesRegex(RuntimeError, "invalid schema"):
            main.initialize_qdrant_client()

        qdrant_client.close.assert_called_once_with()

    @patch.object(main, "validate_configured_collections")
    @patch.object(main, "create_qdrant_client")
    def test_schema_error_is_preserved_when_qdrant_cleanup_fails(
        self, create_client, validate_collections
    ):
        qdrant_client = MagicMock()
        qdrant_client.close.side_effect = RuntimeError("close failed")
        create_client.return_value = qdrant_client
        schema_error = QdrantSchemaValidationError(
            [QdrantValidationIssue("products", "missing_collection", "missing")]
        )
        validate_collections.side_effect = schema_error

        with self.assertLogs(main.logger, level="ERROR") as logs:
            with self.assertRaisesRegex(RuntimeError, "invalid schema") as ctx:
                main.initialize_qdrant_client()

        self.assertIs(ctx.exception.__cause__, schema_error)
        self.assertTrue(any("Error closing Qdrant" in line for line in logs.output))

    @patch.object(main, "validate_configured_collections")
    @patch.object(main, "create_qdrant_client")
    def test_generic_error_is_preserved_when_qdrant_cleanup_fails(
        self, create_client, validate_collections
    ):
        qdrant_client = MagicMock()
        qdrant_client.close.side_effect = RuntimeError("close failed")
        create_client.return_value = qdrant_client
        validation_error = RuntimeError("validation failed")
        validate_collections.side_effect = validation_error

        with self.assertLogs(main.logger, level="ERROR") as logs:
            with self.assertRaisesRegex(RuntimeError, "validation failed") as ctx:
                main.initialize_qdrant_client()

        self.assertIs(ctx.exception.__cause__, validation_error)
        self.assertTrue(any("Error closing Qdrant" in line for line in logs.output))

    @patch.object(main, "_close_qdrant_client_after_startup_failure")
    @patch.object(
        main, "create_qdrant_client", side_effect=RuntimeError("connect failed")
    )
    def test_client_creation_failure_does_not_attempt_cleanup(
        self, create_client, close_client
    ):
        with self.assertRaisesRegex(RuntimeError, "connect failed"):
            main.initialize_qdrant_client()

        create_client.assert_called_once_with(timeout=30)
        close_client.assert_not_called()

    def test_assign_startup_clients_writes_expected_app_state(self):
        app = FastAPI()
        clients = main.StartupClients(
            embed_client=object(),
            qdrant_client=object(),
            collection_quantization_cache={"products": False},
            redis_client=object(),
            reranker=object(),
        )
        executor = MagicMock(spec=ClassificationExecutor)

        main.assign_startup_clients(app, clients, executor)

        self.assertIs(app.state.embed_client, clients.embed_client)
        self.assertIs(app.state.reranker, clients.reranker)
        self.assertIs(app.state.qdrant_client, clients.qdrant_client)
        self.assertEqual(
            app.state.collection_quantization_cache,
            {"products": False},
        )
        self.assertIs(app.state.redis_client, clients.redis_client)
        self.assertIs(app.state.classification_executor, executor)


class MainStartupAsyncClientTests(unittest.IsolatedAsyncioTestCase):
    @patch.object(main, "initialize_startup_clients", new_callable=AsyncMock)
    @patch.object(main, "ClassificationExecutor")
    async def test_lifespan_closes_executor_when_startup_fails(
        self, executor_class, initialize_clients
    ) -> None:
        executor = MagicMock()
        executor.close = AsyncMock()
        executor_class.return_value = executor
        initialize_clients.side_effect = RuntimeError("invalid qdrant")

        with self.assertRaisesRegex(RuntimeError, "invalid qdrant"):
            async with main.lifespan(FastAPI()):
                self.fail("lifespan must not yield after startup failure")

        executor.close.assert_awaited_once_with()

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

    @patch.object(main.redis, "Redis")
    async def test_initialize_redis_client_closes_and_reraises_cancellation(
        self, redis_class
    ):
        redis_client = MagicMock()
        cancelled_ping = asyncio.get_running_loop().create_future()
        cancelled_ping.cancel()
        redis_client.ping.return_value = cancelled_ping
        redis_client.close = AsyncMock()
        redis_class.return_value = redis_client

        with self.assertRaises(asyncio.CancelledError):
            await main.initialize_redis_client()

        redis_client.close.assert_awaited_once_with()

    @patch.object(main, "initialize_redis_client", new_callable=AsyncMock)
    @patch.object(main, "initialize_qdrant_client")
    @patch.object(main, "initialize_embed_client")
    async def test_partial_startup_closes_qdrant_when_redis_step_raises(
        self, initialize_embed, initialize_qdrant, initialize_redis
    ):
        qdrant_client = MagicMock()
        initialize_qdrant.return_value = (qdrant_client, {"products": False})
        initialize_redis.side_effect = RuntimeError("redis init failed")

        with self.assertRaisesRegex(RuntimeError, "redis init failed"):
            await main.initialize_startup_clients()

        qdrant_client.close.assert_called_once_with()

    @patch.object(
        main,
        "initialize_openrouter_reranker",
        side_effect=RuntimeError("reranker init failed"),
    )
    @patch.object(main, "initialize_redis_client", new_callable=AsyncMock)
    @patch.object(main, "initialize_qdrant_client")
    @patch.object(main, "initialize_embed_client")
    async def test_partial_startup_closes_qdrant_and_redis_when_later_step_raises(
        self,
        initialize_embed,
        initialize_qdrant,
        initialize_redis,
        initialize_reranker,
    ):
        qdrant_client = MagicMock()
        redis_client = MagicMock()
        redis_client.close = AsyncMock()
        initialize_qdrant.return_value = (qdrant_client, {"products": False})
        initialize_redis.return_value = redis_client

        with self.assertRaisesRegex(RuntimeError, "reranker init failed"):
            await main.initialize_startup_clients()

        qdrant_client.close.assert_called_once_with()
        redis_client.close.assert_awaited_once_with()

    @patch.object(main, "close_startup_clients", new_callable=AsyncMock)
    @patch.object(main, "initialize_startup_clients", new_callable=AsyncMock)
    @patch.object(main, "ClassificationExecutor")
    async def test_lifespan_closes_clients_when_executor_shutdown_fails(
        self, executor_class, initialize_clients, close_clients
    ):
        clients = main.StartupClients()
        initialize_clients.return_value = clients
        executor = MagicMock()
        executor.close = AsyncMock(side_effect=RuntimeError("executor close failed"))
        executor_class.return_value = executor

        with self.assertRaisesRegex(RuntimeError, "executor close failed"):
            async with main.lifespan(FastAPI()):
                pass

        close_clients.assert_awaited_once_with(clients)

    @patch.object(main, "initialize_startup_clients", new_callable=AsyncMock)
    @patch.object(main, "ClassificationExecutor")
    async def test_lifespan_shutdown_order_is_executor_qdrant_redis(
        self, executor_class, initialize_clients
    ):
        events: list[str] = []
        qdrant_client = MagicMock()
        qdrant_client.close.side_effect = lambda: events.append("qdrant")
        redis_client = MagicMock()

        async def close_redis() -> None:
            events.append("redis")

        redis_client.close = AsyncMock(side_effect=close_redis)
        clients = main.StartupClients(
            qdrant_client=qdrant_client,
            redis_client=redis_client,
        )
        initialize_clients.return_value = clients
        executor = MagicMock()

        async def close_executor() -> None:
            events.append("executor")

        executor.close = AsyncMock(side_effect=close_executor)
        executor_class.return_value = executor

        async with main.lifespan(FastAPI()):
            pass

        self.assertEqual(events, ["executor", "qdrant", "redis"])

    async def test_close_startup_clients_closes_reranker_qdrant_and_redis(self):
        reranker = MagicMock()
        qdrant_client = MagicMock()
        redis_client = MagicMock()
        redis_client.close = AsyncMock()
        clients = main.StartupClients(
            embed_client=None,
            reranker=reranker,
            qdrant_client=qdrant_client,
            collection_quantization_cache={},
            redis_client=redis_client,
        )

        with self.assertLogs(main.logger, level="INFO") as logs:
            await main.close_startup_clients(clients)

        reranker.close.assert_called_once_with()
        qdrant_client.close.assert_called_once_with()
        redis_client.close.assert_awaited_once_with()
        self.assertTrue(
            any("OpenRouter reranker closed" in message for message in logs.output)
        )
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
