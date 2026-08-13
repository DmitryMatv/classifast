import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.classification_service import ClassificationOutcome
from app.classifier import perform_classification
from tests.helpers import build_classification_service


class ClassificationServiceContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_classify_returns_outcome_with_named_fields(self) -> None:
        classification_result = {
            "results": [{"score": 0.9, "payload": {}, "id": "p1"}],
            "version_config": {"base_url": "https://example.com/"},
            "version_name": "v2026",
            "collection_name": "EMDN_2026",
            "config": {},
            "query": "hypodermic needle",
        }
        service = build_classification_service()

        with patch(
            "app.classification_service.perform_classification",
            return_value=classification_result,
        ) as perform_classification_mock:
            outcome = await service.classify(
                query="hypodermic needle",
                classifier_type="EMDN",
                version="v2026",
                top_k=5,
            )

        self.assertIsInstance(outcome, ClassificationOutcome)
        self.assertEqual(outcome.results, classification_result["results"])
        self.assertEqual(
            outcome.version_config, classification_result["version_config"]
        )
        self.assertEqual(outcome.version_name, "v2026")
        self.assertEqual(outcome.collection_name, "EMDN_2026")
        self.assertEqual(outcome.query, "hypodermic needle")
        perform_classification_mock.assert_called_once()

    async def test_classify_submits_pipeline_to_executor_with_bound_infra(
        self,
    ) -> None:
        embed_client = object()
        qdrant_client = object()
        reranker = object()
        quantization_cache = {"EMDN_2026": True}
        executor = MagicMock()
        executor.run = AsyncMock(
            return_value={
                "results": [],
                "version_config": {},
                "version_name": "v2026",
                "collection_name": "EMDN_2026",
                "config": {},
                "query": "needle",
            }
        )
        service = build_classification_service(
            embed_client=embed_client,
            qdrant_client=qdrant_client,
            quantization_cache=quantization_cache,
            reranker=reranker,
            executor=executor,
        )

        with patch("app.classification_service.perform_classification") as pipeline:
            outcome = await service.classify(
                query="needle",
                classifier_type="EMDN",
                version="v2026",
                top_k=7,
            )

        self.assertIsInstance(outcome, ClassificationOutcome)
        executor.run.assert_awaited_once()
        self.assertIs(executor.run.await_args.args[0], pipeline)
        self.assertEqual(
            executor.run.await_args.kwargs,
            {
                "embed_client": embed_client,
                "qdrant_client": qdrant_client,
                "query": "needle",
                "classifier_type": "EMDN",
                "version": "v2026",
                "top_k": 7,
                "quantization_cache": quantization_cache,
                "reranker": reranker,
            },
        )

    async def test_classify_submits_real_pipeline_when_not_mocked(self) -> None:
        executor = MagicMock()
        executor.run = AsyncMock(
            return_value={
                "results": [],
                "version_config": {},
                "version_name": "v1",
                "collection_name": "test",
                "config": {},
                "query": "q",
            }
        )
        service = build_classification_service(executor=executor)

        await service.classify(query="q", classifier_type="UNSPSC")

        self.assertIs(executor.run.await_args.args[0], perform_classification)

    async def test_classify_propagates_executor_errors(self) -> None:
        executor = MagicMock()
        executor.run = AsyncMock(side_effect=RuntimeError("worker down"))
        service = build_classification_service(executor=executor)

        with self.assertRaisesRegex(RuntimeError, "worker down"):
            await service.classify(query="q", classifier_type="UNSPSC")

    async def test_inline_executor_runs_pipeline_on_calling_thread(self) -> None:
        service = build_classification_service()

        with patch(
            "app.classification_service.perform_classification",
            return_value={
                "results": [],
                "version_config": {},
                "version_name": "v1",
                "collection_name": "test",
                "config": {},
                "query": "q",
            },
        ) as pipeline:
            await service.classify(query="q", classifier_type="UNSPSC")

        pipeline.assert_called_once()


if __name__ == "__main__":
    unittest.main()
