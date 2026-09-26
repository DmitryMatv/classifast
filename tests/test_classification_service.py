import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.classification_service import ClassificationOutcome
from app.classifier import perform_classification
from app.classifier_config import CLASSIFIER_CONFIG
from app.query_enhancer import EnhancementOutcome, EnhancementStatus, QueryEnhancer
from tests.helpers import build_classification_service


class ClassificationServiceContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_beta_exact_alphanumeric_id_skips_model_and_semantic_work(self) -> None:
        exact_result = {"id": "exact", "score": 1.0, "payload": {"original_id": "SH203-C20"}}
        enhancer = MagicMock()
        enhancer.enhance = AsyncMock()
        service = build_classification_service(enhancer=enhancer)
        with (
            patch("app.classifier.perform_exact_id_search", return_value=[exact_result]) as exact,
            patch("app.classifier.perform_partial_id_search") as partial,
            patch("app.classifier.get_embedding") as embedding,
        ):
            outcome = await service.classify("SH203-C20", "ETIM", enhancement_enabled=True)
        self.assertEqual(outcome.results[0]["payload"]["original_id"], "SH203-C20")
        self.assertEqual(outcome.query, "SH203-C20")
        self.assertIs(outcome.enhancement_status, EnhancementStatus.SKIPPED)
        exact.assert_called_once()
        partial.assert_not_called()
        embedding.assert_not_called()
        enhancer.enhance.assert_not_awaited()

    async def test_beta_miss_formats_only_applied_description_input_first(self) -> None:
        query = "bolt"
        semantic = "bolt\n\nA threaded fastener"
        config = CLASSIFIER_CONFIG["UNSPSC"]
        enhancer = MagicMock()
        enhancer.enhance = AsyncMock(
            return_value=EnhancementOutcome(semantic, EnhancementStatus.APPLIED)
        )
        reranker = MagicMock()
        reranker.rerank.return_value = [0.8]
        service = build_classification_service(enhancer=enhancer, reranker=reranker)
        semantic_result = {"id": "semantic", "score": 0.4, "payload": {"class_name": "Bolts"}}
        with (
            patch("app.classifier.perform_exact_id_search", return_value=[]) as exact,
            patch("app.classifier.perform_partial_id_search", return_value=[]),
            patch("app.classifier.get_embedding", return_value=[0.1]) as embedding,
            patch("app.classifier.perform_semantic_search", return_value=[semantic_result]),
        ):
            outcome = await service.classify(query, "UNSPSC", enhancement_enabled=True)
        self.assertEqual(outcome.query, query)
        self.assertIs(outcome.enhancement_status, EnhancementStatus.APPLIED)
        exact.assert_called_once()
        enhancer.enhance.assert_awaited_once_with(query, "UNSPSC")
        self.assertEqual(
            embedding.call_args.kwargs["text"],
            f"{semantic}\n\n{config['query_instruction']}",
        )
        self.assertEqual(
            reranker.rerank.call_args.args[0],
            f"{semantic}\n\n{config['rerank_instruction']}",
        )

    async def test_beta_failure_uses_original_legacy_format(self) -> None:
        enhancer = MagicMock()
        enhancer.enhance = AsyncMock(
            return_value=EnhancementOutcome("bolt", EnhancementStatus.FAILED)
        )
        service = build_classification_service(enhancer=enhancer)
        with (
            patch("app.classifier.perform_exact_id_search", return_value=[]),
            patch("app.classifier.perform_partial_id_search", return_value=[]),
            patch("app.classifier.get_embedding", return_value=[0.1]) as embedding,
            patch("app.classifier.perform_semantic_search", return_value=[]),
        ):
            outcome = await service.classify("bolt", "UNSPSC", enhancement_enabled=True)
        self.assertIs(outcome.enhancement_status, EnhancementStatus.FAILED)
        self.assertEqual(
            embedding.call_args.kwargs["text"],
            f"Instruct: {CLASSIFIER_CONFIG['UNSPSC']['query_instruction']}\nQuery:bolt",
        )

    async def test_missing_enhancer_is_reported_as_failure(self) -> None:
        service = build_classification_service()
        with (
            patch("app.classifier.perform_exact_id_search", return_value=[]),
            patch("app.classifier.perform_partial_id_search", return_value=[]),
            patch("app.classifier.get_embedding", return_value=[0.1]),
            patch("app.classifier.perform_semantic_search", return_value=[]),
        ):
            outcome = await service.classify("bolt", "UNSPSC", enhancement_enabled=True)
        self.assertIs(outcome.enhancement_status, EnhancementStatus.FAILED)

    async def test_beta_partial_code_miss_skips_model_and_preserves_id_result(self) -> None:
        def unexpected_request(request: httpx.Request) -> httpx.Response:
            raise AssertionError("Partial code should skip OpenRouter")

        partial_result = {"id": "partial", "score": 0.9, "payload": {"original_id": "SH203-C20"}}
        async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected_request)) as client:
            service = build_classification_service(enhancer=QueryEnhancer("test", client=client))
            with (
                patch("app.classifier.perform_exact_id_search", return_value=[]) as exact,
                patch("app.classifier.perform_partial_id_search", return_value=[partial_result]) as partial,
                patch("app.classifier.get_embedding", return_value=[0.1]) as embedding,
                patch("app.classifier.perform_semantic_search", return_value=[]),
            ):
                outcome = await service.classify("SH203", "ETIM", enhancement_enabled=True)
        self.assertIs(outcome.enhancement_status, EnhancementStatus.SKIPPED)
        self.assertEqual(outcome.results[0]["payload"]["original_id"], "SH203-C20")
        self.assertEqual(outcome.query, "SH203")
        exact.assert_called_once()
        partial.assert_called_once()
        self.assertTrue(embedding.call_args.kwargs["text"].startswith("Instruct: "))

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
                "semantic_query": None,
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
