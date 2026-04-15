import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from qdrant_client import models

import embedders.embedder_remote_UNSPSC_v3 as target


def make_collection_info(size=3, distance=models.Distance.DOT):
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace(size=size, distance=distance)
            )
        )
    )


def make_record(record_id, original_id):
    return SimpleNamespace(id=record_id, payload={"original_id": original_id})


def make_response(vectors):
    return SimpleNamespace(
        embeddings=[SimpleNamespace(values=vector) for vector in vectors]
    )


class FakeClock:
    def __init__(self):
        self.current = 0.0
        self.sleeps = []

    def time(self):
        return self.current

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.current += duration


class EmbedderRemoteUnspscV3Tests(unittest.TestCase):
    def make_data(self, rows):
        return pd.DataFrame(rows)

    def make_client(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value = make_collection_info()
        return client

    def make_embed_client(self):
        return SimpleNamespace(
            models=SimpleNamespace(
                embed_content=MagicMock(return_value=make_response([[1.0, 2.0, 3.0]]))
            )
        )

    def test_existing_point_rerun_updates_payload_and_returns_true(self):
        client = self.make_client()
        client.scroll.side_effect = [([make_record("q1", "1000")], None)]
        embed_client = self.make_embed_client()

        data = self.make_data(
            [
                {
                    "id": 1,
                    "original_id": "1000",
                    "default_class": "Updated name",
                    "default_definition": "Updated definition",
                    "embedding_text": "ignored",
                    "id_level": "Commodity",
                }
            ]
        )

        with patch.object(target, "QdrantClient", return_value=client):
            result = target.create_and_populate_qdrant(
                data=data,
                collection_name="unspsc",
                vector_size=3,
                distance_metric=models.Distance.DOT,
                qdrant_path="/tmp/qdrant",
                embed_client=embed_client,
                embed_model="test-model",
            )

        self.assertTrue(result)
        client.set_payload.assert_called_once()
        payload = client.set_payload.call_args.kwargs["payload"]
        self.assertEqual(payload["id_level"], "Commodity")
        self.assertEqual(payload["class_name"], "Updated name")
        client.upsert.assert_not_called()

    def test_titled_embedding_path_counts_one_rate_limit_slot_per_text(self):
        clock = FakeClock()
        embed_client = SimpleNamespace(
            models=SimpleNamespace(
                embed_content=MagicMock(
                    side_effect=[
                        make_response([[1.0, 2.0, 3.0]]),
                        make_response([[4.0, 5.0, 6.0]]),
                        make_response([[7.0, 8.0, 9.0]]),
                    ]
                )
            )
        )
        rate_limiter = target.MinuteRateLimiter(
            rate_limit=2,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        result = target.get_embeddings_batch_sync(
            embed_client=embed_client,
            model_name="test-model",
            task_type="RETRIEVAL_DOCUMENT",
            texts=["a", "b", "c"],
            titles=["A", "B", "C"],
            embed_dims=3,
            rate_limiter=rate_limiter,
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(embed_client.models.embed_content.call_count, 3)
        self.assertEqual(clock.sleeps, [60.0])

    def test_untitled_embedding_path_uses_one_rate_limit_slot_for_batch(self):
        clock = FakeClock()
        embed_client = SimpleNamespace(
            models=SimpleNamespace(
                embed_content=MagicMock(
                    return_value=make_response(
                        [
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0],
                        ]
                    )
                )
            )
        )
        rate_limiter = target.MinuteRateLimiter(
            rate_limit=1,
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        result = target.get_embeddings_batch_sync(
            embed_client=embed_client,
            model_name="test-model",
            task_type="RETRIEVAL_QUERY",
            texts=["a", "b", "c"],
            embed_dims=3,
            rate_limiter=rate_limiter,
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(embed_client.models.embed_content.call_count, 1)
        self.assertEqual(clock.sleeps, [])
        self.assertEqual(rate_limiter.calls_this_minute, 1)

    def test_embedding_exception_allows_later_batches_but_returns_false(self):
        client = self.make_client()
        client.scroll.side_effect = [([], None)]
        embed_client = self.make_embed_client()

        data = self.make_data(
            [
                {
                    "id": 1,
                    "original_id": "1000",
                    "default_class": "A",
                    "default_definition": "A def",
                    "embedding_text": "A text",
                    "id_level": "Commodity",
                },
                {
                    "id": 2,
                    "original_id": "2000",
                    "default_class": "B",
                    "default_definition": "B def",
                    "embedding_text": "B text",
                    "id_level": "Class",
                },
            ]
        )

        with (
            patch.object(target, "QdrantClient", return_value=client),
            patch.object(
                target,
                "get_embeddings_batch_sync",
                side_effect=[RuntimeError("boom"), [[1.0, 2.0, 3.0]]],
            ),
        ):
            result = target.create_and_populate_qdrant(
                data=data,
                collection_name="unspsc",
                vector_size=3,
                distance_metric=models.Distance.DOT,
                qdrant_path="/tmp/qdrant",
                embed_client=embed_client,
                embed_model="test-model",
                embedding_batch_size=1,
            )

        self.assertFalse(result)
        self.assertEqual(client.upsert.call_count, 1)

    def test_upsert_exception_allows_later_batches_but_returns_false(self):
        client = self.make_client()
        client.scroll.side_effect = [([], None)]
        client.upsert.side_effect = [RuntimeError("boom"), None]
        embed_client = self.make_embed_client()

        data = self.make_data(
            [
                {
                    "id": 1,
                    "original_id": "1000",
                    "default_class": "A",
                    "default_definition": "A def",
                    "embedding_text": "A text",
                    "id_level": "Commodity",
                },
                {
                    "id": 2,
                    "original_id": "2000",
                    "default_class": "B",
                    "default_definition": "B def",
                    "embedding_text": "B text",
                    "id_level": "Class",
                },
            ]
        )

        with (
            patch.object(target, "QdrantClient", return_value=client),
            patch.object(
                target,
                "get_embeddings_batch_sync",
                side_effect=[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]],
            ),
        ):
            result = target.create_and_populate_qdrant(
                data=data,
                collection_name="unspsc",
                vector_size=3,
                distance_metric=models.Distance.DOT,
                qdrant_path="/tmp/qdrant",
                embed_client=embed_client,
                embed_model="test-model",
                embedding_batch_size=1,
            )

        self.assertFalse(result)
        self.assertEqual(client.upsert.call_count, 2)

    def test_embedding_count_mismatch_causes_final_failure(self):
        client = self.make_client()
        client.scroll.side_effect = [([], None)]
        embed_client = self.make_embed_client()

        data = self.make_data(
            [
                {
                    "id": 1,
                    "original_id": "1000",
                    "default_class": "A",
                    "default_definition": "A def",
                    "embedding_text": "A text",
                    "id_level": "Commodity",
                }
            ]
        )

        with (
            patch.object(target, "QdrantClient", return_value=client),
            patch.object(
                target,
                "get_embeddings_batch_sync",
                return_value=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ),
        ):
            result = target.create_and_populate_qdrant(
                data=data,
                collection_name="unspsc",
                vector_size=3,
                distance_metric=models.Distance.DOT,
                qdrant_path="/tmp/qdrant",
                embed_client=embed_client,
                embed_model="test-model",
                embedding_batch_size=1,
            )

        self.assertFalse(result)
        client.upsert.assert_not_called()

    def test_create_and_populate_qdrant_uses_passed_embed_dependencies(self):
        client = self.make_client()
        client.scroll.side_effect = [([], None)]
        embed_client = self.make_embed_client()

        data = self.make_data(
            [
                {
                    "id": 1,
                    "original_id": "1000",
                    "default_class": "A",
                    "default_definition": "A def",
                    "embedding_text": "A text",
                    "id_level": "Commodity",
                }
            ]
        )

        with (
            patch.object(target, "QdrantClient", return_value=client),
            patch.object(target, "EMBED_CLIENT", new="wrong-client"),
            patch.object(target, "EMBED_MODEL", new="wrong-model"),
        ):
            result = target.create_and_populate_qdrant(
                data=data,
                collection_name="unspsc",
                vector_size=3,
                distance_metric=models.Distance.DOT,
                qdrant_path="/tmp/qdrant",
                embed_client=embed_client,
                embed_model="passed-model",
                embedding_batch_size=1,
            )

        self.assertTrue(result)
        embed_call = embed_client.models.embed_content.call_args.kwargs
        self.assertEqual(embed_call["model"], "passed-model")

    def test_get_qdrant_collection_name_uses_env_override(self):
        with patch.dict(
            target.os.environ,
            {"QDRANT_COLLECTION_NAME": "custom_collection"},
            clear=False,
        ):
            self.assertEqual(
                target.get_qdrant_collection_name(),
                "custom_collection",
            )

    def test_get_qdrant_collection_name_uses_default_when_env_missing(self):
        with patch.dict(target.os.environ, {}, clear=True):
            self.assertEqual(
                target.get_qdrant_collection_name(),
                target.DEFAULT_QDRANT_COLLECTION_NAME,
            )

    def test_get_embeddings_batch_sync_raises_on_api_failure(self):
        embed_client = SimpleNamespace(
            models=SimpleNamespace(
                embed_content=MagicMock(side_effect=RuntimeError("Gemini unavailable"))
            )
        )

        with self.assertRaises(RuntimeError):
            target.get_embeddings_batch_sync(
                embed_client=embed_client,
                model_name="test-model",
                task_type="RETRIEVAL_QUERY",
                texts=["a"],
                embed_dims=3,
            )


if __name__ == "__main__":
    unittest.main()
