import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call

from qdrant_client import models

from app import qdrant_schema
from app.id_lookup import (
    ORIGINAL_ID_FIELD,
    ORIGINAL_ID_NORMALIZED_FIELD,
    ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
)


def classifier_test_config(collection_name: str = "products", dims: int = 128) -> dict:
    return {
        "TEST": {
            "embed_dims": dims,
            "versions": {"v1": {"collection_name": collection_name}},
        }
    }


def keyword_index() -> models.PayloadIndexInfo:
    return models.PayloadIndexInfo(
        data_type=models.PayloadSchemaType.KEYWORD,
        params=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
        points=10,
    )


def text_index(params: models.TextIndexParams) -> models.PayloadIndexInfo:
    return models.PayloadIndexInfo(
        data_type=models.PayloadSchemaType.TEXT,
        params=params,
        points=10,
    )


def expected_payload_schema() -> dict[str, models.PayloadIndexInfo]:
    return {
        ORIGINAL_ID_FIELD: keyword_index(),
        ORIGINAL_ID_NORMALIZED_FIELD: text_index(
            qdrant_schema.build_normalized_original_id_text_index_params()
        ),
        ORIGINAL_ID_NORMALIZED_REVERSED_FIELD: text_index(
            qdrant_schema.build_normalized_original_id_text_index_params()
        ),
        "class_name": text_index(qdrant_schema.build_class_name_text_index_params()),
    }


def collection_info(
    *,
    vectors: object | None = None,
    payload_schema: dict | None = None,
    quantized: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=vectors or models.VectorParams(size=128, distance="Cosine")
            ),
            quantization_config=object() if quantized else None,
        ),
        payload_schema=(
            expected_payload_schema() if payload_schema is None else payload_schema
        ),
    )


def client_with(info: object, *, names: tuple[str, ...] = ("products",)) -> MagicMock:
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name=name) for name in names]
    )
    client.get_collection.return_value = info
    return client


class QdrantSchemaValidationTests(unittest.TestCase):
    def test_valid_collection_returns_quantization_cache_without_mutation(self) -> None:
        client = client_with(collection_info(quantized=True))

        result = qdrant_schema.validate_configured_collections(
            client, classifier_test_config()
        )

        self.assertEqual(result, {"products": True})
        client.create_payload_index.assert_not_called()
        client.delete_payload_index.assert_not_called()
        client.set_payload.assert_not_called()
        client.batch_update_points.assert_not_called()

    def test_missing_collection_fails(self) -> None:
        client = client_with(collection_info(), names=())

        with self.assertRaises(qdrant_schema.QdrantSchemaValidationError) as ctx:
            qdrant_schema.validate_configured_collections(
                client, classifier_test_config()
            )

        self.assertEqual(ctx.exception.issues[0].code, "missing_collection")

    def test_unavailable_collection_fails(self) -> None:
        client = client_with(collection_info())
        client.get_collection.side_effect = RuntimeError("down")

        report = qdrant_schema.inspect_configured_collections(
            client, classifier_test_config()
        )

        self.assertEqual(report.issues[0].code, "collection_unavailable")

    def test_missing_and_mismatched_indexes_are_all_reported(self) -> None:
        payload = expected_payload_schema()
        del payload[ORIGINAL_ID_NORMALIZED_FIELD]
        payload["class_name"] = keyword_index()
        client = client_with(collection_info(payload_schema=payload))

        report = qdrant_schema.inspect_configured_collections(
            client, classifier_test_config()
        )

        self.assertEqual(
            {issue.code for issue in report.issues},
            {"missing_payload_index", "payload_index_mismatch"},
        )
        self.assertEqual(len(report.issues), 2)

    def test_wrong_text_tokenizer_fails(self) -> None:
        payload = expected_payload_schema()
        payload["class_name"] = text_index(
            qdrant_schema.build_class_name_text_index_params().model_copy(
                update={"tokenizer": models.TokenizerType.PREFIX}
            )
        )
        client = client_with(collection_info(payload_schema=payload))

        report = qdrant_schema.inspect_configured_collections(
            client, classifier_test_config()
        )

        self.assertEqual(report.issues[0].code, "payload_index_mismatch")

    def test_wrong_keyword_parameter_fails(self) -> None:
        payload = expected_payload_schema()
        payload[ORIGINAL_ID_FIELD] = models.PayloadIndexInfo(
            data_type=models.PayloadSchemaType.KEYWORD,
            params=models.KeywordIndexParams(type="keyword", is_tenant=True),
            points=1,
        )
        client = client_with(collection_info(payload_schema=payload))

        report = qdrant_schema.inspect_configured_collections(
            client, classifier_test_config()
        )

        self.assertEqual(report.issues[0].code, "payload_index_mismatch")

    def test_wrong_vector_dimension_fails(self) -> None:
        client = client_with(
            collection_info(vectors=models.VectorParams(size=64, distance="Cosine"))
        )

        report = qdrant_schema.inspect_configured_collections(
            client, classifier_test_config()
        )

        self.assertEqual(report.issues[0].code, "vector_size_mismatch")

    def test_named_vectors_fail(self) -> None:
        client = client_with(
            collection_info(
                vectors={"default": models.VectorParams(size=128, distance="Cosine")}
            )
        )

        report = qdrant_schema.inspect_configured_collections(
            client, classifier_test_config()
        )

        self.assertEqual(report.issues[0].code, "named_vectors_unsupported")

    def test_conflicting_dimensions_do_not_skip_collection_inspection(self) -> None:
        config = {
            "A": {
                "embed_dims": 128,
                "versions": {"v1": {"collection_name": "products"}},
            },
            "B": {
                "embed_dims": 256,
                "versions": {"v2": {"collection_name": "products"}},
            },
            "C": {
                "embed_dims": 128,
                "versions": {"v1": {"collection_name": "other"}},
            },
        }
        products_payload = expected_payload_schema()
        del products_payload[ORIGINAL_ID_NORMALIZED_FIELD]
        client = client_with(collection_info(), names=("products", "other"))
        client.get_collection.side_effect = [
            collection_info(payload_schema=products_payload),
            collection_info(),
        ]

        report = qdrant_schema.inspect_configured_collections(client, config)

        self.assertEqual(
            {issue.code for issue in report.issues},
            {"invalid_config", "missing_payload_index"},
        )
        self.assertEqual(
            client.get_collection.call_args_list,
            [call("products"), call("other")],
        )
        self.assertEqual(report.quantization_cache, {"other": False})

    def test_invalid_dimension_reference_still_inspects_collection(self) -> None:
        config = {
            "A": {
                "embed_dims": 128,
                "versions": {"v1": {"collection_name": "products"}},
            },
            "B": {
                "embed_dims": None,
                "versions": {"v2": {"collection_name": "products"}},
            },
        }
        payload = expected_payload_schema()
        del payload["class_name"]
        client = client_with(collection_info(payload_schema=payload))

        report = qdrant_schema.inspect_configured_collections(client, config)

        self.assertEqual(
            {issue.code for issue in report.issues},
            {"invalid_config", "missing_payload_index"},
        )
        client.get_collection.assert_called_once_with("products")
        self.assertEqual(report.quantization_cache, {})

    def test_boolean_dimension_is_not_accepted_as_an_integer(self) -> None:
        config = {
            "A": {
                "embed_dims": True,
                "versions": {"v1": {"collection_name": "products"}},
            }
        }
        client = client_with(collection_info())

        report = qdrant_schema.inspect_configured_collections(client, config)

        self.assertEqual(report.issues[0].code, "invalid_config")
        client.get_collection.assert_called_once_with("products")
        self.assertEqual(report.quantization_cache, {})
