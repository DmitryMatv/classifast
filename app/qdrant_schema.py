from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from qdrant_client import QdrantClient, models

from .classifier_config import CLASSIFIER_CONFIG
from .id_lookup import (
    ORIGINAL_ID_FIELD,
    ORIGINAL_ID_NORMALIZED_FIELD,
    ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
)

PAYLOAD_INDEX_FIELDS = (
    ORIGINAL_ID_FIELD,
    ORIGINAL_ID_NORMALIZED_FIELD,
    ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
    "class_name",
)


@dataclass(frozen=True)
class CollectionRequirement:
    collection_name: str
    embed_dims: int | None
    references: tuple[str, ...]


@dataclass(frozen=True)
class QdrantValidationIssue:
    collection_name: str
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.collection_name}: {self.detail} [{self.code}]"


@dataclass(frozen=True)
class QdrantValidationReport:
    quantization_cache: dict[str, bool]
    issues: tuple[QdrantValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


class QdrantSchemaValidationError(RuntimeError):
    def __init__(self, issues: Iterable[QdrantValidationIssue]) -> None:
        self.issues = tuple(issues)
        message = "Qdrant schema validation failed"
        if self.issues:
            message += ":\n" + "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(message)


def build_original_id_index_params() -> models.KeywordIndexParams:
    return models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD)


def build_class_name_text_index_params() -> models.TextIndexParams:
    return models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        min_token_len=1,
        max_token_len=30,
        lowercase=True,
    )


def build_normalized_original_id_text_index_params() -> models.TextIndexParams:
    return models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.PREFIX,
        min_token_len=1,
        max_token_len=64,
        lowercase=True,
    )


def get_payload_index_schema(
    field_name: str,
) -> models.KeywordIndexParams | models.TextIndexParams:
    if field_name == ORIGINAL_ID_FIELD:
        return build_original_id_index_params()
    if field_name in {
        ORIGINAL_ID_NORMALIZED_FIELD,
        ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
    }:
        return build_normalized_original_id_text_index_params()
    if field_name == "class_name":
        return build_class_name_text_index_params()
    raise KeyError(f"Unsupported payload index field: {field_name}")


def get_existing_payload_index(
    collection_info: Any,
    field_name: str,
) -> models.PayloadIndexInfo | None:
    payload_schema = getattr(collection_info, "payload_schema", None) or {}
    return payload_schema.get(field_name)


def normalize_text_index_params(
    params: models.TextIndexParams | None,
) -> dict[str, object]:
    return {
        "type": "text",
        "tokenizer": (
            params.tokenizer
            if params and params.tokenizer is not None
            else models.TokenizerType.WORD
        ),
        "min_token_len": (
            params.min_token_len if params and params.min_token_len is not None else 1
        ),
        "max_token_len": (
            params.max_token_len if params and params.max_token_len is not None else 30
        ),
        "lowercase": (
            params.lowercase if params and params.lowercase is not None else True
        ),
        "ascii_folding": (
            params.ascii_folding
            if params and params.ascii_folding is not None
            else False
        ),
        "phrase_matching": (
            params.phrase_matching
            if params and params.phrase_matching is not None
            else False
        ),
        "stopwords": params.stopwords if params else None,
        "on_disk": (params.on_disk if params and params.on_disk is not None else False),
        "stemmer": params.stemmer if params else None,
        "enable_hnsw": (
            params.enable_hnsw if params and params.enable_hnsw is not None else True
        ),
    }


def normalize_keyword_index_params(
    params: models.KeywordIndexParams | None,
) -> dict[str, object]:
    return {
        "type": "keyword",
        "is_tenant": (
            params.is_tenant if params and params.is_tenant is not None else False
        ),
        "on_disk": (params.on_disk if params and params.on_disk is not None else False),
        "enable_hnsw": (
            params.enable_hnsw if params and params.enable_hnsw is not None else True
        ),
    }


def is_expected_payload_index(
    field_name: str,
    index_info: models.PayloadIndexInfo,
) -> bool:
    if field_name == ORIGINAL_ID_FIELD:
        if index_info.data_type != models.PayloadSchemaType.KEYWORD:
            return False
        params = index_info.params
        if params is not None and not isinstance(params, models.KeywordIndexParams):
            return False
        expected = get_payload_index_schema(field_name)
        if not isinstance(expected, models.KeywordIndexParams):
            return False
        return normalize_keyword_index_params(params) == normalize_keyword_index_params(
            expected
        )

    if field_name in {
        ORIGINAL_ID_NORMALIZED_FIELD,
        ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
        "class_name",
    }:
        if index_info.data_type != models.PayloadSchemaType.TEXT:
            return False
        params = index_info.params
        if params is not None and not isinstance(params, models.TextIndexParams):
            return False
        expected = get_payload_index_schema(field_name)
        if not isinstance(expected, models.TextIndexParams):
            return False
        return normalize_text_index_params(params) == normalize_text_index_params(
            expected
        )

    raise KeyError(f"Unsupported payload index field: {field_name}")


def get_all_collection_names(classifier_config: dict | None = None) -> list[str]:
    config_source = (
        CLASSIFIER_CONFIG if classifier_config is None else classifier_config
    )
    return sorted(
        {
            collection_name
            for config in config_source.values()
            for version_config in config.get("versions", {}).values()
            if (collection_name := version_config.get("collection_name"))
        }
    )


def build_collection_requirements(
    classifier_config: dict | None = None,
    collection_names: set[str] | None = None,
) -> tuple[dict[str, CollectionRequirement], list[QdrantValidationIssue]]:
    config_source = (
        CLASSIFIER_CONFIG if classifier_config is None else classifier_config
    )
    dimensions: dict[str, set[int]] = {}
    references: dict[str, list[str]] = {}
    invalid_dimension_references: dict[str, list[str]] = {}

    for classifier_type, config in config_source.items():
        embed_dims = config.get("embed_dims")
        for version, version_config in config.get("versions", {}).items():
            collection_name = version_config.get("collection_name")
            if not collection_name or (
                collection_names is not None and collection_name not in collection_names
            ):
                continue
            reference = f"{classifier_type}/{version}"
            references.setdefault(collection_name, []).append(reference)
            if (
                isinstance(embed_dims, int)
                and not isinstance(embed_dims, bool)
                and embed_dims > 0
            ):
                dimensions.setdefault(collection_name, set()).add(embed_dims)
            else:
                dimensions.setdefault(collection_name, set())
                invalid_dimension_references.setdefault(collection_name, []).append(
                    f"{reference}={embed_dims!r}"
                )

    requirements: dict[str, CollectionRequirement] = {}
    issues: list[QdrantValidationIssue] = []
    for collection_name, collection_references in references.items():
        configured_dimensions = dimensions[collection_name]
        invalid_references = invalid_dimension_references.get(collection_name, [])
        config_problems: list[str] = []
        if len(configured_dimensions) > 1:
            config_problems.append(
                f"conflicting embedding dimensions {sorted(configured_dimensions)}"
            )
        elif not configured_dimensions:
            config_problems.append("no valid configured embedding dimension")
        if invalid_references:
            config_problems.append(
                "invalid embedding dimensions for " + ", ".join(invalid_references)
            )
        embed_dims = (
            next(iter(configured_dimensions))
            if len(configured_dimensions) == 1 and not invalid_references
            else None
        )
        if config_problems:
            issues.append(
                QdrantValidationIssue(
                    collection_name,
                    "invalid_config",
                    "; ".join(config_problems),
                )
            )
        requirements[collection_name] = CollectionRequirement(
            collection_name=collection_name,
            embed_dims=embed_dims,
            references=tuple(collection_references),
        )

    return requirements, issues


def _validate_collection_info(
    requirement: CollectionRequirement,
    collection_info: Any,
) -> list[QdrantValidationIssue]:
    issues: list[QdrantValidationIssue] = []
    collection_name = requirement.collection_name
    vector_params = collection_info.config.params.vectors

    if isinstance(vector_params, dict):
        issues.append(
            QdrantValidationIssue(
                collection_name,
                "named_vectors_unsupported",
                "uses named vectors, but classifier queries expect one unnamed vector",
            )
        )
    else:
        vector_size = getattr(vector_params, "size", None)
        if requirement.embed_dims is not None and vector_size != requirement.embed_dims:
            issues.append(
                QdrantValidationIssue(
                    collection_name,
                    "vector_size_mismatch",
                    f"vector size is {vector_size!r}; expected {requirement.embed_dims}",
                )
            )

    for field_name in PAYLOAD_INDEX_FIELDS:
        index_info = get_existing_payload_index(collection_info, field_name)
        if index_info is None:
            issues.append(
                QdrantValidationIssue(
                    collection_name,
                    "missing_payload_index",
                    f"missing payload index '{field_name}'",
                )
            )
        elif not is_expected_payload_index(field_name, index_info):
            issues.append(
                QdrantValidationIssue(
                    collection_name,
                    "payload_index_mismatch",
                    f"payload index '{field_name}' does not match the required schema",
                )
            )

    return issues


def inspect_configured_collections(
    client: QdrantClient,
    classifier_config: dict | None = None,
    collection_names: set[str] | None = None,
) -> QdrantValidationReport:
    requirements, issues = build_collection_requirements(
        classifier_config, collection_names
    )
    invalid_config_collections = {
        issue.collection_name for issue in issues if issue.code == "invalid_config"
    }
    quantization_cache: dict[str, bool] = {}

    try:
        collections_result = client.get_collections()
        existing_names = {item.name for item in collections_result.collections}
    except Exception as exc:
        issues.append(
            QdrantValidationIssue(
                "<qdrant>",
                "collection_list_failed",
                f"could not list collections: {exc}",
            )
        )
        return QdrantValidationReport(quantization_cache, tuple(issues))

    for collection_name, requirement in requirements.items():
        if collection_name not in existing_names:
            issues.append(
                QdrantValidationIssue(
                    collection_name,
                    "missing_collection",
                    "configured collection does not exist",
                )
            )
            continue

        try:
            collection_info = client.get_collection(collection_name)
        except Exception as exc:
            issues.append(
                QdrantValidationIssue(
                    collection_name,
                    "collection_unavailable",
                    f"could not inspect collection: {exc}",
                )
            )
            continue

        collection_issues = _validate_collection_info(requirement, collection_info)
        issues.extend(collection_issues)
        if not collection_issues and collection_name not in invalid_config_collections:
            quantization_cache[collection_name] = (
                collection_info.config.quantization_config is not None
            )

    return QdrantValidationReport(quantization_cache, tuple(issues))


def validate_configured_collections(
    client: QdrantClient,
    classifier_config: dict | None = None,
    collection_names: set[str] | None = None,
) -> dict[str, bool]:
    report = inspect_configured_collections(client, classifier_config, collection_names)
    if not report.valid:
        raise QdrantSchemaValidationError(report.issues)
    return report.quantization_cache
