from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

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


def get_existing_payload_index(
    collection_info: Any,
    field_name: str,
) -> models.PayloadIndexInfo | None:
    payload_schema = getattr(collection_info, "payload_schema", None) or {}
    return payload_schema.get(field_name)


def _param_or_default(
    params: models.KeywordIndexParams | models.TextIndexParams | None,
    attr: str,
    default: object,
) -> object:
    if params is None:
        return default
    value = getattr(params, attr, None)
    return default if value is None else value


def normalize_text_index_params(
    params: models.KeywordIndexParams | models.TextIndexParams | None,
) -> dict[str, object]:
    return {
        "type": "text",
        "tokenizer": _param_or_default(params, "tokenizer", models.TokenizerType.WORD),
        "min_token_len": _param_or_default(params, "min_token_len", 1),
        "max_token_len": _param_or_default(params, "max_token_len", 30),
        "lowercase": _param_or_default(params, "lowercase", True),
        "ascii_folding": _param_or_default(params, "ascii_folding", False),
        "phrase_matching": _param_or_default(params, "phrase_matching", False),
        "stopwords": _param_or_default(params, "stopwords", None),
        "on_disk": _param_or_default(params, "on_disk", False),
        "stemmer": _param_or_default(params, "stemmer", None),
        "enable_hnsw": _param_or_default(params, "enable_hnsw", True),
    }


def normalize_keyword_index_params(
    params: models.KeywordIndexParams | models.TextIndexParams | None,
) -> dict[str, object]:
    return {
        "type": "keyword",
        "is_tenant": _param_or_default(params, "is_tenant", False),
        "on_disk": _param_or_default(params, "on_disk", False),
        "enable_hnsw": _param_or_default(params, "enable_hnsw", True),
    }


@dataclass(frozen=True)
class _IndexFieldSpec:
    data_type: models.PayloadSchemaType
    build_params: Callable[[], models.KeywordIndexParams | models.TextIndexParams]
    normalize: Callable[
        [models.KeywordIndexParams | models.TextIndexParams | None],
        dict[str, object],
    ]


_INDEX_FIELD_SPECS: dict[str, _IndexFieldSpec] = {
    ORIGINAL_ID_FIELD: _IndexFieldSpec(
        data_type=models.PayloadSchemaType.KEYWORD,
        build_params=build_original_id_index_params,
        normalize=normalize_keyword_index_params,
    ),
    ORIGINAL_ID_NORMALIZED_FIELD: _IndexFieldSpec(
        data_type=models.PayloadSchemaType.TEXT,
        build_params=build_normalized_original_id_text_index_params,
        normalize=normalize_text_index_params,
    ),
    ORIGINAL_ID_NORMALIZED_REVERSED_FIELD: _IndexFieldSpec(
        data_type=models.PayloadSchemaType.TEXT,
        build_params=build_normalized_original_id_text_index_params,
        normalize=normalize_text_index_params,
    ),
    "class_name": _IndexFieldSpec(
        data_type=models.PayloadSchemaType.TEXT,
        build_params=build_class_name_text_index_params,
        normalize=normalize_text_index_params,
    ),
}


def _get_index_field_spec(field_name: str) -> _IndexFieldSpec:
    spec = _INDEX_FIELD_SPECS.get(field_name)
    if spec is None:
        raise KeyError(f"Unsupported payload index field: {field_name}")
    return spec


def get_payload_index_schema(
    field_name: str,
) -> models.KeywordIndexParams | models.TextIndexParams:
    return _get_index_field_spec(field_name).build_params()


def is_expected_payload_index(
    field_name: str,
    index_info: models.PayloadIndexInfo,
) -> bool:
    spec = _get_index_field_spec(field_name)
    if index_info.data_type != spec.data_type:
        return False
    expected = spec.build_params()
    params = index_info.params
    if params is not None and not isinstance(params, type(expected)):
        return False
    return spec.normalize(params) == spec.normalize(expected)


def _resolve_config_source(classifier_config: dict | None) -> dict:
    return CLASSIFIER_CONFIG if classifier_config is None else classifier_config


def get_all_collection_names(classifier_config: dict | None = None) -> list[str]:
    config_source = _resolve_config_source(classifier_config)
    return sorted(
        {
            collection_name
            for config in config_source.values()
            for version_config in config.get("versions", {}).values()
            if (collection_name := version_config.get("collection_name"))
        }
    )


def _is_valid_embed_dims(embed_dims: Any) -> bool:
    return (
        isinstance(embed_dims, int)
        and not isinstance(embed_dims, bool)
        and embed_dims > 0
    )


@dataclass
class _CollectionUsage:
    references: list[str] = field(default_factory=list)
    dimensions: set[int] = field(default_factory=set)
    invalid_dimension_references: list[str] = field(default_factory=list)

    def record(self, reference: str, embed_dims: Any) -> None:
        self.references.append(reference)
        if _is_valid_embed_dims(embed_dims):
            self.dimensions.add(embed_dims)
        else:
            self.invalid_dimension_references.append(f"{reference}={embed_dims!r}")


def _scan_collection_usage(
    config_source: dict,
    collection_names: set[str] | None,
) -> dict[str, _CollectionUsage]:
    usage: dict[str, _CollectionUsage] = {}
    for classifier_type, config in config_source.items():
        embed_dims = config.get("embed_dims")
        for version, version_config in config.get("versions", {}).items():
            collection_name = version_config.get("collection_name")
            if not collection_name or (
                collection_names is not None and collection_name not in collection_names
            ):
                continue
            usage.setdefault(collection_name, _CollectionUsage()).record(
                f"{classifier_type}/{version}", embed_dims
            )
    return usage


def _config_problems(usage: _CollectionUsage) -> list[str]:
    problems: list[str] = []
    if len(usage.dimensions) > 1:
        problems.append(f"conflicting embedding dimensions {sorted(usage.dimensions)}")
    elif not usage.dimensions:
        problems.append("no valid configured embedding dimension")
    if usage.invalid_dimension_references:
        problems.append(
            "invalid embedding dimensions for "
            + ", ".join(usage.invalid_dimension_references)
        )
    return problems


def _build_collection_requirement(
    collection_name: str,
    usage: _CollectionUsage,
) -> tuple[CollectionRequirement, list[QdrantValidationIssue]]:
    problems = _config_problems(usage)
    embed_dims = (
        next(iter(usage.dimensions))
        if len(usage.dimensions) == 1 and not usage.invalid_dimension_references
        else None
    )
    requirement = CollectionRequirement(
        collection_name=collection_name,
        embed_dims=embed_dims,
        references=tuple(usage.references),
    )
    if not problems:
        return requirement, []
    issue = QdrantValidationIssue(
        collection_name,
        "invalid_config",
        "; ".join(problems),
    )
    return requirement, [issue]


def build_collection_requirements(
    classifier_config: dict | None = None,
    collection_names: set[str] | None = None,
) -> tuple[dict[str, CollectionRequirement], list[QdrantValidationIssue]]:
    config_source = _resolve_config_source(classifier_config)
    requirements: dict[str, CollectionRequirement] = {}
    issues: list[QdrantValidationIssue] = []
    for collection_name, usage in _scan_collection_usage(
        config_source, collection_names
    ).items():
        requirement, collection_issues = _build_collection_requirement(
            collection_name, usage
        )
        requirements[collection_name] = requirement
        issues.extend(collection_issues)
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
