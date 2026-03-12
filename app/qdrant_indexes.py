from qdrant_client import models

ORIGINAL_ID_FIELD = "original_id"
CLASS_NAME_FIELD = "class_name"

TEXT_INDEXED_FIELDS = (CLASS_NAME_FIELD,)
PAYLOAD_INDEX_FIELDS = (ORIGINAL_ID_FIELD, CLASS_NAME_FIELD)


def build_class_name_text_index_params() -> models.TextIndexParams:
    """Return the text index settings used for human-readable payload fields."""
    return models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        min_token_len=1,
        max_token_len=30,
        lowercase=True,
    )


def build_original_id_keyword_index_params() -> models.KeywordIndexParams:
    """Return the exact-match index settings for classification identifiers."""
    return models.KeywordIndexParams(type="keyword")


def get_expected_payload_index_schema(
    field_name: str,
) -> models.TextIndexParams | models.KeywordIndexParams:
    """Return the Qdrant payload index schema required for a field."""
    if field_name == ORIGINAL_ID_FIELD:
        return build_original_id_keyword_index_params()
    if field_name == CLASS_NAME_FIELD:
        return build_class_name_text_index_params()
    raise KeyError(f"Unsupported payload index field: {field_name}")
