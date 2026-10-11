import re

ORIGINAL_ID_FIELD = "original_id"
ORIGINAL_ID_NORMALIZED_FIELD = "original_id_normalized"
ORIGINAL_ID_NORMALIZED_REVERSED_FIELD = "original_id_normalized_reversed"
# Same value as ORIGINAL_ID_NORMALIZED_FIELD, but keyword-indexed for equality
# lookups; Qdrant allows one index per field and that one is a prefix index.
ORIGINAL_ID_NORMALIZED_KEY_FIELD = "original_id_normalized_key"
ID_LOOKUP_FIELDS = (
    ORIGINAL_ID_FIELD,
    ORIGINAL_ID_NORMALIZED_FIELD,
    ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
    ORIGINAL_ID_NORMALIZED_KEY_FIELD,
)

NON_ASCII_ALNUM_PATTERN = re.compile(r"[^0-9A-Za-z]+")
# Letters and digits outside ASCII, which compaction would silently delete.
NON_ASCII_WORD_CHARACTER_PATTERN = re.compile(r"[^\W0-9A-Za-z_]")


def is_code_shaped(value: str) -> bool:
    """True when compaction removes only separators, never description text."""
    return not NON_ASCII_WORD_CHARACTER_PATTERN.search(value)


def compact_original_id(value: object) -> str:
    return NON_ASCII_ALNUM_PATTERN.sub("", str(value).casefold())


def normalize_original_id_for_lookup(value: object) -> str:
    compacted = compact_original_id(value)
    if (
        len(compacted) == 9
        and compacted.isdigit()
        and compacted.startswith("0")
        and compacted[-2] == "0"
        and compacted[-1] != "0"
    ):
        return compacted[1:-2] + compacted[-1]

    normalized = compacted.lstrip("0").rstrip("0")
    return normalized or compacted


def reverse_normalized_id(value: str) -> str:
    return value[::-1]


# Collection builders and the payload migration both write these fields;
# startup validation requires their indexes on every configured collection.
def build_normalized_id_payload(original_id: object) -> dict[str, str]:
    normalized = normalize_original_id_for_lookup(original_id)
    return {
        ORIGINAL_ID_NORMALIZED_FIELD: normalized,
        ORIGINAL_ID_NORMALIZED_REVERSED_FIELD: reverse_normalized_id(normalized),
        ORIGINAL_ID_NORMALIZED_KEY_FIELD: normalized,
    }
