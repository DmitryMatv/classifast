import re

ORIGINAL_ID_FIELD = "original_id"
ORIGINAL_ID_NORMALIZED_FIELD = "original_id_normalized"
ORIGINAL_ID_NORMALIZED_REVERSED_FIELD = "original_id_normalized_reversed"
ID_LOOKUP_FIELDS = (
    ORIGINAL_ID_FIELD,
    ORIGINAL_ID_NORMALIZED_FIELD,
    ORIGINAL_ID_NORMALIZED_REVERSED_FIELD,
)

NON_ASCII_ALNUM_PATTERN = re.compile(r"[^0-9A-Za-z]+")


def normalize_original_id_for_lookup(value: object) -> str:
    compacted = NON_ASCII_ALNUM_PATTERN.sub("", str(value).casefold())
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
