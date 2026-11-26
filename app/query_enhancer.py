import asyncio
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from .classifier import sanitize_query_text
from .classifier_config import CLASSIFIER_CONFIG

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
QUERY_ENHANCEMENT_MODEL = "anthropic/claude-haiku-5.5"
NO_DESCRIPTION_REPLY = "NONE"
CODE_LIKE_PATTERN = re.compile(r"(?i)^(?=.*[a-z])(?=.*\d)[a-z\d][a-z\d.\-_/]*$")


class EnhancementStatus(Enum):
    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class EnhancementOutcome:
    text: str
    status: EnhancementStatus


def _is_code_like(original: str) -> bool:
    stripped = original.strip()
    return bool(
        re.fullmatch(r"[\d\s.\-]+", stripped)
        or (len(stripped) >= 5 and CODE_LIKE_PATTERN.fullmatch(stripped))
    )


def _system_prompt(classifier_type: str) -> str:
    key = classifier_type.upper()
    subject = CLASSIFIER_CONFIG[key]["query_subject"]
    return (
        f"You expand search queries for {key} classification. "
        f"The user's message is a description of {subject}. "
        "Reply with one plain English sentence that says what it is in generic terms: "
        "turn brand names, model names, abbreviations, and trade jargon into the common type, "
        "without repeating the brand. "
        "Keep the details the text gives and add none of your own. "
        "If you do not recognize what the text refers to, "
        "or it could mean different kinds of things, "
        f"reply with {NO_DESCRIPTION_REPLY}. Reply with the sentence only."
    )


def _is_no_description(description: str) -> bool:
    return description.strip(" \"'.").upper() in ("", NO_DESCRIPTION_REPLY)


class QueryEnhancer:
    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=3.0,
        )
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def enhance(self, original: str, classifier_type: str) -> EnhancementOutcome:
        if _is_code_like(original):
            return EnhancementOutcome(original, EnhancementStatus.SKIPPED)

        try:
            sanitized_original = sanitize_query_text(original)
            async with asyncio.timeout(3.0):
                response = await self._client.post(
                    OPENROUTER_CHAT_URL,
                    json={
                        "model": QUERY_ENHANCEMENT_MODEL,
                        "max_tokens": 80,
                        "reasoning": {"effort": "none"},
                        "messages": [
                            {
                                "role": "system",
                                "content": _system_prompt(classifier_type),
                            },
                            {"role": "user", "content": sanitized_original},
                        ],
                    },
                    timeout=3.0,
                )
            response.raise_for_status()
            payload: Any = response.json()
            description = payload["choices"][0]["message"]["content"]
            if not isinstance(description, str):
                return EnhancementOutcome(original, EnhancementStatus.FAILED)
            description = re.sub(r"\s+", " ", description).strip()
            if _is_no_description(description):
                return EnhancementOutcome(original, EnhancementStatus.SKIPPED)
            if len(description) > 240 or any(
                ord(character) < 32 for character in description
            ):
                return EnhancementOutcome(original, EnhancementStatus.FAILED)
            description = sanitize_query_text(description, for_search=True)
            if not description:
                return EnhancementOutcome(original, EnhancementStatus.FAILED)
            if description.casefold() == sanitized_original.casefold():
                return EnhancementOutcome(original, EnhancementStatus.SKIPPED)
            return EnhancementOutcome(
                f"{sanitized_original}\n\n{description}", EnhancementStatus.APPLIED
            )
        except Exception as exc:
            logger.warning("Query enhancement unavailable: %s", type(exc).__name__)
            return EnhancementOutcome(original, EnhancementStatus.FAILED)
