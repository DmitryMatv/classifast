import asyncio
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from .classifier import sanitize_query_text

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
QUERY_ENHANCEMENT_MODEL = "google/gemini-3.1-flash-lite"
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
        or CODE_LIKE_PATTERN.fullmatch(stripped)
    )


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
                        "temperature": 0,
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    f"{sanitized_original}\n\n"
                                    f"For a {classifier_type} classification search, "
                                    "write one short, neutral description of the "
                                    "user's product or service. Use only its common "
                                    "meaning and details supported by the input. "
                                    "Do not guess materials, uses, industry, or "
                                    "specifications. If the meaning is unclear, return "
                                    "an empty string. Output only the description, "
                                    "with no preface."
                                ),
                            }
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
            if not description:
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
