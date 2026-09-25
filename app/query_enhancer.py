import asyncio
import logging
import re
from typing import Any

import httpx

from .classifier import sanitize_query_text

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
QUERY_ENHANCEMENT_MODEL = "google/gemini-3.1-flash-lite"


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

    async def enhance(self, original: str, classifier_type: str) -> str:
        if re.fullmatch(r"[\d\s.\-]+", original.strip()):
            return original

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
                                "role": "system",
                                "content": (
                                    "For a product or service classification search, write one short, "
                                    "neutral description of the user's term. Use only its common meaning "
                                    "and details supported by the term. Do not guess materials, uses, "
                                    "industry, or specifications. If the meaning is unclear, return "
                                    "an empty string. Output only the description, with no preface."
                                ),
                            },
                            {
                                "role": "user",
                                "content": f"Standard: {classifier_type}\nTerm: {sanitized_original}",
                            },
                        ],
                    },
                    timeout=3.0,
                )
            response.raise_for_status()
            payload: Any = response.json()
            description = payload["choices"][0]["message"]["content"]
            if not isinstance(description, str):
                return original
            description = re.sub(r"\s+", " ", description).strip()
            if (
                not description
                or len(description) > 240
                or any(ord(character) < 32 for character in description)
            ):
                return original
            description = sanitize_query_text(description, for_search=True)
            if not description or description.casefold() == sanitized_original.casefold():
                return original
            return f"{sanitized_original.rstrip(' .')}. {description}"
        except Exception as exc:
            logger.warning("Query enhancement unavailable: %s", type(exc).__name__)
            return original
