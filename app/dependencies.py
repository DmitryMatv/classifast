import os

from fastapi.templating import Jinja2Templates
from jwt.jwks_client import PyJWKClient


def group_original_id_tokens(original_id: object) -> list[dict[str, object]]:
    """Mark letter-prefix boundaries and right-aligned pair gaps in digit runs."""
    if original_id is None:
        return []

    id_str = str(original_id)
    tokens: list[dict[str, object]] = [
        {"char": char, "gap_after": False} for char in id_str
    ]

    for index in range(len(id_str) - 1):
        if id_str[index].isalpha() and id_str[index + 1].isdigit():
            tokens[index]["gap_after"] = True

    digit_run_start: int | None = None

    for index in range(len(id_str) + 1):
        if index < len(id_str) and id_str[index].isdigit():
            if digit_run_start is None:
                digit_run_start = index
            continue

        if digit_run_start is None:
            continue

        run_length = index - digit_run_start
        first_group_size = 1 if run_length % 2 else 2
        gap_index = digit_run_start + first_group_size - 1
        while gap_index < index - 1:
            tokens[gap_index]["gap_after"] = True
            gap_index += 2
        digit_run_start = None

    return tokens


# Setup Jinja2 templates
templates = Jinja2Templates(directory="app/templates")
templates.env.filters["group_original_id_tokens"] = group_original_id_tokens

# Clerk Authentication Configuration
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
CLERK_FRONTEND_API = os.getenv("CLERK_FRONTEND_API", "")
if CLERK_FRONTEND_API:
    CLERK_FRONTEND_API = (
        CLERK_FRONTEND_API.replace("https://", "").replace("http://", "").rstrip("/")
    )
CLERK_PERMITTED_ORIGINS = os.getenv("CLERK_PERMITTED_ORIGINS", "")

# Cached JWKS client for Clerk JWT verification
_jwks_client = None


def get_jwks_client() -> PyJWKClient | None:
    global _jwks_client
    if _jwks_client is None and CLERK_FRONTEND_API:
        _jwks_client = PyJWKClient(
            f"https://{CLERK_FRONTEND_API}/.well-known/jwks.json"
        )
    return _jwks_client
