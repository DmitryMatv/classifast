import os

from fastapi.templating import Jinja2Templates
from jwt import PyJWKClient


def group_original_id_tokens(original_id: object) -> list[dict[str, object]]:
    """Split an original_id into characters and mark pair gaps within digit runs."""
    if original_id is None:
        return []

    id_str = str(original_id)
    tokens: list[dict[str, object]] = []
    digit_run_index = 0

    for index, char in enumerate(id_str):
        if char.isdigit():
            digit_run_index += 1
            next_char_is_digit = index + 1 < len(id_str) and id_str[index + 1].isdigit()
            gap_after = next_char_is_digit and digit_run_index % 2 == 0
        else:
            digit_run_index = 0
            gap_after = False

        tokens.append({"char": char, "gap_after": gap_after})

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


def get_jwks_client():
    global _jwks_client
    if _jwks_client is None and CLERK_FRONTEND_API:
        _jwks_client = PyJWKClient(
            f"https://{CLERK_FRONTEND_API}/.well-known/jwks.json"
        )
    return _jwks_client
