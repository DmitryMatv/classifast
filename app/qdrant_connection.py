import os
from urllib.parse import urlsplit

from qdrant_client import QdrantClient


def resolve_qdrant_url() -> str:
    """Resolve the single Qdrant endpoint contract used by runtime and utilities."""
    configured_url = os.getenv("QDRANT_URL", "").strip()
    if configured_url:
        if configured_url.startswith(("http://", "https://")):
            return configured_url.rstrip("/")
        return f"https://{configured_url.rstrip('/')}"

    host = os.getenv("QDRANT_HOST", "localhost").strip() or "localhost"
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return f"http://{host}:{port}"


def create_qdrant_client(*, timeout: int) -> QdrantClient:
    """Create a Qdrant client with shared endpoint and credential resolution."""
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    url = resolve_qdrant_url()
    parsed_url = urlsplit(url)
    # QdrantClient defaults to port 6333 even when a URL specifies a scheme.
    # Portless full URLs must therefore override that default with the
    # scheme's standard port.
    if parsed_url.port is None and parsed_url.scheme in {"http", "https"}:
        default_port = 80 if parsed_url.scheme == "http" else 443
        return QdrantClient(
            url=url,
            port=default_port,
            api_key=api_key or None,
            timeout=timeout,
        )
    return QdrantClient(
        url=url,
        api_key=api_key or None,
        timeout=timeout,
    )
