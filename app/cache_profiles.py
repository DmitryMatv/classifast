import time
from dataclasses import dataclass
from email.utils import formatdate


@dataclass(frozen=True)
class CacheProfile:
    browser_cache_control: str
    cloudflare_cache_control: str
    browser_max_age: int
    emit_expires: bool = True


HTML_PAGE = CacheProfile(
    browser_cache_control="public, max-age=60, stale-while-revalidate=600",
    cloudflare_cache_control="max-age=7200, stale-while-revalidate=86400",
    browser_max_age=60,
)

CLASSIFICATION_RESULT = CacheProfile(
    browser_cache_control="public, max-age=60, stale-while-revalidate=600",
    cloudflare_cache_control="max-age=86400, stale-while-revalidate=86400",
    browser_max_age=60,
)

STATIC_CODE = CacheProfile(
    browser_cache_control="public, max-age=300, stale-while-revalidate=3600",
    cloudflare_cache_control="max-age=43200, stale-while-revalidate=86400",
    browser_max_age=300,
)

STATIC_MEDIA = CacheProfile(
    browser_cache_control="public, max-age=3600, stale-while-revalidate=86400",
    cloudflare_cache_control="max-age=604800, stale-while-revalidate=86400",
    browser_max_age=3600,
)

STATIC_TEXT = CacheProfile(
    browser_cache_control="public, max-age=600, stale-while-revalidate=3600",
    cloudflare_cache_control="max-age=7200, stale-while-revalidate=86400",
    browser_max_age=600,
)

NO_STORE = CacheProfile(
    browser_cache_control="no-store, max-age=0",
    cloudflare_cache_control="no-store",
    browser_max_age=0,
    emit_expires=False,
)


def get_expires_header(max_age_seconds: int) -> str:
    """Generate Expires header value in HTTP-date format (RFC 7231)."""
    return formatdate(time.time() + max_age_seconds, usegmt=True)


def build_cache_headers(profile: CacheProfile) -> dict[str, str]:
    """Build the shared cache header set for a response profile."""
    headers = {
        "Cache-Control": profile.browser_cache_control,
        "Cloudflare-CDN-Cache-Control": profile.cloudflare_cache_control,
    }
    if profile.emit_expires:
        headers["Expires"] = get_expires_header(profile.browser_max_age)
    return headers


def add_vary(headers: dict[str, str], value: str) -> None:
    """Merge a Vary token into an existing header value."""
    existing = headers.get("Vary", "")
    tokens = {part.strip().lower() for part in existing.split(",") if part.strip()}
    if value.lower() in tokens:
        return
    headers["Vary"] = f"{existing}, {value}" if existing else value
