from dataclasses import dataclass


@dataclass(frozen=True)
class CacheProfile:
    browser_cache_control: str
    cloudflare_cache_control: str


HTML_PAGE = CacheProfile(
    browser_cache_control="public, max-age=600, stale-while-revalidate=3600",
    cloudflare_cache_control="max-age=3600, stale-while-revalidate=86400",
)

CLASSIFICATION_RESULT = CacheProfile(
    browser_cache_control="public, max-age=86400, stale-while-revalidate=604800",
    cloudflare_cache_control=("public, max-age=604800, stale-while-revalidate=604800"),
)

STATIC_CODE = CacheProfile(
    browser_cache_control="public, max-age=300, stale-while-revalidate=3600",
    cloudflare_cache_control="max-age=43200, stale-while-revalidate=86400",
)

STATIC_MEDIA = CacheProfile(
    browser_cache_control="public, max-age=3600, stale-while-revalidate=86400",
    cloudflare_cache_control="max-age=604800, stale-while-revalidate=86400",
)

STATIC_TEXT = CacheProfile(
    browser_cache_control="public, max-age=600, stale-while-revalidate=3600",
    cloudflare_cache_control="max-age=7200, stale-while-revalidate=86400",
)

NO_STORE = CacheProfile(
    browser_cache_control="no-store, max-age=0",
    cloudflare_cache_control="no-store",
)


def build_cache_headers(profile: CacheProfile) -> dict[str, str]:
    """Build the shared cache header set for a response profile."""
    return {
        "Cache-Control": profile.browser_cache_control,
        "Cloudflare-CDN-Cache-Control": profile.cloudflare_cache_control,
    }


def add_vary(headers: dict[str, str], value: str) -> None:
    """Merge a Vary token into an existing header value."""
    existing = headers.get("Vary", "")
    tokens = {part.strip().lower() for part in existing.split(",") if part.strip()}
    if value.lower() in tokens:
        return
    headers["Vary"] = f"{existing}, {value}" if existing else value
