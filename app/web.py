import logging
import re
import time
import unicodedata
from datetime import datetime
from urllib.parse import quote, unquote_plus, urlencode

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from .cache_profiles import (
    CLASSIFICATION_RESULT,
    HTML_PAGE,
    NO_STORE,
    build_cache_headers,
)
from .classifier import get_classification_cache_headers, perform_classification
from .classifier_config import CLASSIFIER_CONFIG, ClassifierConfig
from .dependencies import templates
from .usage_tracker import (
    FREE_USER_LIMIT,
    UsageStatus,
    add_quota_headers,
    check_usage,
    increment_usage,
    set_tracking_cookie,
    verify_checkout_token,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_default_top_k(classifier_type: str) -> int:
    """Return the default number of results to show for a classifier page."""
    return 30 if classifier_type.strip().upper() == "UNSPSC" else 10


def slugify(text: str) -> str:
    """
    Slugify utility for SEO-friendly URLs.
    Used only on the server so canonical page URLs have a single source of truth.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text)[:200])
    text = re.sub(r"\s+", " ", text).strip()

    allowed_punctuation = ".,'()-"
    sanitized_characters = [
        character
        for character in text
        if (
            unicodedata.category(character)[:1] in {"L", "M", "N"}
            or character in allowed_punctuation
            or character in {" ", "_"}
        )
    ]
    sanitized_text = "".join(sanitized_characters)
    sanitized_text = re.sub(r"\s+", "_", sanitized_text)
    return sanitized_text.strip("_")


def normalize_query_text(text: str, *, max_length: int | None = None) -> str:
    """Normalize user-entered classifier queries without changing visible meaning."""
    normalized_text = unicodedata.normalize("NFC", str(text or ""))
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    if max_length is not None and len(normalized_text) > max_length:
        normalized_text = normalized_text[:max_length].strip()

    return normalized_text


def normalize_version(config: ClassifierConfig, version: str | None) -> str:
    """Return a valid classifier version, falling back to the default version."""
    versions_list = list(config["versions"].keys())
    default_version = versions_list[0] if versions_list else ""
    if version is not None and version in config["versions"]:
        return version
    return default_version


def normalize_page_top_k(classifier_type: str, top_k: int | None) -> int:
    """Return a valid page top_k, falling back to the classifier default."""
    default_top_k = get_default_top_k(classifier_type)
    if top_k is None or top_k < 1 or top_k > 100:
        return default_top_k
    return top_k


def build_clean_page_url(
    classifier_type: str,
    normalized_description: str,
    version: str | None,
    top_k: int | None,
    *,
    include_top_k: bool = True,
) -> str:
    """Build the canonical in-app page URL for a classifier query."""
    upper_type = classifier_type.strip().upper()
    config = CLASSIFIER_CONFIG.get(upper_type)
    new_url = f"/{upper_type}/"

    slug = slugify(normalized_description.replace("/", " "))
    if slug:
        new_url = f"/{upper_type}/{quote(slug, safe='')}"

    query_params: list[tuple[str, str]] = []
    if config:
        default_version = normalize_version(config, None)
        if version and version != default_version:
            query_params.append(("version", version))

    default_top_k = get_default_top_k(upper_type)
    if include_top_k and top_k is not None and top_k != default_top_k:
        query_params.append(("top_k", str(top_k)))

    if query_params:
        return f"{new_url}?{urlencode(query_params)}"
    return new_url


def build_clean_classifier_url(
    classifier_type: str,
    normalized_description: str,
    version: str | None,
) -> str:
    """Build the canonical history URL for fragment responses."""
    return build_clean_page_url(
        classifier_type,
        normalized_description,
        version,
        top_k=None,
        include_top_k=False,
    )


def resolve_push_url_enabled(
    push_url: bool | None,
    url_change: bool | None,
) -> bool:
    """Resolve server-side HX-Push-Url behavior from explicit query params only."""
    if push_url is not None:
        return push_url

    if url_change is not None:
        return url_change

    return False


def resolve_page_title_enabled(
    push_url: bool | None,
    url_change: bool | None,
    track_usage: bool,
) -> bool:
    """Keep title swaps cache-safe while honoring explicit history flags."""
    if push_url is not None:
        return push_url

    if url_change is not None:
        return url_change

    return track_usage


# Serve the main homepage
@router.get("/", response_class=HTMLResponse)
@router.head("/")  # Add HEAD support
async def read_root(request: Request):
    """Serves the main homepage with Cloudflare-friendly caching."""

    # For HEAD requests, return just headers
    if request.method == "HEAD":
        headers = build_cache_headers(HTML_PAGE)
        headers["Vary"] = "Accept-Encoding"
        headers["Content-Type"] = "text/html; charset=utf-8"
        headers["Link"] = '<https://classifast.com/>; rel="canonical"'
        return Response(headers=headers)

    today = datetime.now()
    response = templates.TemplateResponse(
        request,
        "index.html",
        {"current_year": today.year},
    )

    # Cloudflare-friendly cache headers (same as classifier pages)
    response.headers.update(build_cache_headers(HTML_PAGE))
    response.headers["Vary"] = "Accept-Encoding"
    response.headers["Link"] = '<https://classifast.com/>; rel="canonical"'
    response.headers["X-Robots-Tag"] = "index, follow"

    return response


@router.get("/{classifier_type}", include_in_schema=False)
@router.head("/{classifier_type}", include_in_schema=False)
async def redirect_classifier_page_no_slash(classifier_type: str, request: Request):
    """
    Redirects URLs without trailing slash to versions with trailing slash for SEO consistency.
    Also redirects lowercase classifier types to uppercase.
    """
    upper_type = classifier_type.strip().upper()
    if upper_type in CLASSIFIER_CONFIG:
        query_string = f"?{request.url.query}" if request.url.query else ""
        return RedirectResponse(url=f"/{upper_type}/{query_string}", status_code=301)
    raise HTTPException(status_code=404, detail=f"Type '{classifier_type}' not found")


@router.get("/{classifier_type}/", response_class=HTMLResponse)
@router.head("/{classifier_type}/")
async def show_classifier_page(
    request: Request,
    classifier_type: str,
    version: str | None = None,
    top_k: int | None = None,
):
    """
    Serves the base classifier page.
    """
    return await show_classifier_page_with_query(
        request, classifier_type, "", version, top_k
    )


@router.get("/{classifier_type}/fragment", response_class=HTMLResponse)
async def get_classification_fragment(
    request: Request,
    classifier_type: str,
    product_description: str = Query(..., alias="product_description"),
    top_k: int | None = Query(None, ge=1, le=100),
    version: str = Query(...),
    push_url: bool | None = Query(None),
    track_usage: bool = Query(True),
    url_change: bool | None = Query(None),
):
    """
    GET endpoint for retrieving classification results as an HTML fragment.
    Optimized for HTMX lazy loading and caching.
    """
    # Normalize inputs early to ensure cache hits and prevent unnecessary API calls
    normalized_description = normalize_query_text(product_description, max_length=4000)
    upper_type = classifier_type.strip().upper()
    if top_k is None:
        top_k = get_default_top_k(upper_type)

    logger.info(
        "WEB received GET fragment request for '%s' with version '%s'. Push URL: %s. Track usage: %s",
        upper_type,
        version,
        push_url,
        track_usage,
    )

    if "track_usage" not in request.query_params and url_change is not None:
        track_usage = url_change
    new_url = build_clean_classifier_url(upper_type, normalized_description, version)
    push_url_enabled = resolve_push_url_enabled(
        push_url=push_url,
        url_change=url_change,
    )
    page_title_enabled = resolve_page_title_enabled(
        push_url=push_url,
        url_change=url_change,
        track_usage=track_usage,
    )

    # Handle checkout return with token verification (also on fragment requests)
    checkout_success = request.query_params.get("checkout")
    checkout_token = request.query_params.get("checkout_token")
    if checkout_success == "success" and checkout_token:
        redis_client = getattr(request.app.state, "redis_client", None)
        await verify_checkout_token(checkout_token, request, redis_client)

    # Check usage limits before processing (only for user queries, not examples)
    redis_client = getattr(request.app.state, "redis_client", None)

    if track_usage:
        usage_status = await check_usage(request, redis_client)

        if not usage_status.allowed:
            response = templates.TemplateResponse(
                request,
                "paywall.html",
                {
                    "limit": usage_status.limit,
                    "is_authenticated": usage_status.is_authenticated,
                    "free_user_limit": FREE_USER_LIMIT,
                },
            )
            response.headers.update(build_cache_headers(NO_STORE))
            if push_url_enabled:
                response.headers["HX-Push-Url"] = new_url
            add_quota_headers(response, usage_status)
            if usage_status.tracking_id:
                set_tracking_cookie(response, usage_status.tracking_id)
            return response
    else:
        usage_status = UsageStatus(
            allowed=True,
            remaining=-1,
            limit=-1,
            is_authenticated=False,
            is_pro=False,
            tracking_id=None,
        )

    # Handle empty query gracefully
    # normalized_description was already set above for URL building
    if not normalized_description:
        response = templates.TemplateResponse(
            request,
            "results.html",
            {
                "query": normalized_description,
                "results_for_query": [],
            },
        )
        response.headers.update(build_cache_headers(CLASSIFICATION_RESULT))
        response.headers["Vary"] = "Accept-Encoding"
        add_quota_headers(response, usage_status)
        return response

    start_total_time = time.perf_counter()

    try:
        quantization_cache = getattr(
            request.app.state, "collection_quantization_cache", {}
        )
        # Use shared classification service with ZeroEntropy reranking
        zclient = getattr(request.app.state, "zclient", None)
        result = perform_classification(
            embed_client=request.app.state.embed_client,
            qdrant_client=request.app.state.qdrant_client,
            query=normalized_description,
            classifier_type=upper_type,
            version=version,
            top_k=top_k,
            quantization_cache=quantization_cache,
            zclient=zclient,
        )

        classification_results = result["results"]

    except HTTPException:
        # Let HTTP exceptions propagate
        raise
    except Exception as e:
        logger.error("Error during '%s' fragment classification: %s", upper_type, e)
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )

    end_total_time = time.perf_counter()
    total_request_time = end_total_time - start_total_time

    # Calculate dynamic page title for OOB swap
    page_title = None
    if page_title_enabled:
        page_title = f"{upper_type} codes for '{normalized_description.title()}'"

    # Render the results partial with normalized query
    response = templates.TemplateResponse(
        request,
        "results.html",
        {
            "query": normalized_description,
            "results_for_query": classification_results,
            "base_url": result["version_config"].get("base_url", ""),
            "tooltip": result["version_config"].get("tooltip", ""),
            "total_request_time": total_request_time,
            "page_title": page_title,
        },
    )

    # Cache classification results to save expensive API calls (ZeroEntropy, Gemini, Qdrant)
    # Cloudflare edge cache handles this - cache hits don't consume quotas
    cache_headers = get_classification_cache_headers()
    response.headers.update(cache_headers)

    # Set HTMX header to update URL in browser address bar (new_url was built earlier)
    if push_url_enabled:
        response.headers["HX-Push-Url"] = new_url

    # Increment usage counter for real user-triggered searches.
    if track_usage:
        await increment_usage(request, redis_client, usage_status)
    add_quota_headers(response, usage_status)

    return response


@router.get("/{classifier_type}/search", include_in_schema=False)
async def redirect_classifier_search(
    classifier_type: str,
    product_description: str = Query(""),
    version: str | None = None,
    top_k: int | None = None,
):
    """Redirect first-party search requests to the canonical classifier page URL."""
    upper_type = classifier_type.strip().upper()
    config = CLASSIFIER_CONFIG.get(upper_type)
    if not config:
        raise HTTPException(
            status_code=404, detail=f"Classifier '{classifier_type}' not found"
        )

    normalized_description = normalize_query_text(product_description, max_length=4000)
    normalized_version = normalize_version(config, version)
    normalized_top_k = normalize_page_top_k(upper_type, top_k)

    canonical_page_url = build_clean_page_url(
        upper_type,
        normalized_description,
        normalized_version,
        normalized_top_k,
    )
    return RedirectResponse(url=canonical_page_url, status_code=303)


@router.get("/{classifier_type}/{search_query:path}", response_class=HTMLResponse)
@router.head("/{classifier_type}/{search_query:path}")
async def show_classifier_page_with_query(
    request: Request,
    classifier_type: str,
    search_query: str = "",
    version: str | None = None,
    top_k: int | None = None,
):
    """
    Serves the specific classifier page with clean URL structure.
    Handles both base URLs like /NAICS and search URLs like /NAICS/gamedev-studio
    Also redirects lowercase classifier types to uppercase.
    """
    # Normalize classifier type early
    upper_type = classifier_type.strip().upper()
    config = CLASSIFIER_CONFIG.get(upper_type)
    if not config:
        raise HTTPException(
            status_code=404, detail=f"Classifier '{classifier_type}' not found"
        )

    # Redirect lowercase to uppercase for SEO consistency
    if classifier_type != upper_type:
        query_string = f"?{request.url.query}" if request.url.query else ""
        redirect_url = f"/{upper_type}/"
        if search_query:
            redirect_url += f"{search_query}"
        return RedirectResponse(url=f"{redirect_url}{query_string}", status_code=301)

    # Use the uppercase classifier_type from here
    effective_classifier_type = upper_type

    # Handle checkout return with token verification
    checkout_success = request.query_params.get("checkout")
    checkout_token = request.query_params.get("checkout_token")
    if checkout_success == "success" and checkout_token:
        redis_client = getattr(request.app.state, "redis_client", None)
        await verify_checkout_token(checkout_token, request, redis_client)

    # Handle empty search query for base URLs
    decoded_search_query = ""
    if search_query and search_query.strip():
        decoded_search_query = (
            unquote_plus(search_query).rstrip("/").replace("/", " ").replace("_", " ")
        )
    decoded_search_query = normalize_query_text(decoded_search_query, max_length=4000)
    normalized_top_k = normalize_page_top_k(effective_classifier_type, top_k)
    normalized_version = normalize_version(config, version)

    canonical_page_url = build_clean_page_url(
        effective_classifier_type,
        decoded_search_query,
        normalized_version,
        normalized_top_k,
    )
    requested_path_and_query = request.url.path
    if request.url.query:
        requested_path_and_query += f"?{request.url.query}"
    if requested_path_and_query != canonical_page_url:
        return RedirectResponse(url=canonical_page_url, status_code=301)

    canonical_url = "https://classifast.com" + build_clean_page_url(
        effective_classifier_type,
        decoded_search_query,
        normalized_version,
        normalized_top_k,
        include_top_k=False,
    )

    # For HEAD requests, return just headers
    if request.method == "HEAD":
        headers = build_cache_headers(HTML_PAGE)
        headers["Vary"] = "Accept-Encoding"
        headers["Content-Type"] = "text/html; charset=utf-8"
        headers["Link"] = f'<{canonical_url}>; rel="canonical"'
        return Response(headers=headers)

    first_version = normalize_version(config, None)

    # Initialize results data structure
    results_data = {
        "results_for_query": [],
        "query": decoded_search_query,
        "base_url": "",
        "tooltip": "",
        "total_request_time": 0,
    }

    # Determine if we should trigger a search on load
    # This is true if we have a URL search query OR if we're falling back to the example
    trigger_search_on_load = False

    if decoded_search_query:
        trigger_search_on_load = True
    else:
        # If no search query (base URL), use example query
        example_query = normalize_query_text(
            config["example"].replace("Example:", "").strip(),
            max_length=4000,
        )
        if example_query:
            results_data["query"] = example_query
            trigger_search_on_load = True

    today = datetime.now()
    current_year = today.year
    current_month_name = today.strftime("%B")

    response = templates.TemplateResponse(
        request,
        "classifier_page.html",
        {
            "classifier_type": effective_classifier_type,
            "title": config["title"],
            "heading": config["heading"],
            "description": config["description"],
            "versions": list(config["versions"].keys()),
            "example": config["example"],
            "url_params": {
                "search": decoded_search_query,
                "version": (
                    normalized_version if normalized_version != first_version else ""
                ),
                "top_k": normalized_top_k,
            },
            "trigger_search_on_load": trigger_search_on_load,
            "canonical_url": canonical_url,
            "current_year": current_year,
            "current_month_name": current_month_name,
            **results_data,
        },
    )

    # Cloudflare-friendly cache headers (aligned with homepage)
    response.headers.update(build_cache_headers(HTML_PAGE))
    response.headers["Vary"] = "Accept-Encoding"
    response.headers["Link"] = f'<{canonical_url}>; rel="canonical"'
    response.headers["X-Robots-Tag"] = "index, follow"

    return response
