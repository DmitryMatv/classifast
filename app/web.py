import logging
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote_plus, urlencode

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from app.cache_profiles import CacheProfile

from .cache_profiles import (
    CLASSIFICATION_RESULT,
    HTML_PAGE,
    NO_STORE,
    STATIC_MEDIA,
    STATIC_TEXT,
    build_cache_headers,
)
from .classifier import get_classification_cache_headers, perform_classification
from .classifier_config import CLASSIFIER_CONFIG
from .dependencies import templates
from .mapping_store import (
    MappingProduct,
    get_mapping_product,
    list_mapping_products,
)
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
BASE_DIR = Path(__file__).resolve().parent.parent


def get_default_top_k(classifier_type: str) -> int:
    """Return the default number of results to show for a classifier page."""
    return 30 if classifier_type.strip().upper() == "UNSPSC" else 10


def slugify(text: str) -> str:
    """
    Slugify utility for SEO-friendly URLs.
    Matches the logic used in show_classifier_page_with_query and frontend JS.
    """
    if not text:
        return ""
    # Sanitize input: limit length and remove harmful characters
    text = str(text)[:200]  # Limit to 200 chars max
    # Normalize internal whitespace first (collapse multiple spaces/newlines into single space)
    text = re.sub(r"\s+", " ", text)
    # Preserve punctuation that sanitize_query_text accepts so URL slugs round-trip
    # cleanly back into the classifier textbox.
    text = re.sub(r"[^\w\s.,:;'()-]", "", text)
    text = re.sub(r"[\s]+", "_", text)
    return text.strip("_")


def build_page_headers(canonical_url: str) -> dict[str, str]:
    headers = build_cache_headers(HTML_PAGE)
    headers["Vary"] = "Accept-Encoding"
    headers["Content-Type"] = "text/html; charset=utf-8"
    headers["Link"] = f'<{canonical_url}>; rel="canonical"'
    headers["X-Robots-Tag"] = "index, follow"
    return headers


def build_mapping_canonical_url(slug: str | None = None) -> str:
    canonical_url = "https://classifast.com/mapping"
    if slug:
        canonical_url += f"/{quote(slug, safe='')}"
    if not canonical_url.endswith("/"):
        canonical_url += "/"
    return canonical_url


def get_sample_cache_profile(sample_path: str) -> CacheProfile:
    if sample_path.endswith(".csv"):
        return STATIC_TEXT
    return STATIC_MEDIA


def build_classification_results_context(
    request: Request,
    classifier_type: str,
    query: str,
    version: str,
    top_k: int,
) -> dict[str, object]:
    """Build the template context used to render classification results."""
    normalized_query = re.sub(r"\s+", " ", query).strip()
    upper_type = classifier_type.strip().upper()

    if not normalized_query:
        return {
            "query": normalized_query,
            "results_for_query": [],
            "base_url": "",
            "tooltip": "",
            "total_request_time": 0,
        }

    start_total_time = time.perf_counter()
    quantization_cache = getattr(request.app.state, "collection_quantization_cache", {})
    zclient = getattr(request.app.state, "zclient", None)
    result = perform_classification(
        embed_client=request.app.state.embed_client,
        qdrant_client=request.app.state.qdrant_client,
        query=normalized_query,
        classifier_type=upper_type,
        version=version,
        top_k=top_k,
        quantization_cache=quantization_cache,
        zclient=zclient,
    )
    total_request_time = time.perf_counter() - start_total_time

    return {
        "query": normalized_query,
        "results_for_query": result["results"],
        "base_url": result["version_config"].get("base_url", ""),
        "tooltip": result["version_config"].get("tooltip", ""),
        "total_request_time": total_request_time,
        "classifier_type": upper_type,
    }


def _normalize_product_description(product_description: str) -> str:
    return re.sub(r"\s+", " ", product_description).strip()


def _get_classifier_config_or_404(classifier_type: str) -> tuple[str, dict]:
    upper_type = classifier_type.strip().upper()
    config = CLASSIFIER_CONFIG.get(upper_type)
    if not config:
        raise HTTPException(
            status_code=404, detail=f"Classifier '{classifier_type}' not found"
        )
    return upper_type, config


def _resolve_classifier_options(
    upper_type: str,
    config: dict,
    version: str | None,
    top_k: int | None,
) -> tuple[str, int, str, int]:
    default_top_k = get_default_top_k(upper_type)
    resolved_top_k = default_top_k if top_k is None else top_k

    versions_list = list(config["versions"].keys())
    default_version = versions_list[0] if versions_list else ""
    resolved_version = default_version if version is None else version

    return resolved_version, resolved_top_k, default_version, default_top_k


def _resolve_fragment_flags(
    request: Request,
    push_url: bool | None,
    track_usage: bool,
    url_change: bool | None,
) -> tuple[bool, bool]:
    resolved_push_url = True if push_url is None and url_change is None else push_url
    if resolved_push_url is None:
        resolved_push_url = url_change

    resolved_track_usage = track_usage
    if "track_usage" not in request.query_params and url_change is not None:
        resolved_track_usage = url_change

    return bool(resolved_push_url), resolved_track_usage


def _get_redis_client(request: Request):
    return getattr(request.app.state, "redis_client", None)


async def _maybe_verify_checkout_return(request: Request, redis_client) -> None:
    checkout_success = request.query_params.get("checkout")
    checkout_token = request.query_params.get("checkout_token")
    if checkout_success == "success" and checkout_token:
        await verify_checkout_token(checkout_token, request, redis_client)


def _build_fragment_push_url(
    upper_type: str,
    normalized_description: str,
    version: str,
    default_version: str,
    top_k: int,
    default_top_k: int,
) -> str:
    slug = slugify(normalized_description.replace("/", " "))
    new_url = f"/{upper_type}"
    if slug:
        new_url += f"/{quote(slug, safe='')}"

    params: dict[str, str | int] = {}
    if version and version != default_version:
        params["version"] = version
    if top_k != default_top_k:
        params["top_k"] = top_k
    if params:
        new_url += f"?{urlencode(params)}"

    return new_url


def _build_unmetered_usage_status() -> UsageStatus:
    return UsageStatus(
        allowed=True,
        remaining=-1,
        limit=-1,
        is_authenticated=False,
        is_pro=False,
        tracking_id=None,
    )


def _render_paywall_fragment(
    request: Request,
    usage_status: UsageStatus,
    push_url: bool,
    new_url: str,
):
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
    if push_url:
        response.headers["HX-Push-Url"] = new_url
    add_quota_headers(response, usage_status)

    return response


def _render_empty_results_fragment(
    request: Request,
    normalized_description: str,
    usage_status: UsageStatus,
):
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


def _render_classification_results_fragment(
    request: Request,
    results_context: dict,
    page_title: str | None,
    push_url: bool,
    new_url: str,
    usage_status: UsageStatus,
):
    response = templates.TemplateResponse(
        request,
        "results.html",
        {
            **results_context,
            "page_title": page_title,
        },
    )
    response.headers.update(get_classification_cache_headers())
    if push_url:
        response.headers["HX-Push-Url"] = new_url
    add_quota_headers(response, usage_status)
    return response


def get_related_mapping_products(product: MappingProduct) -> list[MappingProduct]:
    related_products: list[MappingProduct] = []
    for slug in product.related_slugs:
        related_product = get_mapping_product(slug)
        if related_product:
            related_products.append(related_product)
    return related_products


# Serve the main homepage
@router.get("/", response_class=HTMLResponse)
@router.head("/")  # Add HEAD support
async def read_root(request: Request):
    """Serves the main homepage with Cloudflare-friendly caching."""

    # For HEAD requests, return just headers
    if request.method == "HEAD":
        return Response(headers=build_page_headers("https://classifast.com/"))

    today = datetime.now()
    response = templates.TemplateResponse(
        request,
        "index.html",
        {"current_year": today.year},
    )

    # Cloudflare-friendly cache headers (same as classifier pages)
    response.headers.update(build_page_headers("https://classifast.com/"))

    return response


@router.get("/mapping", include_in_schema=False)
@router.head("/mapping", include_in_schema=False)
async def redirect_mapping_index_no_slash(request: Request):
    query_string = f"?{request.url.query}" if request.url.query else ""
    return RedirectResponse(url=f"/mapping/{query_string}", status_code=301)


@router.get("/mappings", include_in_schema=False)
@router.head("/mappings", include_in_schema=False)
@router.get("/mappings/", include_in_schema=False)
@router.head("/mappings/", include_in_schema=False)
async def redirect_legacy_mapping_index(request: Request):
    query_string = f"?{request.url.query}" if request.url.query else ""
    return RedirectResponse(url=f"/mapping/{query_string}", status_code=301)


@router.get("/mapping/", response_class=HTMLResponse)
@router.head("/mapping/")
async def show_mapping_index(request: Request):
    canonical_url = build_mapping_canonical_url()
    if request.method == "HEAD":
        return Response(headers=build_page_headers(canonical_url))

    today = datetime.now()
    products = list_mapping_products()
    response = templates.TemplateResponse(
        request,
        "mapping_index.html",
        {
            "products": products,
            "featured_products": [product for product in products if product.featured],
            "canonical_url": canonical_url,
            "current_year": today.year,
        },
    )
    response.headers.update(build_page_headers(canonical_url))
    return response


@router.get("/mapping/{slug}", include_in_schema=False)
@router.head("/mapping/{slug}", include_in_schema=False)
async def redirect_mapping_product_no_slash(slug: str, request: Request):
    product = get_mapping_product(slug)
    if not product:
        raise HTTPException(status_code=404, detail="Mapping product not found")
    query_string = f"?{request.url.query}" if request.url.query else ""
    return RedirectResponse(
        url=f"/mapping/{product.slug}/{query_string}", status_code=301
    )


@router.get("/mappings/{slug}", include_in_schema=False)
@router.head("/mappings/{slug}", include_in_schema=False)
@router.get("/mappings/{slug}/", include_in_schema=False)
@router.head("/mappings/{slug}/", include_in_schema=False)
async def redirect_legacy_mapping_product(slug: str, request: Request):
    product = get_mapping_product(slug)
    if not product:
        raise HTTPException(status_code=404, detail="Mapping product not found")
    query_string = f"?{request.url.query}" if request.url.query else ""
    return RedirectResponse(
        url=f"/mapping/{product.slug}/{query_string}", status_code=301
    )


@router.get("/mapping/{slug}/sample")
async def download_mapping_sample(slug: str):
    product = get_mapping_product(slug)
    if not product:
        raise HTTPException(status_code=404, detail="Mapping product not found")

    file_path = BASE_DIR / product.sample_file_path
    # Resolve to absolute path and verify it's within BASE_DIR
    file_path = file_path.resolve()
    if not file_path.is_relative_to(BASE_DIR):
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Sample file not found")

    response = FileResponse(file_path, filename=file_path.name)
    response.headers.update(
        build_cache_headers(get_sample_cache_profile(product.sample_file_path))
    )
    response.headers["Vary"] = "Accept-Encoding"
    return response


@router.get("/mappings/{slug}/sample", include_in_schema=False)
async def redirect_legacy_mapping_sample(slug: str):
    product = get_mapping_product(slug)
    if not product:
        raise HTTPException(status_code=404, detail="Mapping product not found")
    return RedirectResponse(url=f"/mapping/{product.slug}/sample", status_code=301)


@router.get("/mapping/{slug}/", response_class=HTMLResponse)
@router.head("/mapping/{slug}/")
async def show_mapping_product_page(request: Request, slug: str):
    product = get_mapping_product(slug)
    if not product:
        raise HTTPException(status_code=404, detail="Mapping product not found")

    canonical_url = build_mapping_canonical_url(product.slug)
    if request.method == "HEAD":
        return Response(headers=build_page_headers(canonical_url))

    today = datetime.now()
    response = templates.TemplateResponse(
        request,
        "mapping_product.html",
        {
            "product": product,
            "canonical_url": canonical_url,
            "related_products": get_related_mapping_products(product),
            "all_products": list_mapping_products(),
            "current_year": today.year,
        },
    )
    response.headers.update(build_page_headers(canonical_url))
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
    version: str | None = Query(None),
    push_url: bool | None = Query(None),
    track_usage: bool = Query(True),
    url_change: bool | None = Query(None),
):
    """
    GET endpoint for retrieving classification results as an HTML fragment.
    Optimized for HTMX lazy loading and caching.
    """
    normalized_description = _normalize_product_description(product_description)
    upper_type, config = _get_classifier_config_or_404(classifier_type)
    version, top_k, default_version, default_top_k = _resolve_classifier_options(
        upper_type, config, version, top_k
    )

    logger.info(
        "WEB received GET fragment request for '%s' with version '%s'. Push URL: %s. Track usage: %s",
        upper_type,
        version,
        push_url,
        track_usage,
    )

    push_url, track_usage = _resolve_fragment_flags(
        request, push_url, track_usage, url_change
    )
    redis_client = _get_redis_client(request)
    await _maybe_verify_checkout_return(request, redis_client)
    new_url = _build_fragment_push_url(
        upper_type,
        normalized_description,
        version,
        default_version,
        top_k,
        default_top_k,
    )

    if track_usage:
        usage_status = await check_usage(request, redis_client)
        if not usage_status.allowed:
            return _render_paywall_fragment(request, usage_status, push_url, new_url)
    else:
        usage_status = _build_unmetered_usage_status()

    if not normalized_description:
        return _render_empty_results_fragment(
            request, normalized_description, usage_status
        )

    try:
        results_context = build_classification_results_context(
            request=request,
            classifier_type=upper_type,
            query=normalized_description,
            version=version,
            top_k=top_k,
        )
    except HTTPException:
        # Let HTTP exceptions propagate
        raise
    except Exception as e:
        logger.error("Error during '%s' fragment classification: %s", upper_type, e)
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )

    page_title = (
        f"{upper_type} codes for '{normalized_description.title()}'"
        if push_url
        else None
    )
    response = _render_classification_results_fragment(
        request, results_context, page_title, push_url, new_url, usage_status
    )
    if track_usage:
        await increment_usage(request, redis_client, usage_status)

    return response


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
    default_top_k = get_default_top_k(effective_classifier_type)

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
        # Normalize internal whitespace (collapse multiple spaces/newlines into single space)
        decoded_search_query = re.sub(r"\s+", " ", decoded_search_query).strip()
        # Sanitize the decoded query
        # Relaxed sanitization: allow characters like apostrophes, but keep length limit
        if len(decoded_search_query) > 4000:
            decoded_search_query = decoded_search_query[:4000]
            decoded_search_query = decoded_search_query.strip()

    # Build canonical URL
    # URL-encode slug to handle non-Latin characters in HTTP headers
    canonical_url = f"https://classifast.com/{effective_classifier_type}"
    if decoded_search_query:
        slug = slugify(decoded_search_query)
        canonical_url += f"/{quote(slug, safe='')}"

    # Ensure trailing slash for consistency with redirects and sitemap
    if not canonical_url.endswith("/"):
        canonical_url += "/"

    # For HEAD requests, return just headers
    if request.method == "HEAD":
        return Response(headers=build_page_headers(canonical_url))

    # Validate top_k parameter
    if top_k is None or top_k < 1 or top_k > 100:
        top_k = default_top_k

    # Get first version for default handling
    versions_list = list(config["versions"].keys())
    first_version = versions_list[0] if versions_list else ""
    validated_version = version if version in config["versions"] else first_version

    # Initialize results data structure
    results_data = {
        "results_for_query": [],
        "query": decoded_search_query,
        "base_url": "",
        "tooltip": "",
        "total_request_time": 0,
    }

    raw_example = config["example"].strip()
    display_example = raw_example if raw_example else ""

    # Track whether the base page is seeded from the configured example text.
    # This must always be initialized before the template context is built.
    default_example_prefill = False

    trigger_search_on_load = False

    if decoded_search_query:
        trigger_search_on_load = True
    else:
        # If no search query (base URL), use example query
        example_query = raw_example
        if example_query:
            default_example_prefill = True
            results_data["query"] = example_query
            try:
                results_data = build_classification_results_context(
                    request=request,
                    classifier_type=effective_classifier_type,
                    query=example_query,
                    version=validated_version,
                    top_k=top_k,
                )
            except Exception as e:
                logger.warning(
                    "SSR fallback for '%s' page classification due to %s: %s",
                    effective_classifier_type,
                    type(e).__name__,
                    e,
                )
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
            "example": display_example,
            "url_params": {
                "search": decoded_search_query,
                "version": (
                    validated_version
                    if validated_version and validated_version != first_version
                    else ""
                ),
                "top_k": top_k,
            },
            "default_example_prefill": default_example_prefill,
            "trigger_search_on_load": trigger_search_on_load,
            "default_top_k": default_top_k,
            "first_version": first_version,
            "canonical_url": canonical_url,
            "current_year": current_year,
            "current_month_name": current_month_name,
            **results_data,
        },
    )

    # Cloudflare-friendly cache headers (aligned with homepage)
    response.headers.update(build_page_headers(canonical_url))

    return response
