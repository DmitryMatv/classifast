import logging
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote_plus, urlencode

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

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


def get_sample_cache_profile(sample_path: str):
    if sample_path.endswith(".csv"):
        return STATIC_TEXT
    return STATIC_MEDIA


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
    # Normalize inputs early to ensure cache hits and prevent unnecessary API calls
    normalized_description = re.sub(r"\s+", " ", product_description).strip()
    upper_type = classifier_type.strip().upper()
    config = CLASSIFIER_CONFIG.get(upper_type)
    if not config:
        raise HTTPException(
            status_code=404, detail=f"Classifier '{classifier_type}' not found"
        )

    default_top_k = get_default_top_k(upper_type)
    if top_k is None:
        top_k = default_top_k

    versions_list = list(config["versions"].keys())
    default_version = versions_list[0] if versions_list else ""
    if version is None:
        version = default_version

    logger.info(
        "WEB received GET fragment request for '%s' with version '%s'. Push URL: %s. Track usage: %s",
        upper_type,
        version,
        push_url,
        track_usage,
    )

    if push_url is None:
        push_url = True if url_change is None else url_change
    if "track_usage" not in request.query_params and url_change is not None:
        track_usage = url_change

    # Handle checkout return with token verification (also on fragment requests)
    checkout_success = request.query_params.get("checkout")
    checkout_token = request.query_params.get("checkout_token")
    if checkout_success == "success" and checkout_token:
        redis_client = getattr(request.app.state, "redis_client", None)
        await verify_checkout_token(checkout_token, request, redis_client)

    # Build the new URL early so we can set it before usage check
    # This ensures the URL updates even if user hits the paywall
    slug = slugify(normalized_description.replace("/", " "))
    new_url = f"/{upper_type}"
    if slug:
        # URL-encode slug to handle non-Latin characters (Chinese, Arabic, etc.)
        # HTTP headers require Latin-1 encoding
        new_url += f"/{quote(slug, safe='')}"

    params: dict[str, str | int] = {}
    if version and version != default_version:
        params["version"] = version
    if top_k != default_top_k:
        params["top_k"] = top_k
    if params:
        new_url += f"?{urlencode(params)}"

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
            if push_url:
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
    if push_url:
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
    if push_url:
        response.headers["HX-Push-Url"] = new_url

    # Increment usage counter for real user-triggered searches.
    if track_usage:
        await increment_usage(request, redis_client, usage_status)
    add_quota_headers(response, usage_status)

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

    # Determine if we should trigger a search on load
    # This is true if we have a URL search query OR if we're falling back to the example
    trigger_search_on_load = False
    default_example_prefill = False

    if decoded_search_query:
        trigger_search_on_load = True
    else:
        # If no search query (base URL), use example query
        example_query = raw_example
        if example_query:
            results_data["query"] = example_query
            trigger_search_on_load = True
            default_example_prefill = True

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
                "version": version if version and version != first_version else "",
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
