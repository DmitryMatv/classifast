import logging
import re
import time
from urllib.parse import quote, unquote_plus, urlencode

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from .classifier import perform_classification
from .classifier_config import CLASSIFIER_CONFIG
from .dependencies import limiter, templates
from .usage_tracker import (
    FREE_USER_DAILY_LIMIT,
    add_quota_headers,
    check_usage,
    increment_usage,
    set_tracking_cookie,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def slugify(text: str) -> str:
    """
    Slugify utility for SEO-friendly URLs.
    Matches the logic used in show_classifier_page_with_query and frontend JS.
    """
    if not text:
        return ""
    # Sanitize input: limit length and remove harmful characters
    text = str(text)[:200]  # Limit to 200 chars max
    # Preserve periods, commas, apostrophes, and parentheses while removing other special characters
    text = re.sub(r"[^\w\s.,'()-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")


# Serve the main homepage
@router.get("/", response_class=HTMLResponse)
@router.head("/")  # Add HEAD support
async def read_root(request: Request):
    """Serves the main homepage with Cloudflare-friendly caching."""

    # For HEAD requests, return just headers
    if request.method == "HEAD":
        headers = {
            "Cache-Control": "public, max-age=86400, s-maxage=604800, stale-if-error=86400",
            "Vary": "Accept-Encoding",
            "Content-Type": "text/html; charset=utf-8",
            "Link": '<https://classifast.com/>; rel="canonical"',
        }
        return Response(headers=headers)

    response = templates.TemplateResponse("index.html", {"request": request})

    # Cloudflare-friendly cache headers (same as classifier pages)
    response.headers["Cache-Control"] = (
        "public, max-age=86400, s-maxage=604800, stale-if-error=86400"
    )
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
    upper_type = classifier_type.upper()
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
    top_k: int = 10,
):
    """
    Serves the base classifier page.
    """
    return await show_classifier_page_with_query(
        request, classifier_type, "", version, top_k
    )


@router.get("/{classifier_type}/fragment", response_class=HTMLResponse)
@limiter.limit("60/minute")
async def get_classification_fragment(
    request: Request,
    classifier_type: str,
    product_description: str = Query(..., alias="product_description"),
    top_k: int = Query(10),
    version: str = Query(...),
    prevent_url_change: bool = Query(False),
):
    """
    GET endpoint for retrieving classification results as an HTML fragment.
    Optimized for HTMX lazy loading and caching.
    """
    logger.info(
        "Received GET fragment request for '%s' with version '%s'. Prevent URL change: %s",
        classifier_type,
        version,
        prevent_url_change,
    )

    # Check usage limits before processing
    redis_client = getattr(request.app.state, "redis_client", None)
    usage_status = await check_usage(request, redis_client)

    if not usage_status.allowed:
        # Return paywall template
        response = templates.TemplateResponse(
            "paywall.html",
            {
                "request": request,
                "limit": usage_status.limit,
                "is_authenticated": usage_status.is_authenticated,
                "free_user_limit": FREE_USER_DAILY_LIMIT,
            },
        )
        add_quota_headers(response, usage_status)
        if usage_status.tracking_id:
            set_tracking_cookie(response, usage_status.tracking_id)
        return response

    # Reuse the logic from handle_classify but adapted for GET
    # Handle empty query gracefully - also remove trailing slashes and replace with spaces
    normalized_description = product_description.strip()
    if not normalized_description:
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "query": product_description,
                "results_for_query": [],
            },
        )

    start_total_time = time.perf_counter()

    try:
        # Use shared classification service
        result = await perform_classification(
            embed_client=request.app.state.embed_client,
            qdrant_client=request.app.state.qdrant_client,
            query=normalized_description,
            classifier_type=classifier_type,
            version=version,
            top_k=top_k,
            quantization_cache=getattr(
                request.app.state, "collection_quantization_cache", None
            ),
        )

        classification_results = result["results"]

    except HTTPException:
        # Let HTTP exceptions propagate
        raise
    except Exception as e:
        logger.error(
            "Error during '%s' fragment classification: %s", classifier_type, e
        )
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )

    end_total_time = time.perf_counter()
    total_request_time = end_total_time - start_total_time

    # Calculate dynamic page title for OOB swap
    page_title = None
    if not prevent_url_change:
        page_title = (
            f"{classifier_type.upper()} codes for '{normalized_description.title()}'"
        )

    # Render the results partial
    response = templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "query": product_description,
            "results_for_query": classification_results,
            "base_url": result["version_config"].get("base_url", ""),
            "tooltip": result["version_config"].get("tooltip", ""),
            "total_request_time": total_request_time,
            "page_title": page_title,
        },
    )

    # Add strong caching headers for this fragment
    # This is safe because it's for specific query/version combinations
    response.headers["Cache-Control"] = (
        "public, max-age=86400, s-maxage=604800, stale-while-revalidate=3600, stale-if-error=86400"
    )
    response.headers["Vary"] = "Accept-Encoding"

    # Calculate new URL for HTMX to push (server-side URL updating)
    # Use uppercase classifier_type for URLs
    upper_type = classifier_type.upper()
    slug = slugify(normalized_description.replace("/", " "))
    new_url = f"/{upper_type}"
    if slug:
        # URL-encode slug to handle non-Latin characters (Chinese, Arabic, etc.)
        # HTTP headers require Latin-1 encoding
        new_url += f"/{quote(slug, safe='')}"

    # Handle version query param
    config = CLASSIFIER_CONFIG.get(upper_type)
    if config:
        versions_list = list(config.get("versions", {}).keys())
        default_version = versions_list[0] if versions_list else None

        # Only append version if it's not the default one
        if version and version != default_version:
            new_url += f"?{urlencode({'version': version})}"

    # Set HTMX header to update URL in browser address bar
    if not prevent_url_change:
        response.headers["HX-Push-Url"] = new_url

    # Increment usage counter and set tracking cookie
    await increment_usage(request, redis_client, usage_status)
    add_quota_headers(response, usage_status)
    if usage_status.tracking_id:
        set_tracking_cookie(response, usage_status.tracking_id)

    return response


@router.get("/{classifier_type}/{search_query:path}", response_class=HTMLResponse)
@router.head("/{classifier_type}/{search_query:path}")
async def show_classifier_page_with_query(
    request: Request,
    classifier_type: str,
    search_query: str = "",
    version: str | None = None,
    top_k: int = 10,
):
    """
    Serves the specific classifier page with clean URL structure.
    Handles both base URLs like /NAICS and search URLs like /NAICS/gamedev-studio
    Also redirects lowercase classifier types to uppercase.
    """
    upper_type = classifier_type.upper()
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
    classifier_type = upper_type

    # Handle empty search query for base URLs
    decoded_search_query = ""
    if search_query and search_query.strip():
        decoded_search_query = (
            unquote_plus(search_query)
            .rstrip("/")
            .replace("/", " ")
            .replace("-", " ")
            .strip()
        )
        # Sanitize the decoded query
        # Relaxed sanitization: allow characters like apostrophes, but keep length limit
        if len(decoded_search_query) > 4000:
            decoded_search_query = decoded_search_query[:4000]

        decoded_search_query = decoded_search_query.strip()

    # Build canonical URL
    # URL-encode slug to handle non-Latin characters in HTTP headers
    canonical_url = f"https://classifast.com/{classifier_type}"
    if decoded_search_query:
        slug = slugify(decoded_search_query)
        canonical_url += f"/{quote(slug, safe='')}"

    # Ensure trailing slash for consistency with redirects and sitemap
    if not canonical_url.endswith("/"):
        canonical_url += "/"

    # For HEAD requests, return just headers
    if request.method == "HEAD":
        headers = {
            "Cache-Control": "public, max-age=86400, s-maxage=604800, stale-if-error=86400",
            "Vary": "Accept-Encoding",
            "Content-Type": "text/html; charset=utf-8",
            "Link": f'<{canonical_url}>; rel="canonical"',
        }
        return Response(headers=headers)

    # Validate top_k parameter
    if top_k < 1 or top_k > 100:
        top_k = 10

    # Get first version for default handling
    versions_list = list(config.get("versions", {}).keys())
    first_version = versions_list[0] if versions_list else ""

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
        example_query = config.get("example", "").replace("Example:", "").strip()
        if example_query:
            results_data["query"] = example_query
            trigger_search_on_load = True

    response = templates.TemplateResponse(
        "classifier_page.html",
        {
            "request": request,
            "classifier_type": classifier_type,
            "title": config["title"],
            "heading": config["heading"],
            "description": config["description"],
            "versions": list(config.get("versions", {}).keys()),
            "example": config["example"],
            "url_params": {
                "search": decoded_search_query,
                "version": version if version and version != first_version else "",
                "top_k": top_k,
            },
            "trigger_search_on_load": trigger_search_on_load,
            "canonical_url": canonical_url,
            **results_data,
        },
    )

    # Cloudflare-friendly cache headers (aligned with homepage)
    response.headers["Cache-Control"] = (
        "public, max-age=86400, s-maxage=604800, stale-if-error=86400"
    )
    response.headers["Vary"] = "Accept-Encoding"
    response.headers["Link"] = f'<{canonical_url}>; rel="canonical"'
    response.headers["X-Robots-Tag"] = "index, follow"

    return response
