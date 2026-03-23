import logging
import re
import time
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
from .classifier_config import CLASSIFIER_CONFIG
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
    # Preserve periods, commas, apostrophes, and parentheses while removing other special characters
    text = re.sub(r"[^\w\s.,'()-]", "", text)
    text = re.sub(r"[\s]+", "_", text)
    return text.strip("_")


def _decode_search_query(search_query: str) -> str:
    if not search_query or not search_query.strip():
        return ""

    decoded_search_query = (
        unquote_plus(search_query).rstrip("/").replace("/", " ").replace("_", " ")
    )
    decoded_search_query = re.sub(r"\s+", " ", decoded_search_query).strip()
    if len(decoded_search_query) > 4000:
        decoded_search_query = decoded_search_query[:4000].strip()
    return decoded_search_query


def _build_classifier_canonical_url(
    classifier_type: str, decoded_search_query: str
) -> str:
    canonical_url = f"https://classifast.com/{classifier_type}"
    if decoded_search_query:
        canonical_url += f"/{quote(slugify(decoded_search_query), safe='')}"
    if not canonical_url.endswith("/"):
        canonical_url += "/"
    return canonical_url


def _build_classifier_relative_canonical_path(
    classifier_type: str, decoded_search_query: str
) -> str:
    canonical_path = f"/{classifier_type}"
    if decoded_search_query:
        canonical_path += f"/{quote(slugify(decoded_search_query), safe='')}"
    if not canonical_path.endswith("/"):
        canonical_path += "/"
    return canonical_path


def _get_default_version(config: dict) -> str:
    versions_list = list(config["versions"].keys())
    return versions_list[0] if versions_list else ""


def _get_example_query(config: dict) -> str:
    return config["example"].replace("Example:", "").strip()


def _build_classifier_page_state(
    request: Request,
    classifier_type: str,
    search_query: str,
    version: str | None,
    top_k: int,
) -> dict:
    config = CLASSIFIER_CONFIG[classifier_type]
    decoded_search_query = _decode_search_query(search_query)
    default_version = _get_default_version(config)
    recognized_query_params = {"version", "top_k", "checkout", "checkout_token"}
    query_param_keys = set(request.query_params.keys())
    has_version_param = "version" in query_param_keys
    has_top_k_param = "top_k" in query_param_keys
    has_checkout_param = "checkout" in query_param_keys
    has_checkout_token_param = "checkout_token" in query_param_keys
    has_checkout_params = has_checkout_param or has_checkout_token_param
    has_any_query_params = bool(request.query_params)
    has_non_content_query_params = bool(query_param_keys - recognized_query_params)
    is_generated_search_page = bool(decoded_search_query)
    is_content_variant_url = has_version_param or has_top_k_param
    should_redirect_to_canonical = (
        has_non_content_query_params and not has_checkout_params
    )
    should_ssr_initial_results = (
        not is_generated_search_page and not has_any_query_params
    )
    page_robots_directive = (
        "index, follow"
        if not is_generated_search_page and not is_content_variant_url
        else "noindex, follow"
    )
    cache_profile = HTML_PAGE
    is_indexable_canonical_page = (
        not is_generated_search_page
        and not is_content_variant_url
        and not has_any_query_params
    )
    if has_checkout_params:
        page_robots_directive = "noindex, follow"
        cache_profile = NO_STORE
        is_indexable_canonical_page = False
    elif should_redirect_to_canonical:
        page_robots_directive = "noindex, follow"
        is_indexable_canonical_page = False
    initial_results_query = decoded_search_query or _get_example_query(config)

    return {
        "canonical_url": _build_classifier_canonical_url(
            classifier_type, decoded_search_query
        ),
        "canonical_path": _build_classifier_relative_canonical_path(
            classifier_type, decoded_search_query
        ),
        "cache_profile": cache_profile,
        "config": config,
        "decoded_search_query": decoded_search_query,
        "default_version": default_version,
        "has_any_query_params": has_any_query_params,
        "has_checkout_params": has_checkout_params,
        "has_non_content_query_params": has_non_content_query_params,
        "initial_results_query": initial_results_query,
        "is_content_variant_url": is_content_variant_url,
        "is_generated_search_page": is_generated_search_page,
        "is_indexable_canonical_page": is_indexable_canonical_page,
        "page_robots_directive": page_robots_directive,
        "selected_version": version or default_version,
        "should_redirect_to_canonical": should_redirect_to_canonical,
        "should_ssr_initial_results": should_ssr_initial_results,
        "top_k": top_k if 1 <= top_k <= 100 else 10,
    }


def _build_ssr_results_context(
    request: Request,
    classifier_type: str,
    query: str,
    version: str | None,
    top_k: int,
) -> dict:
    normalized_query = re.sub(r"\s+", " ", query).strip()
    if not normalized_query:
        return {
            "query": "",
            "results_for_query": [],
            "base_url": "",
            "tooltip": "",
            "total_request_time": 0,
        }

    start_total_time = time.perf_counter()
    quantization_cache = getattr(request.app.state, "collection_quantization_cache", {})
    zclient = getattr(request.app.state, "zclient", None)
    result = perform_classification(
        embed_client=getattr(request.app.state, "embed_client", None),
        qdrant_client=getattr(request.app.state, "qdrant_client", None),
        query=normalized_query,
        classifier_type=classifier_type,
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
    }


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
        headers["X-Robots-Tag"] = "index, follow"
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
    top_k: int = 10,
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
    top_k: int = Query(10, ge=1, le=100),
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
    normalized_description = re.sub(r"\s+", " ", product_description).strip()
    upper_type = classifier_type.strip().upper()

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

    # Handle version query param
    config = CLASSIFIER_CONFIG.get(upper_type)
    if config:
        versions_list = list(config["versions"].keys())
        default_version = versions_list[0] if versions_list else None

        # Only append version if it's not the default one
        if version and version != default_version:
            new_url += f"?{urlencode({'version': version})}"

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
    top_k: int = 10,
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
    page_state = _build_classifier_page_state(
        request, effective_classifier_type, search_query, version, top_k
    )
    canonical_url = page_state["canonical_url"]
    canonical_path = page_state["canonical_path"]

    if page_state["should_redirect_to_canonical"]:
        return RedirectResponse(url=canonical_path, status_code=301)

    # For HEAD requests, return just headers
    if request.method == "HEAD":
        headers = build_cache_headers(page_state["cache_profile"])
        headers["Vary"] = "Accept-Encoding"
        headers["Content-Type"] = "text/html; charset=utf-8"
        headers["Link"] = f'<{canonical_url}>; rel="canonical"'
        headers["X-Robots-Tag"] = page_state["page_robots_directive"]
        return Response(headers=headers)

    # Handle checkout return with token verification
    checkout_success = request.query_params.get("checkout")
    checkout_token = request.query_params.get("checkout_token")
    if checkout_success == "success" and checkout_token:
        redis_client = getattr(request.app.state, "redis_client", None)
        await verify_checkout_token(checkout_token, request, redis_client)

    results_data = {
        "results_for_query": [],
        "query": page_state["initial_results_query"],
        "base_url": "",
        "tooltip": "",
        "total_request_time": 0,
    }
    used_ssr_initial_results = False
    effective_robots_directive = page_state["page_robots_directive"]
    effective_cache_profile = page_state["cache_profile"]
    if page_state["should_ssr_initial_results"]:
        try:
            results_data = _build_ssr_results_context(
                request,
                effective_classifier_type,
                page_state["initial_results_query"],
                page_state["selected_version"],
                page_state["top_k"],
            )
            used_ssr_initial_results = True
        except HTTPException as exc:
            logger.warning(
                "Falling back to HTMX initial load for '%s' landing page after SSR failure: %s",
                effective_classifier_type,
                exc.detail,
            )
            effective_robots_directive = "noindex, follow"
            effective_cache_profile = NO_STORE
        except Exception as e:
            logger.warning(
                "Falling back to HTMX initial load for '%s' landing page after SSR failure: %s",
                effective_classifier_type,
                e,
            )
            effective_robots_directive = "noindex, follow"
            effective_cache_profile = NO_STORE

    today = datetime.now()
    current_year = today.year
    current_month_name = today.strftime("%B")
    primary_version_label = page_state["default_version"]

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
                "search": page_state["decoded_search_query"],
                "version": (
                    version
                    if version and version != page_state["default_version"]
                    else ""
                ),
                "top_k": page_state["top_k"],
            },
            "meta_robots_content": effective_robots_directive,
            "primary_version_label": primary_version_label,
            "should_ssr_initial_results": used_ssr_initial_results,
            "should_trigger_initial_results_load": (
                not used_ssr_initial_results
                and bool(page_state["initial_results_query"])
            ),
            "canonical_url": canonical_url,
            "current_year": current_year,
            "current_month_name": current_month_name,
            **results_data,
        },
    )

    # Cloudflare-friendly cache headers (aligned with homepage)
    response.headers.update(build_cache_headers(effective_cache_profile))
    response.headers["Vary"] = "Accept-Encoding"
    response.headers["Link"] = f'<{canonical_url}>; rel="canonical"'
    response.headers["X-Robots-Tag"] = effective_robots_directive

    return response
