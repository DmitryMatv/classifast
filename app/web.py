import html
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlparse

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from starlette.templating import _TemplateResponse

from app.cache_profiles import CacheProfile

from .cache_profiles import (
    HTML_PAGE,
    NO_STORE,
    STATIC_MEDIA,
    STATIC_TEXT,
    build_cache_headers,
)
from .classifier_config import CLASSIFIER_CONFIG
from .classifier_page_delivery import (
    REMOVED_CLASSIFIER_TYPES,
    build_classification_results_context,
    build_classifier_canonical_url,
    build_classifier_page_context,
    build_classifier_redirect_url,
    build_fragment_page_title,
    build_fragment_push_url,
    decode_search_query,
    get_classifier_or_404,
    get_default_top_k,
    get_homepage_popular_lookup_links,
    maybe_seed_classifier_page_results,
    normalize_product_description,
    render_classification_results_fragment,
    render_empty_results_fragment,
    resolve_classifier_options,
    resolve_fragment_push_url,
    should_ssr,
)
from .dependencies import templates
from .google_crawlers import is_verified_google_search_crawler_request
from .mapping_store import (
    MappingProduct,
    get_mapping_product,
    list_mapping_products,
)
from .usage_tracker import (
    FREE_USER_LIMIT,
    QuotaUnavailableError,
    UsageStatus,
    add_quota_headers,
    reserve_usage,
    verify_checkout_token,
)

logger = logging.getLogger(__name__)

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parent.parent


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


def _get_redis_client(request: Request):
    return getattr(request.app.state, "redis_client", None)


async def _maybe_verify_checkout_return(request: Request, redis_client) -> None:
    checkout_success = request.query_params.get("checkout")
    checkout_token = request.query_params.get("checkout_token")
    if checkout_success == "success" and checkout_token:
        await verify_checkout_token(checkout_token, request, redis_client)


def _render_paywall_fragment(
    request: Request,
    usage_status: UsageStatus,
    push_url: bool,
    new_url: str,
) -> _TemplateResponse:
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


def _render_status_fragment(
    message: str,
    status_code: int,
) -> HTMLResponse:
    response = HTMLResponse(
        content=(
            '<div class="bg-white border border-amber-200 rounded-lg p-6 shadow">'
            f'<p class="text-gray-700">{html.escape(message)}</p>'
            "</div>"
        ),
        status_code=status_code,
    )
    response.headers.update(build_cache_headers(NO_STORE))
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
        {
            "current_year": today.year,
            "popular_lookup_links": get_homepage_popular_lookup_links(),
        },
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
    if upper_type in REMOVED_CLASSIFIER_TYPES:
        raise HTTPException(
            status_code=410,
            detail=f"Classifier '{classifier_type}' is no longer available",
        )
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
    url_change: bool | None = Query(None),
):
    """
    GET endpoint for retrieving classification results as an HTML fragment.
    Optimized for HTMX lazy loading and caching.
    """
    normalized_description = normalize_product_description(product_description)
    upper_type, config = get_classifier_or_404(classifier_type)
    version, top_k, default_version = resolve_classifier_options(
        config,
        version,
        top_k,
        get_default_top_k(upper_type),
        allow_invalid_version=True,
    )

    logger.info(
        "WEB received GET fragment request for '%s' with version '%s'. Push URL: %s",
        upper_type,
        version,
        push_url,
    )

    push_url = resolve_fragment_push_url(push_url, url_change)
    redis_client = _get_redis_client(request)
    await _maybe_verify_checkout_return(request, redis_client)
    new_url = build_fragment_push_url(
        upper_type,
        normalized_description,
        version,
        default_version,
        top_k,
        get_default_top_k(upper_type),
    )

    if not normalized_description:
        return render_empty_results_fragment(request, normalized_description)

    is_verified_crawler = await is_verified_google_search_crawler_request(request)
    if is_verified_crawler:
        logger.info("Bypassing quota for verified Google search crawler")
    else:
        try:
            usage_status = await reserve_usage(request, redis_client)
            if not usage_status.allowed:
                return _render_paywall_fragment(
                    request, usage_status, push_url, new_url
                )
        except QuotaUnavailableError as e:
            logger.warning("Quota unavailable for '%s' fragment: %s", upper_type, e)
            return _render_status_fragment(
                str(e),
                status_code=503,
            )

    try:
        results_context = await build_classification_results_context(
            request=request,
            classifier_type=upper_type,
            query=normalized_description,
            version=version,
            top_k=top_k,
        )
    except HTTPException as exc:
        exc.headers = {
            **(exc.headers or {}),
            **build_cache_headers(NO_STORE),
        }
        raise
    except Exception as e:
        logger.error("Error during '%s' fragment classification: %s", upper_type, e)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}",
            headers=build_cache_headers(NO_STORE),
        )

    page_title = (
        build_fragment_page_title(upper_type, normalized_description)
        if push_url
        else None
    )
    response = render_classification_results_fragment(
        request, results_context, page_title, push_url, new_url
    )

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
    upper_type, config = get_classifier_or_404(classifier_type)

    normalized_search_query = search_query.rstrip("/")
    canonical_search_query = (
        f"{normalized_search_query}/" if normalized_search_query else ""
    )
    if classifier_type != upper_type or search_query != canonical_search_query:
        redirect_url = build_classifier_redirect_url(
            upper_type, search_query, request.url.query
        )
        return RedirectResponse(url=redirect_url, status_code=301)

    default_top_k = get_default_top_k(upper_type)
    redis_client = _get_redis_client(request)
    await _maybe_verify_checkout_return(request, redis_client)

    decoded_search_query = decode_search_query(search_query)
    canonical_url = build_classifier_canonical_url(upper_type, decoded_search_query)

    if request.url.path != urlparse(canonical_url).path:
        redirect_url = build_classifier_redirect_url(
            upper_type, search_query, request.url.query
        )
        return RedirectResponse(url=redirect_url, status_code=301)

    if request.method == "HEAD":
        return Response(headers=build_page_headers(canonical_url))

    validated_version, top_k, first_version = resolve_classifier_options(
        config, version, top_k, default_top_k
    )
    raw_example = config["example"].strip()
    display_example = raw_example if raw_example else ""
    allow_query_ssr = should_ssr(
        decoded_search_query,
        bool(request.query_params),
        canonical_url,
    )
    (
        results_data,
        default_example_prefill,
        trigger_search_on_load,
        ssr_state,
    ) = await maybe_seed_classifier_page_results(
        request,
        upper_type,
        decoded_search_query,
        raw_example,
        validated_version,
        top_k,
        allow_query_ssr=allow_query_ssr,
    )

    response = templates.TemplateResponse(
        request,
        "classifier_page.html",
        build_classifier_page_context(
            upper_type,
            config,
            display_example,
            decoded_search_query,
            validated_version,
            first_version,
            top_k,
            default_top_k,
            canonical_url,
            results_data,
            default_example_prefill,
            trigger_search_on_load,
            results_loaded=ssr_state == "success",
        ),
    )
    response.headers.update(build_page_headers(canonical_url))
    if ssr_state == "failure":
        response.headers.update(build_cache_headers(NO_STORE))
        response.headers["X-Robots-Tag"] = "noindex, nofollow"

    return response
