import logging
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Literal
from urllib.parse import quote, unquote_plus, urlencode, urlparse
from xml.etree import ElementTree

from fastapi import HTTPException, Request
from starlette.templating import _TemplateResponse

from .cache_profiles import CLASSIFICATION_RESULT, build_cache_headers
from .classifier import get_classification_cache_headers
from .classifier_config import CLASSIFIER_CONFIG, ClassifierConfig
from .dependencies import templates

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SSRState = Literal["not_attempted", "success", "failure"]


def _load_sitemap_query_paths() -> frozenset[str]:
    sitemap_path = BASE_DIR / "app" / "static" / "sitemap.xml"
    try:
        root = ElementTree.parse(sitemap_path).getroot()
    except (OSError, ElementTree.ParseError) as exc:
        logger.warning("Unable to load sitemap SEO query allowlist: %s", exc)
        return frozenset()

    query_paths: set[str] = set()
    for loc in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
        if not loc.text:
            continue

        path = urlparse(loc.text).path
        path_parts = [part for part in path.split("/") if part]
        if len(path_parts) == 2 and path_parts[0] in CLASSIFIER_CONFIG:
            query_paths.add(path)

    return frozenset(query_paths)


SITEMAP_QUERY_PATHS = _load_sitemap_query_paths()

# Keep anchor text curated while using the sitemap as the source of truth for
# which query pages are canonical and eligible for server-rendered results.
POPULAR_LOOKUP_CATALOG: dict[str, tuple[tuple[str, str], ...]] = {
    "UNSPSC": (
        ("Laptop computers", "/UNSPSC/laptop_computer/"),
        ("Desktop computers", "/UNSPSC/desktop_computer/"),
        ("Office chairs", "/UNSPSC/office_chair/"),
        ("Office desks", "/UNSPSC/office_desk/"),
        ("Copy paper", "/UNSPSC/copy_paper/"),
        ("Printer toner", "/UNSPSC/printer_toner/"),
        ("Safety gloves", "/UNSPSC/safety_gloves/"),
        ("Industrial pumps", "/UNSPSC/industrial_pump/"),
        ("Centrifugal pumps", "/UNSPSC/centrifugal_pump/"),
        ("Valves", "/UNSPSC/valve/"),
        ("Electric motors", "/UNSPSC/electric_motor/"),
        ("Air compressors", "/UNSPSC/air_compressor/"),
        ("Forklifts", "/UNSPSC/forklift/"),
        ("Generators", "/UNSPSC/generator/"),
        ("Network switches", "/UNSPSC/network_switch/"),
        ("Server racks", "/UNSPSC/server_rack/"),
        ("Laser printers", "/UNSPSC/laser_printer/"),
        ("Tablet computers", "/UNSPSC/tablet_computer/"),
        ("Printers", "/UNSPSC/printer/"),
        ("Ergonomic office chairs", "/UNSPSC/ergonomic_office_chair/"),
    ),
    "NAICS": (
        ("Property management", "/NAICS/property_management/"),
        ("Software development", "/NAICS/software_development/"),
        ("Construction", "/NAICS/construction/"),
        ("Restaurants", "/NAICS/restaurant/"),
        ("Accounting services", "/NAICS/accounting/"),
        ("Trucking", "/NAICS/trucking/"),
        ("Real estate", "/NAICS/real_estate/"),
        ("Plumbing contractors", "/NAICS/plumbing/"),
    ),
    "HS": (
        ("Smartphones", "/HS/smartphone/"),
        ("Coffee beans", "/HS/coffee_beans/"),
        ("Laptops", "/HS/laptops/"),
        ("Pharmaceuticals", "/HS/pharmaceuticals/"),
        ("Auto parts", "/HS/auto_parts/"),
        ("Furniture", "/HS/furniture/"),
        ("Televisions", "/HS/televisions/"),
        ("Medical devices", "/HS/medical_devices/"),
    ),
    "CN": (
        ("Frozen mangoes", "/CN/frozen_mangoes/"),
        ("Olive oil", "/CN/olive_oil/"),
        ("Wine", "/CN/wine/"),
        ("Pharmaceuticals", "/CN/pharmaceuticals/"),
        ("Electric vehicles", "/CN/electric_vehicles/"),
        ("Solar panels", "/CN/solar_panels/"),
        ("Electronics", "/CN/electronics/"),
        ("Medical devices", "/CN/medical_devices/"),
    ),
    "HTS": (
        ("Smartphones", "/HTS/smartphone/"),
        ("Hydraulic tools", "/HTS/hydraulic_tools/"),
        ("Auto parts", "/HTS/auto_parts/"),
        ("Steel products", "/HTS/steel_products/"),
        ("Coffee beans", "/HTS/coffee_beans/"),
        ("Electronics", "/HTS/electronics/"),
        ("Footwear", "/HTS/footwear/"),
        ("Medical devices", "/HTS/medical_devices/"),
    ),
    "GPC": (
        ("Smartphones", "/GPC/smartphone/"),
        ("Shampoo", "/GPC/shampoo/"),
        ("Coffee", "/GPC/coffee/"),
        ("Laptops", "/GPC/laptop/"),
        ("Toothpaste", "/GPC/toothpaste/"),
        ("Milk", "/GPC/milk/"),
        ("Bread", "/GPC/bread/"),
        ("Televisions", "/GPC/television/"),
    ),
    "GMDN": (
        ("Syringes", "/GMDN/syringe/"),
        ("Nebulizers", "/GMDN/nebulizer/"),
        ("Catheters", "/GMDN/catheter/"),
        ("Surgical masks", "/GMDN/surgical_mask/"),
        ("Pacemakers", "/GMDN/pacemaker/"),
        ("Blood pressure monitors", "/GMDN/blood_pressure_monitor/"),
        ("Infusion pumps", "/GMDN/infusion_pump/"),
        ("Stethoscopes", "/GMDN/stethoscope/"),
    ),
    "EMDN": (
        ("Syringes", "/EMDN/syringe/"),
        ("Nebulizers", "/EMDN/nebulizer/"),
        ("Catheters", "/EMDN/catheter/"),
        ("Surgical gloves", "/EMDN/surgical_gloves/"),
        ("Defibrillators", "/EMDN/defibrillator/"),
        ("Ultrasound scanners", "/EMDN/ultrasound_scanner/"),
        ("Stethoscopes", "/EMDN/stethoscope/"),
        ("Wheelchairs", "/EMDN/wheelchair/"),
    ),
    "ETIM": (
        ("Circuit breakers", "/ETIM/circuit_breaker/"),
        ("Cables", "/ETIM/cable/"),
        ("LED lamps", "/ETIM/LED_lamp/"),
        ("Switches", "/ETIM/switch/"),
        ("Power supplies", "/ETIM/power_supply/"),
        ("Connectors", "/ETIM/connector/"),
        ("Transformers", "/ETIM/transformer/"),
        ("Fuses", "/ETIM/fuse/"),
    ),
    "ISIC": (
        ("Pharmacies", "/ISIC/pharmacy/"),
        ("Forestry", "/ISIC/forestry/"),
        ("Software development", "/ISIC/software_development/"),
        ("Construction", "/ISIC/construction/"),
        ("Retail trade", "/ISIC/retail_trade/"),
        ("Manufacturing", "/ISIC/manufacturing/"),
        ("Education", "/ISIC/education/"),
        ("Financial services", "/ISIC/financial_services/"),
    ),
    "NACE": (
        ("Pharmacies", "/NACE/pharmacy/"),
        ("Software development", "/NACE/software_development/"),
        ("Construction", "/NACE/construction/"),
        ("Retail", "/NACE/retail/"),
        ("Used car dealerships", "/NACE/used_car_dealership/"),
        ("Manufacturing", "/NACE/manufacturing/"),
        ("Education", "/NACE/education/"),
        ("Financial services", "/NACE/financial_services/"),
    ),
    "CPV": (
        ("Office supplies", "/CPV/office_supplies/"),
        ("IT services", "/CPV/IT_services/"),
        ("Construction works", "/CPV/construction_works/"),
        ("Medical equipment", "/CPV/medical_equipment/"),
        ("Cleaning services", "/CPV/cleaning_services/"),
        ("Vehicles", "/CPV/vehicles/"),
        ("Software development", "/CPV/software_development/"),
        ("Consulting services", "/CPV/consulting_services/"),
    ),
    "NSN": (
        ("Batteries", "/NSN/battery/"),
        ("Bolts", "/NSN/bolt/"),
        ("Filters", "/NSN/filter/"),
        ("Hoses", "/NSN/hose/"),
        ("Pumps", "/NSN/pump/"),
        ("Valves", "/NSN/valve/"),
        ("Engines", "/NSN/engine/"),
        ("Generators", "/NSN/generator/"),
    ),
}


def get_popular_lookup_links(classifier_type: str) -> list[dict[str, str]]:
    """Return curated lookup links whose canonical pages are in the sitemap."""
    upper_type = classifier_type.strip().upper()
    return [
        {
            "classifier_type": upper_type,
            "label": label,
            "url": path,
        }
        for label, path in POPULAR_LOOKUP_CATALOG.get(upper_type, ())
        if path in SITEMAP_QUERY_PATHS
    ]


def get_homepage_popular_lookup_links() -> list[dict[str, str]]:
    """Return a small cross-standard set of high-value homepage lookups."""
    unspsc_links = {link["label"]: link for link in get_popular_lookup_links("UNSPSC")}
    homepage_unspsc_labels = (
        "Laptop computers",
        "Office chairs",
        "Industrial pumps",
        "Safety gloves",
        "Printers",
        "Network switches",
    )
    return (
        [
            unspsc_links[label]
            for label in homepage_unspsc_labels
            if label in unspsc_links
        ]
        + get_popular_lookup_links("HS")[:1]
        + get_popular_lookup_links("NAICS")[:1]
    )


def get_default_top_k(classifier_type: str) -> int:
    """Return the default number of results to show for a classifier page."""
    return 10


def slugify(text: str) -> str:
    """
    Slugify utility for SEO-friendly URLs.
    Matches the logic used in show_classifier_page_with_query and frontend JS.
    """
    if not text:
        return ""
    # Sanitize input: limit length and remove harmful characters
    text = text[:200]  # Limit to 200 chars max
    # Normalize internal whitespace first (collapse multiple spaces/newlines into single space)
    text = re.sub(r"\s+", " ", text)
    # Preserve punctuation that sanitize_query_text accepts so URL slugs round-trip
    # cleanly back into the classifier textbox.
    text = re.sub(r"[^\w\s.,:;'()-]", "", text)
    text = re.sub(r"[\s]+", "_", text)
    return text.strip("_")


def _build_classifier_search_slug(decoded_query: str, classifier_type: str) -> str:
    """Use a known underscore URL when it is an existing sitemap canonical."""
    slug = slugify(decoded_query)
    underscore_slug = slugify(decoded_query.replace("-", " "))
    underscore_path = f"/{classifier_type}/{quote(underscore_slug, safe='')}/"
    if underscore_slug != slug and underscore_path in SITEMAP_QUERY_PATHS:
        return underscore_slug
    return slug


def decode_search_query(search_query: str) -> str:
    if not search_query or not search_query.strip():
        return ""

    decoded_query = (
        unquote_plus(search_query).rstrip("/").replace("/", " ").replace("_", " ")
    )
    decoded_query = re.sub(r"\s+", " ", decoded_query).strip()
    if len(decoded_query) > 4000:
        decoded_query = decoded_query[:4000].strip()
    return decoded_query


def build_classifier_redirect_url(
    upper_type: str,
    search_query: str,
    query_string: str,
) -> str:
    redirect_url = f"/{upper_type}/"
    normalized_search_query = search_query.rstrip("/")
    if normalized_search_query:
        decoded_query = decode_search_query(normalized_search_query)
        slug = _build_classifier_search_slug(decoded_query, upper_type)
        redirect_url += f"{quote(slug, safe='')}/"
    if query_string:
        redirect_url += f"?{query_string}"
    return redirect_url


def build_classifier_canonical_url(classifier_type: str, decoded_query: str) -> str:
    canonical_url = f"https://classifast.com/{classifier_type}"
    if decoded_query:
        slug = _build_classifier_search_slug(decoded_query, classifier_type)
        canonical_url += f"/{quote(slug, safe='')}"
    if not canonical_url.endswith("/"):
        canonical_url += "/"
    return canonical_url


def build_fragment_push_url(
    upper_type: str,
    normalized_description: str,
    version: str,
    default_version: str,
    top_k: int,
    default_top_k: int,
) -> str:
    slug = _build_classifier_search_slug(
        normalized_description.replace("/", " "), upper_type
    )
    new_url = f"/{upper_type}/"
    if slug:
        new_url += f"{quote(slug, safe='')}/"

    params: dict[str, str | int] = {}
    if version and version != default_version:
        params["version"] = version
    if top_k != default_top_k:
        params["top_k"] = top_k
    if params:
        new_url += f"?{urlencode(params)}"

    return new_url


def normalize_product_description(product_description: str) -> str:
    return re.sub(r"\s+", " ", product_description).strip()


def get_classifier_or_404(classifier_type: str) -> tuple[str, ClassifierConfig]:
    upper_type = classifier_type.strip().upper()
    config = CLASSIFIER_CONFIG.get(upper_type)
    if not config:
        raise HTTPException(
            status_code=404, detail=f"Classifier '{classifier_type}' not found"
        )
    return upper_type, config


def resolve_classifier_options(
    config: ClassifierConfig,
    version: str | None,
    top_k: int | None,
    default_top_k: int,
    *,
    allow_invalid_version: bool = False,
) -> tuple[str, int, str]:
    """Resolve version and top_k against a standard's config.

    Page requests silently fall back to the first version; fragment requests
    let the classification pipeline reject unknown versions. The top_k clamp
    is defense-in-depth: the fragment endpoint's ``ge=1, le=100`` query
    validation rejects out-of-range values before this is reached.
    """
    resolved_top_k = (
        default_top_k if top_k is None or top_k < 1 or top_k > 100 else top_k
    )

    versions_list = list(config["versions"].keys())
    first_version: str = versions_list[0] if versions_list else ""
    invalid_version = not allow_invalid_version and version not in config["versions"]
    if version is None or invalid_version:
        validated_version = first_version
    else:
        validated_version = version

    return validated_version, resolved_top_k, first_version


def resolve_fragment_push_url(
    push_url: bool | None,
    url_change: bool | None,
) -> bool:
    if push_url is not None:
        return push_url
    if url_change is not None:
        return url_change
    return True


def should_ssr(
    decoded_search_query: str,
    has_query_params: bool,
    canonical_url: str,
) -> bool:
    """Decide whether a query page is eligible for server-rendered results."""
    return (
        bool(decoded_search_query)
        and not has_query_params
        and urlparse(canonical_url).path in SITEMAP_QUERY_PATHS
    )


async def build_classification_results_context(
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
            "append_code_to_url": True,
            "code_url_suffix": "",
            "tooltip": "",
            "total_request_time": 0,
        }

    start_total_time = perf_counter()
    outcome = await request.app.state.classification_service.classify(
        query=normalized_query,
        classifier_type=upper_type,
        version=version,
        top_k=top_k,
    )
    total_request_time = perf_counter() - start_total_time

    return {
        "query": normalized_query,
        "results_for_query": outcome.results,
        "base_url": outcome.version_config.get("base_url", ""),
        "append_code_to_url": outcome.version_config.get("append_code_to_url", True),
        "code_url_suffix": outcome.version_config.get("code_url_suffix", ""),
        "tooltip": outcome.version_config.get("tooltip", ""),
        "total_request_time": total_request_time,
        "classifier_type": upper_type,
    }


def build_empty_classifier_results(decoded_query: str) -> dict[str, object]:
    return {
        "results_for_query": [],
        "query": decoded_query,
        "base_url": "",
        "code_url_suffix": "",
        "tooltip": "",
        "total_request_time": 0,
    }


async def maybe_seed_classifier_page_results(
    request: Request,
    classifier_type: str,
    decoded_query: str,
    example_query: str,
    version: str,
    top_k: int,
    *,
    allow_query_ssr: bool = False,
) -> tuple[dict[str, object], bool, bool, SSRState]:
    results_data = build_empty_classifier_results(decoded_query)
    if decoded_query and not allow_query_ssr:
        return results_data, False, True, "not_attempted"

    query = decoded_query or example_query
    if not query:
        return results_data, False, False, "not_attempted"

    results_data["query"] = query
    try:
        seeded_results = await build_classification_results_context(
            request=request,
            classifier_type=classifier_type,
            query=query,
            version=version,
            top_k=top_k,
        )
        return seeded_results, not decoded_query, False, "success"
    except Exception as e:
        logger.warning(
            "SSR fallback for '%s' page classification due to %s: %s",
            classifier_type,
            type(e).__name__,
            e,
        )
        return results_data, not decoded_query, True, "failure"


def build_classifier_page_context(
    classifier_type: str,
    config: ClassifierConfig,
    display_example: str,
    decoded_query: str,
    validated_version: str,
    first_version: str,
    top_k: int,
    default_top_k: int,
    canonical_url: str,
    results_data: dict[str, object],
    default_example_prefill: bool,
    trigger_search_on_load: bool,
    results_loaded: bool,
) -> dict[str, object]:
    today = datetime.now()
    return {
        "classifier_type": classifier_type,
        "title": config["title"],
        "heading": config["heading"],
        "description": config["description"],
        "versions": list(config["versions"].keys()),
        "example": display_example,
        "url_params": {
            "search": decoded_query,
            "version": (
                validated_version
                if validated_version and validated_version != first_version
                else ""
            ),
            "top_k": top_k,
        },
        "default_example_prefill": default_example_prefill,
        "trigger_search_on_load": trigger_search_on_load,
        "results_loaded": results_loaded,
        "default_top_k": default_top_k,
        "first_version": first_version,
        "canonical_url": canonical_url,
        "current_year": today.year,
        "current_month_name": today.strftime("%B"),
        "popular_lookup_links": get_popular_lookup_links(classifier_type),
        **results_data,
    }


def build_fragment_page_title(classifier_type: str, query: str) -> str:
    return f"{classifier_type} codes for '{query.title()}'"


def render_empty_results_fragment(
    request: Request,
    normalized_description: str,
) -> _TemplateResponse:
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
    return response


def render_classification_results_fragment(
    request: Request,
    results_context: dict,
    page_title: str | None,
    push_url: bool,
    new_url: str,
) -> _TemplateResponse:
    response = templates.TemplateResponse(
        request,
        "results.html",
        {
            **results_context,
            "page_title": page_title,
        },
    )
    response.headers.update(get_classification_cache_headers())
    response.headers["Cache-Tag"] = "classification-results"
    if push_url:
        response.headers["HX-Push-Url"] = new_url
    return response
