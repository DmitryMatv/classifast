import asyncio
import ipaddress
import logging
import os
import time
from typing import TypeAlias

import httpx
from fastapi import Request

logger = logging.getLogger(__name__)

GOOGLE_COMMON_CRAWLERS_URL = (
    "https://developers.google.com/static/crawling/ipranges/common-crawlers.json"
)
SUPPORTED_GOOGLE_CRAWLER_TOKENS = ("Googlebot", "Google-InspectionTool")
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}

IPNetwork: TypeAlias = ipaddress.IPv4Network | ipaddress.IPv6Network

_range_lock = asyncio.Lock()
_cached_networks: list[IPNetwork] | None = None
_cached_at = 0.0


def _google_crawler_bypass_enabled() -> bool:
    value = os.getenv("GOOGLE_CRAWLER_BYPASS_ENABLED", "true")
    return value.strip().lower() in TRUE_ENV_VALUES


def _trust_cloudflare_connecting_ip() -> bool:
    """Whether the deployment guarantees all origin traffic traverses Cloudflare."""
    value = os.getenv("GOOGLE_CRAWLER_TRUST_CF_CONNECTING_IP", "false")
    return value.strip().lower() in TRUE_ENV_VALUES


def _range_cache_ttl_seconds() -> int:
    try:
        return max(0, int(os.getenv("GOOGLE_CRAWLER_IP_RANGE_TTL_SECONDS", "86400")))
    except ValueError:
        logger.warning("Invalid GOOGLE_CRAWLER_IP_RANGE_TTL_SECONDS; using default")
        return 86400


def _range_negative_ttl_seconds() -> int:
    try:
        return max(
            0,
            int(os.getenv("GOOGLE_CRAWLER_IP_RANGE_NEGATIVE_TTL_SECONDS", "300")),
        )
    except ValueError:
        logger.warning(
            "Invalid GOOGLE_CRAWLER_IP_RANGE_NEGATIVE_TTL_SECONDS; using default"
        )
        return 300


def _range_ttl_for_cache() -> int:
    if _cached_networks == []:
        return _range_negative_ttl_seconds()
    return _range_cache_ttl_seconds()


def _range_fetch_timeout_seconds() -> float:
    try:
        return max(
            0.1,
            float(os.getenv("GOOGLE_CRAWLER_IP_RANGE_TIMEOUT_SECONDS", "2.0")),
        )
    except ValueError:
        logger.warning("Invalid GOOGLE_CRAWLER_IP_RANGE_TIMEOUT_SECONDS; using default")
        return 2.0


def _has_supported_google_user_agent(user_agent: str) -> bool:
    return any(token in user_agent for token in SUPPORTED_GOOGLE_CRAWLER_TOKENS)


def _get_client_ip(request: Request) -> str:
    # CF-Connecting-IP is trustworthy only when direct origin access is blocked.
    # Application code cannot establish that infrastructure property, so require
    # an explicit deployment opt-in before treating the header as authoritative.
    if _trust_cloudflare_connecting_ip():
        cf_ip = request.headers.get("cf-connecting-ip")
        if cf_ip:
            return cf_ip
    return request.client.host if request.client else "unknown"


def _parse_google_crawler_networks(payload: dict[str, object]) -> list[IPNetwork]:
    prefixes = payload.get("prefixes")
    if not isinstance(prefixes, list):
        return []

    networks: list[IPNetwork] = []
    for entry in prefixes:
        if not isinstance(entry, dict):
            continue

        prefix = entry.get("ipv4Prefix") or entry.get("ipv6Prefix")
        if not isinstance(prefix, str):
            continue

        try:
            network = ipaddress.ip_network(prefix)
        except ValueError:
            logger.warning("Ignoring invalid Google crawler IP prefix: %s", prefix)
            continue
        networks.append(network)

    return networks


async def _fetch_google_common_crawler_networks() -> list[IPNetwork]:
    async with httpx.AsyncClient(timeout=_range_fetch_timeout_seconds()) as client:
        response = await client.get(GOOGLE_COMMON_CRAWLERS_URL)
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, dict):
        return []

    return _parse_google_crawler_networks(payload)


def _cache_negative_google_crawler_ranges(now: float) -> list[IPNetwork]:
    global _cached_at, _cached_networks

    _cached_networks = []
    _cached_at = now
    return _cached_networks


async def _get_google_common_crawler_networks() -> list[IPNetwork]:
    global _cached_at, _cached_networks

    now = time.monotonic()
    ttl = _range_ttl_for_cache()
    if _cached_networks is not None and now - _cached_at < ttl:
        return _cached_networks

    async with _range_lock:
        now = time.monotonic()
        ttl = _range_ttl_for_cache()
        if _cached_networks is not None and now - _cached_at < ttl:
            return _cached_networks

        try:
            networks = await _fetch_google_common_crawler_networks()
        except Exception as exc:
            logger.warning("Failed to refresh Google crawler IP ranges: %s", exc)
            return _cache_negative_google_crawler_ranges(now)

        if not networks:
            logger.warning("Google crawler IP range refresh returned no usable ranges")
            return _cache_negative_google_crawler_ranges(now)

        _cached_networks = networks
        _cached_at = now
        return networks


async def is_verified_google_search_crawler_request(request: Request) -> bool:
    if not _google_crawler_bypass_enabled():
        return False

    user_agent = request.headers.get("user-agent", "")
    if not _has_supported_google_user_agent(user_agent):
        return False

    try:
        client_ip = ipaddress.ip_address(_get_client_ip(request))
    except ValueError:
        return False

    networks = await _get_google_common_crawler_networks()
    return any(client_ip in network for network in networks)
