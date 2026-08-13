from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.classifier import perform_classification, validate_and_prepare_classification
from app.classifier_config import CLASSIFIER_CONFIG
from app.usage_tracker import UsageStatus
from app.web import router
from tests.helpers import build_classification_service

BASE_DIR = Path(__file__).resolve().parents[1]
EMDN_VERSION = "EMDN v2026 (English)"
EMDN_COLLECTION = "EMDN_2026_EN_Qwen3-8B_v1"
EMDN_EC_URL = (
    "https://health.ec.europa.eu/medical-devices-topics-interest/"
    "european-medical-devices-nomenclature-emdn_en"
)
EMDN_DOWNLOAD_URL = "https://webgate.ec.europa.eu/dyna2/emdn/build/EMDN%20v2026_EN.xlsx"
EMDN_CODE_BASE_URL = "https://webgate.ec.europa.eu/dyna2/emdn/"
EUDAMED_URL = "https://ec.europa.eu/tools/eudamed/eudamed"
EUDAMED_SIGN_IN_URL = "https://webgate.ec.europa.eu/eudamed"


def build_test_app() -> FastAPI:
    app = FastAPI()
    app.mount(
        "/static",
        StaticFiles(directory=BASE_DIR / "app" / "static"),
        name="static",
    )
    app.include_router(router)
    app.state.classification_service = build_classification_service()
    app.state.redis_client = object()
    return app


def empty_classification_result() -> dict:
    return {
        "results": [],
        "version_config": {"base_url": "", "tooltip": ""},
        "version_name": EMDN_VERSION,
        "collection_name": EMDN_COLLECTION,
        "query": "test query",
    }


def test_emdn_classifier_configuration_contract() -> None:
    config = CLASSIFIER_CONFIG["EMDN"]

    assert config["title"] == "EMDN Code Finder"
    assert config["heading"] == "Find the right EMDN code for a medical device."
    assert config["embed_model_name"] == "Qwen/Qwen3-Embedding-8B"
    assert config["embed_dims"] == 2048
    assert config["versions"][EMDN_VERSION]["collection_name"] == EMDN_COLLECTION
    assert config["versions"][EMDN_VERSION]["base_url"] == EMDN_CODE_BASE_URL
    assert config["versions"][EMDN_VERSION]["code_url_suffix"] == "#title"
    assert "terminal emdn term" in config["query_instruction"].lower()
    assert "hierarchy context" in config["query_instruction"].lower()
    assert "ancestor hierarchy" in config["rerank_instruction"].lower()
    assert "level 7" not in config["query_instruction"].lower()
    assert "level 7" not in config["rerank_instruction"].lower()

    prepared = validate_and_prepare_classification("emdn", EMDN_VERSION)
    assert prepared["collection_name"] == EMDN_COLLECTION


def test_emdn_exact_code_lookup_shortcuts_embedding() -> None:
    exact = [
        {
            "id": "point-id",
            "score": 1.0,
            "payload": {
                "original_id": "A0101010101",
                "class_name": "HYPODERMIC SYRINGE NEEDLES, WITH SAFETY SYSTEMS",
                "definition": (
                    "DEVICES FOR ADMINISTRATION, WITHDRAWAL AND COLLECTION "
                    "> NEEDLES > NEEDLES FOR INFUSION AND SAMPLING"
                ),
                "level": 6,
                "terminal": True,
            },
        }
    ]

    with (
        patch("app.classifier.perform_exact_id_search", return_value=exact),
        patch("app.classifier.get_embedding") as embedding,
    ):
        result = perform_classification(
            embed_client=object(),
            qdrant_client=object(),
            query="A0101010101",
            classifier_type="EMDN",
            version=EMDN_VERSION,
            top_k=10,
            quantization_cache={},
            reranker=None,
        )

    assert result["results"][0]["payload"]["original_id"] == "A0101010101"
    embedding.assert_not_called()


def test_emdn_semantic_search_does_not_add_terminal_or_level_filter() -> None:
    semantic_results = [
        {
            "id": "semantic-point",
            "score": 0.9,
            "payload": {
                "original_id": "A0101010101",
                "class_name": "HYPODERMIC SYRINGE NEEDLES, WITH SAFETY SYSTEMS",
                "definition": "ADMINISTRATION DEVICES > NEEDLES",
                "level": 6,
                "terminal": True,
            },
        }
    ]

    with (
        patch("app.classifier.perform_exact_id_search", return_value=[]),
        patch("app.classifier.perform_partial_id_search", return_value=[]),
        patch("app.classifier.get_embedding", return_value=[0.1, 0.2]),
        patch(
            "app.classifier.perform_semantic_search",
            return_value=semantic_results,
        ) as search,
    ):
        result = perform_classification(
            embed_client=object(),
            qdrant_client=object(),
            query="sterile safety hypodermic needle",
            classifier_type="EMDN",
            version=EMDN_VERSION,
            top_k=10,
            quantization_cache={},
            reranker=None,
        )

    assert result["results"][0]["id"] == "semantic-point"
    assert result["results"][0]["payload"] == semantic_results[0]["payload"]
    assert "query_filter" not in search.call_args.kwargs


@pytest.mark.anyio
async def test_emdn_page_navigation_resources_and_no_logo() -> None:
    app = build_test_app()
    transport = httpx.ASGITransport(app=app)

    with patch(
        "app.classification_service.perform_classification",
        return_value=empty_classification_result(),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            follow_redirects=False,
        ) as client:
            homepage = await client.get("/")
            page = await client.get("/EMDN/")
            lowercase_redirect = await client.get("/emdn")
            other_classifier_page = await client.get("/GMDN/")

    assert homepage.status_code == 200
    assert homepage.text.count('href="http://testserver/EMDN/"') >= 2

    assert page.status_code == 200
    assert "EMDN Code Finder" in page.text
    assert "Find the right EMDN code for a medical device." in page.text
    assert CLASSIFIER_CONFIG["EMDN"]["example"] in page.text
    assert 'href="http://testserver/EMDN/"' in page.text
    assert 'data-classifier-logo="true"' not in page.text
    assert f'href="{EMDN_EC_URL}"' in page.text
    assert f'href="{EMDN_DOWNLOAD_URL}"' in page.text
    assert f'href="{EUDAMED_URL}"' in page.text
    assert f'href="{EUDAMED_SIGN_IN_URL}"' in page.text
    assert "Sign in to EUDAMED" in page.text
    assert 'aria-label="EMDN resources"' in page.text
    assert 'aria-label="EMDN Classification Tool"' not in page.text

    assert lowercase_redirect.status_code == 301
    assert lowercase_redirect.headers["location"] == "/EMDN/"

    assert other_classifier_page.status_code == 200
    assert 'aria-label="EMDN Classification Tool"' in other_classifier_page.text
    assert (
        "European nomenclature for assigning medical device codes"
        in other_classifier_page.text
    )
    assert "used with EUDAMED." in other_classifier_page.text


@pytest.mark.anyio
async def test_emdn_fragment_groups_code_without_metadata_badges() -> None:
    app = build_test_app()
    result = {
        "results": [
            {
                "score": 0.95,
                "payload": {
                    "original_id": "A0101010101",
                    "class_name": ("HYPODERMIC SYRINGE NEEDLES, WITH SAFETY SYSTEMS"),
                    "definition": (
                        "DEVICES FOR ADMINISTRATION, WITHDRAWAL AND COLLECTION "
                        "> NEEDLES > HYPODERMIC NEEDLES"
                    ),
                    "category": "A",
                    "category_name": (
                        "DEVICES FOR ADMINISTRATION, WITHDRAWAL AND COLLECTION"
                    ),
                    "level": 6,
                    "terminal": True,
                    "parent_code": "A01010101",
                },
            }
        ],
        "version_config": {"base_url": "", "tooltip": ""},
        "version_name": EMDN_VERSION,
        "collection_name": EMDN_COLLECTION,
        "query": "sterile safety hypodermic needle",
    }
    usage = UsageStatus(
        allowed=True,
        remaining=9,
        limit=10,
        is_authenticated=False,
        is_pro=False,
        tracking_id="emdn-test",
    )
    transport = httpx.ASGITransport(app=app)

    with (
        patch("app.classification_service.perform_classification", return_value=result),
        patch("app.web.reserve_usage", new=AsyncMock(return_value=usage)),
        patch(
            "app.web.is_verified_google_search_crawler_request",
            new=AsyncMock(return_value=False),
        ),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get(
                "/EMDN/fragment",
                params={
                    "product_description": "sterile safety hypodermic needle",
                    "version": EMDN_VERSION,
                    "top_k": 10,
                    "push_url": "false",
                },
            )

    assert response.status_code == 200
    assert response.text.count("code-spacer-halves") == 5
    assert "onclick=\"window.copyOriginalId('A0101010101', this)\"" in response.text
    assert "HYPODERMIC SYRINGE NEEDLES, WITH SAFETY SYSTEMS" in response.text
    assert "DEVICES FOR ADMINISTRATION, WITHDRAWAL AND COLLECTION" in response.text
    assert "&gt; NEEDLES &gt; HYPODERMIC NEEDLES" in response.text
    assert "Terminal" not in response.text
    assert "Level 6" not in response.text
    assert ">6<" not in response.text
    assert "lucide-external-link-icon" not in response.text


def test_emdn_sitemap_entry_is_present() -> None:
    sitemap = (BASE_DIR / "app" / "static" / "sitemap.xml").read_text(encoding="utf-8")

    assert "<loc>https://classifast.com/EMDN/</loc>" in sitemap
