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

BASE_DIR = Path(__file__).resolve().parents[1]
GMDN_VERSION = "AccessGUDID Full Release (July 6, 2026)"


def build_test_app() -> FastAPI:
    app = FastAPI()
    app.mount(
        "/static",
        StaticFiles(directory=BASE_DIR / "app" / "static"),
        name="static",
    )
    app.include_router(router)
    app.state.embed_client = object()
    app.state.qdrant_client = object()
    app.state.collection_quantization_cache = {}
    app.state.zclient = None
    app.state.redis_client = object()
    return app


def empty_classification_result() -> dict:
    return {
        "results": [],
        "version_config": {"base_url": "", "tooltip": ""},
    }


def test_gmdn_classifier_configuration_contract() -> None:
    config = CLASSIFIER_CONFIG["GMDN"]

    assert config["title"] == "GMDN Code Finder"
    assert config["embed_model_name"] == "Qwen/Qwen3-Embedding-8B"
    assert config["embed_dims"] == 2048
    assert config["versions"][GMDN_VERSION]["collection_name"] == (
        "GMDN_GUDID_20260706_Qwen3-8B_v1"
    )
    assert "active" not in config["query_instruction"].lower()
    assert "obsolete" not in config["query_instruction"].lower()
    assert "implantable" not in config["query_instruction"].lower()

    prepared = validate_and_prepare_classification("gmdn", GMDN_VERSION)
    assert prepared["collection_name"] == "GMDN_GUDID_20260706_Qwen3-8B_v1"


def test_gmdn_exact_code_lookup_shortcuts_embedding() -> None:
    exact = [
        {
            "id": "point-id",
            "score": 1.0,
            "payload": {
                "original_id": "46653",
                "class_name": "Spinal fixation plate",
                "definition": "An implantable plate.",
                "status": "Obsolete",
                "implantable": True,
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
            query="46653",
            classifier_type="GMDN",
            version=GMDN_VERSION,
            top_k=10,
            quantization_cache={},
            zclient=None,
        )

    assert result["results"][0]["payload"]["original_id"] == "46653"
    assert result["results"][0]["payload"]["status"] == "Obsolete"
    embedding.assert_not_called()


def test_active_and_obsolete_terms_participate_equally_in_semantic_search() -> None:
    semantic_results = [
        {
            "id": "obsolete",
            "score": 0.9,
            "payload": {
                "original_id": "11168",
                "class_name": "Dentifrice",
                "definition": "A tooth-cleaning substance.",
                "status": "Obsolete",
                "implantable": False,
            },
        },
        {
            "id": "active",
            "score": 0.8,
            "payload": {
                "original_id": "10003",
                "class_name": "Abdominal binder",
                "definition": "Supports the abdomen.",
                "status": "Active",
                "implantable": False,
            },
        },
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
            query="medical support device",
            classifier_type="GMDN",
            version=GMDN_VERSION,
            top_k=2,
            quantization_cache={},
            zclient=None,
        )

    assert [item["id"] for item in result["results"]] == ["obsolete", "active"]
    assert search.call_args.kwargs["search_exact"] is True
    assert "query_filter" not in search.call_args.kwargs


@pytest.mark.anyio
async def test_gmdn_page_and_navigation_are_available() -> None:
    app = build_test_app()
    transport = httpx.ASGITransport(app=app)

    with patch(
        "app.web.perform_classification",
        return_value=empty_classification_result(),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            homepage = await client.get("/")
            page = await client.get("/GMDN/")

    assert homepage.status_code == 200
    assert 'href="http://testserver/GMDN/"' in homepage.text
    assert page.status_code == 200
    assert "GMDN Code Finder" in page.text
    assert "Find the right GMDN code for a medical device." in page.text
    assert CLASSIFIER_CONFIG["GMDN"]["example"] in page.text
    assert 'href="http://testserver/GMDN/"' in page.text


@pytest.mark.anyio
async def test_gmdn_fragment_renders_separate_metadata_labels() -> None:
    app = build_test_app()
    result = {
        "results": [
            {
                "score": 0.95,
                "payload": {
                    "original_id": "46653",
                    "class_name": "Spinal fixation plate",
                    "definition": "An implantable plate.",
                    "status": "Active",
                    "implantable": True,
                },
            },
            {
                "score": 0.85,
                "payload": {
                    "original_id": "10003",
                    "class_name": "Abdominal binder",
                    "definition": "Supports the abdomen.",
                    "status": "Obsolete",
                    "implantable": False,
                },
            },
        ],
        "version_config": {"base_url": "", "tooltip": ""},
    }
    usage = UsageStatus(
        allowed=True,
        remaining=9,
        limit=10,
        is_authenticated=False,
        is_pro=False,
        tracking_id="gmdn-test",
    )
    transport = httpx.ASGITransport(app=app)

    with (
        patch("app.web.perform_classification", return_value=result),
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
                "/GMDN/fragment",
                params={
                    "product_description": "medical device",
                    "version": GMDN_VERSION,
                    "top_k": 10,
                    "push_url": "false",
                },
            )

    assert response.status_code == 200
    assert "Implantable" in response.text
    assert "Non-implantable" in response.text
    assert "Active" in response.text
    assert "Obsolete" in response.text


@pytest.mark.anyio
async def test_unspsc_level_rendering_remains_unchanged() -> None:
    app = build_test_app()
    version = next(iter(CLASSIFIER_CONFIG["UNSPSC"]["versions"]))
    result = {
        "results": [
            {
                "score": 0.9,
                "payload": {
                    "original_id": "43211503",
                    "class_name": "Notebook computers",
                    "definition": "Portable computers.",
                    "id_level": "Commodity",
                },
            }
        ],
        "version_config": {"base_url": "", "tooltip": ""},
    }
    usage = UsageStatus(
        allowed=True,
        remaining=9,
        limit=10,
        is_authenticated=False,
        is_pro=False,
        tracking_id="unspsc-test",
    )
    transport = httpx.ASGITransport(app=app)

    with (
        patch("app.web.perform_classification", return_value=result),
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
                "/UNSPSC/fragment",
                params={
                    "product_description": "notebook computer",
                    "version": version,
                    "top_k": 10,
                    "push_url": "false",
                },
            )

    assert response.status_code == 200
    assert "Commodity" in response.text
    assert "Implantable" not in response.text
    assert "Non-implantable" not in response.text
