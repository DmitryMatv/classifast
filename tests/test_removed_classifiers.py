from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.classifier_config import CLASSIFIER_CONFIG
from app.classifier_page_delivery import REMOVED_CLASSIFIER_TYPES
from app.web import router
from tests.helpers import build_classification_service

BASE_DIR = Path(__file__).resolve().parents[1]


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


def test_removed_classifiers_are_absent_from_config() -> None:
    assert "GMDN" in REMOVED_CLASSIFIER_TYPES
    assert REMOVED_CLASSIFIER_TYPES.isdisjoint(CLASSIFIER_CONFIG)


@pytest.mark.anyio
async def test_removed_classifier_pages_return_410() -> None:
    app = build_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        follow_redirects=False,
    ) as client:
        base_page = await client.get("/GMDN/")
        no_slash = await client.get("/gmdn")
        query_page = await client.get("/GMDN/syringe/")

    assert base_page.status_code == 410
    assert no_slash.status_code == 410
    assert query_page.status_code == 410


@pytest.mark.anyio
async def test_removed_classifier_fragment_returns_410() -> None:
    app = build_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        fragment = await client.get(
            "/GMDN/fragment",
            params={"product_description": "sterile catheter"},
        )

    assert fragment.status_code == 410
