"""Every local collection builder must satisfy the runtime Qdrant contract.

The builders live in the ignored embedders/ directory, so this test checks
whichever ones the checkout has and skips when there are none.
"""

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from qdrant_client import models

from app import qdrant_schema
from app.id_lookup import build_normalized_id_payload

EMBEDDERS_DIR = Path(__file__).resolve().parents[1] / "embedders"
BUILDER_PATHS = sorted(EMBEDDERS_DIR.glob("embedder_remote_*.py"))
# EMDN builds payloads from a dataclass; tests/test_emdn_embedder.py covers it.
ROW_PAYLOAD_PATHS = [path for path in BUILDER_PATHS if "EMDN" not in path.name]

pytestmark = pytest.mark.skipif(
    not BUILDER_PATHS, reason="embedders/ is not present in this checkout"
)


def _load_builder(path: Path) -> ModuleType:
    module_name = f"_builder_{path.stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        pytest.skip(f"{path.name} needs the uninstalled package {exc.name!r}")
    finally:
        sys.modules.pop(module_name, None)
    return module


@pytest.mark.parametrize("path", BUILDER_PATHS, ids=lambda path: path.stem)
def test_builder_creates_every_required_index_as_startup_expects(path: Path) -> None:
    client = MagicMock()

    _load_builder(path)._create_payload_indexes(client, "collection")

    created = {
        call.kwargs["field_name"]: call.kwargs["field_schema"]
        for call in client.create_payload_index.call_args_list
    }
    for field_name in qdrant_schema.PAYLOAD_INDEX_FIELDS:
        assert field_name in created, field_name
        schema = created[field_name]
        index_info = models.PayloadIndexInfo(
            data_type=getattr(schema.type, "value", schema.type),
            params=schema,
            points=0,
        )
        assert qdrant_schema.is_expected_payload_index(field_name, index_info), (
            field_name
        )


@pytest.mark.parametrize("path", ROW_PAYLOAD_PATHS, ids=lambda path: path.stem)
def test_builder_payload_carries_normalized_id_fields(path: Path) -> None:
    row = defaultdict(str, original_id="43.21.15-03")

    payload = _load_builder(path)._build_point_payload(row)

    assert payload["original_id"] == "43.21.15-03"
    for field_name, value in build_normalized_id_payload("43.21.15-03").items():
        assert payload[field_name] == value, field_name
