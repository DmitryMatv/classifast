from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from openpyxl import Workbook
from qdrant_client import models

from embedders import embedder_remote_EMDN_hf as target

BASE_ROWS = [
    (
        "A",
        "ADMINISTRATION DEVICES",
        "A",
        "ADMINISTRATION DEVICES",
        1,
        "NO",
    ),
    (
        "A",
        "ADMINISTRATION DEVICES",
        "A01",
        "NEEDLES",
        2,
        "NO",
    ),
    (
        "A",
        "ADMINISTRATION DEVICES",
        "A0101",
        "INFUSION NEEDLES",
        3,
        "YES",
    ),
    (
        "A",
        "ADMINISTRATION DEVICES",
        "A02",
        "ABDOMINAL BINDERS",
        2,
        "YES",
    ),
]


def write_workbook(
    path: Path,
    rows: list[tuple[object, ...]] | None = None,
    headers: tuple[str, ...] = target.EXPECTED_COLUMNS,
) -> Path:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = target.EXPECTED_SHEET_NAME
    worksheet.append([None, None, "EMDN_v.2026"])
    worksheet.append(list(headers))
    for row in BASE_ROWS if rows is None else rows:
        worksheet.append(list(row))
    workbook.save(path)
    workbook.close()
    return path


def make_prepared_term(
    code: str = "A0101",
    name: str = "INFUSION NEEDLES",
) -> target.PreparedEmdnTerm:
    normalized = target._normalize_original_id(code)
    hierarchy = "Category: ADMINISTRATION DEVICES\nLevel 2: NEEDLES"
    return target.PreparedEmdnTerm(
        id=target.build_point_id(code),
        original_id=code,
        original_id_normalized=normalized,
        original_id_normalized_reversed=normalized[::-1],
        class_name=name,
        definition="ADMINISTRATION DEVICES > NEEDLES",
        category="A",
        category_name="ADMINISTRATION DEVICES",
        level=3,
        terminal=True,
        parent_code="A01",
        hierarchy=hierarchy,
        embedding_text=target.build_embedding_text(name, hierarchy),
    )


def test_valid_workbook_maps_only_terminal_terms_and_ignores_first_row(
    tmp_path: Path,
) -> None:
    path = write_workbook(tmp_path / "emdn.xlsx")

    prepared = target.load_and_prepare_emdn_data(path)

    assert [term.original_id for term in prepared] == ["A0101", "A02"]
    first = prepared[0]
    assert first.class_name == "INFUSION NEEDLES"
    assert first.category == "A"
    assert first.category_name == "ADMINISTRATION DEVICES"
    assert first.level == 3
    assert first.terminal is True
    assert first.parent_code == "A01"
    assert first.original_id_normalized == "a0101"
    assert first.original_id_normalized_reversed == "1010a"
    assert first.definition == "ADMINISTRATION DEVICES > NEEDLES"


def test_embedding_text_contains_title_and_ancestor_hierarchy_without_codes(
    tmp_path: Path,
) -> None:
    prepared = target.load_and_prepare_emdn_data(write_workbook(tmp_path / "emdn.xlsx"))

    assert prepared[0].embedding_text == (
        "Title: INFUSION NEEDLES\n"
        "Hierarchy:\n"
        "Category: ADMINISTRATION DEVICES\n"
        "Level 2: NEEDLES"
    )
    assert "A0101" not in prepared[0].embedding_text
    assert "Terminal" not in prepared[0].embedding_text


def test_point_ids_are_deterministic_and_namespaced() -> None:
    assert target.build_point_id("A0101") == target.build_point_id("A0101")
    assert target.build_point_id("A0101") != target.build_point_id("A0102")
    assert target.build_point_id("A0101") != target.build_point_id("B0101")


def test_duplicate_titles_under_different_codes_are_preserved(tmp_path: Path) -> None:
    rows = BASE_ROWS + [
        (
            "A",
            "ADMINISTRATION DEVICES",
            "A03",
            "ABDOMINAL BINDERS",
            2,
            "YES",
        )
    ]

    prepared = target.load_and_prepare_emdn_data(
        write_workbook(tmp_path / "emdn.xlsx", rows)
    )

    assert [
        term.original_id for term in prepared if term.class_name == "ABDOMINAL BINDERS"
    ] == [
        "A02",
        "A03",
    ]


def test_duplicate_code_is_rejected(tmp_path: Path) -> None:
    rows = BASE_ROWS + [BASE_ROWS[-1]]

    with pytest.raises(ValueError, match="duplicate EMDN code"):
        target.load_and_prepare_emdn_data(write_workbook(tmp_path / "emdn.xlsx", rows))


def test_missing_column_is_rejected(tmp_path: Path) -> None:
    headers = target.EXPECTED_COLUMNS[:-1]
    rows = [row[:-1] for row in BASE_ROWS]

    with pytest.raises(ValueError, match="column count"):
        target.load_and_prepare_emdn_data(
            write_workbook(tmp_path / "emdn.xlsx", rows, headers)
        )


@pytest.mark.parametrize(
    ("row_index", "column_index", "value", "message"),
    [
        (2, 2, "A1", "does not match"),
        (2, 0, "B", "does not match"),
        (2, 4, 8, "LEVEL"),
        (2, 5, "MAYBE", "terminal value"),
        (2, 3, "", "EMDN term is empty"),
        (2, 1, "", "category description is empty"),
    ],
)
def test_invalid_row_values_are_rejected(
    tmp_path: Path,
    row_index: int,
    column_index: int,
    value: object,
    message: str,
) -> None:
    rows = [list(row) for row in BASE_ROWS]
    rows[row_index][column_index] = value

    with pytest.raises(ValueError, match=message):
        target.load_and_prepare_emdn_data(
            write_workbook(
                tmp_path / "emdn.xlsx",
                [tuple(row) for row in rows],
            )
        )


def test_missing_parent_is_rejected(tmp_path: Path) -> None:
    rows = [BASE_ROWS[0], BASE_ROWS[2], BASE_ROWS[3]]

    with pytest.raises(ValueError, match="parent code 'A01' is missing"):
        target.load_and_prepare_emdn_data(write_workbook(tmp_path / "emdn.xlsx", rows))


def test_terminal_flag_must_agree_with_child_relationships(tmp_path: Path) -> None:
    rows = [list(row) for row in BASE_ROWS]
    rows[1][5] = "YES"

    with pytest.raises(ValueError, match="must be NO"):
        target.load_and_prepare_emdn_data(
            write_workbook(
                tmp_path / "emdn.xlsx",
                [tuple(row) for row in rows],
            )
        )


def test_embedding_response_shape_and_dimensions_are_validated() -> None:
    assert target.normalize_embedding_response(
        [[1, 2], [3, 4]],
        expected_count=2,
        embed_dims=2,
    ) == [[1.0, 2.0], [3.0, 4.0]]
    assert target.normalize_embedding_response(
        [1, 2],
        expected_count=1,
        embed_dims=2,
    ) == [[1.0, 2.0]]

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        target.normalize_embedding_response(
            [[1], [2]],
            expected_count=2,
            embed_dims=2,
        )
    with pytest.raises(RuntimeError, match="Expected 2 embeddings"):
        target.normalize_embedding_response(
            [[1, 2]],
            expected_count=2,
            embed_dims=2,
        )


def test_hugging_face_batch_parameters_are_forwarded() -> None:
    client = MagicMock()
    client.feature_extraction.return_value = [[1, 2], [3, 4]]

    result = target.get_embeddings_batch_sync(
        embed_client=client,
        model_name="Qwen/Qwen3-Embedding-8B",
        texts=["first", "second"],
        embed_dims=2,
    )

    assert result == [[1.0, 2.0], [3.0, 4.0]]
    client.feature_extraction.assert_called_once_with(
        ["first", "second"],
        model="Qwen/Qwen3-Embedding-8B",
        dimensions=2,
    )


def test_point_payload_preserves_emdn_metadata() -> None:
    payload = target.build_point_payload(make_prepared_term())

    assert payload == {
        "original_id": "A0101",
        "original_id_normalized": "a0101",
        "original_id_normalized_reversed": "1010a",
        "class_name": "INFUSION NEEDLES",
        "definition": "ADMINISTRATION DEVICES > NEEDLES",
        "category": "A",
        "category_name": "ADMINISTRATION DEVICES",
        "level": 3,
        "terminal": True,
        "parent_code": "A01",
        "hierarchy": "Category: ADMINISTRATION DEVICES\nLevel 2: NEEDLES",
    }


def test_collection_creation_uses_dot_vectors_on_disk() -> None:
    client = MagicMock()
    client.collection_exists.return_value = False

    assert target._ensure_collection(
        client,
        "EMDN",
        2048,
        models.Distance.DOT,
    )

    config = client.create_collection.call_args.kwargs
    assert config["collection_name"] == "EMDN"
    assert config["vectors_config"].size == 2048
    assert config["vectors_config"].distance == models.Distance.DOT
    assert config["vectors_config"].on_disk is True
    assert config["on_disk_payload"] is True
    assert "quantization_config" not in config


def test_incompatible_existing_collection_is_rejected() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value.config.params.vectors = models.VectorParams(
        size=1024,
        distance=models.Distance.COSINE,
    )

    assert not target._ensure_collection(
        client,
        "EMDN",
        2048,
        models.Distance.DOT,
    )
    client.create_collection.assert_not_called()


def test_required_payload_indexes_are_created() -> None:
    client = MagicMock()

    assert target._create_payload_indexes(client, "EMDN")

    calls = {
        call.kwargs["field_name"]: call.kwargs["field_schema"]
        for call in client.create_payload_index.call_args_list
    }
    assert set(calls) == {
        "original_id",
        "original_id_normalized",
        "original_id_normalized_reversed",
        "class_name",
        "category",
        "level",
        "terminal",
    }
    assert isinstance(calls["original_id"], models.KeywordIndexParams)
    assert calls["original_id_normalized"].tokenizer == models.TokenizerType.PREFIX
    assert calls["class_name"].tokenizer == models.TokenizerType.WORD
    assert isinstance(calls["category"], models.KeywordIndexParams)
    assert isinstance(calls["level"], models.IntegerIndexParams)
    assert isinstance(calls["terminal"], models.BoolIndexParams)


def test_existing_points_are_skipped() -> None:
    data = [
        make_prepared_term("A0101", "FIRST"),
        make_prepared_term("A0102", "SECOND"),
    ]
    client = MagicMock()
    client.count.return_value.count = 2

    with (
        patch.object(target, "_connect_qdrant", return_value=client),
        patch.object(target, "_ensure_collection", return_value=True),
        patch.object(target, "_create_payload_indexes", return_value=True),
        patch.object(
            target,
            "_fetch_existing_points",
            return_value={"A0101": "existing-point"},
        ),
        patch.object(
            target,
            "get_embeddings_batch_sync",
            return_value=[[0.1, 0.2]],
        ) as embed,
    ):
        success = target.create_and_populate_qdrant(
            data=data,
            collection_name="EMDN",
            vector_size=2,
            distance_metric=models.Distance.DOT,
            qdrant_url="https://qdrant.example",
            qdrant_api_key=None,
            embed_client=MagicMock(),
            embed_model="model",
        )

    assert success
    assert embed.call_args.kwargs["texts"] == [data[1].embedding_text]
    points = client.upsert.call_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].payload["original_id"] == "A0102"
    client.close.assert_called_once()


@pytest.mark.parametrize("failed_stage", ["embedding", "upsert"])
def test_failed_batches_return_failure(failed_stage: str) -> None:
    data = [make_prepared_term()]
    client = MagicMock()
    client.count.return_value.count = 1
    if failed_stage == "upsert":
        client.upsert.side_effect = RuntimeError("upsert failed")

    embedding_side_effect: object
    if failed_stage == "embedding":
        embedding_side_effect = RuntimeError("embedding failed")
    else:
        embedding_side_effect = [[0.1, 0.2]]

    with (
        patch.object(target, "_connect_qdrant", return_value=client),
        patch.object(target, "_ensure_collection", return_value=True),
        patch.object(target, "_create_payload_indexes", return_value=True),
        patch.object(target, "_fetch_existing_points", return_value={}),
        patch.object(
            target,
            "get_embeddings_batch_sync",
            side_effect=(
                embedding_side_effect
                if isinstance(embedding_side_effect, Exception)
                else None
            ),
            return_value=(
                embedding_side_effect
                if not isinstance(embedding_side_effect, Exception)
                else None
            ),
        ),
    ):
        success = target.create_and_populate_qdrant(
            data=data,
            collection_name="EMDN",
            vector_size=2,
            distance_metric=models.Distance.DOT,
            qdrant_url="https://qdrant.example",
            qdrant_api_key=None,
            embed_client=MagicMock(),
            embed_model="model",
        )

    assert not success
    client.close.assert_called_once()


def test_prepare_only_makes_no_external_calls(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = write_workbook(tmp_path / "emdn.xlsx")

    with (
        patch.object(target, "InferenceClient") as inference_client,
        patch.object(target, "_connect_qdrant") as connect_qdrant,
    ):
        exit_code = target.main(["--data-path", str(path), "--prepare-only"])

    assert exit_code == 0
    inference_client.assert_not_called()
    connect_qdrant.assert_not_called()
    assert "no external services were called" in capsys.readouterr().out
