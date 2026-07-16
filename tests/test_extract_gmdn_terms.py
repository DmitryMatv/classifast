import csv
from pathlib import Path

import pytest

from utilities.extract_gmdn_terms import EXPECTED_HEADER, extract_gmdn_terms


def write_source(path: Path, rows: list[tuple[str, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="|", lineterminator="\n")
        writer.writerow(EXPECTED_HEADER)
        writer.writerows(rows)


def read_output(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def test_extracts_unique_sorted_codes_and_marks_any_obsolete(tmp_path: Path) -> None:
    source = tmp_path / "gmdnTerms.txt"
    output = tmp_path / "gmdn_codes.csv"
    write_source(
        source,
        [
            ("di-1", "Second term", "Old definition", "20000", "Obsolete", ""),
            (
                "di-2",
                "First, term",
                'Definition with "quotes"',
                "10000",
                "Active",
                "true",
            ),
            ("di-3", "First, term", 'Definition with "quotes"', "10000", "Active", ""),
            ("di-4", "Second term", "Current definition", "20000", "Active", "true"),
            ("di-5", "Third term", "Obsolete definition", "30000", "Obsolete", ""),
            ("di-6", "Fourth term", "Current fourth definition", "40000", "Active", ""),
            (
                "di-7",
                "Fourth term",
                "Retired fourth definition",
                "40000",
                "Obsolete",
                "true",
            ),
        ],
    )

    stats = extract_gmdn_terms(source, output)

    assert stats.source_rows == 7
    assert stats.unique_codes == 4
    assert stats.active_codes == 1
    assert stats.obsolete_codes == 3
    assert stats.implantable_codes == 3
    assert stats.non_implantable_codes == 1
    assert read_output(output) == [
        {
            "gmdn_code": "10000",
            "gmdn_name": "First, term",
            "gmdn_definition": 'Definition with "quotes"',
            "status": "Active",
            "implantable": "true",
        },
        {
            "gmdn_code": "20000",
            "gmdn_name": "Second term",
            "gmdn_definition": "Old definition",
            "status": "Obsolete",
            "implantable": "true",
        },
        {
            "gmdn_code": "30000",
            "gmdn_name": "Third term",
            "gmdn_definition": "Obsolete definition",
            "status": "Obsolete",
            "implantable": "false",
        },
        {
            "gmdn_code": "40000",
            "gmdn_name": "Fourth term",
            "gmdn_definition": "Retired fourth definition",
            "status": "Obsolete",
            "implantable": "true",
        },
    ]


def test_rejects_unexpected_header(tmp_path: Path) -> None:
    source = tmp_path / "gmdnTerms.txt"
    output = tmp_path / "gmdn_codes.csv"
    source.write_text("gmdnCode|gmdnPTName\n10000|Term\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected header"):
        extract_gmdn_terms(source, output)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (("di", "", "Definition", "10000", "Active", ""), "gmdnPTName is empty"),
        (
            ("di", "Term", "Definition", "10000", "Retired", ""),
            "unsupported gmdnCodeStatus",
        ),
        (("di", "Term", "Definition", "1234", "Active", ""), "five-digit number"),
        (
            ("di", "Term", "Definition", "10000", "Active", "maybe"),
            "unsupported implantable value",
        ),
    ],
)
def test_rejects_invalid_required_values(
    tmp_path: Path, row: tuple[str, ...], message: str
) -> None:
    source = tmp_path / "gmdnTerms.txt"
    output = tmp_path / "gmdn_codes.csv"
    write_source(source, [row])

    with pytest.raises(ValueError, match=message):
        extract_gmdn_terms(source, output)


def test_conflicting_same_status_metadata_does_not_replace_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "gmdnTerms.txt"
    output = tmp_path / "gmdn_codes.csv"
    output.write_text("existing output\n", encoding="utf-8")
    write_source(
        source,
        [
            ("di-1", "Term", "First definition", "10000", "Active", ""),
            ("di-2", "Term", "Different definition", "10000", "Active", ""),
        ],
    )

    with pytest.raises(ValueError, match="conflicting Active metadata"):
        extract_gmdn_terms(source, output)

    assert output.read_text(encoding="utf-8") == "existing output\n"
