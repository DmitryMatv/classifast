#!/usr/bin/env python3
"""Extract a unique GMDN terminology catalog from an AccessGUDID release."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GMDN_ROOT = REPO_ROOT / "data" / "GMDN"
DEFAULT_OUTPUT = DEFAULT_GMDN_ROOT / "gmdn_codes.csv"
EXPECTED_HEADER = (
    "PrimaryDI",
    "gmdnPTName",
    "gmdnPTDefinition",
    "gmdnCode",
    "gmdnCodeStatus",
    "implantable",
)
OUTPUT_HEADER = (
    "gmdn_code",
    "gmdn_name",
    "gmdn_definition",
    "status",
    "implantable",
)
VALID_STATUSES = frozenset({"Active", "Obsolete"})


@dataclass(frozen=True)
class GmdnTerm:
    code: str
    name: str
    definition: str
    status: str
    implantable: bool


@dataclass(frozen=True)
class ExtractionStats:
    source_rows: int
    unique_codes: int
    active_codes: int
    obsolete_codes: int
    implantable_codes: int
    non_implantable_codes: int


def discover_input(gmdn_root: Path = DEFAULT_GMDN_ROOT) -> Path:
    """Find the single AccessGUDID gmdnTerms.txt below the GMDN data folder."""
    matches = sorted(gmdn_root.rglob("gmdnTerms.txt"))
    if not matches:
        raise FileNotFoundError(f"No gmdnTerms.txt found below {gmdn_root}")
    if len(matches) > 1:
        rendered = "\n  ".join(str(path) for path in matches)
        raise ValueError(
            "Multiple gmdnTerms.txt files found; select one with --input:\n"
            f"  {rendered}"
        )
    return matches[0]


def _required_value(
    row: dict[str | None, str | list[str] | None], field: str, row_number: int
) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Row {row_number}: {field} is empty")
    return value.strip()


def _implantable_value(
    row: dict[str | None, str | list[str] | None], row_number: int
) -> bool:
    value = row.get("implantable")
    if not isinstance(value, str):
        raise ValueError(f"Row {row_number}: implantable is missing")

    normalized = value.strip().lower()
    if normalized in {"", "false"}:
        return False
    if normalized == "true":
        return True
    raise ValueError(f"Row {row_number}: unsupported implantable value {value!r}")


def _read_terms(input_path: Path) -> tuple[dict[str, GmdnTerm], int]:
    terms: dict[str, GmdnTerm] = {}
    source_rows = 0

    with input_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="|")
        if tuple(reader.fieldnames or ()) != EXPECTED_HEADER:
            actual = ", ".join(reader.fieldnames or ()) or "<missing>"
            expected = ", ".join(EXPECTED_HEADER)
            raise ValueError(
                f"Unexpected header in {input_path}. Expected: {expected}. Got: {actual}"
            )

        for row_number, row in enumerate(reader, start=2):
            source_rows += 1
            if None in row:
                raise ValueError(
                    f"Row {row_number}: expected {len(EXPECTED_HEADER)} columns"
                )

            code = _required_value(row, "gmdnCode", row_number)
            name = _required_value(row, "gmdnPTName", row_number)
            definition = _required_value(row, "gmdnPTDefinition", row_number)
            status = _required_value(row, "gmdnCodeStatus", row_number)
            implantable = _implantable_value(row, row_number)

            if status not in VALID_STATUSES:
                raise ValueError(
                    f"Row {row_number}: unsupported gmdnCodeStatus {status!r}"
                )
            if not (len(code) == 5 and code.isascii() and code.isdigit()):
                raise ValueError(
                    f"Row {row_number}: gmdnCode must be a five-digit number, got {code!r}"
                )

            candidate = GmdnTerm(code, name, definition, status, implantable)
            existing = terms.get(code)
            if existing is None:
                terms[code] = candidate
            elif existing.status == candidate.status:
                if (
                    existing.name != candidate.name
                    or existing.definition != candidate.definition
                ):
                    raise ValueError(
                        f"Row {row_number}: conflicting {status} metadata for GMDN code {code}"
                    )
                terms[code] = replace(
                    existing,
                    implantable=existing.implantable or candidate.implantable,
                )
            elif candidate.status == "Obsolete":
                terms[code] = replace(
                    candidate,
                    implantable=existing.implantable or candidate.implantable,
                )
            else:
                terms[code] = replace(
                    existing,
                    implantable=existing.implantable or candidate.implantable,
                )

    return terms, source_rows


def _write_terms(output_path: Path, terms: dict[str, GmdnTerm]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            writer = csv.writer(destination, lineterminator="\n")
            writer.writerow(OUTPUT_HEADER)
            for code in sorted(terms, key=int):
                term = terms[code]
                writer.writerow(
                    (
                        term.code,
                        term.name,
                        term.definition,
                        term.status,
                        "true" if term.implantable else "false",
                    )
                )

        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def extract_gmdn_terms(input_path: Path, output_path: Path) -> ExtractionStats:
    """Extract one canonical row per GMDN code and atomically write the CSV."""
    terms, source_rows = _read_terms(input_path)
    _write_terms(output_path, terms)

    active_codes = sum(term.status == "Active" for term in terms.values())
    obsolete_codes = sum(term.status == "Obsolete" for term in terms.values())
    implantable_codes = sum(term.implantable for term in terms.values())
    return ExtractionStats(
        source_rows=source_rows,
        unique_codes=len(terms),
        active_codes=active_codes,
        obsolete_codes=obsolete_codes,
        implantable_codes=implantable_codes,
        non_implantable_codes=len(terms) - implantable_codes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract unique GMDN codes, names, definitions, and statuses from gmdnTerms.txt."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="AccessGUDID gmdnTerms.txt (auto-discovered below data/GMDN by default)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination CSV (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input or discover_input()
    stats = extract_gmdn_terms(input_path, args.output)
    print(f"Read {stats.source_rows:,} source rows from {input_path}")
    print(
        f"Wrote {stats.unique_codes:,} unique GMDN codes "
        f"({stats.active_codes:,} Active, {stats.obsolete_codes:,} Obsolete)"
    )
    print(
        f"Implantable: {stats.implantable_codes:,} true, "
        f"{stats.non_implantable_codes:,} false"
    )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
