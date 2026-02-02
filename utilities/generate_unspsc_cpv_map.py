#!/usr/bin/env python3
"""
UNSPSC to CPV Mapping Generator

This script maps all UNSPSC Segments (58) and Families (559) to CPV codes
using semantic classification. Total: 617 records to classify.

Input: data/unspsc-english-v260801.1.xlsx (row 13+ has headers)
Output: data/unspsc_to_cpv_mapping.csv

For each UNSPSC record:
- If Segment (no Family): uses Segment Title + Segment Definition
- If Family (has Family, no Class): uses Family Title + Family Definition
- Classifies against CPV using semantic search
- Stores top 3 CPV matches with scores

Progress is saved every 100 rows in case of interruption.
"""

import asyncio
import csv
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from google import genai
from qdrant_client import AsyncQdrantClient

from app.classifier import perform_classification
from app.classifier_config import CLASSIFIER_CONFIG

# Configuration

# Configuration
INPUT_FILE = Path("data/unspsc-english-v260801.1.xlsx")
OUTPUT_FILE = Path("data/unspsc_to_cpv_mapping.csv")
PROGRESS_FILE = Path("data/unspsc_to_cpv_mapping_progress.csv")
ROWS_PER_BATCH = 100

# Load environment variables (explicit path to avoid Python 3.14 frame issues)
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


def load_unspsc_data() -> pd.DataFrame:
    """Load UNSPSC Excel and filter to Segments + Families only (Class is empty)."""
    print(f"Loading UNSPSC data from {INPUT_FILE}...")

    # Row 13 (0-indexed: 12) contains column headers
    df = pd.read_excel(INPUT_FILE, header=12)

    # Filter rows where 'Class' is NaN (empty) - these are Segments and Families
    segment_family = df[df["Class"].isna()].copy()

    print(
        f"Loaded {len(segment_family)} UNSPSC Segments + Families (58 segments + 559 families)"
    )

    return segment_family


def build_query(row: pd.Series) -> str:
    """Build classification query from UNSPSC row.

    Format:
    - Segment only: Segment Title - Segment Definition
    - Family:
      Family Title - Family Definition
      (Segment Title - Segment Definition)

    Family name comes first, Segment context in parentheses on new line.
    Parentheses signal that Segment info is secondary/contextual.
    If definition is missing, the "-" separator is omitted.
    """
    # Check if it's a Segment (no Family) or Family (has Family)
    if pd.isna(row["Family"]):
        # It's a Segment - single line format
        title = row["Segment Title"] if pd.notna(row["Segment Title"]) else ""
        definition = (
            row["Segment Definition"] if pd.notna(row["Segment Definition"]) else ""
        )
        if title and definition:
            query = f"{title} - {definition}"
        else:
            query = title or definition
    else:
        # It's a Family - two line format
        # Line 1: Family Title - Family Definition (PRIMARY)
        family_title = row["Family Title"] if pd.notna(row["Family Title"]) else ""
        family_definition = (
            row["Family Definition"] if pd.notna(row["Family Definition"]) else ""
        )

        # Line 2: (Segment Title - Segment Definition) (SECONDARY/CONTEXT)
        segment_title = row["Segment Title"] if pd.notna(row["Segment Title"]) else ""
        segment_definition = (
            row["Segment Definition"] if pd.notna(row["Segment Definition"]) else ""
        )

        # Build line 1 (Family) - PRIMARY INFORMATION
        if family_title and family_definition:
            line1 = f"{family_title} - {family_definition}"
        else:
            line1 = family_title or family_definition

        # Build line 2 (Segment) - SECONDARY CONTEXT IN PARENTHESES
        if segment_title and segment_definition:
            line2 = f"({segment_title} - {segment_definition})"
        elif segment_title:
            line2 = f"({segment_title})"
        elif segment_definition:
            line2 = f"({segment_definition})"
        else:
            line2 = ""

        # Combine with newline
        if line1 and line2:
            query = f"{line1}\n{line2}"
        else:
            query = line1 or line2

    query = query.strip()

    # Truncate if too long (max 3900 to stay under 4000 limit)
    # Priority: Keep Family line intact, truncate Segment if needed
    if len(query) > 3900:
        lines = query.split("\n")
        if len(lines) == 2 and len(lines[0]) < 3900:
            # Keep Family line, truncate Segment line
            remaining = 3900 - len(lines[0]) - 1  # -1 for newline
            lines[1] = lines[1][:remaining]
            query = "\n".join(lines)
        else:
            query = query[:3900]

    return query


def get_unspsc_code(row: pd.Series) -> str:
    """Get the UNSPSC code (Segment or Family)."""
    if pd.isna(row["Family"]):
        return str(int(row["Segment"]))
    else:
        return str(int(row["Family"]))


def get_unspsc_title(row: pd.Series) -> str:
    """Get the UNSPSC title."""
    if pd.isna(row["Family"]):
        return row["Segment Title"] if pd.notna(row["Segment Title"]) else ""
    else:
        return row["Family Title"] if pd.notna(row["Family Title"]) else ""


def get_unspsc_definition(row: pd.Series) -> str:
    """Get the UNSPSC definition."""
    if pd.isna(row["Family"]):
        return row["Segment Definition"] if pd.notna(row["Segment Definition"]) else ""
    else:
        return row["Family Definition"] if pd.notna(row["Family Definition"]) else ""


def get_unspsc_type(row: pd.Series) -> str:
    """Get the UNSPSC type (Segment or Family)."""
    return "Segment" if pd.isna(row["Family"]) else "Family"


def save_results(
    results: List[Dict], output_file: Path, is_progress: bool = False
) -> None:
    """Save classification results to CSV."""
    if not results:
        return

    # Define CSV columns
    fieldnames = [
        "unspsc_code",
        "unspsc_type",
        "unspsc_title",
        "unspsc_definition",
        "cpv_match_1_code",
        "cpv_match_1_name",
        "cpv_match_1_score",
        "cpv_match_2_code",
        "cpv_match_2_name",
        "cpv_match_2_score",
        "cpv_match_3_code",
        "cpv_match_3_name",
        "cpv_match_3_score",
    ]

    file_exists = output_file.exists()
    mode = "a" if file_exists else "w"

    with open(output_file, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or (is_progress and not file_exists):
            writer.writeheader()
        writer.writerows(results)

    print(f"  Saved {len(results)} results to {output_file}")


def load_existing_results(output_file: Path) -> set:
    """Load already processed UNSPSC codes to avoid re-processing."""
    processed = set()

    if not output_file.exists():
        return processed

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["unspsc_code"])
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")

    print(f"Found {len(processed)} already processed records")
    return processed


async def classify_single(
    embed_client: genai.Client,
    qdrant_client: AsyncQdrantClient,
    row: pd.Series,
    quantization_cache: Dict[str, bool],
) -> Optional[Dict]:
    """Classify a single UNSPSC record against CPV."""
    query = build_query(row)

    if not query:
        print(f"  Warning: Empty query for {get_unspsc_code(row)}, skipping")
        return None

    try:
        # Call the classifier
        result = await perform_classification(
            embed_client=embed_client,
            qdrant_client=qdrant_client,
            query=query,
            classifier_type="CPV",
            version=None,  # Use default version
            top_k=3,
            quantization_cache=quantization_cache,
        )

        # Extract top 3 matches
        matches = result["results"][:3]

        # Build result row
        row_data = {
            "unspsc_code": get_unspsc_code(row),
            "unspsc_type": get_unspsc_type(row),
            "unspsc_title": get_unspsc_title(row),
            "unspsc_definition": get_unspsc_definition(row),
        }

        # Add CPV matches
        for i, match in enumerate(matches):
            payload = match.get("payload", {})
            row_data[f"cpv_match_{i + 1}_code"] = payload.get("original_id", "")
            row_data[f"cpv_match_{i + 1}_name"] = payload.get("class_name", "")
            row_data[f"cpv_match_{i + 1}_score"] = round(match.get("score", 0), 4)

        # Fill empty matches with empty strings
        for i in range(len(matches), 3):
            row_data[f"cpv_match_{i + 1}_code"] = ""
            row_data[f"cpv_match_{i + 1}_name"] = ""
            row_data[f"cpv_match_{i + 1}_score"] = ""

        return row_data

    except Exception as e:
        print(f"  Error classifying {get_unspsc_code(row)}: {e}")
        return None


async def main():
    """Main entry point."""
    print("=" * 70)
    print("UNSPSC to CPV Mapping Generator")
    print("=" * 70)

    # Check environment variables
    gemini_key = os.getenv("GEMINI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")

    if not gemini_key:
        print("Error: GEMINI_API_KEY not set in environment")
        sys.exit(1)

    if not qdrant_url:
        print("Error: QDRANT_URL not set in environment")
        sys.exit(1)

    print(f"Qdrant URL: {qdrant_url}")
    print(f"Gemini API Key: {'*' * 10}{gemini_key[-4:]}")

    # Initialize clients
    print("\nInitializing clients...")
    embed_client = genai.Client(api_key=gemini_key)
    qdrant_client = AsyncQdrantClient(
        url=qdrant_url,
        port=443,
        https=True,
        api_key=qdrant_key,
        timeout=60,
    )

    # Check if CPV collection has quantization
    quantization_cache: Dict[str, bool] = {}
    cpv_config = CLASSIFIER_CONFIG.get("CPV", {})
    cpv_version = (
        list(cpv_config.get("versions", {}).keys())[0]
        if cpv_config.get("versions")
        else ""
    )
    cpv_collection = (
        cpv_config.get("versions", {}).get(cpv_version, {}).get("collection_name", "")
    )

    if cpv_collection:
        try:
            collection_info = await qdrant_client.get_collection(cpv_collection)
            has_quantization = collection_info.config.quantization_config is not None
            quantization_cache[cpv_collection] = has_quantization
            print(f"CPV collection '{cpv_collection}' quantization: {has_quantization}")
        except Exception as e:
            print(f"Warning: Could not check quantization config: {e}")

    # Load UNSPSC data
    df = load_unspsc_data()

    # Load already processed codes
    processed_codes = load_existing_results(OUTPUT_FILE)

    # Filter out already processed
    df_to_process = df[~df.apply(get_unspsc_code, axis=1).isin(processed_codes)].copy()

    if len(df_to_process) == 0:
        print("\nAll records already processed!")
        print(f"Output file: {OUTPUT_FILE}")
        await qdrant_client.close()
        return

    print(f"\nProcessing {len(df_to_process)} remaining records...")
    print(f"Output will be saved to: {OUTPUT_FILE}")
    print(f"Progress saved every {ROWS_PER_BATCH} rows\n")

    # Process in batches
    batch_results = []
    total_processed = len(processed_codes)

    for idx, (_, row) in enumerate(df_to_process.iterrows(), 1):
        unspsc_code = get_unspsc_code(row)
        unspsc_type = get_unspsc_type(row)
        query_preview = build_query(row)[:80]

        print(
            f"[{total_processed + idx}/{len(df)}] {unspsc_type} {unspsc_code}: {query_preview}..."
        )

        # Classify
        result = await classify_single(
            embed_client, qdrant_client, row, quantization_cache
        )

        if result:
            batch_results.append(result)
            print(
                f"  -> CPV: {result.get('cpv_match_1_code', 'N/A')} ({result.get('cpv_match_1_name', 'N/A')[:50]}...)"
            )

        # Save progress every N rows
        if len(batch_results) >= ROWS_PER_BATCH:
            save_results(batch_results, OUTPUT_FILE)
            batch_results = []
            print(
                f"\n*** Progress saved! {total_processed + idx} records completed ***\n"
            )

    # Save remaining results
    if batch_results:
        save_results(batch_results, OUTPUT_FILE)

    # Cleanup
    await qdrant_client.close()

    print("\n" + "=" * 70)
    print("Mapping complete!")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total records: {len(df)}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Checkpoint results have been saved.")
        sys.exit(0)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
