#!/usr/bin/env python3
"""
UNSPSC to CPV Mapping Generator

This script maps all UNSPSC Segments (58) and Families (559) to CPV codes
using semantic classification. Total: 617 records to classify.

Input: data/unspsc-english-v260801.1.xlsx (row 13+ has headers)
Output: data/unspsc_to_cpv_mapping.csv

For each UNSPSC record:
- Uses clean UNSPSC Title only (Segment Title or Family Title)
- Classifies against CPV using semantic search
- Stores top 3 CPV matches with scores

Progress is saved every 100 rows in case of interruption.
"""

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for app imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient

from app.classifier import get_embeddings_batch, perform_semantic_search_batch
from app.classifier_config import CLASSIFIER_CONFIG

# Configuration
INPUT_FILE = Path(__file__).parent.parent / "data/unspsc-english-v260801.1.xlsx"
OUTPUT_FILE = Path(__file__).parent / "unspsc_to_cpv_mapping.csv"
ROWS_PER_BATCH = 128
EMBEDDING_BATCH_SIZE = 32  # Process embeddings in batches of 32 for efficiency

# Load environment variables from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def load_unspsc_data() -> pd.DataFrame:
    """Load UNSPSC Excel and filter to Segments + Families only (Class is empty)."""
    print(f"Loading UNSPSC data from {INPUT_FILE}...")

    # Row 13 (0-indexed: 12) contains column headers
    df = pd.read_excel(INPUT_FILE, header=12)

    # Filter rows where 'Class' is NaN (empty) - these are Segments and Families
    segment_family = df[df["Class"].isna()].copy()

    print(f"Loaded {len(segment_family)} UNSPSC Segments + Families")

    return segment_family


def build_query(row: pd.Series) -> str:
    """Build classification query from UNSPSC row.

    Uses title as primary information with definition as clarifying context.
    Title appears first for higher semantic weight in embeddings.

    For Families, includes parent Segment information at the end for additional
    context without overwhelming the Family's own semantic meaning.
    """
    title = get_unspsc_title(row)
    definition = get_unspsc_definition(row)

    # Build base query with title (markdown header + title case for semantic weight)
    query = f"# {title}"
    if definition:
        query += f"\n\nDefinition: {definition}"
    else:
        query += f"\n\nDefinition: {title}"

    # For Families, append Segment context at the end (lowest semantic weight)
    if pd.notna(row["Family"]):
        segment_title = get_segment_title(row)
        segment_definition = get_segment_definition(row)

        if segment_title:
            query += f"\n\nParent category: {segment_title}"
            if segment_definition:
                query += f"\nParent category definition: {segment_definition}"

    return query


def get_unspsc_code(row: pd.Series) -> str:
    """Get the UNSPSC code (Segment or Family)."""
    if pd.isna(row["Family"]):
        if pd.isna(row["Segment"]):
            return ""
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


def get_segment_title(row: pd.Series) -> str:
    """Get the Segment title (for Families to get parent segment context)."""
    return row["Segment Title"] if pd.notna(row["Segment Title"]) else ""


def get_segment_definition(row: pd.Series) -> str:
    """Get the Segment definition (for Families to get parent segment context)."""
    return row["Segment Definition"] if pd.notna(row["Segment Definition"]) else ""


def get_unspsc_type(row: pd.Series) -> str:
    """Get the UNSPSC type (Segment or Family)."""
    return "Segment" if pd.isna(row["Family"]) else "Family"


def save_results(results: List[Dict], output_file: Path) -> None:
    """Save classification results to CSV."""
    if not results:
        return

    # Define CSV columns
    fieldnames = [
        "unspsc_code",
        "unspsc_type",
        "unspsc_title",
        "unspsc_definition",
        "embedding_text",
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
        if not file_exists:
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


def classify_batch(
    embed_client: genai.Client,
    qdrant_client: QdrantClient,
    rows: List[pd.Series],
    cpv_collection: str,
    cpv_config: Dict,
    quantization_cache: Dict[str, bool],
) -> List[Optional[Dict]]:
    """
    Classify multiple UNSPSC records against CPV using batch embedding and search.

    Args:
        embed_client: The Google GenAI client
        qdrant_client: The Qdrant client
        rows: List of DataFrame rows to classify
        cpv_collection: Name of the CPV collection
        cpv_config: CPV classifier configuration
        quantization_cache: Cache of collection quantization status

    Returns:
        List of classification results (one per input row, None if failed)
    """
    if not rows:
        return []

    # Build queries for all rows
    queries = []
    valid_indices = []

    for i, row in enumerate(rows):
        query = build_query(row)
        if query:
            queries.append(query)
            valid_indices.append(i)
        else:
            print(f"  Warning: Empty query for {get_unspsc_code(row)}, skipping")

    if not queries:
        return [None] * len(rows)

    try:
        # Generate embeddings for all queries in one batch call
        embeddings = get_embeddings_batch(
            embed_client=embed_client,
            model_name=cpv_config.get("embed_model_name", ""),
            texts=queries,
            task_type="RETRIEVAL_QUERY",
            embed_dims=cpv_config.get("embed_dims"),
        )

        # Perform semantic search for all embeddings in one batch call
        has_quantization = quantization_cache.get(cpv_collection, False)
        all_matches = perform_semantic_search_batch(
            qdrant_client=qdrant_client,
            collection_name=cpv_collection,
            query_embeddings=embeddings,
            top_k=3,
            has_quantization=has_quantization,
        )

        # Build results for all rows
        results = [None] * len(rows)

        for result_idx, row_idx in enumerate(valid_indices):
            row = rows[row_idx]
            matches = all_matches[result_idx]
            query = queries[result_idx]

            # Build result row
            row_data = {
                "unspsc_code": get_unspsc_code(row),
                "unspsc_type": get_unspsc_type(row),
                "unspsc_title": get_unspsc_title(row),
                "unspsc_definition": get_unspsc_definition(row),
                "embedding_text": query,
            }

            # Add CPV matches (limit to top 3, even if more returned)
            top_matches = matches[:3]
            for i, match in enumerate(top_matches):
                payload = match.get("payload", {})
                row_data[f"cpv_match_{i + 1}_code"] = payload.get("original_id", "")
                row_data[f"cpv_match_{i + 1}_name"] = payload.get("class_name", "")
                row_data[f"cpv_match_{i + 1}_score"] = round(match.get("score", 0), 4)

            # Fill empty matches with empty strings
            for i in range(len(top_matches), 3):
                row_data[f"cpv_match_{i + 1}_code"] = ""
                row_data[f"cpv_match_{i + 1}_name"] = ""
                row_data[f"cpv_match_{i + 1}_score"] = ""

            results[row_idx] = row_data

        return results

    except Exception as e:
        print(f"  Error in batch classification: {e}")
        # Return None for all rows in the batch on failure
        return [None] * len(rows)


def main():
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
    qdrant_client = QdrantClient(
        url=qdrant_url,
        port=443,
        https=True,
        api_key=qdrant_key,
        timeout=60,
    )

    # Get CPV configuration
    quantization_cache: Dict[str, bool] = {}
    cpv_config = CLASSIFIER_CONFIG.get("CPV", {})
    cpv_version = (
        list(cpv_config.get("versions", {}).keys())[0]
        if cpv_config.get("versions")
        else ""
    )
    cpv_version_config = cpv_config.get("versions", {}).get(cpv_version, {})
    cpv_collection = cpv_version_config.get("collection_name", "")

    if not cpv_collection:
        print("Error: Could not determine CPV collection name")
        sys.exit(1)

    print(f"Using CPV collection: {cpv_collection}")

    # Check if CPV collection has quantization
    try:
        collection_info = qdrant_client.get_collection(cpv_collection)
        has_quantization = collection_info.config.quantization_config is not None
        quantization_cache[cpv_collection] = has_quantization
        print(f"CPV collection quantization: {has_quantization}")
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
        return

    print(f"\nProcessing {len(df_to_process)} remaining records...")
    print(f"Output will be saved to: {OUTPUT_FILE}")
    print(f"Progress saved every {ROWS_PER_BATCH} rows")
    print(f"Embedding batch size: {EMBEDDING_BATCH_SIZE} (API calls per batch)\n")

    # Process in embedding batches for efficiency
    batch_results = []
    total_processed = len(processed_codes)

    # Convert DataFrame to list of rows for batch processing
    rows_to_process = [row for _, row in df_to_process.iterrows()]
    total_rows = len(rows_to_process)

    # Process in embedding batches
    for batch_start in range(0, total_rows, EMBEDDING_BATCH_SIZE):
        batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, total_rows)
        current_batch = rows_to_process[batch_start:batch_end]

        # Print batch info
        first_code = get_unspsc_code(current_batch[0])
        last_code = get_unspsc_code(current_batch[-1])
        print(
            f"\n[Batch {batch_start // EMBEDDING_BATCH_SIZE + 1}] Processing {len(current_batch)} records ({first_code} to {last_code})"
        )

        # Classify batch
        results = classify_batch(
            embed_client,
            qdrant_client,
            current_batch,
            cpv_collection,
            cpv_config,
            quantization_cache,
        )

        # Process results
        for i, (row, result) in enumerate(zip(current_batch, results)):
            idx = batch_start + i + 1
            unspsc_code = get_unspsc_code(row)

            if result:
                batch_results.append(result)
                print(
                    f"  [{total_processed + idx}/{len(df)}] {unspsc_code} -> CPV: {result.get('cpv_match_1_code', 'N/A')} ({result.get('cpv_match_1_name', 'N/A')[:40]}...)"
                )
            else:
                print(f"  [{total_processed + idx}/{len(df)}] {unspsc_code} -> FAILED")

        # Save progress every N rows
        if len(batch_results) >= ROWS_PER_BATCH:
            save_results(batch_results, OUTPUT_FILE)
            batch_results = []
            print(
                f"\n*** Progress saved! {total_processed + batch_end} records completed ***\n"
            )

    # Save remaining results
    if batch_results:
        save_results(batch_results, OUTPUT_FILE)

    qdrant_client.close()

    print("\n" + "=" * 70)
    print("Mapping complete!")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total records: {len(df)}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Checkpoint results have been saved.")
        sys.exit(0)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
