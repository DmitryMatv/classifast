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
from typing import Any, Dict, List, Optional

# Add parent directory to path for app imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from zeroentropy import ZeroEntropy

from app.classifier import get_embedding, perform_semantic_search
from app.classifier_config import CLASSIFIER_CONFIG

# Configuration
INPUT_FILE = Path(__file__).parent.parent / "data/unspsc-english-v260801.1.xlsx"
OUTPUT_FILE = Path(__file__).parent / "unspsc_to_cpv_mapping.csv"
ROWS_PER_BATCH = 100

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
        "cpv_match_1_zeroentropy_score",
        "cpv_match_2_code",
        "cpv_match_2_name",
        "cpv_match_2_score",
        "cpv_match_2_zeroentropy_score",
        "cpv_match_3_code",
        "cpv_match_3_name",
        "cpv_match_3_score",
        "cpv_match_3_zeroentropy_score",
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


def rerank_with_zeroentropy(
    zclient: ZeroEntropy,
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    rerank_top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Rerank semantic search results using ZeroEntropy rerank API.

    Args:
        zclient: Initialized ZeroEntropy client
        query: The search query (UNSPSC title/description)
        candidates: List of candidate matches from semantic search
        top_k: Number of top results to return after reranking
        rerank_top_n: Number of candidates to send to ZeroEntropy

    Returns:
        List of reranked candidates with zeroentropy_relevance_score field.
        Falls back to original candidates if API fails.
    """
    if not candidates or not zclient:
        # Return candidates with default zeroentropy score for consistency
        return [{**c, "zeroentropy_relevance_score": 0.0} for c in candidates[:top_k]]

    # Limit candidates to rerank
    candidates_to_rerank = candidates[: min(rerank_top_n, len(candidates))]
    remaining_candidates = candidates[len(candidates_to_rerank) :]

    # Prepare documents for ZeroEntropy - use just class_name as text
    # but keep track of index to map back to original candidates
    documents = []
    for i, candidate in enumerate(candidates_to_rerank):
        payload = candidate.get("payload", {})
        class_name = payload.get("class_name", "")
        documents.append(class_name)

    try:
        print(f"    Reranking top {len(documents)} candidates with ZeroEntropy...")

        # Call ZeroEntropy rerank API
        response = zclient.models.rerank(
            model="zerank-2",
            query=query,
            documents=documents,
        )

        # Map rerank results back to original candidates
        reranked_candidates = []
        seen_indices = set()

        # Process results from ZeroEntropy
        for result in response.results:
            original_index = result.index
            seen_indices.add(original_index)
            original_candidate = candidates_to_rerank[original_index].copy()

            # Add ZeroEntropy relevance score (normalize to 0-100 scale)
            relevance_score = result.relevance_score * 100
            original_candidate["zeroentropy_relevance_score"] = round(
                relevance_score, 4
            )

            reranked_candidates.append(original_candidate)

        # Add any candidates that weren't returned (shouldn't happen, but safe)
        for i, candidate in enumerate(candidates_to_rerank):
            if i not in seen_indices:
                candidate_copy = candidate.copy()
                candidate_copy["zeroentropy_relevance_score"] = 0.0
                reranked_candidates.append(candidate_copy)

        if reranked_candidates:
            print(
                f"    ZeroEntropy reranking complete. Top result: {reranked_candidates[0].get('payload', {}).get('original_id', 'N/A')} (score: {reranked_candidates[0].get('zeroentropy_relevance_score', 0):.2f})"
            )
        else:
            print("    ZeroEntropy reranking complete. No results returned.")

    except Exception as e:
        print(
            f"    Warning: ZeroEntropy reranking failed: {e}, using semantic search scores"
        )
        # Return original candidates with zeroentropy_relevance_score = 0
        reranked_candidates = []
        for c in candidates_to_rerank:
            c_copy = c.copy()
            c_copy["zeroentropy_relevance_score"] = 0.0
            reranked_candidates.append(c_copy)

    # Add remaining candidates (not reranked) with 0 score
    for c in remaining_candidates:
        c_copy = c.copy()
        c_copy["zeroentropy_relevance_score"] = 0.0
        reranked_candidates.append(c_copy)

    return reranked_candidates[:top_k]


def classify_single(
    embed_client: genai.Client,
    qdrant_client: QdrantClient,
    zclient: Optional[ZeroEntropy],
    row: pd.Series,
    cpv_collection: str,
    cpv_config: Dict,
    quantization_cache: Dict[str, bool],
) -> Optional[Dict]:
    """Classify a single UNSPSC record against CPV using semantic search + ZeroEntropy reranking."""
    query = build_query(row)

    if not query:
        print(f"  Warning: Empty query for {get_unspsc_code(row)}, skipping")
        return None

    try:
        # Generate embedding for the query
        query_embedding = get_embedding(
            embed_client=embed_client,
            model_name=cpv_config.get("embed_model_name", ""),
            text=query,
            task_type="RETRIEVAL_QUERY",
            embed_dims=cpv_config.get("embed_dims"),
        )

        # Perform semantic search - get top N for reranking
        has_quantization = quantization_cache.get(cpv_collection, False)
        matches = perform_semantic_search(
            qdrant_client=qdrant_client,
            collection_name=cpv_collection,
            query_embedding=query_embedding,
            top_k=100,  # Get more candidates for better reranking results
            has_quantization=has_quantization,
        )

        # Rerank with ZeroEntropy if available
        if zclient and matches:
            reranked_matches = rerank_with_zeroentropy(
                zclient=zclient,
                query=query,
                candidates=matches,
                top_k=3,  # Get top 3 to match CSV fieldnames
                rerank_top_n=50,  # Rerank top N candidates for better results
            )
        else:
            # No ZeroEntropy available, use original semantic search scores
            reranked_matches = [
                {**match, "zeroentropy_relevance_score": 0.0} for match in matches[:3]
            ]

        # Build result row
        row_data = {
            "unspsc_code": get_unspsc_code(row),
            "unspsc_type": get_unspsc_type(row),
            "unspsc_title": get_unspsc_title(row),
            "unspsc_definition": get_unspsc_definition(row),
            "embedding_text": query,
        }

        # Add CPV matches with both semantic and ZeroEntropy scores
        for i, match in enumerate(reranked_matches):
            payload = match.get("payload", {})
            row_data[f"cpv_match_{i + 1}_code"] = payload.get("original_id", "")
            row_data[f"cpv_match_{i + 1}_name"] = payload.get("class_name", "")
            row_data[f"cpv_match_{i + 1}_score"] = round(match.get("score", 0), 4)
            row_data[f"cpv_match_{i + 1}_zeroentropy_score"] = round(
                match.get("zeroentropy_relevance_score", 0), 4
            )

        # Fill empty matches with empty strings
        for i in range(len(reranked_matches), 3):
            row_data[f"cpv_match_{i + 1}_code"] = ""
            row_data[f"cpv_match_{i + 1}_name"] = ""
            row_data[f"cpv_match_{i + 1}_score"] = ""
            row_data[f"cpv_match_{i + 1}_zeroentropy_score"] = ""

        return row_data

    except Exception as e:
        print(f"  Error classifying {get_unspsc_code(row)}: {e}")
        return None


def main():
    """Main entry point."""
    print("=" * 70)
    print("UNSPSC to CPV Mapping Generator")
    print("=" * 70)

    # Check environment variables
    gemini_key = os.getenv("GEMINI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    zeroentropy_key = os.getenv("ZEROENTROPY_API_KEY")

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

    # Initialize ZeroEntropy client (optional - reads ZEROENTROPY_API_KEY from env)
    zclient = None
    if zeroentropy_key:
        zclient = ZeroEntropy()
        print(
            f"ZeroEntropy API Key: {'*' * 10}{zeroentropy_key[-4:]} (reranking enabled)"
        )
    else:
        print(
            "ZeroEntropy API Key: NOT SET (reranking disabled - using semantic scores only)"
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
    print(f"Progress saved every {ROWS_PER_BATCH} rows\n")

    # Process in batches
    batch_results = []
    total_processed = len(processed_codes)

    for idx, (_, row) in enumerate(df_to_process.iterrows(), 1):
        unspsc_code = get_unspsc_code(row)
        # unspsc_type = get_unspsc_type(row)
        query_preview = build_query(row)[:80]

        print("=" * 70)
        print(f"[{total_processed + idx}/{len(df)}] {unspsc_code}: {query_preview}...")

        # Classify
        result = classify_single(
            embed_client,
            qdrant_client,
            zclient,
            row,
            cpv_collection,
            cpv_config,
            quantization_cache,
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
