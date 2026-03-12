#!/usr/bin/env python3
"""
CPV to UNSPSC Mapping Generator

This script maps all CPV codes (~9,454) to UNSPSC categories using semantic classification.

Input: data/cpv_2008_ver_2013.xlsx (CODE + EN columns)
Output: mapping/cpv_to_unspsc_mapping.csv

For each CPV record:
- Uses CPV English name as query
- Classifies against UNSPSC using semantic search
- Stores top 3 UNSPSC matches with scores

Progress is saved every 10 rows in case of interruption.
"""

import csv
import logging
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import tenacity
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException as QdrantApiException
from zeroentropy import ZeroEntropy, APIError as ZeroEntropyAPIError

from app.classifier import (
    get_embedding,
    perform_semantic_search,
    rerank_with_zeroentropy,
)

INPUT_FILE = Path(__file__).parent.parent / "data/cpv_2008_ver_2013.xlsx"
UNSPSC_FILE = Path(__file__).parent.parent / "data/unspsc-english-v260801.1.xlsx"
OUTPUT_FILE = Path(__file__).parent / "cpv_to_unspsc_mapping.csv"

# Keep retrieval and rerank tunable independently for mapping-quality experiments.
SEMANTIC_RETRIEVE_LIMIT = 300
RERANK_CANDIDATE_LIMIT = 100
OUTPUT_TOP_K = 3

# Set to False to use payload-based title/definition text only for reranking.
USE_UNSPSC_RERANK_CONTEXT = False

ROWS_PER_BATCH = 10


def get_cpv_level(code: str) -> str:
    """
    Determine the hierarchy level of a CPV code.

    CPV structure:
    - Division: XX000000-Y (first 2 digits)
    - Group: XXX00000-Y (first 3 digits)
    - Class: XXXX0000-Y (first 4 digits)
    - Category: XXXXX000-Y (first 5 digits)
    - Sub-category: XXXXXXXY (more specific)
    """
    if not code or not isinstance(code, str):
        return "unknown"
    clean_code = code.replace("-", "")
    if len(clean_code) < 8:
        return "unknown"

    significant = clean_code[:8]
    trailing_zeros = len(significant) - len(significant.rstrip("0"))

    if trailing_zeros == 6:
        return "division"
    elif trailing_zeros == 5:
        return "group"
    elif trailing_zeros == 4:
        return "class"
    elif trailing_zeros == 3:
        return "category"
    else:
        return "subcategory"


def find_parent_code(
    pattern_prefix: str, expected_level: str, parent_index: Dict[str, Dict[str, str]]
) -> Optional[str]:
    """Find a code in the dataset using pre-indexed lookup."""
    prefix_len = {"division": 2, "group": 3, "class": 4, "category": 5}.get(
        expected_level
    )
    if prefix_len and expected_level in parent_index:
        key = pattern_prefix[:prefix_len]
        return parent_index[expected_level].get(key)
    return None


def build_parent_index(
    code_to_description: Dict[str, str],
) -> Dict[str, Dict[str, str]]:
    """Build index mapping (level, prefix) -> full_code for fast parent lookup."""
    index: Dict[str, Dict[str, str]] = {
        "division": {},
        "group": {},
        "class": {},
        "category": {},
    }
    for full_code in code_to_description.keys():
        level = get_cpv_level(full_code)
        if level in index:
            clean = full_code.replace("-", "")
            if level == "division":
                index[level][clean[:2]] = full_code
            elif level == "group":
                index[level][clean[:3]] = full_code
            elif level == "class":
                index[level][clean[:4]] = full_code
            elif level == "category":
                index[level][clean[:5]] = full_code
    return index


CPV_PARENT_LEVELS = {
    "division": [],
    "group": ["division"],
    "class": ["group", "division"],
    "category": ["class", "group", "division"],
    "subcategory": ["category", "class", "group", "division"],
}

CPV_LEVEL_PATTERNS = {
    "division": (2, "000000"),
    "group": (3, "00000"),
    "class": (4, "0000"),
    "category": (5, "000"),
}


def build_hierarchy_text(
    code: str,
    description: str,
    code_to_description: dict,
    parent_index: Dict[str, Dict[str, str]],
) -> str:
    """
    Build hierarchical embedding text for a CPV code.

    Format:
    {Description} ({Parent1} < {Parent2} < {Parent3} ...)

    Example:
    Papaya seeds (Seeds < Farming < Agriculture)
    """
    clean_code = code.replace("-", "")
    level = get_cpv_level(code)

    parents = []
    seen_parent_codes = set()

    for level_name in CPV_PARENT_LEVELS.get(level, []):
        prefix_len, suffix = CPV_LEVEL_PATTERNS[level_name]
        pattern = clean_code[:prefix_len] + suffix
        parent_code = find_parent_code(pattern, level_name, parent_index)
        if (
            parent_code
            and parent_code != code
            and parent_code not in seen_parent_codes
            and parent_code in code_to_description
        ):
            parents.append(code_to_description[parent_code])
            seen_parent_codes.add(parent_code)

    if parents:
        hierarchy_chain = " < ".join(parents)
        return f"{description} (Hierarchy: {hierarchy_chain})"
    else:
        return description


def get_unspsc_level(code: str) -> str:
    """Determine the hierarchy level of a UNSPSC code."""
    if not code:
        return "unknown"
    try:
        code_int = int(code)
    except (ValueError, TypeError):
        return "unknown"

    if code_int % 1000000 == 0:
        return "segment"
    elif code_int % 10000 == 0:
        return "family"
    elif code_int % 100 == 0:
        return "class"
    else:
        return "commodity"


def _make_unspsc_entry(
    level: str,
    code: str,
    title: str,
    definition: str,
    segment: Optional[str],
    family: Optional[str],
    class_code: Optional[str],
    commodity: Optional[str],
    segment_title: str,
    family_title: str,
    class_title: str,
    commodity_title: str,
    segment_def: str,
    family_def: str,
    class_def: str,
    commodity_def: str,
) -> Dict[str, Any]:
    """Build a single UNSPSC lookup entry dict."""
    return {
        "title": title,
        "definition": definition,
        "level": level,
        "segment": segment,
        "family": family,
        "class": class_code,
        "commodity": commodity,
        "segment_title": segment_title,
        "family_title": family_title,
        "class_title": class_title,
        "commodity_title": commodity_title,
        "segment_def": segment_def,
        "family_def": family_def,
        "class_def": class_def,
        "commodity_def": commodity_def,
    }


def load_unspsc_data() -> Dict[str, Dict]:
    """Load UNSPSC Excel and build hierarchy lookup tables.

    Returns:
        Dict mapping UNSPSC code (str) to entry dict from _make_unspsc_entry.
    """
    if not UNSPSC_FILE.exists():
        raise FileNotFoundError(f"UNSPSC data file not found: {UNSPSC_FILE}")

    print(f"Loading UNSPSC data from {UNSPSC_FILE}...")

    df = pd.read_excel(UNSPSC_FILE, header=12)

    unspsc_lookup: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        segment = (
            str(int(row["Segment"])).zfill(8) if pd.notna(row["Segment"]) else None
        )
        family = str(int(row["Family"])).zfill(8) if pd.notna(row["Family"]) else None
        class_code = str(int(row["Class"])).zfill(8) if pd.notna(row["Class"]) else None
        commodity = (
            str(int(row["Commodity"])).zfill(8) if pd.notna(row["Commodity"]) else None
        )

        segment_title = (
            str(row["Segment Title"]) if pd.notna(row["Segment Title"]) else ""
        )
        segment_def = (
            str(row["Segment Definition"])
            if pd.notna(row["Segment Definition"])
            else ""
        )

        family_title = str(row["Family Title"]) if pd.notna(row["Family Title"]) else ""
        family_def = (
            str(row["Family Definition"]) if pd.notna(row["Family Definition"]) else ""
        )

        class_title = str(row["Class Title"]) if pd.notna(row["Class Title"]) else ""
        class_def = (
            str(row["Class Definition"]) if pd.notna(row["Class Definition"]) else ""
        )

        commodity_title = (
            str(row["Commodity Title"]) if pd.notna(row["Commodity Title"]) else ""
        )
        commodity_def = (
            str(row["Commodity Definition"])
            if pd.notna(row["Commodity Definition"])
            else ""
        )

        if commodity and commodity_title and commodity not in unspsc_lookup:
            unspsc_lookup[commodity] = _make_unspsc_entry(
                level="commodity",
                code=commodity,
                title=commodity_title,
                definition=commodity_def,
                segment=segment,
                family=family,
                class_code=class_code,
                commodity=commodity,
                segment_title=segment_title,
                family_title=family_title,
                class_title=class_title,
                commodity_title=commodity_title,
                segment_def=segment_def,
                family_def=family_def,
                class_def=class_def,
                commodity_def=commodity_def,
            )

        if class_code and class_title and class_code not in unspsc_lookup:
            unspsc_lookup[class_code] = _make_unspsc_entry(
                level="class",
                code=class_code,
                title=class_title,
                definition=class_def,
                segment=segment,
                family=family,
                class_code=class_code,
                commodity=None,
                segment_title=segment_title,
                family_title=family_title,
                class_title=class_title,
                commodity_title="",
                segment_def=segment_def,
                family_def=family_def,
                class_def=class_def,
                commodity_def="",
            )

        if family and family_title and family not in unspsc_lookup:
            unspsc_lookup[family] = _make_unspsc_entry(
                level="family",
                code=family,
                title=family_title,
                definition=family_def,
                segment=segment,
                family=family,
                class_code=None,
                commodity=None,
                segment_title=segment_title,
                family_title=family_title,
                class_title="",
                commodity_title="",
                segment_def=segment_def,
                family_def=family_def,
                class_def="",
                commodity_def="",
            )

        if segment and segment_title and segment not in unspsc_lookup:
            unspsc_lookup[segment] = _make_unspsc_entry(
                level="segment",
                code=segment,
                title=segment_title,
                definition=segment_def,
                segment=segment,
                family=None,
                class_code=None,
                commodity=None,
                segment_title=segment_title,
                family_title="",
                class_title="",
                commodity_title="",
                segment_def=segment_def,
                family_def="",
                class_def="",
                commodity_def="",
            )

    print(
        f"Loaded {len(unspsc_lookup)} UNSPSC codes (Segments + Families + Classes + Commodities)"
    )
    return unspsc_lookup


def build_unspsc_hierarchy_text(unspsc_code: str, unspsc_lookup: Optional[Dict]) -> str:
    """Build hierarchical text for a UNSPSC code.

    Format varies by level:
    - Segment: {Title} (Definition: ...)
    - Family: {Title} (Definition: ...) (Parent category: {Segment})
    - Class: {Title} (Definition: ...) (Hierarchy: {Family} < {Segment})
    - Commodity: {Title} (Definition: ...) (Hierarchy: {Class} < {Family} < {Segment})

    Example for commodity 10101501 (Cats):
    Cats (Hierarchy: Livestock < Live animals < Live Plant and Animal Material...)
    """
    if unspsc_lookup is None or unspsc_code not in unspsc_lookup:
        return ""

    data = unspsc_lookup[unspsc_code]
    level = data["level"]
    title = data["title"]
    definition = data.get("definition", "")

    details = []
    if definition:
        details.append(f"Definition: {definition}")

    if level == "segment":
        if details:
            return f"{title} ({') ('.join(details)})"
        return title

    if level == "family":
        segment_title = data.get("segment_title", "")
        if segment_title:
            details.append(f"Parent category: {segment_title}")
        if details:
            return f"{title} ({') ('.join(details)})"
        return title

    parents = []

    if level == "commodity" and data["class_title"]:
        parents.append(data["class_title"])
    if level in ["commodity", "class"] and data["family_title"]:
        parents.append(data["family_title"])
    if level in ["commodity", "class"] and data["segment_title"]:
        parents.append(data["segment_title"])

    if parents:
        details.append(f"Hierarchy: {' < '.join(parents)}")

    if details:
        return f"{title} ({') ('.join(details)})"
    return title


def build_unspsc_rerank_document(
    candidate: Dict[str, Any],
    unspsc_lookup: Optional[Dict[str, Dict]] = None,
    use_context: bool = True,
) -> str:
    """Build rerank text for a UNSPSC candidate with context-aware fallback."""
    payload = candidate.get("payload", {})

    if use_context and unspsc_lookup:
        original_id = payload.get("original_id", "")
        hierarchy_text = build_unspsc_hierarchy_text(original_id, unspsc_lookup)
        if hierarchy_text:
            return hierarchy_text

    class_name = payload.get("class_name", "")
    definition = payload.get("definition", "")

    if class_name and definition:
        return f"{class_name} - Definition: {definition}"
    if definition:
        return definition
    return class_name


def load_cpv_data() -> tuple[pd.DataFrame, Dict[str, str], Dict[str, Dict[str, str]]]:
    """Load CPV Excel and return CODE + EN columns plus code_to_description mapping and parent_index."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"CPV data file not found: {INPUT_FILE}")

    print(f"Loading CPV data from {INPUT_FILE}...")

    df = pd.read_excel(INPUT_FILE)

    df = df[["CODE", "EN"]].copy()
    df = df.dropna(subset=["CODE", "EN"])

    code_to_description = {}
    for _, row in df.iterrows():
        code = str(row["CODE"])
        desc = str(row["EN"])
        if code and desc:
            code_to_description[code] = desc

    parent_index = build_parent_index(code_to_description)

    print(f"Loaded {len(df)} CPV codes")
    print(f"Built hierarchy mapping with {len(code_to_description)} codes")
    print(
        f"Built parent index with {sum(len(v) for v in parent_index.values())} entries"
    )

    return df, code_to_description, parent_index


def save_results(results: List[Dict], output_file: Path) -> None:
    """Save classification results to CSV."""
    if not results:
        return

    fieldnames = [
        "cpv_code",
        "cpv_name",
        "cpv_level",
        "embedding_text",
        "unspsc_match_1_code",
        "unspsc_match_1_title",
        "unspsc_match_1_type",
        "unspsc_match_1_score",
        "unspsc_match_1_zeroentropy_score",
        "unspsc_match_2_code",
        "unspsc_match_2_title",
        "unspsc_match_2_type",
        "unspsc_match_2_score",
        "unspsc_match_2_zeroentropy_score",
        "unspsc_match_3_code",
        "unspsc_match_3_title",
        "unspsc_match_3_type",
        "unspsc_match_3_score",
        "unspsc_match_3_zeroentropy_score",
    ]

    file_exists = output_file.exists()
    mode = "a" if file_exists else "w"

    if file_exists:
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        if existing_header is None:
            # File is empty, treat as new file
            mode = "w"
            file_exists = False
        elif existing_header != fieldnames:
            raise ValueError(
                f"CSV header mismatch. Expected {fieldnames}, got {existing_header}. "
                "Delete the output file and re-run, or reconcile manually."
            )

    with open(output_file, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"  Saved {len(results)} results to {output_file}")


def load_existing_results(output_file: Path) -> set:
    """Load already processed CPV codes to avoid re-processing."""
    processed = set()

    if not output_file.exists():
        return processed

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["cpv_code"])
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")

    print(f"Found {len(processed)} already processed records")
    return processed


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=2, min=2, max=30),
    retry=tenacity.retry_if_exception_type((QdrantApiException, ZeroEntropyAPIError)),
    before_sleep=tenacity.before_sleep_log(
        logging.getLogger(__name__), logging.WARNING
    ),
)
def classify_single(
    embed_client: genai.Client,
    qdrant_client: QdrantClient,
    zclient: Optional[ZeroEntropy],
    row: pd.Series,
    unspsc_collection: str,
    quantization_cache: Dict[str, bool],
    code_to_description: Dict[str, str],
    parent_index: Dict[str, Dict[str, str]],
    unspsc_lookup: Optional[Dict[str, Dict]] = None,
) -> Optional[Dict]:
    """Classify a single CPV record against UNSPSC using semantic search + ZeroEntropy reranking."""
    cpv_code = str(row["CODE"])
    cpv_name = str(row["EN"])

    if not cpv_name.strip():
        print(f"  Warning: Empty name for {cpv_code}, skipping")
        return None

    embedding_text = build_hierarchy_text(
        cpv_code, cpv_name, code_to_description, parent_index
    )
    query = embedding_text

    document_builder = partial(
        build_unspsc_rerank_document,
        unspsc_lookup=unspsc_lookup,
        use_context=USE_UNSPSC_RERANK_CONTEXT,
    )

    try:
        query_embedding = get_embedding(
            embed_client=embed_client,
            model_name="gemini-embedding-001",
            text=query,
            task_type="RETRIEVAL_QUERY",
            embed_dims=3072,
        )

        has_quantization = quantization_cache.get(unspsc_collection, False)
        matches = perform_semantic_search(
            qdrant_client=qdrant_client,
            collection_name=unspsc_collection,
            query_embedding=query_embedding,
            top_k=SEMANTIC_RETRIEVE_LIMIT,
            has_quantization=has_quantization,
        )

        if zclient and matches:
            reranked_matches = rerank_with_zeroentropy(
                zclient=zclient,
                query=query,
                candidates=matches,
                top_k=OUTPUT_TOP_K,
                rerank_top_n=RERANK_CANDIDATE_LIMIT,
                document_builder=document_builder,
            )
            # Multiply scores by 100 to match expected scale
            for match in reranked_matches:
                match["zeroentropy_relevance_score"] = round(
                    match.get("zeroentropy_relevance_score", 0) * 100, 4
                )
        else:
            reranked_matches = [
                {**match, "zeroentropy_relevance_score": 0.0}
                for match in matches[:OUTPUT_TOP_K]
            ]

        cpv_level = get_cpv_level(cpv_code)

        row_data = {
            "cpv_code": cpv_code,
            "cpv_name": cpv_name,
            "cpv_level": cpv_level,
            "embedding_text": query,
        }

        for i, match in enumerate(reranked_matches):
            payload = match.get("payload", {})
            row_data[f"unspsc_match_{i + 1}_code"] = payload.get("original_id", "")
            row_data[f"unspsc_match_{i + 1}_title"] = payload.get("class_name", "")
            row_data[f"unspsc_match_{i + 1}_type"] = payload.get("id_level", "")
            row_data[f"unspsc_match_{i + 1}_score"] = round(match.get("score", 0), 4)
            row_data[f"unspsc_match_{i + 1}_zeroentropy_score"] = round(
                match.get("zeroentropy_relevance_score", 0), 4
            )

        for i in range(len(reranked_matches), OUTPUT_TOP_K):
            row_data[f"unspsc_match_{i + 1}_code"] = ""
            row_data[f"unspsc_match_{i + 1}_title"] = ""
            row_data[f"unspsc_match_{i + 1}_type"] = ""
            row_data[f"unspsc_match_{i + 1}_score"] = ""
            row_data[f"unspsc_match_{i + 1}_zeroentropy_score"] = ""

        return row_data

    except Exception as e:
        logging.getLogger(__name__).warning(
            f"Error classifying {cpv_code}: {e}", exc_info=True
        )
        return None


def main():
    """Main entry point."""
    # Load environment variables from project root
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    print("=" * 70)
    print("CPV to UNSPSC Mapping Generator")
    print("=" * 70)

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
    print(
        f"Gemini API Key: {'*' * 10}{(gemini_key[-4:] if len(gemini_key) >= 4 else gemini_key)}"
    )

    print("\nInitializing clients...")
    embed_client = genai.Client(api_key=gemini_key)
    qdrant_client = QdrantClient(
        url=qdrant_url,
        port=443,
        https=True,
        api_key=qdrant_key,
        timeout=60,
    )

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

    quantization_cache: Dict[str, bool] = {}

    unspsc_collection = "UNSPSC_UNv260801-1-eng_new001-3072_v1"

    try:
        collection_info = qdrant_client.get_collection(unspsc_collection)
        has_quantization = collection_info.config.quantization_config is not None
        quantization_cache[unspsc_collection] = has_quantization
        print(f"UNSPSC collection quantization: {has_quantization}")
    except Exception as e:
        print(f"Warning: Could not check quantization config: {e}")

    df, code_to_description, parent_index = load_cpv_data()

    unspsc_lookup = None
    if zclient and USE_UNSPSC_RERANK_CONTEXT:
        unspsc_lookup = load_unspsc_data()

    processed_codes = load_existing_results(OUTPUT_FILE)

    df_to_process = df[~df["CODE"].astype(str).isin(processed_codes)].copy()

    if len(df_to_process) == 0:
        print("\nAll records already processed!")
        print(f"Output file: {OUTPUT_FILE}")
        return

    print(f"\nProcessing {len(df_to_process)} remaining records...")
    print(f"Output will be saved to: {OUTPUT_FILE}")
    print(f"Progress saved every {ROWS_PER_BATCH} rows\n")

    batch_results: List[Dict[str, Any]] = []
    total_processed = len(processed_codes)

    success_count = 0
    try:
        for idx, (_, row) in enumerate(df_to_process.iterrows(), 1):
            cpv_code = str(row["CODE"])
            query_preview = str(row["EN"])[:80]

            print("=" * 70)
            print(f"[{total_processed + idx}/{len(df)}] {cpv_code}: {query_preview}...")

            result = classify_single(
                embed_client,
                qdrant_client,
                zclient,
                row,
                unspsc_collection,
                quantization_cache,
                code_to_description,
                parent_index,
                unspsc_lookup,
            )

            if result:
                batch_results.append(result)
                success_count += 1
                print(
                    f"  -> UNSPSC: {result.get('unspsc_match_1_code', 'N/A')} ({result.get('unspsc_match_1_title', 'N/A')[:50]}...)"
                )

            if len(batch_results) >= ROWS_PER_BATCH:
                save_results(batch_results, OUTPUT_FILE)
                batch_results = []
                print(
                    f"\n*** Progress saved! {total_processed + idx} records completed ***\n"
                )
    finally:
        if batch_results:
            save_results(batch_results, OUTPUT_FILE)
            print(
                f"\n*** Final progress saved! {total_processed + success_count} records completed ***\n"
            )

    qdrant_client.close()

    print("\n" + "=" * 70)
    print("Mapping complete!")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total records: {len(df)}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
