#!/usr/bin/env python3
"""
Test script to verify title functionality in the embedding API
"""
import os
import asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv


async def test_title_functionality():
    """Test the title parameter with RETRIEVAL_DOCUMENT task type"""

    # Load environment variables
    load_dotenv()

    # Initialize client
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        print("✓ Client initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize client: {e}")
        return

    # Test data
    test_texts = [
        "This is a product description for a circuit breaker",
        "This is a product description for a switch",
    ]
    test_titles = ["Circuit Breaker Product", "Switch Product"]

    print("\n--- Testing title functionality ---")

    # Test 1: Basic EmbedContentConfig with RETRIEVAL_DOCUMENT
    try:
        print("\n1. Testing basic RETRIEVAL_DOCUMENT config...")
        config = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        response = await client.aio.models.embed_content(
            model="gemini-embedding-001",
            contents=test_texts[:1],
            config=config,
        )
        print(f"✓ Basic RETRIEVAL_DOCUMENT config works")
        print(f"  Embedding dimension: {len(response.embeddings[0].values)}")
    except Exception as e:
        print(f"✗ Basic RETRIEVAL_DOCUMENT failed: {e}")

    # Test 2: EmbedContentConfig with title parameter
    try:
        print("\n2. Testing RETRIEVAL_DOCUMENT with title...")
        config_with_title = types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT", title=test_titles[0]
        )
        response = await client.aio.models.embed_content(
            model="gemini-embedding-001",
            contents=[test_texts[0]],
            config=config_with_title,
        )
        print(f"✓ RETRIEVAL_DOCUMENT with title works")
        print(f"  Embedding dimension: {len(response.embeddings[0].values)}")
    except Exception as e:
        print(f"✗ RETRIEVAL_DOCUMENT with title failed: {e}")

    # Test 3: Compare embeddings with and without title
    try:
        print("\n3. Testing embedding differences with/without title...")

        # Without title
        config_no_title = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        response_no_title = await client.aio.models.embed_content(
            model="gemini-embedding-001",
            contents=[test_texts[0]],
            config=config_no_title,
        )

        # With title
        config_with_title = types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT", title=test_titles[0]
        )
        response_with_title = await client.aio.models.embed_content(
            model="gemini-embedding-001",
            contents=[test_texts[0]],
            config=config_with_title,
        )

        # Compare embeddings
        emb_no_title = response_no_title.embeddings[0].values
        emb_with_title = response_with_title.embeddings[0].values

        # Calculate similarity (dot product)
        similarity = sum(a * b for a, b in zip(emb_no_title, emb_with_title))
        print(f"✓ Embeddings generated successfully")
        print(f"  Similarity between embeddings: {similarity:.4f}")
        print(f"  Embeddings are {'identical' if similarity > 0.999 else 'different'}")

    except Exception as e:
        print(f"✗ Embedding comparison failed: {e}")

    # Test 4: Test our implementation from classifier.py
    try:
        print("\n4. Testing our get_embeddings_batch implementation...")
        from app.classifier import get_embeddings_batch

        # Test with titles
        embeddings_with_titles = await get_embeddings_batch(
            embed_client=client,
            model_name="gemini-embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
            texts=test_texts,
            titles=test_titles,
        )

        # Test without titles
        embeddings_without_titles = await get_embeddings_batch(
            embed_client=client,
            model_name="gemini-embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
            texts=test_texts,
        )

        print(f"✓ get_embeddings_batch works")
        print(f"  With titles: {len(embeddings_with_titles)} embeddings")
        print(f"  Without titles: {len(embeddings_without_titles)} embeddings")

        if embeddings_with_titles and embeddings_without_titles:
            # Compare first embeddings
            similarity = sum(
                a * b
                for a, b in zip(embeddings_with_titles[0], embeddings_without_titles[0])
            )
            print(f"  First embedding similarity: {similarity:.4f}")

    except Exception as e:
        print(f"✗ get_embeddings_batch test failed: {e}")

    print("\n--- Test completed ---")


if __name__ == "__main__":
    asyncio.run(test_title_functionality())
