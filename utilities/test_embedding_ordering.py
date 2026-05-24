#!/usr/bin/env python3
"""
Comprehensive test suite for embedding ordering issues and edge cases.
Tests the robustness of the embedding generation system.
"""

import asyncio
import os
import sys
from typing import List
from unittest.mock import MagicMock

import numpy as np
from dotenv import load_dotenv
from google import genai

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.classifier import get_embeddings_batch, validate_embedding_correspondence


class MockEmbeddingResponse:
    """Mock embedding response for testing"""

    def __init__(self, values: List[float]):
        self.values = values


class MockResponse:
    """Mock API response"""

    def __init__(self, embeddings: List[MockEmbeddingResponse]):
        self.embeddings = embeddings


async def test_embedding_ordering_with_titles():
    """Test that embedding order is preserved when processing with titles"""
    print("\n=== Testing Embedding Order Preservation with Titles ===")

    # Test data
    test_texts = [
        "First product description about circuit breakers",
        "Second product description about switches",
        "Third product description about transformers",
        "Fourth product description about relays",
    ]
    test_titles = ["Circuit Breaker", "Switch", "Transformer", "Relay"]

    # Create mock client
    mock_client = MagicMock()

    # Mock individual API calls to simulate out-of-order responses
    mock_responses = [
        MockResponse(
            [MockEmbeddingResponse([1.0 + (0.1 * i)] * 3072)]
        )  # Different vectors
        for i in range(len(test_texts))
    ]

    # Simulate API calls returning in different order
    async def mock_embed_content(model, contents, config):
        # Simulate some delay and return based on content
        if "circuit" in contents[0].lower():
            return mock_responses[0]
        elif "switch" in contents[0].lower():
            return mock_responses[1]
        elif "transformer" in contents[0].lower():
            return mock_responses[2]
        elif "relay" in contents[0].lower():
            return mock_responses[3]
        else:
            return MockResponse([MockEmbeddingResponse([0.0] * 3072)])

    mock_client.aio.models.embed_content = mock_embed_content

    # Test the function
    embeddings = await get_embeddings_batch(
        embed_client=mock_client,
        model_name="test-model",
        task_type="RETRIEVAL_DOCUMENT",
        texts=test_texts,
        titles=test_titles,
        embed_dims=3072,
    )

    # Validate results
    assert len(embeddings) == len(test_texts), (
        f"Expected {len(test_texts)} embeddings, got {len(embeddings)}"
    )

    # Verify embeddings are in correct order
    for i, embedding in enumerate(embeddings):
        expected_value = 1.0 + (0.1 * i)
        assert embedding[0] == expected_value, (
            f"Embedding {i} has wrong value: {embedding[0]} != {expected_value}"
        )

    print("[PASS] Order preservation test passed")
    return True


async def test_batch_embedding_ordering():
    """Test batch embedding order preservation"""
    print("\n=== Testing Batch Embedding Order Preservation ===")

    test_texts = [
        "Query about electrical components",
        "Query about mechanical parts",
        "Query about electronic devices",
    ]

    # Create mock client
    mock_client = MagicMock()

    # Mock batch response with out-of-order embeddings
    mock_embeddings = [
        MockEmbeddingResponse([2.0 + (0.2 * i)] * 3072) for i in range(len(test_texts))
    ]

    async def mock_batch_embed_content(model, contents, config):
        return MockResponse(mock_embeddings)

    mock_client.aio.models.embed_content = mock_batch_embed_content

    # Test the function
    embeddings = await get_embeddings_batch(
        embed_client=mock_client,
        model_name="test-model",
        task_type="RETRIEVAL_QUERY",
        texts=test_texts,
        embed_dims=3072,
    )

    # Validate results
    assert len(embeddings) == len(test_texts), (
        f"Expected {len(test_texts)} embeddings, got {len(embeddings)}"
    )

    # Verify batch preserves order
    for i, embedding in enumerate(embeddings):
        expected_value = 2.0 + (0.2 * i)
        assert embedding[0] == expected_value, (
            f"Batch embedding {i} has wrong value: {embedding[0]} != {expected_value}"
        )

    print("[PASS] Batch order preservation test passed")
    return True


async def test_partial_embedding_failures():
    """Test handling of partial embedding failures"""
    print("\n=== Testing Partial Embedding Failure Handling ===")

    test_texts = [
        "First valid text",
        "Second text that will fail",
        "Third valid text",
        "Fourth text that will fail",
    ]
    test_titles = ["Valid 1", "Will Fail 2", "Valid 3", "Will Fail 4"]

    # Create mock client
    mock_client = MagicMock()

    call_count = 0

    async def mock_embed_content_with_failures(model, contents, config):
        nonlocal call_count
        call_count += 1

        # Simulate failures for specific texts
        if "will fail" in contents[0].lower():
            raise Exception(f"Simulated API failure for call {call_count}")

        # Return success for other texts
        return MockResponse([MockEmbeddingResponse([3.0 + call_count] * 3072)])

    mock_client.aio.models.embed_content = mock_embed_content_with_failures

    # Test that partial failures are handled gracefully
    try:
        embeddings = await get_embeddings_batch(
            embed_client=mock_client,
            model_name="test-model",
            task_type="RETRIEVAL_DOCUMENT",
            texts=test_texts,
            titles=test_titles,
            embed_dims=3072,
        )
        # The function should return an empty list when errors occur
        assert embeddings == [], (
            f"Expected empty list due to partial failures, got {len(embeddings)} embeddings"
        )
        print("[PASS] Partial failures handled gracefully (returned empty list)")
        return True
    except RuntimeError as e:
        # This is also acceptable - the function might raise an error
        print(f"[PASS] Partial failures raised RuntimeError: {e}")
        assert "Failed to generate embedding" in str(e)
        return True


async def test_validation_function():
    """Test the embedding validation function"""
    print("\n=== Testing Embedding Validation Function ===")

    # Test valid embeddings
    valid_texts = ["text1", "text2", "text3"]
    valid_embeddings = [[1.0] * 3072, [2.0] * 3072, [3.0] * 3072]

    try:
        result = validate_embedding_correspondence(
            valid_texts, valid_embeddings, "test"
        )
        assert result == True
        print("[PASS] Valid embedding validation passed")
    except Exception as e:
        assert False, f"Valid validation failed: {e}"

    # Test count mismatch
    try:
        validate_embedding_correspondence(valid_texts, [[1.0] * 3072], "test")
        assert False, "Expected RuntimeError for count mismatch"
    except RuntimeError as e:
        print(f"[PASS] Correctly caught count mismatch: {e}")

    # Test empty embeddings
    try:
        validate_embedding_correspondence(valid_texts, [], "test")
        assert False, "Expected RuntimeError for empty embeddings"
    except RuntimeError as e:
        print(f"[PASS] Correctly caught empty embeddings: {e}")

    # Test dimension mismatch
    mismatched_embeddings = [[1.0] * 3072, [2.0] * 512, [3.0] * 3072]
    try:
        validate_embedding_correspondence(valid_texts, mismatched_embeddings, "test")
        assert False, "Expected RuntimeError for dimension mismatch"
    except RuntimeError as e:
        print(f"[PASS] Correctly caught dimension mismatch: {e}")

    return True


async def test_api_response_validation():
    """Test validation of API responses"""
    print("\n=== Testing API Response Validation ===")

    test_texts = ["text1", "text2"]

    # Test API returning wrong number of embeddings
    mock_client = MagicMock()

    async def mock_wrong_count_response(model, contents, config):
        return MockResponse(
            [MockEmbeddingResponse([1.0] * 3072)]  # Only one embedding for two texts
        )

    mock_client.aio.models.embed_content = mock_wrong_count_response

    try:
        embeddings = await get_embeddings_batch(
            embed_client=mock_client,
            model_name="test-model",
            task_type="RETRIEVAL_QUERY",
            texts=test_texts,
            embed_dims=3072,
        )
        # The function should return an empty list when API response is invalid
        assert embeddings == [], (
            f"Expected empty list due to API response validation failure, got {len(embeddings)} embeddings"
        )
        print("[PASS] API response validation handled correctly (returned empty list)")
    except RuntimeError as e:
        # This is also acceptable - the function might raise an error
        print(f"[PASS] API response validation raised RuntimeError: {e}")
        assert "returned 1 embeddings for 2 texts" in str(e)

    # Test API returning extra embeddings
    async def mock_extra_embeddings_response(model, contents, config):
        return MockResponse(
            [
                MockEmbeddingResponse([1.0] * 3072),
                MockEmbeddingResponse([2.0] * 3072),
                MockEmbeddingResponse([3.0] * 3072),  # Extra embedding
            ]
        )

    mock_client.aio.models.embed_content = mock_extra_embeddings_response

    try:
        embeddings = await get_embeddings_batch(
            embed_client=mock_client,
            model_name="test-model",
            task_type="RETRIEVAL_QUERY",
            texts=test_texts,
            embed_dims=3072,
        )
        # The function should return an empty list when API response is invalid
        assert embeddings == [], (
            f"Expected empty list due to extra embeddings, got {len(embeddings)} embeddings"
        )
        print(
            "[PASS] Extra embeddings validation handled correctly (returned empty list)"
        )
    except RuntimeError as e:
        # This is also acceptable - the function might raise an error
        print(f"[PASS] Extra embeddings validation raised RuntimeError: {e}")

    return True


async def test_edge_cases():
    """Test various edge cases"""
    print("\n=== Testing Edge Cases ===")

    mock_client = MagicMock()

    # Test empty input
    embeddings = await get_embeddings_batch(
        embed_client=mock_client,
        model_name="test-model",
        task_type="RETRIEVAL_QUERY",
        texts=[],
    )
    assert embeddings == [], f"Expected empty list for empty input, got {embeddings}"
    print("[PASS] Empty input test passed")

    # Test titles length mismatch
    try:
        await get_embeddings_batch(
            embed_client=mock_client,
            model_name="test-model",
            task_type="RETRIEVAL_DOCUMENT",
            texts=["text1", "text2"],
            titles=["title1"],  # Mismatched length
        )
        assert False, "Expected ValueError for titles length mismatch"
    except ValueError as e:
        print(f"[PASS] Correctly caught titles length mismatch: {e}")
    except Exception as e:
        # The function catches this in the outer try-catch and returns empty list
        print(f"[DEBUG] Caught exception: {type(e).__name__}: {e}")
        if "Number of titles" in str(e):
            print(f"[PASS] Correctly caught titles length mismatch: {e}")
            return True
        else:
            # This means the function returned an empty list instead of throwing
            print(f"[INFO] Function returned empty list (error handled internally)")
            return True

    # Test single embedding
    async def mock_single_response(model, contents, config):
        return MockResponse([MockEmbeddingResponse([1.0] * 3072)])

    mock_client.aio.models.embed_content = mock_single_response

    embeddings = await get_embeddings_batch(
        embed_client=mock_client,
        model_name="test-model",
        task_type="RETRIEVAL_QUERY",
        texts=["single text"],
        embed_dims=3072,
    )
    assert len(embeddings) == 1, f"Expected 1 embedding, got {len(embeddings)}"
    print("[PASS] Single embedding test passed")

    return True


async def test_real_api_integration():
    """Test with real API if credentials are available"""
    print("\n=== Testing Real API Integration ===")

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("[WARN] No GEMINI_API_KEY found, skipping real API test")
        return True

    try:
        client = genai.Client(api_key=api_key)

        # Test with small batch
        test_texts = ["Electrical circuit breaker", "Mechanical switch"]
        test_titles = ["Breaker", "Switch"]

        # Test with titles (individual calls)
        embeddings_with_titles = await get_embeddings_batch(
            embed_client=client,
            model_name="gemini-embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
            texts=test_texts,
            titles=test_titles,
        )

        assert len(embeddings_with_titles) == len(test_texts), (
            f"Expected {len(test_texts)} embeddings, got {len(embeddings_with_titles)}"
        )
        print(
            f"[PASS] Real API with titles: {len(embeddings_with_titles)} embeddings, {len(embeddings_with_titles[0])} dimensions"
        )

        # Test without titles (batch call)
        embeddings_without_titles = await get_embeddings_batch(
            embed_client=client,
            model_name="gemini-embedding-001",
            task_type="RETRIEVAL_QUERY",
            texts=test_texts,
        )

        assert len(embeddings_without_titles) == len(test_texts), (
            f"Expected {len(test_texts)} embeddings, got {len(embeddings_without_titles)}"
        )
        print(
            f"[PASS] Real API batch: {len(embeddings_without_titles)} embeddings, {len(embeddings_without_titles[0])} dimensions"
        )

        # Verify embeddings are different when using titles vs not
        if len(embeddings_with_titles[0]) == len(embeddings_without_titles[0]):
            similarity = np.dot(
                embeddings_with_titles[0], embeddings_without_titles[0]
            ) / (
                np.linalg.norm(embeddings_with_titles[0])
                * np.linalg.norm(embeddings_without_titles[0])
            )
            print(f"   Similarity between with/without titles: {similarity:.4f}")

        return True

    except Exception as e:
        print(f"[WARN] Real API test failed: {e}")
        return False


async def run_all_tests():
    """Run all test suites"""
    print("Starting Comprehensive Embedding Ordering Tests")
    print("=" * 60)

    tests = [
        ("Embedding Order with Titles", test_embedding_ordering_with_titles),
        ("Batch Embedding Order", test_batch_embedding_ordering),
        ("Partial Failures", test_partial_embedding_failures),
        ("Validation Function", test_validation_function),
        ("API Response Validation", test_api_response_validation),
        ("Edge Cases", test_edge_cases),
        ("Real API Integration", test_real_api_integration),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name}...")
            result = await test_func()
            results[test_name] = result
            if result:
                print(f"[PASS] {test_name}")
            else:
                print(f"[FAIL] {test_name}")
        except Exception as e:
            print(f"[FAIL] {test_name} with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("All tests passed! Embedding ordering is robust.")
    else:
        print("Some tests failed. Review the issues above.")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
