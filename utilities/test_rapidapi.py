#!/usr/bin/env python3
"""
Test script for RapidAPI endpoints
Usage: python test_rapidapi.py
"""

import requests
import json
import sys
import os

# Configuration - adjust these values for your setup
BASE_URL = "http://localhost:8001/api/v1/rapid"
API_KEY = "test-key"  # This would be your RapidAPI key in production
PROXY_SECRET = os.getenv("RAPIDAPI_SECRET")  # Use value from .env for testing


def test_health():
    """Test the health endpoint"""
    print("🩺 Testing /ping endpoint...")
    try:
        headers = {"X-RapidAPI-Key": API_KEY}
        if PROXY_SECRET:
            headers["X-RapidAPI-Proxy-Secret"] = PROXY_SECRET
        response = requests.get(f"{BASE_URL}/ping", headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_standards():
    """Test the standards listing endpoint"""
    print("📋 Testing /standards endpoint...")
    try:
        headers = {"X-RapidAPI-Key": API_KEY}
        if PROXY_SECRET:
            headers["X-RapidAPI-Proxy-Secret"] = PROXY_SECRET
        response = requests.get(f"{BASE_URL}/standards", headers=headers)
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Available standards: {list(data['standards'].keys())}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Standards test failed: {e}")
        return False


def test_classify():
    """Test the classification endpoint"""
    print("🔍 Testing /classify endpoint...")

    test_cases = [
        {"query": "Laptop computer 15 inch screen 8GB RAM", "standard": "unspsc"},
        {"query": "Miniature circuit breaker 20A 3P", "standard": "etim"},
        {"query": "Game development studio", "standard": "naics"},
    ]

    headers = {"X-RapidAPI-Key": API_KEY, "Content-Type": "application/json"}

    if PROXY_SECRET:
        headers["X-RapidAPI-Proxy-Secret"] = PROXY_SECRET

    for case in test_cases:
        print(f"\n--- Testing {case['standard']}: {case['query'][:30]}... ---")

        params = {"query": case["query"], "standard": case["standard"], "top_k": 5}

        try:
            response = requests.get(
                f"{BASE_URL}/classify", params=params, headers=headers
            )

            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"Processing time: {data['processing_time']:.3f}s")
                print(f"Results found: {len(data['results'])}")

                if data["results"]:
                    print("Top results:")
                    for i, result in enumerate(data["results"][:3], 1):
                        print(
                            f"  {i}. {result['code']} - {result['name'][:50]}... (score: {result['score']:.3f})"
                        )
                return True
            else:
                print(f"❌ Error: {response.text}")
                return False

        except Exception as e:
            print(f"❌ Classification failed: {e}")
            return False

    return True


def main():
    """Run all tests"""
    print("🧪 Testing RapidAPI endpoints...\n")

    # Check if server is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        print(
            "❌ Server not running. Start with: uvicorn app.main:app --reload --port 8001"
        )
        sys.exit(1)

    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Standards", test_standards),
        ("Classification", test_classify),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"{name}")
        print("=" * 50)
        result = test_func()
        results.append(result)

    print(f"\n{'='*50}")
    print("📊 SUMMARY")
    print("=" * 50)
    for i, (name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()
