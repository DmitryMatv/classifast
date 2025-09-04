#!/usr/bin/env python3
"""
Test script to debug Cloudflare header issues with RapidAPI
Usage: python test_cloudflare_headers.py YOUR_RAPIDAPI_KEY
"""

import requests
import json
import sys
import os


def test_cloudflare_headers(api_key):
    """Test the debug-headers endpoint to see what headers reach the server"""

    # Test with your actual RapidAPI endpoint
    # Replace this with your actual RapidAPI URL
    rapidapi_url = "https://classifast.p.rapidapi.com/api/v1/rapid/debug-headers"

    print("🧪 Testing Cloudflare header compatibility...")
    print(f"Testing URL: {rapidapi_url}")
    print("=" * 50)

    # Test 1: Direct call without Cloudflare (if possible)
    try:
        print("\n1. Testing debug-headers endpoint...")
        headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Proxy-Secret": os.getenv("RAPIDAPI_SECRET", ""),
        }

        response = requests.get(rapidapi_url, headers=headers)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("\nReceived headers:")
            for key, value in data["received_headers"].items():
                if "rapid" in key.lower() or "key" in key.lower():
                    print(f"  🔍 {key}: {value}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


def test_standards_endpoint(api_key):
    """Test the standards endpoint to see if authentication works"""

    rapidapi_url = "https://classifast.p.rapidapi.com/api/v1/rapid/standards"

    print("\n2. Testing standards endpoint...")
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Proxy-Secret": os.getenv("RAPIDAPI_SECRET", ""),
    }

    try:
        response = requests.get(rapidapi_url, headers=headers)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("✅ Standards endpoint accessible!")
            print("Available standards:", list(data["standards"].keys()))
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


def test_local_debug():
    """Test local debug endpoint if available"""
    local_url = "http://localhost:8001/api/v1/rapid/debug-headers"

    print("\n3. Testing local debug endpoint...")
    headers = {
        "X-RapidAPI-Key": "test-key",
        "X-RapidAPI-Proxy-Secret": os.getenv("RAPIDAPI_SECRET", ""),
    }

    try:
        response = requests.get(local_url, headers=headers)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("\nLocal headers:")
            for key, value in data["received_headers"].items():
                if "rapid" in key.lower() or "key" in key.lower():
                    print(f"  ✅ {key}: {value}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Local test failed: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_cloudflare_headers.py YOUR_RAPIDAPI_KEY")
        print("Get your API key from RapidAPI dashboard")
        sys.exit(1)

    api_key = sys.argv[1]

    print("Cloudflare Header Debugging Tool")
    print("=" * 40)

    # Test local debug endpoint first
    local_works = test_local_debug()

    # Test actual RapidAPI endpoint
    rapid_works = test_cloudflare_headers(api_key)

    # Test standards endpoint
    standards_works = test_standards_endpoint(api_key)

    print("\n" + "=" * 40)
    print("📊 SUMMARY")
    print("=" * 40)
    print(f"Local debug endpoint: {'✅ PASS' if local_works else '❌ FAIL'}")
    print(f"RapidAPI debug endpoint: {'✅ PASS' if rapid_works else '❌ FAIL'}")
    print(f"RapidAPI standards endpoint: {'✅ PASS' if standards_works else '❌ FAIL'}")

    if not rapid_works:
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check if Cloudflare Transform Rules are preserving custom headers")
        print("2. Set CLOUDFLARE_COMPATIBILITY_MODE=true in your .env")
        print("3. Check Cloudflare Logs for header modification")
        print("4. Verify RAPIDAPI_SECRET matches RapidAPI dashboard")


if __name__ == "__main__":
    main()
