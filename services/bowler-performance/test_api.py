"""
Simple test script to verify API endpoints
Run this after starting the server to test basic functionality
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_root():
    """Test root endpoint"""
    print("Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_health():
    """Test health check endpoint"""
    print("Testing health check endpoint...")
    response = requests.get(f"{BASE_URL}/api/v1/health/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_docs():
    """Test if docs are accessible"""
    print("Testing API documentation...")
    response = requests.get(f"{BASE_URL}/api/v1/docs")
    print(f"Docs Status: {response.status_code}")
    print(f"Docs URL: {BASE_URL}/api/v1/docs\n")

if __name__ == "__main__":
    print("=" * 50)
    print("Cricket Bowling Analysis API - Test Suite")
    print("=" * 50 + "\n")
    
    try:
        test_root()
        test_health()
        test_docs()
        print("✅ All basic tests passed!")
        print(f"\n📚 Visit {BASE_URL}/api/v1/docs for interactive API documentation")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the server.")
        print("Make sure the server is running with: python run.py")
    except Exception as e:
        print(f"❌ Error: {e}")
