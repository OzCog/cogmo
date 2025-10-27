#!/usr/bin/env python3
"""
Validation script for AtomSpace REST API
Demonstrates all endpoints and verifies functionality
"""

import requests
import json
import time
import sys
from threading import Thread
import subprocess

def start_server():
    """Start the API server in background"""
    import simple_restapi
    simple_restapi.initialize_test_data()
    simple_restapi.app.run(host='127.0.0.1', port=5000, debug=False)

def test_api():
    """Test all API endpoints"""
    base_url = "http://127.0.0.1:5000"
    
    # Wait for server to start
    time.sleep(2)
    
    tests_passed = 0
    tests_total = 0
    
    def test_endpoint(description, method, url, data=None, expected_status=200):
        nonlocal tests_passed, tests_total
        tests_total += 1
        
        try:
            if method == 'GET':
                response = requests.get(url, timeout=5)
            elif method == 'POST':
                response = requests.post(url, json=data, timeout=5)
            elif method == 'DELETE':
                response = requests.delete(url, timeout=5)
            
            if response.status_code == expected_status:
                print(f"✅ {description}")
                tests_passed += 1
                return response.json()
            else:
                print(f"❌ {description} - Status: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ {description} - Error: {e}")
            return None
    
    print("🧪 Testing AtomSpace REST API Endpoints")
    print("=" * 50)
    
    # Test 1: Root endpoint
    result = test_endpoint(
        "GET / - API info",
        'GET', f"{base_url}/"
    )
    
    # Test 2: Get all atoms
    result = test_endpoint(
        "GET /api/v1.1/atoms - Get all atoms",
        'GET', f"{base_url}/api/v1.1/atoms"
    )
    
    if result:
        print(f"   Found {result['result']['total']} atoms")
    
    # Test 3: Get atom types
    result = test_endpoint(
        "GET /api/v1.1/types - Get atom types",
        'GET', f"{base_url}/api/v1.1/types"
    )
    
    if result:
        print(f"   Available types: {len(result['types'])}")
    
    # Test 4: Get statistics
    result = test_endpoint(
        "GET /api/v1.1/stats - Get statistics",
        'GET', f"{base_url}/api/v1.1/stats"
    )
    
    if result:
        print(f"   Total atoms: {result['total_atoms']}")
    
    # Test 5: Validate atomspace
    result = test_endpoint(
        "GET /api/v1.1/validate - Validate atomspace",
        'GET', f"{base_url}/api/v1.1/validate"
    )
    
    # Test 6: Create new atom
    new_atom_data = {
        "type": "ConceptNode",
        "name": "test_concept",
        "truthvalue": {
            "details": {
                "strength": 0.9,
                "confidence": 0.8
            }
        }
    }
    
    result = test_endpoint(
        "POST /api/v1.1/atoms - Create new atom",
        'POST', f"{base_url}/api/v1.1/atoms",
        data=new_atom_data,
        expected_status=201
    )
    
    created_atom_id = None
    if result and 'result' in result and 'atom' in result['result']:
        created_atom_id = result['result']['atom']['id']
        print(f"   Created atom ID: {created_atom_id}")
    
    # Test 7: Filter atoms by type
    result = test_endpoint(
        "GET /api/v1.1/atoms?type=ConceptNode - Filter by type",
        'GET', f"{base_url}/api/v1.1/atoms?type=ConceptNode"
    )
    
    if result:
        print(f"   ConceptNodes found: {result['result']['total']}")
    
    # Test 8: Filter atoms by name
    result = test_endpoint(
        "GET /api/v1.1/atoms?name=test_concept - Filter by name",
        'GET', f"{base_url}/api/v1.1/atoms?name=test_concept"
    )
    
    if result:
        print(f"   Atoms named 'test_concept': {result['result']['total']}")
    
    # Test 9: Delete atom (use the one we just created)
    if created_atom_id:
        result = test_endpoint(
            f"DELETE /api/v1.1/atoms - Delete created atom",
            'DELETE', f"{base_url}/api/v1.1/atoms/{created_atom_id}"
        )
    
    print("=" * 50)
    print(f"🎯 Test Results: {tests_passed}/{tests_total} passed")
    
    if tests_passed == tests_total:
        print("🎉 All tests passed! AtomSpace REST API is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == '__main__':
    print("🚀 Starting AtomSpace REST API validation...")
    
    # Start server in background thread
    server_thread = Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Run tests
    success = test_api()
    
    sys.exit(0 if success else 1)