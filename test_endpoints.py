#!/usr/bin/env python3
"""Quick test to verify endpoints and signed URLs"""

import requests
import json

BASE_URL = "http://localhost:8000"

print("="*80)
print("TESTING BACKEND ENDPOINTS")
print("="*80)

# Test 1: Health check
print("\n1. Testing health endpoint...")
try:
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        print("✅ Server is running")
    else:
        print(f"❌ Server health check failed: {response.status_code}")
except Exception as e:
    print(f"❌ Cannot connect to server: {e}")
    print("   Make sure server is running on http://localhost:8000")
    exit(1)

# Test 2: List files to get a file_id
print("\n2. Getting file list...")
response = requests.get(f"{BASE_URL}/files")
if response.status_code == 200:
    data = response.json()
    if data['total'] > 0:
        print(f"✅ Found {data['total']} files")
        file_id = data['files'][0]['id']
        file_name = data['files'][0]['name']
        print(f"   Using file: {file_name}")
        print(f"   File ID: {file_id}")
    else:
        print("❌ No files found. Upload a file first.")
        exit(1)
else:
    print(f"❌ Failed to list files: {response.status_code}")
    exit(1)

# Test 3: Test the view-url endpoint
print("\n3. Testing /files/{file_id}/view-url endpoint...")
try:
    response = requests.get(f"{BASE_URL}/files/{file_id}/view-url")
    if response.status_code == 200:
        data = response.json()
        print("✅ Endpoint is working!")
        print(f"   File: {data['file_name']}")
        print(f"   Has signed URL: {'signed_url' in data}")
        if 'signed_url' in data:
            print(f"   URL length: {len(data['signed_url'])}")
            print(f"   URL preview: {data['signed_url'][:80]}...")
    else:
        print(f"❌ Endpoint failed: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"❌ Error calling endpoint: {e}")

# Test 4: Test chat endpoint
print("\n4. Testing /chat endpoint...")
try:
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": "Who founded Femigrants foundation?", "chat_context": []}
    )
    if response.status_code == 200:
        data = response.json()
        print("✅ Chat endpoint working!")
        print(f"   Response: {data['response'][:100]}...")
        print(f"   Context items: {len(data['context_used'])}")
        
        if data['context_used']:
            context = data['context_used'][0]
            print(f"\n   First context item:")
            print(f"   - Has 'signed_url' key: {'signed_url' in context}")
            print(f"   - Has 'file_id' key: {'file_id' in context}")
            
            if 'signed_url' in context and context['signed_url']:
                print(f"   ✅ Signed URL present in response!")
                print(f"      URL: {context['signed_url'][:80]}...")
            else:
                print(f"   ❌ Signed URL is NULL or missing")
                print(f"      Context keys: {context.keys()}")
                
                # Check in metadata
                if 'metadata' in context and 'signed_url' in context['metadata']:
                    print(f"   ✅ Signed URL found in metadata")
                    print(f"      URL: {context['metadata']['signed_url'][:80]}...")
                else:
                    print(f"   ❌ Signed URL also not in metadata")
    else:
        print(f"❌ Chat endpoint failed: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"❌ Error calling chat endpoint: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nIf the /files/{file_id}/view-url endpoint is not found:")
print("  → Restart your server: python3 main.py")
print("\nIf signed URLs are NULL in chat response:")
print("  → Check server logs for 'Signed URL: Present/None'")
print("  → Pinecone context() might not include signed URLs by default")
print("\nFor frontend integration:")
print("  → Use context.signed_url or context.metadata.signed_url")
print("  → Or call /files/{file_id}/view-url to get fresh URL")

