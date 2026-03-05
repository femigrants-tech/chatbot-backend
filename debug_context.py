#!/usr/bin/env python3
"""Debug why context() returns 0 results"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
assistant = pc.assistant.Assistant(assistant_name=os.getenv('PINECONE_ASSISTANT_NAME', 'femigrants-assistant'))

# Try different variations
queries = [
    "Who founded Femigrants foundation?",
    "Who founded Femigrants foundation",
    "Femigrants foundation founder",
    "founder of Femigrants"
]

for query in queries:
    print("="*80)
    print(f"Query: {query}")
    print("="*80)
    
    try:
        result = assistant.context(query=query)
        print(f"Result type: {type(result)}")
        print(f"Is dict: {isinstance(result, dict)}")
        
        if isinstance(result, dict):
            print(f"Has 'snippets' key: {'snippets' in result}")
            snippets_raw = result.get('snippets', [])
            print(f"Snippets type: {type(snippets_raw)}")
            print(f"Snippets length: {len(snippets_raw) if hasattr(snippets_raw, '__len__') else 'no __len__'}")
            
            # Convert to list if needed
            snippets = list(snippets_raw) if snippets_raw else []
            print(f"After list conversion: {len(snippets)}")
            
            if snippets:
                print(f"\nFirst snippet type: {type(snippets[0])}")
                print(f"First snippet score: {snippets[0].get('score') if isinstance(snippets[0], dict) else 'N/A'}")
                content = snippets[0].get('content', '') if isinstance(snippets[0], dict) else str(snippets[0])
                print(f"Content preview: {content[:200]}...")
            else:
                print("No snippets after conversion")
        else:
            print(f"Result is not a dict: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()

