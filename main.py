"""
Local development entry point.
The actual app lives in api/index.py (used by Vercel).
Run locally with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.index import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
