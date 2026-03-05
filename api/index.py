"""
Vercel serverless function entry point.
Imports the FastAPI app from main.py so Vercel can serve it.
"""
import sys
import os

# Add the parent directory to sys.path so we can import main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
