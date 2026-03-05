"""
Minimal test endpoint to verify Vercel Python runtime works.
Access at: /api/test
"""
import sys
import os

def handler(request):
    """Vercel Python handler"""
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": f'{{"status": "ok", "python": "{sys.version}", "platform": "{sys.platform}"}}'
    }
