"""
Root FastAPI application entrypoint for FastAPI Cloud & standard ASGI hosts.
Exports `app` instance from `app.main:app`.
"""

from app.main import app

__all__ = ["app"]
