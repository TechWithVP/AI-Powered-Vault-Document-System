"""Aggregates the authenticated v1 resource routers.

Health endpoints are mounted separately in the application factory because
operational probes must not require an API key.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import chat, documents, metrics

api_router = APIRouter()
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(metrics.router)
