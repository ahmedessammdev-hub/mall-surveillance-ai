"""
FastAPI Application Factory.

Creates the FastAPI app with all routers, middleware, and lifespan events.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import alerts, analytics, cameras, events, health, search, tracks
from config import settings

logger = logging.getLogger(__name__)


def create_app(app_state=None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        app_state: Application state object with runtime components.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("FastAPI backend starting")
        yield
        logger.info("FastAPI backend shutting down")

    app = FastAPI(
        title="Mall Surveillance AI",
        description="AI-powered intelligent surveillance system for shopping malls",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store app_state for dependency injection
    app.state.app_state = app_state

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.backend.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(cameras.router, prefix="/api/cameras", tags=["Cameras"])
    app.include_router(events.router, prefix="/api/events", tags=["Events"])
    app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
    app.include_router(tracks.router, prefix="/api/tracks", tags=["Tracks"])
    app.include_router(search.router, prefix="/api/search", tags=["Search"])
    app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
    app.include_router(health.router, prefix="/api/system-health", tags=["System Health"])

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": "Mall Surveillance AI",
            "version": "0.1.0",
            "status": "running",
        }

    return app
