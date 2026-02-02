"""FastAPI application for ML Gap Finder."""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.api.routes import (
    gaps_router,
    evidence_router,
    hypotheses_router,
    literature_router,
)
from src.api.schemas import HealthResponse

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(
        "Starting ML Gap Finder",
        llm_provider=settings.llm.provider.value,
        environment=settings.environment,
    )
    yield
    logger.info("Shutting down ML Gap Finder")


app = FastAPI(
    title="ML Gap Finder API",
    description="ML Literature Gap Finder & Hypothesis Generator",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(gaps_router)
app.include_router(evidence_router)
app.include_router(hypotheses_router)
app.include_router(literature_router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "name": "ML Gap Finder API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    # Check database connections
    databases = {
        "neo4j": False,
        "postgres": False,
        "qdrant": False,
        "redis": False,
    }

    try:
        from src.db.neo4j import Neo4jClient
        async with Neo4jClient() as neo4j:
            await neo4j.run_query("RETURN 1")
            databases["neo4j"] = True
    except Exception:
        pass

    try:
        from src.db.postgres import PostgresClient
        async with PostgresClient() as postgres:
            await postgres.fetch_one("SELECT 1")
            databases["postgres"] = True
    except Exception:
        pass

    try:
        from src.db.qdrant import QdrantVectorStore
        qdrant = QdrantVectorStore()
        qdrant.client.get_collections()
        databases["qdrant"] = True
    except Exception:
        pass

    try:
        from src.db.redis import RedisCache
        async with RedisCache() as redis:
            await redis.set("health_check", "ok", expire=10)
            databases["redis"] = True
    except Exception:
        pass

    all_healthy = all(databases.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="0.1.0",
        llm_provider=settings.llm.provider.value,
        databases=databases,
    )


@app.get("/api/v1/status", tags=["status"])
async def api_status():
    """API status and configuration."""
    return {
        "llm_provider": settings.llm.provider.value,
        "llm_model": (
            settings.llm.anthropic_model
            if settings.llm.provider.value == "anthropic"
            else settings.llm.ollama_model
        ),
        "environment": settings.environment,
        "endpoints": {
            "gaps": "/api/v1/gaps",
            "evidence": "/api/v1/evidence",
            "hypotheses": "/api/v1/hypotheses",
            "literature": "/api/v1/literature",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
