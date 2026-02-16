"""FastAPI application for ML Gap Finder."""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.api.routes import (
    evidence_router,
    gaps_router,
    hypotheses_router,
    literature_router,
)
from src.api.schemas import HealthResponse
from src.db.neo4j import Neo4jClient
from src.db.postgres import PostgresClient
from src.db.qdrant import QdrantVectorStore
from src.db.redis import RedisCache

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(
        "Starting ML Gap Finder",
        llm_provider=settings.llm.provider.value,
        environment=settings.environment,
    )

    # Initialize shared database clients
    neo4j = Neo4jClient()
    postgres = PostgresClient()
    redis = RedisCache()
    qdrant = QdrantVectorStore()

    await neo4j.connect()
    await postgres.connect()
    await redis.connect()

    app.state.neo4j = neo4j
    app.state.postgres = postgres
    app.state.redis = redis
    app.state.qdrant = qdrant

    # Auto-initialize database schemas
    try:
        await _init_postgres_schema(postgres)
        logger.info("PostgreSQL schema initialized")
    except Exception as e:
        logger.warning("PostgreSQL schema init failed", error=str(e))

    yield

    # Cleanup shared connections
    await neo4j.disconnect()
    await postgres.disconnect()
    await redis.disconnect()
    logger.info("Shutting down ML Gap Finder")


async def _init_postgres_schema(postgres: PostgresClient) -> None:
    """Initialize PostgreSQL tables if they don't exist."""
    schema = """
    CREATE TABLE IF NOT EXISTS papers (
        id SERIAL PRIMARY KEY,
        arxiv_id VARCHAR(20) UNIQUE NOT NULL,
        title TEXT NOT NULL,
        abstract TEXT,
        authors JSONB,
        year INTEGER,
        venue VARCHAR(255),
        categories VARCHAR(50)[],
        citation_count INTEGER DEFAULT 0,
        full_text TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
    CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id ON papers(arxiv_id);

    CREATE TABLE IF NOT EXISTS extracted_methods (
        id SERIAL PRIMARY KEY,
        paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
        method_name VARCHAR(255) NOT NULL,
        method_type VARCHAR(50),
        extraction_confidence FLOAT,
        pwc_validated BOOLEAN DEFAULT FALSE,
        context_snippet TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_extracted_methods_paper_id
        ON extracted_methods(paper_id);

    CREATE TABLE IF NOT EXISTS hypotheses (
        id SERIAL PRIMARY KEY,
        gap_description TEXT NOT NULL,
        hypothesis_text TEXT NOT NULL,
        mechanism TEXT,
        assumptions JSONB,
        evidence_paper_ids INTEGER[],
        citation_accuracy FLOAT,
        gap_verified BOOLEAN,
        temporal_validated BOOLEAN,
        coherence_score INTEGER CHECK (coherence_score BETWEEN 1 AND 5),
        evidence_relevance_score INTEGER CHECK (
            evidence_relevance_score BETWEEN 1 AND 5
        ),
        specificity_score INTEGER CHECK (specificity_score BETWEEN 1 AND 5),
        human_rating INTEGER CHECK (human_rating BETWEEN 1 AND 5),
        human_rater_id VARCHAR(50),
        created_at TIMESTAMP DEFAULT NOW(),
        model_version VARCHAR(50)
    );

    CREATE TABLE IF NOT EXISTS evaluation_runs (
        id SERIAL PRIMARY KEY,
        run_date TIMESTAMP DEFAULT NOW(),
        test_category VARCHAR(50),
        metric_name VARCHAR(100),
        metric_value FLOAT,
        sample_size INTEGER,
        model_version VARCHAR(50),
        notes TEXT
    );
    """
    await postgres.execute(schema)


def get_neo4j(request: Request) -> Neo4jClient:
    """FastAPI dependency for shared Neo4j client."""
    return request.app.state.neo4j


def get_postgres(request: Request) -> PostgresClient:
    """FastAPI dependency for shared PostgreSQL client."""
    return request.app.state.postgres


def get_redis(request: Request) -> RedisCache:
    """FastAPI dependency for shared Redis client."""
    return request.app.state.redis


def get_qdrant(request: Request) -> QdrantVectorStore:
    """FastAPI dependency for shared Qdrant client."""
    return request.app.state.qdrant


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
    allow_origins=["*"],
    allow_credentials=False,
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
async def health_check(request: Request) -> HealthResponse:
    """Health check endpoint."""
    databases = {
        "neo4j": False,
        "postgres": False,
        "qdrant": False,
        "redis": False,
    }

    try:
        await request.app.state.neo4j.run_query("RETURN 1")
        databases["neo4j"] = True
    except Exception as e:
        logger.warning("Neo4j health check failed", error=str(e))

    try:
        await request.app.state.postgres.fetch_one("SELECT 1")
        databases["postgres"] = True
    except Exception as e:
        logger.warning("PostgreSQL health check failed", error=str(e))

    try:
        request.app.state.qdrant.client.get_collections()
        databases["qdrant"] = True
    except Exception as e:
        logger.warning("Qdrant health check failed", error=str(e))

    try:
        await request.app.state.redis.set("health_check", "ok", expire=10)
        databases["redis"] = True
    except Exception as e:
        logger.warning("Redis health check failed", error=str(e))

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
