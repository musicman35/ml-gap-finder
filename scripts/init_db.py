#!/usr/bin/env python3
"""Database initialization script for ML Gap Finder.

This script initializes all database schemas:
- PostgreSQL: tables for papers, methods, hypotheses, evaluations
- Neo4j: constraints and indexes for knowledge graph
- Qdrant: vector collections for embeddings
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config.settings import settings
from src.db.neo4j import Neo4jClient
from src.db.postgres import PostgresClient

logger = structlog.get_logger()


# PostgreSQL Schema
POSTGRES_SCHEMA = """
-- Papers table
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
CREATE INDEX IF NOT EXISTS idx_papers_categories ON papers USING GIN(categories);

-- Extracted methods table
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

CREATE INDEX IF NOT EXISTS idx_extracted_methods_paper_id ON extracted_methods(paper_id);
CREATE INDEX IF NOT EXISTS idx_extracted_methods_name ON extracted_methods(method_name);

-- Generated hypotheses table
CREATE TABLE IF NOT EXISTS hypotheses (
    id SERIAL PRIMARY KEY,
    gap_description TEXT NOT NULL,
    hypothesis_text TEXT NOT NULL,
    mechanism TEXT,
    assumptions JSONB,
    evidence_paper_ids INTEGER[],

    -- Tier 1 scores
    citation_accuracy FLOAT,
    gap_verified BOOLEAN,
    temporal_validated BOOLEAN,

    -- Tier 2 scores (LLM-as-judge)
    coherence_score INTEGER CHECK (coherence_score BETWEEN 1 AND 5),
    evidence_relevance_score INTEGER CHECK (evidence_relevance_score BETWEEN 1 AND 5),
    specificity_score INTEGER CHECK (specificity_score BETWEEN 1 AND 5),

    -- Tier 3 (human rating)
    human_rating INTEGER CHECK (human_rating BETWEEN 1 AND 5),
    human_rater_id VARCHAR(50),

    created_at TIMESTAMP DEFAULT NOW(),
    model_version VARCHAR(50)
);

-- Evaluation runs table
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

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_date ON evaluation_runs(run_date);
CREATE INDEX IF NOT EXISTS idx_evaluation_runs_category ON evaluation_runs(test_category);
"""

# Neo4j Schema
NEO4J_SCHEMA = [
    # Constraints
    "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE",
    "CREATE CONSTRAINT method_id IF NOT EXISTS FOR (m:Method) REQUIRE m.method_id IS UNIQUE",
    "CREATE CONSTRAINT dataset_id IF NOT EXISTS FOR (d:Dataset) REQUIRE d.dataset_id IS UNIQUE",
    "CREATE CONSTRAINT task_id IF NOT EXISTS FOR (t:Task) REQUIRE t.task_id IS UNIQUE",
    # Indexes
    "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
    "CREATE INDEX method_type IF NOT EXISTS FOR (m:Method) ON (m.type)",
    "CREATE INDEX task_domain IF NOT EXISTS FOR (t:Task) ON (t.domain)",
]


async def init_postgres():
    """Initialize PostgreSQL schema."""
    logger.info("Initializing PostgreSQL schema...")

    async with PostgresClient() as client:
        await client.execute(POSTGRES_SCHEMA)

    logger.info("PostgreSQL schema initialized successfully")


async def init_neo4j():
    """Initialize Neo4j schema."""
    logger.info("Initializing Neo4j schema...")

    async with Neo4jClient() as client:
        for query in NEO4J_SCHEMA:
            try:
                await client.run_query(query)
                logger.debug(f"Executed: {query[:50]}...")
            except Exception as e:
                # Constraint/index may already exist
                logger.warning(f"Neo4j query warning: {e}")

    logger.info("Neo4j schema initialized successfully")


def init_qdrant():
    """Initialize Qdrant collections."""
    logger.info("Initializing Qdrant collections...")

    client = QdrantClient(url=settings.qdrant_url)

    collections = {
        "paper_abstracts": {
            "size": 384,  # all-MiniLM-L6-v2 dimension
            "distance": Distance.COSINE,
        },
        "method_descriptions": {
            "size": 384,
            "distance": Distance.COSINE,
        },
    }

    for name, config in collections.items():
        try:
            # Check if collection exists
            existing = client.get_collections().collections
            existing_names = [c.name for c in existing]

            if name not in existing_names:
                client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=config["size"],
                        distance=config["distance"],
                    ),
                )
                logger.info(f"Created Qdrant collection: {name}")
            else:
                logger.info(f"Qdrant collection already exists: {name}")
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection {name}: {e}")
            raise

    logger.info("Qdrant collections initialized successfully")


async def main():
    """Run all database initialization."""
    logger.info("Starting database initialization...")

    try:
        # Initialize PostgreSQL
        await init_postgres()

        # Initialize Neo4j
        await init_neo4j()

        # Initialize Qdrant (sync)
        init_qdrant()

        logger.info("All databases initialized successfully!")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
