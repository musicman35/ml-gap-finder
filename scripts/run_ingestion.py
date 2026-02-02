#!/usr/bin/env python3
"""Data ingestion runner for ML Gap Finder.

This script orchestrates the full data ingestion pipeline:
1. Harvest papers from arXiv
2. Enrich with Semantic Scholar data
3. Fetch method/dataset info from Papers With Code
4. Extract methods from abstracts
5. Build knowledge graph in Neo4j
6. Generate embeddings for Qdrant

Usage:
    # Full ingestion
    uv run python scripts/run_ingestion.py --mode=full

    # Incremental (recent papers only)
    uv run python scripts/run_ingestion.py --mode=incremental --days=7

    # Sample run (for testing)
    uv run python scripts/run_ingestion.py --mode=sample --limit=100
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from sentence_transformers import SentenceTransformer

from config.settings import settings
from src.db.postgres import PostgresClient
from src.db.neo4j import Neo4jClient
from src.db.qdrant import QdrantVectorStore
from src.db.redis import RedisCache
from src.ingestion.arxiv import ArxivHarvester
from src.ingestion.semantic_scholar import SemanticScholarClient
from src.ingestion.papers_with_code import PapersWithCodeClient
from src.ingestion.method_extractor import MethodExtractor

logger = structlog.get_logger()


class IngestionPipeline:
    """Orchestrates the full data ingestion pipeline."""

    def __init__(
        self,
        postgres: PostgresClient,
        neo4j: Neo4jClient,
        qdrant: QdrantVectorStore,
        redis: RedisCache,
    ):
        """Initialize pipeline with database clients."""
        self.postgres = postgres
        self.neo4j = neo4j
        self.qdrant = qdrant
        self.redis = redis

        # Initialize harvesters
        self.arxiv = ArxivHarvester()
        self.semantic_scholar = SemanticScholarClient(cache=redis)
        self.pwc = PapersWithCodeClient(cache=redis)
        self.method_extractor = MethodExtractor()

        # Embedding model
        self._embedding_model = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model."""
        if self._embedding_model is None:
            logger.info("Loading embedding model...")
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedding_model

    async def run_arxiv_harvest(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        max_papers: int | None = None,
    ) -> int:
        """Stage 1: Harvest papers from arXiv.

        Args:
            start_date: Start of date range.
            end_date: End of date range.
            max_papers: Maximum papers to fetch.

        Returns:
            Number of papers harvested.
        """
        logger.info("Starting arXiv harvest", start_date=start_date, end_date=end_date)

        count = await self.arxiv.harvest_and_store(
            postgres=self.postgres,
            start_date=start_date,
            end_date=end_date,
            max_papers=max_papers,
        )

        logger.info("arXiv harvest complete", papers_stored=count)
        return count

    async def run_semantic_scholar_enrichment(
        self,
        batch_size: int = 100,
    ) -> int:
        """Stage 2: Enrich papers with Semantic Scholar data.

        Args:
            batch_size: Number of papers to process per batch.

        Returns:
            Number of papers enriched.
        """
        logger.info("Starting Semantic Scholar enrichment")

        # Get papers without citation data
        papers = await self.postgres.fetch_all(
            "SELECT arxiv_id FROM papers WHERE citation_count = 0 LIMIT 1000"
        )

        enriched = 0
        for paper in papers:
            arxiv_id = paper["arxiv_id"]

            try:
                s2_paper = await self.semantic_scholar.get_paper_by_arxiv_id(arxiv_id)
                if s2_paper:
                    await self.postgres.update_citation_count(
                        arxiv_id, s2_paper.citation_count
                    )
                    enriched += 1

                    if enriched % 100 == 0:
                        logger.info("Enrichment progress", enriched=enriched)

            except Exception as e:
                logger.warning(
                    "Failed to enrich paper",
                    arxiv_id=arxiv_id,
                    error=str(e),
                )

        logger.info("Semantic Scholar enrichment complete", papers_enriched=enriched)
        return enriched

    async def run_pwc_integration(
        self,
        batch_size: int = 50,
    ) -> int:
        """Stage 3: Fetch method/dataset info from Papers With Code.

        Args:
            batch_size: Number of papers to process per batch.

        Returns:
            Number of papers with PWC data.
        """
        logger.info("Starting Papers With Code integration")

        papers = await self.postgres.fetch_all(
            "SELECT id, arxiv_id FROM papers LIMIT 500"
        )

        integrated = 0
        for paper in papers:
            arxiv_id = paper["arxiv_id"]

            try:
                pwc_paper = await self.pwc.get_paper_by_arxiv_id(arxiv_id)
                if pwc_paper and pwc_paper.methods:
                    # Store PWC methods for validation
                    for method_name in pwc_paper.methods:
                        await self.postgres.insert_extracted_method({
                            "paper_id": paper["id"],
                            "method_name": method_name,
                            "method_type": "unknown",
                            "extraction_confidence": 1.0,
                            "context_snippet": "From Papers With Code",
                        })
                    integrated += 1

            except Exception as e:
                logger.warning(
                    "Failed to get PWC data",
                    arxiv_id=arxiv_id,
                    error=str(e),
                )

        logger.info("PWC integration complete", papers_with_pwc=integrated)
        return integrated

    async def run_method_extraction(
        self,
        use_llm: bool = True,
        batch_size: int = 50,
    ) -> int:
        """Stage 4: Extract methods from paper abstracts.

        Args:
            use_llm: Whether to use LLM for extraction.
            batch_size: Papers per batch.

        Returns:
            Number of methods extracted.
        """
        logger.info("Starting method extraction", use_llm=use_llm)

        papers = await self.postgres.fetch_all(
            """
            SELECT p.id, p.arxiv_id, p.abstract
            FROM papers p
            LEFT JOIN extracted_methods em ON p.id = em.paper_id
            WHERE em.id IS NULL AND p.abstract IS NOT NULL
            LIMIT 200
            """
        )

        total_methods = 0
        for paper in papers:
            if not paper["abstract"]:
                continue

            try:
                methods = await self.method_extractor.extract(
                    abstract=paper["abstract"],
                    use_llm=use_llm,
                )

                for method in methods:
                    await self.postgres.insert_extracted_method({
                        "paper_id": paper["id"],
                        "method_name": method.name,
                        "method_type": method.method_type,
                        "extraction_confidence": method.confidence,
                        "context_snippet": method.context_snippet,
                    })
                    total_methods += 1

            except Exception as e:
                logger.warning(
                    "Failed to extract methods",
                    arxiv_id=paper["arxiv_id"],
                    error=str(e),
                )

        logger.info("Method extraction complete", methods_extracted=total_methods)
        return total_methods

    async def run_graph_construction(self) -> int:
        """Stage 5: Build knowledge graph in Neo4j.

        Returns:
            Number of relationships created.
        """
        logger.info("Starting graph construction")

        # Get papers with methods
        papers = await self.postgres.fetch_all(
            """
            SELECT p.arxiv_id, p.title, p.authors, p.year, p.venue,
                   p.categories, p.citation_count
            FROM papers p
            LIMIT 1000
            """
        )

        relationships = 0

        for paper in papers:
            try:
                # Create paper node
                await self.neo4j.create_paper_node({
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "authors": paper.get("authors") or [],
                    "year": paper.get("year"),
                    "venue": paper.get("venue"),
                    "categories": paper.get("categories") or [],
                    "citation_count": paper.get("citation_count", 0),
                })

                # Get methods for this paper
                methods = await self.postgres.fetch_all(
                    "SELECT * FROM extracted_methods WHERE paper_id = (SELECT id FROM papers WHERE arxiv_id = %s)",
                    (paper["arxiv_id"],),
                )

                for method in methods:
                    method_id = f"method_{method['method_name'].lower().replace(' ', '_')}"

                    # Create method node
                    await self.neo4j.create_method_node({
                        "method_id": method_id,
                        "name": method["method_name"],
                        "type": method.get("method_type", "unknown"),
                        "pwc_id": None,
                        "description": method.get("context_snippet", ""),
                        "embedding_id": None,
                    })

                    # Create relationship
                    await self.neo4j.create_uses_relationship(
                        paper["arxiv_id"],
                        method_id,
                    )
                    relationships += 1

            except Exception as e:
                logger.warning(
                    "Failed to add to graph",
                    arxiv_id=paper["arxiv_id"],
                    error=str(e),
                )

        logger.info("Graph construction complete", relationships=relationships)
        return relationships

    def run_embedding_generation(
        self,
        batch_size: int = 100,
    ) -> int:
        """Stage 6: Generate embeddings for Qdrant.

        Args:
            batch_size: Papers per batch.

        Returns:
            Number of embeddings generated.
        """
        logger.info("Starting embedding generation")

        # This would be async in practice, simplified for now
        # In production, you'd batch process papers from PostgreSQL

        embeddings_count = 0
        # Implementation would fetch abstracts, generate embeddings,
        # and store in Qdrant

        logger.info("Embedding generation complete", embeddings=embeddings_count)
        return embeddings_count

    async def run_full_pipeline(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        max_papers: int | None = None,
        use_llm: bool = True,
    ) -> dict:
        """Run the complete ingestion pipeline.

        Args:
            start_date: Start of date range.
            end_date: End of date range.
            max_papers: Maximum papers to process.
            use_llm: Whether to use LLM for method extraction.

        Returns:
            Pipeline statistics.
        """
        stats = {}

        # Stage 1: arXiv harvest
        stats["arxiv_papers"] = await self.run_arxiv_harvest(
            start_date=start_date,
            end_date=end_date,
            max_papers=max_papers,
        )

        # Stage 2: Semantic Scholar enrichment
        stats["enriched_papers"] = await self.run_semantic_scholar_enrichment()

        # Stage 3: PWC integration
        stats["pwc_papers"] = await self.run_pwc_integration()

        # Stage 4: Method extraction
        stats["extracted_methods"] = await self.run_method_extraction(use_llm=use_llm)

        # Stage 5: Graph construction
        stats["graph_relationships"] = await self.run_graph_construction()

        # Stage 6: Embedding generation
        stats["embeddings"] = self.run_embedding_generation()

        return stats


async def main():
    """Run the ingestion pipeline."""
    parser = argparse.ArgumentParser(description="ML Gap Finder Data Ingestion")
    parser.add_argument(
        "--mode",
        choices=["full", "incremental", "sample"],
        default="sample",
        help="Ingestion mode",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days back for incremental mode",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max papers for sample mode",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM-based method extraction",
    )
    args = parser.parse_args()

    logger.info("Starting ingestion", mode=args.mode)

    # Set date range based on mode
    end_date = datetime.now()
    if args.mode == "full":
        start_date = datetime(2020, 1, 1)
        max_papers = None
    elif args.mode == "incremental":
        start_date = end_date - timedelta(days=args.days)
        max_papers = None
    else:  # sample
        start_date = datetime(2024, 1, 1)
        max_papers = args.limit

    # Initialize database clients
    async with PostgresClient() as postgres:
        async with Neo4jClient() as neo4j:
            async with RedisCache() as redis:
                qdrant = QdrantVectorStore()

                pipeline = IngestionPipeline(
                    postgres=postgres,
                    neo4j=neo4j,
                    qdrant=qdrant,
                    redis=redis,
                )

                stats = await pipeline.run_full_pipeline(
                    start_date=start_date,
                    end_date=end_date,
                    max_papers=max_papers,
                    use_llm=not args.no_llm,
                )

                logger.info("Ingestion complete", stats=stats)


if __name__ == "__main__":
    asyncio.run(main())
