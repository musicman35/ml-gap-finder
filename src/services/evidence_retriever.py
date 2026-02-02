"""Evidence retrieval service for finding supporting papers."""

from dataclasses import dataclass, field
from typing import Any

import structlog

from src.db.neo4j import Neo4jClient
from src.db.postgres import PostgresClient
from src.db.qdrant import QdrantVectorStore

logger = structlog.get_logger()


@dataclass
class PaperEvidence:
    """Evidence from a single paper."""

    arxiv_id: str
    title: str
    year: int
    citation_count: int
    excerpt: str
    relevance_score: float


@dataclass
class MetricResult:
    """A metric result from a paper."""

    metric_name: str
    value: float
    dataset: str
    is_improvement: bool


@dataclass
class EvidenceBundle:
    """Collection of evidence for a claim."""

    papers: list[PaperEvidence] = field(default_factory=list)
    metrics: dict[str, list[MetricResult]] = field(default_factory=dict)
    confidence: float = 0.0
    claim_support_strength: str = "weak"  # "weak", "moderate", "strong"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "papers": [
                {
                    "arxiv_id": p.arxiv_id,
                    "title": p.title,
                    "year": p.year,
                    "citation_count": p.citation_count,
                    "excerpt": p.excerpt,
                    "relevance_score": p.relevance_score,
                }
                for p in self.papers
            ],
            "metrics": {
                k: [
                    {
                        "metric_name": m.metric_name,
                        "value": m.value,
                        "dataset": m.dataset,
                        "is_improvement": m.is_improvement,
                    }
                    for m in v
                ]
                for k, v in self.metrics.items()
            },
            "confidence": self.confidence,
            "claim_support_strength": self.claim_support_strength,
        }


@dataclass
class CitationValidation:
    """Result of citation validation."""

    is_valid: bool
    cited_paper_exists: bool
    claim_supported: bool
    similarity_score: float
    explanation: str


class EvidenceRetrieverService:
    """Service for retrieving evidence supporting method claims."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        postgres_client: PostgresClient,
        qdrant_client: QdrantVectorStore,
        embedding_model=None,
    ):
        """Initialize evidence retriever.

        Args:
            neo4j_client: Neo4j client for graph queries.
            postgres_client: PostgreSQL client for paper data.
            qdrant_client: Qdrant client for semantic search.
            embedding_model: Sentence transformer model.
        """
        self.neo4j = neo4j_client
        self.postgres = postgres_client
        self.qdrant = qdrant_client
        self._embedding_model = embedding_model

    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedding_model

    def _compute_embedding(self, text: str) -> list[float]:
        """Compute embedding for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        return self.embedding_model.encode(text).tolist()

    async def get_evidence(
        self,
        method: str,
        task: str,
        claim_type: str = "improves",
        max_papers: int = 10,
    ) -> EvidenceBundle:
        """Find papers with evidence for method-task claims.

        Args:
            method: Method name or ID.
            task: Task name.
            claim_type: Type of claim ("improves", "outperforms", "enables").
            max_papers: Maximum papers to retrieve.

        Returns:
            Bundle of evidence supporting the claim.
        """
        logger.info(
            "Getting evidence",
            method=method,
            task=task,
            claim_type=claim_type,
        )

        # Get papers from graph
        papers = await self.neo4j.get_method_papers(
            method_id=method,
            task=task,
            limit=max_papers,
        )

        evidence_papers = []
        for paper_info in papers:
            # Get full paper details from PostgreSQL
            paper = await self.postgres.get_paper_by_arxiv_id(paper_info["arxiv_id"])
            if paper:
                evidence_papers.append(
                    PaperEvidence(
                        arxiv_id=paper["arxiv_id"],
                        title=paper["title"],
                        year=paper.get("year", 0),
                        citation_count=paper.get("citation_count", 0),
                        excerpt=paper.get("abstract", "")[:500],
                        relevance_score=1.0,  # Would be computed from semantic similarity
                    )
                )

        # Also do semantic search for related papers
        query = f"{method} {task} {claim_type}"
        query_embedding = self._compute_embedding(query)

        semantic_results = self.qdrant.search_papers_by_abstract(
            query_vector=query_embedding,
            top_k=max_papers,
        )

        for result in semantic_results:
            arxiv_id = result["payload"].get("arxiv_id")
            if arxiv_id and not any(p.arxiv_id == arxiv_id for p in evidence_papers):
                paper = await self.postgres.get_paper_by_arxiv_id(arxiv_id)
                if paper:
                    evidence_papers.append(
                        PaperEvidence(
                            arxiv_id=paper["arxiv_id"],
                            title=paper["title"],
                            year=paper.get("year", 0),
                            citation_count=paper.get("citation_count", 0),
                            excerpt=paper.get("abstract", "")[:500],
                            relevance_score=result["score"],
                        )
                    )

        # Sort by relevance and limit
        evidence_papers.sort(key=lambda p: p.relevance_score, reverse=True)
        evidence_papers = evidence_papers[:max_papers]

        # Calculate confidence and support strength
        confidence = self._calculate_confidence(evidence_papers)
        support_strength = self._determine_support_strength(confidence, len(evidence_papers))

        bundle = EvidenceBundle(
            papers=evidence_papers,
            confidence=confidence,
            claim_support_strength=support_strength,
        )

        logger.info(
            "Evidence retrieved",
            paper_count=len(evidence_papers),
            confidence=confidence,
            support_strength=support_strength,
        )

        return bundle

    def _calculate_confidence(self, papers: list[PaperEvidence]) -> float:
        """Calculate confidence score based on evidence.

        Args:
            papers: List of evidence papers.

        Returns:
            Confidence score between 0 and 1.
        """
        if not papers:
            return 0.0

        # Factors: number of papers, citation counts, relevance scores
        paper_factor = min(len(papers) / 10.0, 1.0)

        avg_citations = sum(p.citation_count for p in papers) / len(papers)
        citation_factor = min(avg_citations / 100.0, 1.0)

        avg_relevance = sum(p.relevance_score for p in papers) / len(papers)

        confidence = (paper_factor * 0.3 + citation_factor * 0.3 + avg_relevance * 0.4)
        return round(confidence, 3)

    def _determine_support_strength(
        self,
        confidence: float,
        paper_count: int,
    ) -> str:
        """Determine claim support strength.

        Args:
            confidence: Confidence score.
            paper_count: Number of supporting papers.

        Returns:
            Support strength category.
        """
        if confidence >= 0.7 and paper_count >= 5:
            return "strong"
        elif confidence >= 0.4 and paper_count >= 3:
            return "moderate"
        else:
            return "weak"

    async def validate_citation(
        self,
        citing_paper_id: str,
        cited_paper_id: str,
        claimed_contribution: str,
    ) -> CitationValidation:
        """Verify that a citation supports the claimed contribution.

        Args:
            citing_paper_id: arXiv ID of citing paper.
            cited_paper_id: arXiv ID of cited paper.
            claimed_contribution: What the citation claims to support.

        Returns:
            Validation result.
        """
        logger.info(
            "Validating citation",
            citing=citing_paper_id,
            cited=cited_paper_id,
        )

        # Check if cited paper exists
        cited_paper = await self.postgres.get_paper_by_arxiv_id(cited_paper_id)
        if not cited_paper:
            return CitationValidation(
                is_valid=False,
                cited_paper_exists=False,
                claim_supported=False,
                similarity_score=0.0,
                explanation="Cited paper not found in database",
            )

        # Compute semantic similarity between claim and paper abstract
        claim_embedding = self._compute_embedding(claimed_contribution)
        abstract = cited_paper.get("abstract", "")

        if abstract:
            abstract_embedding = self._compute_embedding(abstract)

            # Cosine similarity
            import numpy as np
            similarity = np.dot(claim_embedding, abstract_embedding) / (
                np.linalg.norm(claim_embedding) * np.linalg.norm(abstract_embedding)
            )
        else:
            similarity = 0.0

        # Determine if claim is supported
        claim_supported = similarity > 0.5

        return CitationValidation(
            is_valid=claim_supported,
            cited_paper_exists=True,
            claim_supported=claim_supported,
            similarity_score=float(similarity),
            explanation=(
                f"Semantic similarity: {similarity:.2f}. "
                f"{'Claim appears supported by cited paper.' if claim_supported else 'Claim may not be directly supported.'}"
            ),
        )

    async def get_method_success_evidence(
        self,
        method: str,
        task: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Get evidence of method success (for hypothesis generation).

        Args:
            method: Method name.
            task: Optional task filter.
            top_k: Number of top papers.

        Returns:
            List of evidence summaries.
        """
        papers = await self.neo4j.get_method_papers(
            method_id=method,
            task=task,
            limit=top_k,
        )

        evidence = []
        for paper_info in papers:
            paper = await self.postgres.get_paper_by_arxiv_id(paper_info["arxiv_id"])
            if paper:
                evidence.append({
                    "paper_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "year": paper.get("year"),
                    "citations": paper.get("citation_count", 0),
                    "excerpt": paper.get("abstract", "")[:300],
                })

        return evidence
