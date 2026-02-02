"""Literature positioning service for situating approaches."""

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.db.neo4j import Neo4jClient
from src.db.postgres import PostgresClient
from src.db.qdrant import QdrantVectorStore
from src.llm.client import BaseLLMClient, get_llm_client
from src.llm.prompts import PromptTemplates

logger = structlog.get_logger()


@dataclass
class SimilarPaper:
    """A semantically similar paper."""

    arxiv_id: str
    title: str
    year: int
    similarity_score: float
    abstract_excerpt: str


@dataclass
class MethodLineage:
    """Lineage of a method through papers."""

    method_name: str
    origin_paper: str | None
    evolution_papers: list[str] = field(default_factory=list)


@dataclass
class PositioningResult:
    """Result of literature positioning."""

    most_similar_papers: list[SimilarPaper]
    method_lineage: list[MethodLineage]
    differentiation_points: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "most_similar_papers": [
                {
                    "arxiv_id": p.arxiv_id,
                    "title": p.title,
                    "year": p.year,
                    "similarity_score": p.similarity_score,
                    "abstract_excerpt": p.abstract_excerpt,
                }
                for p in self.most_similar_papers
            ],
            "method_lineage": [
                {
                    "method_name": m.method_name,
                    "origin_paper": m.origin_paper,
                    "evolution_papers": m.evolution_papers,
                }
                for m in self.method_lineage
            ],
            "differentiation_points": self.differentiation_points,
        }


@dataclass
class RelatedWorkSection:
    """A section in a related work outline."""

    title: str
    theme: str
    papers_to_cite: list[str]
    transition: str


@dataclass
class RelatedWorkOutline:
    """Structured related work outline."""

    sections: list[RelatedWorkSection]
    positioning_summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sections": [
                {
                    "title": s.title,
                    "theme": s.theme,
                    "papers_to_cite": s.papers_to_cite,
                    "transition": s.transition,
                }
                for s in self.sections
            ],
            "positioning_summary": self.positioning_summary,
        }


class LiteraturePositionerService:
    """Service for positioning approaches within existing literature."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        postgres_client: PostgresClient,
        qdrant_client: QdrantVectorStore,
        llm_client: BaseLLMClient | None = None,
        embedding_model=None,
    ):
        """Initialize literature positioner.

        Args:
            neo4j_client: Neo4j client for graph queries.
            postgres_client: PostgreSQL client for paper data.
            qdrant_client: Qdrant client for semantic search.
            llm_client: Optional LLM client.
            embedding_model: Sentence transformer model.
        """
        self.neo4j = neo4j_client
        self.postgres = postgres_client
        self.qdrant = qdrant_client
        self._llm_client = llm_client
        self._embedding_model = embedding_model

    @property
    def llm_client(self) -> BaseLLMClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedding_model

    def _compute_embedding(self, text: str) -> list[float]:
        """Compute embedding for text."""
        return self.embedding_model.encode(text).tolist()

    async def position_approach(
        self,
        approach_description: str,
        methods: list[str],
        top_k: int = 10,
    ) -> PositioningResult:
        """Find how an approach relates to existing work.

        Args:
            approach_description: Description of the proposed approach.
            methods: List of methods used in the approach.
            top_k: Number of similar papers to find.

        Returns:
            Positioning result with similar papers and lineage.
        """
        logger.info(
            "Positioning approach",
            methods=methods,
            description_length=len(approach_description),
        )

        # Find semantically similar papers
        query_embedding = self._compute_embedding(approach_description)
        semantic_results = self.qdrant.search_papers_by_abstract(
            query_vector=query_embedding,
            top_k=top_k,
        )

        similar_papers = []
        for result in semantic_results:
            arxiv_id = result["payload"].get("arxiv_id")
            if arxiv_id:
                paper = await self.postgres.get_paper_by_arxiv_id(arxiv_id)
                if paper:
                    similar_papers.append(
                        SimilarPaper(
                            arxiv_id=arxiv_id,
                            title=paper["title"],
                            year=paper.get("year", 0),
                            similarity_score=result["score"],
                            abstract_excerpt=paper.get("abstract", "")[:300],
                        )
                    )

        # Find method lineage
        method_lineage = []
        for method in methods:
            lineage = await self._get_method_lineage(method)
            if lineage:
                method_lineage.append(lineage)

        # Generate differentiation points using LLM
        differentiation_points = await self._generate_differentiation(
            approach_description,
            similar_papers[:5],
        )

        result = PositioningResult(
            most_similar_papers=similar_papers,
            method_lineage=method_lineage,
            differentiation_points=differentiation_points,
        )

        logger.info(
            "Positioning complete",
            similar_count=len(similar_papers),
            lineage_count=len(method_lineage),
        )

        return result

    async def _get_method_lineage(self, method: str) -> MethodLineage | None:
        """Get the lineage of a method through papers.

        Args:
            method: Method name.

        Returns:
            Method lineage or None.
        """
        # Query for papers that propose or use this method
        query = """
            MATCH (p:Paper)-[r:PROPOSES]->(m:Method)
            WHERE m.name =~ $method_pattern
            RETURN p.arxiv_id as arxiv_id, p.year as year
            ORDER BY p.year ASC
            LIMIT 10
        """

        results = await self.neo4j.run_query(
            query,
            {"method_pattern": f"(?i).*{method}.*"},
        )

        if not results:
            return None

        papers = [r["arxiv_id"] for r in results]
        origin = papers[0] if papers else None

        return MethodLineage(
            method_name=method,
            origin_paper=origin,
            evolution_papers=papers[1:] if len(papers) > 1 else [],
        )

    async def _generate_differentiation(
        self,
        approach_description: str,
        similar_papers: list[SimilarPaper],
    ) -> list[str]:
        """Generate differentiation points using LLM.

        Args:
            approach_description: Description of the approach.
            similar_papers: Most similar papers.

        Returns:
            List of differentiation points.
        """
        if not similar_papers:
            return ["Novel approach with no closely related work found."]

        similar_summaries = "\n".join([
            f"- {p.title}: {p.abstract_excerpt}"
            for p in similar_papers[:5]
        ])

        prompt = f"""Given this proposed approach:
{approach_description}

And these similar existing papers:
{similar_summaries}

List 3-5 key ways this approach differs from or improves upon the existing work.
Format as a numbered list.
"""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                system="You are an expert at analyzing research contributions.",
                temperature=0.5,
            )

            # Parse numbered list
            points = []
            for line in response.split("\n"):
                line = line.strip()
                if re.match(r"^\d+[\.\)]\s*", line):
                    point = re.sub(r"^\d+[\.\)]\s*", "", line)
                    if point:
                        points.append(point)

            return points if points else [response[:200]]

        except Exception as e:
            logger.warning("Failed to generate differentiation", error=str(e))
            return ["Unable to generate differentiation points."]

    async def generate_related_work_outline(
        self,
        approach_description: str,
        max_citations: int = 20,
    ) -> RelatedWorkOutline:
        """Generate a structured related work outline with citations.

        Args:
            approach_description: Description of the approach.
            max_citations: Maximum papers to cite.

        Returns:
            Related work outline.
        """
        logger.info("Generating related work outline")

        # Find relevant papers
        query_embedding = self._compute_embedding(approach_description)
        results = self.qdrant.search_papers_by_abstract(
            query_vector=query_embedding,
            top_k=max_citations,
        )

        # Get paper details
        available_papers = []
        for result in results:
            arxiv_id = result["payload"].get("arxiv_id")
            if arxiv_id:
                paper = await self.postgres.get_paper_by_arxiv_id(arxiv_id)
                if paper:
                    available_papers.append({
                        "arxiv_id": arxiv_id,
                        "title": paper["title"],
                        "year": paper.get("year"),
                        "abstract": paper.get("abstract", "")[:200],
                    })

        # Generate outline using LLM
        papers_text = "\n".join([
            f"[{p['arxiv_id']}] {p['title']} ({p['year']}): {p['abstract']}"
            for p in available_papers
        ])

        prompt = PromptTemplates.RELATED_WORK_OUTLINE.format(
            approach_description=approach_description,
            available_papers=papers_text,
        )

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                system="You are an expert at structuring academic literature reviews.",
                temperature=0.5,
            )

            return self._parse_related_work_response(response)

        except Exception as e:
            logger.warning("Failed to generate outline", error=str(e))
            return RelatedWorkOutline(
                sections=[],
                positioning_summary="Unable to generate outline.",
            )

    def _parse_related_work_response(self, response: str) -> RelatedWorkOutline:
        """Parse LLM response into related work outline.

        Args:
            response: LLM response text.

        Returns:
            Parsed outline.
        """
        sections = []

        # Find section patterns
        section_pattern = r"#### 2\.\d+ (.+?)\n(.*?)(?=#### 2\.\d+|### Summary|\Z)"
        matches = re.findall(section_pattern, response, re.DOTALL)

        for title, content in matches:
            theme_match = re.search(r"Main theme:\s*(.+)", content)
            papers_match = re.search(r"Key papers to cite:\s*(.+)", content)
            transition_match = re.search(r"Transition[^:]*:\s*(.+)", content)

            section = RelatedWorkSection(
                title=title.strip(),
                theme=theme_match.group(1).strip() if theme_match else "",
                papers_to_cite=(
                    papers_match.group(1).strip().split(",")
                    if papers_match else []
                ),
                transition=transition_match.group(1).strip() if transition_match else "",
            )
            sections.append(section)

        # Find summary
        summary_match = re.search(r"### Summary\n(.*)", response, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else ""

        return RelatedWorkOutline(
            sections=sections,
            positioning_summary=summary,
        )
