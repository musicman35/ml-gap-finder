"""Gap detection service for finding underexplored method combinations."""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.db.neo4j import Neo4jClient
from src.db.redis import RedisCache

logger = structlog.get_logger()


def _escape_regex(text: str) -> str:
    """Escape regex metacharacters for safe use in Neo4j regex patterns."""
    return re.escape(text)


@dataclass
class MethodInfo:
    """Information about a method."""

    method_id: str
    name: str
    method_type: str
    paper_count: int


@dataclass
class GapResult:
    """Result of gap detection."""

    gap_id: str
    is_gap: bool
    method_a: MethodInfo
    method_b: MethodInfo
    task: str
    gap_score: float
    method_a_paper_count: int
    method_b_paper_count: int
    combination_paper_count: int
    evidence_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gap_id": self.gap_id,
            "is_gap": self.is_gap,
            "method_a": {
                "method_id": self.method_a.method_id,
                "name": self.method_a.name,
                "type": self.method_a.method_type,
                "paper_count": self.method_a.paper_count,
            },
            "method_b": {
                "method_id": self.method_b.method_id,
                "name": self.method_b.name,
                "type": self.method_b.method_type,
                "paper_count": self.method_b.paper_count,
            },
            "task": self.task,
            "gap_score": self.gap_score,
            "method_a_paper_count": self.method_a_paper_count,
            "method_b_paper_count": self.method_b_paper_count,
            "combination_paper_count": self.combination_paper_count,
            "evidence_summary": self.evidence_summary,
        }


class GapDetectorService:
    """Service for detecting research gaps in method combinations."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        cache: RedisCache | None = None,
    ):
        """Initialize gap detector.

        Args:
            neo4j_client: Neo4j database client.
            cache: Optional Redis cache.
        """
        self.neo4j = neo4j_client
        self.cache = cache

    def _generate_gap_id(
        self,
        method_a: str,
        method_b: str,
        task: str,
    ) -> str:
        """Generate unique gap ID.

        Args:
            method_a: First method ID.
            method_b: Second method ID.
            task: Task name.

        Returns:
            Unique gap identifier.
        """
        # Ensure consistent ordering
        methods = sorted([method_a, method_b])
        key = f"{methods[0]}:{methods[1]}:{task}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    async def find_gaps(
        self,
        method_a: str,
        method_b: str,
        task: str,
        min_individual_papers: int = 5,
        max_combination_papers: int = 2,
    ) -> GapResult:
        """Find if a specific method combination is a gap.

        Args:
            method_a: First method name or ID.
            method_b: Second method name or ID.
            task: Task name.
            min_individual_papers: Minimum papers using each method individually.
            max_combination_papers: Maximum papers combining both methods.

        Returns:
            Gap detection result.
        """
        gap_id = self._generate_gap_id(method_a, method_b, task)

        # Check cache first
        if self.cache:
            cached = await self.cache.get_gap_result(gap_id)
            if cached:
                logger.debug("Gap result from cache", gap_id=gap_id)
                return GapResult(**cached)

        logger.info(
            "Finding gap",
            method_a=method_a,
            method_b=method_b,
            task=task,
        )

        # Query for individual method paper counts
        query_a = """
            MATCH (m:Method)
            WHERE m.name =~ $method_pattern OR m.method_id = $method_id
            OPTIONAL MATCH (p:Paper)-[:USES|PROPOSES]->(m)
            OPTIONAL MATCH (p)-[:EVALUATES_ON]->(:Dataset)-[:BENCHMARK_FOR]->(t:Task)
            WHERE t.name =~ $task_pattern
            RETURN m.method_id as method_id, m.name as name, m.type as type,
                   count(DISTINCT p) as paper_count
        """

        # Get method A info
        escaped_a = _escape_regex(method_a)
        escaped_b = _escape_regex(method_b)
        escaped_task = _escape_regex(task)

        result_a = await self.neo4j.run_query(
            query_a,
            {
                "method_pattern": f"(?i).*{escaped_a}.*",
                "method_id": method_a,
                "task_pattern": f"(?i).*{escaped_task}.*",
            },
        )

        # Get method B info
        result_b = await self.neo4j.run_query(
            query_a,
            {
                "method_pattern": f"(?i).*{escaped_b}.*",
                "method_id": method_b,
                "task_pattern": f"(?i).*{escaped_task}.*",
            },
        )

        if not result_a or not result_b:
            # Methods not found, return empty result
            return GapResult(
                gap_id=gap_id,
                is_gap=False,
                method_a=MethodInfo(method_a, method_a, "unknown", 0),
                method_b=MethodInfo(method_b, method_b, "unknown", 0),
                task=task,
                gap_score=0.0,
                method_a_paper_count=0,
                method_b_paper_count=0,
                combination_paper_count=0,
            )

        method_a_info = result_a[0] if result_a else {}
        method_b_info = result_b[0] if result_b else {}

        # Query for combination papers
        combination_query = """
            MATCH (p:Paper)-[:USES|PROPOSES]->(m1:Method)
            MATCH (p)-[:USES|PROPOSES]->(m2:Method)
            WHERE (m1.name =~ $method_a_pattern OR m1.method_id = $method_a_id)
              AND (m2.name =~ $method_b_pattern OR m2.method_id = $method_b_id)
              AND m1 <> m2
            OPTIONAL MATCH (p)-[:EVALUATES_ON]->(:Dataset)-[:BENCHMARK_FOR]->(t:Task)
            WHERE t.name =~ $task_pattern
            RETURN count(DISTINCT p) as combined_papers
        """

        combination_result = await self.neo4j.run_query(
            combination_query,
            {
                "method_a_pattern": f"(?i).*{escaped_a}.*",
                "method_a_id": method_a,
                "method_b_pattern": f"(?i).*{escaped_b}.*",
                "method_b_id": method_b,
                "task_pattern": f"(?i).*{escaped_task}.*",
            },
        )

        papers_a = method_a_info.get("paper_count", 0)
        papers_b = method_b_info.get("paper_count", 0)
        combined_papers = (
            combination_result[0]["combined_papers"] if combination_result else 0
        )

        # Calculate gap score
        gap_score = 0.0
        is_gap = False
        if papers_a >= min_individual_papers and papers_b >= min_individual_papers:
            if combined_papers <= max_combination_papers:
                gap_score = (papers_a * papers_b) / (combined_papers + 1.0)
                is_gap = True

        result = GapResult(
            gap_id=gap_id,
            is_gap=is_gap,
            method_a=MethodInfo(
                method_id=method_a_info.get("method_id", method_a),
                name=method_a_info.get("name", method_a),
                method_type=method_a_info.get("type", "unknown"),
                paper_count=papers_a,
            ),
            method_b=MethodInfo(
                method_id=method_b_info.get("method_id", method_b),
                name=method_b_info.get("name", method_b),
                method_type=method_b_info.get("type", "unknown"),
                paper_count=papers_b,
            ),
            task=task,
            gap_score=gap_score,
            method_a_paper_count=papers_a,
            method_b_paper_count=papers_b,
            combination_paper_count=combined_papers,
        )

        # Cache result
        if self.cache:
            await self.cache.cache_gap_result(gap_id, result.to_dict())

        logger.info(
            "Gap detection complete",
            gap_id=gap_id,
            is_gap=is_gap,
            gap_score=gap_score,
        )

        return result

    async def discover_gaps(
        self,
        task: str,
        method_type: str | None = None,
        min_individual_papers: int = 5,
        max_combination_papers: int = 2,
        top_k: int = 10,
    ) -> list[GapResult]:
        """Automatically discover potential gaps for a task.

        Args:
            task: Task name to find gaps for.
            method_type: Optional filter by method type.
            min_individual_papers: Minimum papers for each method.
            max_combination_papers: Maximum combined papers to be a gap.
            top_k: Number of top gaps to return.

        Returns:
            List of gap results sorted by gap score.
        """
        logger.info(
            "Discovering gaps",
            task=task,
            method_type=method_type,
            top_k=top_k,
        )

        # Use the Neo4j query from the spec
        results = await self.neo4j.find_method_gaps(
            task=task,
            min_individual_papers=min_individual_papers,
            max_combination_papers=max_combination_papers,
            top_k=top_k,
        )

        gaps = []
        for row in results:
            gap_id = self._generate_gap_id(
                row["method_1_id"],
                row["method_2_id"],
                row["task"],
            )

            gap = GapResult(
                gap_id=gap_id,
                is_gap=True,
                method_a=MethodInfo(
                    method_id=row["method_1_id"],
                    name=row["method_1"],
                    method_type="unknown",
                    paper_count=row["papers_m1"],
                ),
                method_b=MethodInfo(
                    method_id=row["method_2_id"],
                    name=row["method_2"],
                    method_type="unknown",
                    paper_count=row["papers_m2"],
                ),
                task=row["task"],
                gap_score=row["gap_score"],
                method_a_paper_count=row["papers_m1"],
                method_b_paper_count=row["papers_m2"],
                combination_paper_count=row["combined_papers"],
            )
            gaps.append(gap)

        logger.info("Discovered gaps", count=len(gaps))
        return gaps

    async def get_gap_details(
        self,
        gap_id: str,
    ) -> GapResult | None:
        """Get detailed information about a specific gap.

        Args:
            gap_id: Gap identifier.

        Returns:
            Gap result or None if not found.
        """
        if self.cache:
            cached = await self.cache.get_gap_result(gap_id)
            if cached:
                return GapResult(**cached)
        return None

    async def count_combination_papers(
        self,
        method_a: str,
        method_b: str,
        task: str,
    ) -> int:
        """Count papers that combine two methods for a task.

        Args:
            method_a: First method.
            method_b: Second method.
            task: Task name.

        Returns:
            Number of papers combining both methods.
        """
        query = """
            MATCH (p:Paper)-[:USES|PROPOSES]->(m1:Method)
            MATCH (p)-[:USES|PROPOSES]->(m2:Method)
            WHERE (m1.name =~ $method_a_pattern OR m1.method_id = $method_a_id)
              AND (m2.name =~ $method_b_pattern OR m2.method_id = $method_b_id)
              AND m1 <> m2
            OPTIONAL MATCH (p)-[:EVALUATES_ON]->(:Dataset)-[:BENCHMARK_FOR]->(t:Task)
            WHERE t.name =~ $task_pattern
            RETURN count(DISTINCT p) as count
        """

        escaped_a = _escape_regex(method_a)
        escaped_b = _escape_regex(method_b)
        escaped_task = _escape_regex(task)

        result = await self.neo4j.run_query(
            query,
            {
                "method_a_pattern": f"(?i).*{escaped_a}.*",
                "method_a_id": method_a,
                "method_b_pattern": f"(?i).*{escaped_b}.*",
                "method_b_id": method_b,
                "task_pattern": f"(?i).*{escaped_task}.*",
            },
        )

        return result[0]["count"] if result else 0
