"""Neo4j client for knowledge graph operations."""

from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from config.settings import settings


class Neo4jClient:
    """Async Neo4j client wrapper for knowledge graph operations."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        """Initialize Neo4j client.

        Args:
            uri: Neo4j bolt URI. Defaults to settings.neo4j_uri.
            user: Neo4j username. Defaults to settings.neo4j_user.
            password: Neo4j password. Defaults to settings.neo4j_password.
        """
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Establish database connection."""
        self._driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
        )

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aenter__(self) -> "Neo4jClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    async def run_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results.

        Args:
            query: Cypher query string.
            parameters: Query parameters.

        Returns:
            List of result records as dictionaries.
        """
        if not self._driver:
            raise RuntimeError("Not connected to database")

        async with self._driver.session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def create_paper_node(self, paper: dict[str, Any]) -> None:
        """Create or update a Paper node.

        Args:
            paper: Paper data dictionary.
        """
        query = """
            MERGE (p:Paper {arxiv_id: $arxiv_id})
            SET p.title = $title,
                p.authors = $authors,
                p.year = $year,
                p.venue = $venue,
                p.categories = $categories,
                p.citation_count = $citation_count,
                p.updated_at = datetime()
        """
        await self.run_query(query, paper)

    async def create_method_node(self, method: dict[str, Any]) -> None:
        """Create or update a Method node.

        Args:
            method: Method data dictionary.
        """
        query = """
            MERGE (m:Method {method_id: $method_id})
            SET m.name = $name,
                m.type = $type,
                m.pwc_id = $pwc_id,
                m.description = $description,
                m.embedding_id = $embedding_id
        """
        await self.run_query(query, method)

    async def create_dataset_node(self, dataset: dict[str, Any]) -> None:
        """Create or update a Dataset node.

        Args:
            dataset: Dataset data dictionary.
        """
        query = """
            MERGE (d:Dataset {dataset_id: $dataset_id})
            SET d.name = $name,
                d.task = $task,
                d.domain = $domain,
                d.pwc_id = $pwc_id,
                d.size = $size
        """
        await self.run_query(query, dataset)

    async def create_task_node(self, task: dict[str, Any]) -> None:
        """Create or update a Task node.

        Args:
            task: Task data dictionary.
        """
        query = """
            MERGE (t:Task {task_id: $task_id})
            SET t.name = $name,
                t.domain = $domain,
                t.metrics = $metrics
        """
        await self.run_query(query, task)

    async def create_proposes_relationship(
        self,
        paper_arxiv_id: str,
        method_id: str,
        novelty_claim: bool = True,
        confidence: float = 1.0,
    ) -> None:
        """Create PROPOSES relationship between Paper and Method."""
        query = """
            MATCH (p:Paper {arxiv_id: $paper_arxiv_id})
            MATCH (m:Method {method_id: $method_id})
            MERGE (p)-[r:PROPOSES]->(m)
            SET r.novelty_claim = $novelty_claim,
                r.confidence = $confidence
        """
        await self.run_query(
            query,
            {
                "paper_arxiv_id": paper_arxiv_id,
                "method_id": method_id,
                "novelty_claim": novelty_claim,
                "confidence": confidence,
            },
        )

    async def create_uses_relationship(
        self,
        paper_arxiv_id: str,
        method_id: str,
        as_baseline: bool = False,
    ) -> None:
        """Create USES relationship between Paper and Method."""
        query = """
            MATCH (p:Paper {arxiv_id: $paper_arxiv_id})
            MATCH (m:Method {method_id: $method_id})
            MERGE (p)-[r:USES]->(m)
            SET r.as_baseline = $as_baseline
        """
        await self.run_query(
            query,
            {
                "paper_arxiv_id": paper_arxiv_id,
                "method_id": method_id,
                "as_baseline": as_baseline,
            },
        )

    async def create_cites_relationship(
        self,
        citing_arxiv_id: str,
        cited_arxiv_id: str,
    ) -> None:
        """Create CITES relationship between Papers."""
        query = """
            MATCH (p1:Paper {arxiv_id: $citing_arxiv_id})
            MATCH (p2:Paper {arxiv_id: $cited_arxiv_id})
            MERGE (p1)-[:CITES]->(p2)
        """
        await self.run_query(
            query,
            {
                "citing_arxiv_id": citing_arxiv_id,
                "cited_arxiv_id": cited_arxiv_id,
            },
        )

    async def find_method_gaps(
        self,
        task: str,
        min_individual_papers: int = 5,
        max_combination_papers: int = 2,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Find method combination gaps for a task.

        Args:
            task: Task name to search for gaps.
            min_individual_papers: Minimum papers using each method individually.
            max_combination_papers: Maximum papers combining both methods.
            top_k: Number of top gaps to return.

        Returns:
            List of gap results with method pairs and scores.
        """
        query = """
            MATCH (m1:Method)<-[:USES|PROPOSES]-(p1:Paper)-[:EVALUATES_ON]->
                  (:Dataset)-[:BENCHMARK_FOR]->(t:Task {name: $task})
            MATCH (m2:Method)<-[:USES|PROPOSES]-(p2:Paper)-[:EVALUATES_ON]->
                  (:Dataset)-[:BENCHMARK_FOR]->(t)
            WHERE m1.method_id < m2.method_id
            WITH m1, m2, t,
                 COUNT(DISTINCT p1) as papers_m1,
                 COUNT(DISTINCT p2) as papers_m2

            OPTIONAL MATCH (pc:Paper)-[:USES|PROPOSES]->(m1),
                          (pc)-[:USES|PROPOSES]->(m2)
            WHERE (pc)-[:EVALUATES_ON]->(:Dataset)-[:BENCHMARK_FOR]->(t)
            WITH m1, m2, t, papers_m1, papers_m2,
                 COUNT(DISTINCT pc) as combined_papers

            WHERE papers_m1 >= $min_individual
              AND papers_m2 >= $min_individual
              AND combined_papers <= $max_combined

            RETURN m1.name as method_1,
                   m1.method_id as method_1_id,
                   m2.name as method_2,
                   m2.method_id as method_2_id,
                   t.name as task,
                   papers_m1,
                   papers_m2,
                   combined_papers,
                   (papers_m1 * papers_m2) / (combined_papers + 1.0) as gap_score
            ORDER BY gap_score DESC
            LIMIT $top_k
        """
        return await self.run_query(
            query,
            {
                "task": task,
                "min_individual": min_individual_papers,
                "max_combined": max_combination_papers,
                "top_k": top_k,
            },
        )

    async def get_method_papers(
        self,
        method_id: str,
        task: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get papers that use or propose a method.

        Args:
            method_id: Method identifier.
            task: Optional task filter.
            limit: Maximum papers to return.

        Returns:
            List of paper records.
        """
        if task:
            query = """
                MATCH (p:Paper)-[:USES|PROPOSES]->(m:Method {method_id: $method_id})
                MATCH (p)-[:EVALUATES_ON]->(:Dataset)-[:BENCHMARK_FOR]->(t:Task {name: $task})
                RETURN p.arxiv_id as arxiv_id,
                       p.title as title,
                       p.year as year,
                       p.citation_count as citation_count
                ORDER BY p.citation_count DESC
                LIMIT $limit
            """
            params = {"method_id": method_id, "task": task, "limit": limit}
        else:
            query = """
                MATCH (p:Paper)-[:USES|PROPOSES]->(m:Method {method_id: $method_id})
                RETURN p.arxiv_id as arxiv_id,
                       p.title as title,
                       p.year as year,
                       p.citation_count as citation_count
                ORDER BY p.citation_count DESC
                LIMIT $limit
            """
            params = {"method_id": method_id, "limit": limit}

        return await self.run_query(query, params)

    async def count_papers(self) -> int:
        """Count total Paper nodes."""
        result = await self.run_query("MATCH (p:Paper) RETURN count(p) as count")
        return result[0]["count"] if result else 0

    async def count_methods(self) -> int:
        """Count total Method nodes."""
        result = await self.run_query("MATCH (m:Method) RETURN count(m) as count")
        return result[0]["count"] if result else 0

    async def count_relationships(self) -> int:
        """Count total relationships."""
        result = await self.run_query("MATCH ()-[r]->() RETURN count(r) as count")
        return result[0]["count"] if result else 0
