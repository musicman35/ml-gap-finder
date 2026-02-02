"""Async PostgreSQL client for ML Gap Finder."""

from contextlib import asynccontextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row

from config.settings import settings


class PostgresClient:
    """Async PostgreSQL client wrapper."""

    def __init__(self, connection_string: str | None = None):
        """Initialize PostgreSQL client.

        Args:
            connection_string: Database URL. Defaults to settings.database_url.
        """
        self.connection_string = connection_string or settings.database_url
        self._conn: psycopg.AsyncConnection | None = None

    async def connect(self) -> None:
        """Establish database connection."""
        self._conn = await psycopg.AsyncConnection.connect(
            self.connection_string,
            row_factory=dict_row,
        )

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> "PostgresClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        if not self._conn:
            raise RuntimeError("Not connected to database")
        async with self._conn.transaction():
            yield

    async def execute(self, query: str, params: tuple | None = None) -> None:
        """Execute a query without returning results.

        Args:
            query: SQL query to execute.
            params: Query parameters.
        """
        if not self._conn:
            raise RuntimeError("Not connected to database")
        async with self._conn.cursor() as cur:
            await cur.execute(query, params)
        await self._conn.commit()

    async def fetch_one(
        self, query: str, params: tuple | None = None
    ) -> dict[str, Any] | None:
        """Fetch a single row.

        Args:
            query: SQL query to execute.
            params: Query parameters.

        Returns:
            Row as dictionary or None if not found.
        """
        if not self._conn:
            raise RuntimeError("Not connected to database")
        async with self._conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchone()

    async def fetch_all(
        self, query: str, params: tuple | None = None
    ) -> list[dict[str, Any]]:
        """Fetch all rows.

        Args:
            query: SQL query to execute.
            params: Query parameters.

        Returns:
            List of rows as dictionaries.
        """
        if not self._conn:
            raise RuntimeError("Not connected to database")
        async with self._conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchall()

    async def fetch_many(
        self, query: str, params: tuple | None = None, size: int = 100
    ) -> list[dict[str, Any]]:
        """Fetch a batch of rows.

        Args:
            query: SQL query to execute.
            params: Query parameters.
            size: Number of rows to fetch.

        Returns:
            List of rows as dictionaries.
        """
        if not self._conn:
            raise RuntimeError("Not connected to database")
        async with self._conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchmany(size)

    async def insert_paper(self, paper: dict[str, Any]) -> int:
        """Insert a paper and return its ID.

        Args:
            paper: Paper data dictionary.

        Returns:
            Inserted paper ID.
        """
        query = """
            INSERT INTO papers (arxiv_id, title, abstract, authors, year, venue, categories)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (arxiv_id) DO UPDATE SET
                title = EXCLUDED.title,
                abstract = EXCLUDED.abstract,
                authors = EXCLUDED.authors,
                year = EXCLUDED.year,
                venue = EXCLUDED.venue,
                categories = EXCLUDED.categories,
                updated_at = NOW()
            RETURNING id
        """
        import json
        params = (
            paper["arxiv_id"],
            paper["title"],
            paper.get("abstract"),
            json.dumps(paper.get("authors", [])),
            paper.get("year"),
            paper.get("venue"),
            paper.get("categories", []),
        )
        result = await self.fetch_one(query, params)
        await self._conn.commit()
        return result["id"]

    async def get_paper_by_arxiv_id(self, arxiv_id: str) -> dict[str, Any] | None:
        """Get a paper by arXiv ID.

        Args:
            arxiv_id: arXiv identifier.

        Returns:
            Paper data or None if not found.
        """
        query = "SELECT * FROM papers WHERE arxiv_id = %s"
        return await self.fetch_one(query, (arxiv_id,))

    async def update_citation_count(self, arxiv_id: str, count: int) -> None:
        """Update citation count for a paper.

        Args:
            arxiv_id: arXiv identifier.
            count: Citation count.
        """
        query = """
            UPDATE papers SET citation_count = %s, updated_at = NOW()
            WHERE arxiv_id = %s
        """
        await self.execute(query, (count, arxiv_id))

    async def insert_extracted_method(self, method: dict[str, Any]) -> int:
        """Insert an extracted method.

        Args:
            method: Method data dictionary.

        Returns:
            Inserted method ID.
        """
        query = """
            INSERT INTO extracted_methods
            (paper_id, method_name, method_type, extraction_confidence, context_snippet)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """
        params = (
            method["paper_id"],
            method["method_name"],
            method.get("method_type"),
            method.get("extraction_confidence"),
            method.get("context_snippet"),
        )
        result = await self.fetch_one(query, params)
        await self._conn.commit()
        return result["id"]

    async def insert_hypothesis(self, hypothesis: dict[str, Any]) -> int:
        """Insert a generated hypothesis.

        Args:
            hypothesis: Hypothesis data dictionary.

        Returns:
            Inserted hypothesis ID.
        """
        import json
        query = """
            INSERT INTO hypotheses
            (gap_description, hypothesis_text, mechanism, assumptions,
             evidence_paper_ids, model_version)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        params = (
            hypothesis["gap_description"],
            hypothesis["hypothesis_text"],
            hypothesis.get("mechanism"),
            json.dumps(hypothesis.get("assumptions", [])),
            hypothesis.get("evidence_paper_ids", []),
            hypothesis.get("model_version"),
        )
        result = await self.fetch_one(query, params)
        await self._conn.commit()
        return result["id"]
