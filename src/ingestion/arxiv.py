"""arXiv paper harvester using the arXiv API."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator

import arxiv
import httpx
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import settings
from src.db.postgres import PostgresClient

logger = structlog.get_logger()


def _ensure_timezone_aware(dt: datetime | None) -> datetime | None:
    """Ensure datetime is timezone-aware (UTC).

    Args:
        dt: Datetime to check/convert.

    Returns:
        Timezone-aware datetime or None.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# ML-related arXiv categories
ML_CATEGORIES = ["cs.LG", "cs.CL", "cs.CV", "stat.ML"]


@dataclass
class PaperMetadata:
    """Represents metadata for an arXiv paper."""

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: datetime
    updated: datetime
    pdf_url: str

    @property
    def year(self) -> int:
        """Get publication year."""
        return self.published.year

    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "year": self.year,
            "venue": "arXiv",
        }


class ArxivHarvester:
    """Harvests papers from arXiv API with rate limiting."""

    def __init__(
        self,
        categories: list[str] | None = None,
        rate_limit_delay: float = 3.0,
    ):
        """Initialize arXiv harvester.

        Args:
            categories: List of arXiv categories to harvest.
            rate_limit_delay: Delay between requests in seconds.
        """
        self.categories = categories or ML_CATEGORIES
        self.rate_limit_delay = rate_limit_delay
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=rate_limit_delay,
            num_retries=3,
        )

    def _build_query(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> str:
        """Build arXiv API query string.

        Args:
            start_date: Start of date range.
            end_date: End of date range.

        Returns:
            Query string for arXiv API.
        """
        # Category query
        cat_query = " OR ".join([f"cat:{cat}" for cat in self.categories])
        query = f"({cat_query})"

        return query

    def _parse_result(self, result: arxiv.Result) -> PaperMetadata:
        """Parse arXiv result into PaperMetadata.

        Args:
            result: arXiv search result.

        Returns:
            Parsed paper metadata.
        """
        # Extract arxiv_id from entry_id (e.g., http://arxiv.org/abs/2301.00001v1 -> 2301.00001)
        arxiv_id = result.entry_id.split("/")[-1]
        if "v" in arxiv_id:
            arxiv_id = arxiv_id.split("v")[0]

        return PaperMetadata(
            arxiv_id=arxiv_id,
            title=result.title.replace("\n", " ").strip(),
            abstract=result.summary.replace("\n", " ").strip(),
            authors=[author.name for author in result.authors],
            categories=[cat for cat in result.categories],
            published=result.published,
            updated=result.updated,
            pdf_url=result.pdf_url,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((httpx.HTTPError, ConnectionError)),
    )
    async def fetch_papers(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        max_results: int = 1000,
    ) -> list[PaperMetadata]:
        """Fetch papers from arXiv within date range.

        Args:
            start_date: Start of date range.
            end_date: End of date range.
            max_results: Maximum number of papers to fetch.

        Returns:
            List of paper metadata.
        """
        query = self._build_query(start_date, end_date)
        logger.info(
            "Fetching papers from arXiv",
            query=query[:100],
            max_results=max_results,
        )

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers = []
        # Ensure dates are timezone-aware for comparison
        start_date_tz = _ensure_timezone_aware(start_date)
        end_date_tz = _ensure_timezone_aware(end_date)

        for result in self.client.results(search):
            # Filter by date if specified
            if start_date_tz and result.published < start_date_tz:
                continue
            if end_date_tz and result.published > end_date_tz:
                continue

            try:
                paper = self._parse_result(result)
                papers.append(paper)
            except Exception as e:
                logger.warning(
                    "Failed to parse paper",
                    entry_id=result.entry_id,
                    error=str(e),
                )

            # Rate limiting is handled by arxiv.Client
            if len(papers) >= max_results:
                break

        logger.info("Fetched papers from arXiv", count=len(papers))
        return papers

    async def fetch_papers_stream(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        batch_size: int = 100,
    ) -> AsyncIterator[list[PaperMetadata]]:
        """Stream papers from arXiv in batches.

        Args:
            start_date: Start of date range.
            end_date: End of date range.
            batch_size: Number of papers per batch.

        Yields:
            Batches of paper metadata.
        """
        query = self._build_query(start_date, end_date)
        logger.info("Starting paper stream from arXiv", query=query[:100])

        search = arxiv.Search(
            query=query,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        batch = []
        # Ensure dates are timezone-aware for comparison
        start_date_tz = _ensure_timezone_aware(start_date)
        end_date_tz = _ensure_timezone_aware(end_date)

        for result in self.client.results(search):
            # Filter by date if specified
            if start_date_tz and result.published < start_date_tz:
                continue
            if end_date_tz and result.published > end_date_tz:
                continue

            try:
                paper = self._parse_result(result)
                batch.append(paper)
            except Exception as e:
                logger.warning(
                    "Failed to parse paper",
                    entry_id=result.entry_id,
                    error=str(e),
                )

            if len(batch) >= batch_size:
                yield batch
                batch = []
                await asyncio.sleep(0.1)  # Yield control

        # Yield remaining papers
        if batch:
            yield batch

    async def harvest_and_store(
        self,
        postgres: PostgresClient,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        max_papers: int | None = None,
    ) -> int:
        """Harvest papers and store in PostgreSQL.

        Args:
            postgres: PostgreSQL client.
            start_date: Start of date range.
            end_date: End of date range.
            max_papers: Maximum papers to harvest.

        Returns:
            Number of papers stored.
        """
        total_stored = 0

        async for batch in self.fetch_papers_stream(
            start_date=start_date,
            end_date=end_date,
        ):
            for paper in batch:
                try:
                    await postgres.insert_paper(paper.to_dict())
                    total_stored += 1
                except Exception as e:
                    logger.warning(
                        "Failed to store paper",
                        arxiv_id=paper.arxiv_id,
                        error=str(e),
                    )

                if max_papers and total_stored >= max_papers:
                    logger.info("Reached max papers limit", count=total_stored)
                    return total_stored

            logger.info("Stored batch", batch_size=len(batch), total=total_stored)

        logger.info("Harvest complete", total_papers=total_stored)
        return total_stored

    async def fetch_paper_by_id(self, arxiv_id: str) -> PaperMetadata | None:
        """Fetch a single paper by arXiv ID.

        Args:
            arxiv_id: arXiv paper ID.

        Returns:
            Paper metadata or None if not found.
        """
        search = arxiv.Search(id_list=[arxiv_id])

        try:
            results = list(self.client.results(search))
            if results:
                return self._parse_result(results[0])
        except Exception as e:
            logger.warning(
                "Failed to fetch paper",
                arxiv_id=arxiv_id,
                error=str(e),
            )

        return None
