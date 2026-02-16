"""Semantic Scholar API client for paper enrichment."""

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from src.db.redis import RedisCache

logger = structlog.get_logger()

SEMANTIC_SCHOLAR_API_BASE = "https://api.semanticscholar.org/graph/v1"


@dataclass
class SemanticScholarPaper:
    """Semantic Scholar paper data."""

    paper_id: str
    arxiv_id: str | None
    title: str
    citation_count: int
    influential_citation_count: int
    references: list[str]
    citations: list[str]

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "SemanticScholarPaper":
        """Create from API response."""
        external_ids = data.get("externalIds", {})
        return cls(
            paper_id=data.get("paperId", ""),
            arxiv_id=external_ids.get("ArXiv"),
            title=data.get("title", ""),
            citation_count=data.get("citationCount", 0),
            influential_citation_count=data.get("influentialCitationCount", 0),
            references=[ref.get("paperId") for ref in data.get("references", []) if ref.get("paperId")],
            citations=[cit.get("paperId") for cit in data.get("citations", []) if cit.get("paperId")],
        )


class SemanticScholarClient:
    """Client for Semantic Scholar API with rate limiting."""

    def __init__(
        self,
        api_key: str | None = None,
        cache: RedisCache | None = None,
    ):
        """Initialize Semantic Scholar client.

        Args:
            api_key: Optional API key for higher rate limits.
            cache: Optional Redis cache for responses.
        """
        self.api_key = api_key or settings.semantic_scholar_api_key
        self.cache = cache
        self.base_url = SEMANTIC_SCHOLAR_API_BASE

        # Rate limiting: 100 requests per 5 minutes without API key
        self._request_count = 0
        self._window_start = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        current_time = asyncio.get_event_loop().time()
        window_duration = 300  # 5 minutes
        max_requests = 100 if not self.api_key else 1000

        if current_time - self._window_start > window_duration:
            self._window_start = current_time
            self._request_count = 0

        if self._request_count >= max_requests:
            sleep_time = window_duration - (current_time - self._window_start)
            if sleep_time > 0:
                logger.info("Rate limit reached, sleeping", seconds=sleep_time)
                await asyncio.sleep(sleep_time)
                self._window_start = asyncio.get_event_loop().time()
                self._request_count = 0

        self._request_count += 1

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Make API request with rate limiting.

        Args:
            method: HTTP method.
            endpoint: API endpoint.
            params: Query parameters.
            json_data: JSON body for POST requests.

        Returns:
            API response data or None on error.
        """
        await self._check_rate_limit()

        url = f"{self.base_url}{endpoint}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=self._get_headers(),
                params=params,
                json=json_data,
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            return response.json()

    async def get_paper(
        self,
        paper_id: str,
        fields: list[str] | None = None,
    ) -> SemanticScholarPaper | None:
        """Get paper by Semantic Scholar ID or arXiv ID.

        Args:
            paper_id: Paper ID or arXiv ID (prefix with "ARXIV:").
            fields: Fields to retrieve.

        Returns:
            Paper data or None if not found.
        """
        # Check cache first
        cache_key = f"s2:paper:{paper_id}"
        if self.cache:
            cached = await self.cache.get_json(cache_key)
            if cached:
                return SemanticScholarPaper.from_api_response(cached)

        default_fields = [
            "paperId",
            "externalIds",
            "title",
            "citationCount",
            "influentialCitationCount",
            "references.paperId",
            "citations.paperId",
        ]
        fields = fields or default_fields

        data = await self._request(
            "GET",
            f"/paper/{paper_id}",
            params={"fields": ",".join(fields)},
        )

        if data:
            # Cache the response
            if self.cache:
                await self.cache.set_json(cache_key, data, expire=3600)
            return SemanticScholarPaper.from_api_response(data)

        return None

    async def get_paper_by_arxiv_id(
        self,
        arxiv_id: str,
    ) -> SemanticScholarPaper | None:
        """Get paper by arXiv ID.

        Args:
            arxiv_id: arXiv paper ID.

        Returns:
            Paper data or None if not found.
        """
        return await self.get_paper(f"ARXIV:{arxiv_id}")

    async def batch_get_papers(
        self,
        paper_ids: list[str],
        fields: list[str] | None = None,
    ) -> list[SemanticScholarPaper]:
        """Get multiple papers in a batch request.

        Args:
            paper_ids: List of paper IDs.
            fields: Fields to retrieve.

        Returns:
            List of paper data.
        """
        default_fields = [
            "paperId",
            "externalIds",
            "title",
            "citationCount",
            "influentialCitationCount",
        ]
        fields = fields or default_fields

        # Batch endpoint accepts up to 500 IDs
        results = []
        for i in range(0, len(paper_ids), 500):
            batch = paper_ids[i : i + 500]

            data = await self._request(
                "POST",
                "/paper/batch",
                params={"fields": ",".join(fields)},
                json_data={"ids": batch},
            )

            if data:
                for paper_data in data:
                    if paper_data:
                        results.append(SemanticScholarPaper.from_api_response(paper_data))

        return results

    async def search_papers(
        self,
        query: str,
        limit: int = 100,
        fields: list[str] | None = None,
    ) -> list[SemanticScholarPaper]:
        """Search for papers.

        Args:
            query: Search query.
            limit: Maximum results to return.
            fields: Fields to retrieve.

        Returns:
            List of matching papers.
        """
        default_fields = [
            "paperId",
            "externalIds",
            "title",
            "citationCount",
        ]
        fields = fields or default_fields

        papers = []
        offset = 0

        while len(papers) < limit:
            data = await self._request(
                "GET",
                "/paper/search",
                params={
                    "query": query,
                    "limit": min(100, limit - len(papers)),
                    "offset": offset,
                    "fields": ",".join(fields),
                },
            )

            if not data or not data.get("data"):
                break

            for paper_data in data["data"]:
                papers.append(SemanticScholarPaper.from_api_response(paper_data))

            offset += len(data["data"])

            if len(data["data"]) < 100:
                break

        return papers[:limit]

    async def get_citations(
        self,
        paper_id: str,
        limit: int = 100,
    ) -> list[str]:
        """Get papers that cite a given paper.

        Args:
            paper_id: Paper ID.
            limit: Maximum citations to return.

        Returns:
            List of citing paper IDs.
        """
        data = await self._request(
            "GET",
            f"/paper/{paper_id}/citations",
            params={
                "limit": min(1000, limit),
                "fields": "paperId",
            },
        )

        if not data or not data.get("data"):
            return []

        return [cit.get("citingPaper", {}).get("paperId") for cit in data["data"] if cit.get("citingPaper")]

    async def get_references(
        self,
        paper_id: str,
        limit: int = 100,
    ) -> list[str]:
        """Get papers referenced by a given paper.

        Args:
            paper_id: Paper ID.
            limit: Maximum references to return.

        Returns:
            List of referenced paper IDs.
        """
        data = await self._request(
            "GET",
            f"/paper/{paper_id}/references",
            params={
                "limit": min(1000, limit),
                "fields": "paperId",
            },
        )

        if not data or not data.get("data"):
            return []

        return [ref.get("citedPaper", {}).get("paperId") for ref in data["data"] if ref.get("citedPaper")]
