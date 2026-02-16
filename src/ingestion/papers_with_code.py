"""Papers With Code API client for method and dataset information."""

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

from src.db.redis import RedisCache

logger = structlog.get_logger()

PAPERS_WITH_CODE_API_BASE = "https://paperswithcode.com/api/v1"


@dataclass
class PWCMethod:
    """Papers With Code method data."""

    method_id: str
    name: str
    full_name: str
    description: str
    paper_url: str | None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "PWCMethod":
        """Create from API response."""
        return cls(
            method_id=data.get("id", ""),
            name=data.get("name", ""),
            full_name=data.get("full_name", ""),
            description=data.get("description", ""),
            paper_url=data.get("paper", {}).get("url") if data.get("paper") else None,
        )


@dataclass
class PWCDataset:
    """Papers With Code dataset data."""

    dataset_id: str
    name: str
    full_name: str
    description: str
    paper_count: int
    modalities: list[str]

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "PWCDataset":
        """Create from API response."""
        return cls(
            dataset_id=data.get("id", ""),
            name=data.get("name", ""),
            full_name=data.get("full_name", ""),
            description=data.get("description", ""),
            paper_count=data.get("num_papers", 0),
            modalities=data.get("modalities", []),
        )


@dataclass
class PWCTask:
    """Papers With Code task data."""

    task_id: str
    name: str
    description: str
    paper_count: int

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "PWCTask":
        """Create from API response."""
        return cls(
            task_id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            paper_count=data.get("num_papers", 0),
        )


@dataclass
class PWCPaper:
    """Papers With Code paper data."""

    paper_id: str
    arxiv_id: str | None
    title: str
    abstract: str
    methods: list[str]
    tasks: list[str]
    datasets: list[str]

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "PWCPaper":
        """Create from API response."""
        return cls(
            paper_id=data.get("id", ""),
            arxiv_id=data.get("arxiv_id"),
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            methods=[m.get("name", "") for m in data.get("methods", [])],
            tasks=[t.get("name", "") for t in data.get("tasks", [])],
            datasets=[d.get("name", "") for d in data.get("datasets", [])],
        )


class PapersWithCodeClient:
    """Client for Papers With Code API."""

    def __init__(self, cache: RedisCache | None = None):
        """Initialize Papers With Code client.

        Args:
            cache: Optional Redis cache for responses.
        """
        self.cache = cache
        self.base_url = PAPERS_WITH_CODE_API_BASE

        # Rate limiting: 60 requests per minute
        self._request_count = 0
        self._window_start = 0.0

    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        current_time = asyncio.get_event_loop().time()
        window_duration = 60  # 1 minute
        max_requests = 60

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
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list | None:
        """Make API request with rate limiting.

        Args:
            endpoint: API endpoint.
            params: Query parameters.

        Returns:
            API response data or None on error.
        """
        await self._check_rate_limit()

        url = f"{self.base_url}{endpoint}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url,
                params=params,
                headers={"Accept": "application/json"},
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            return response.json()

    async def get_paper(self, paper_id: str) -> PWCPaper | None:
        """Get paper by Papers With Code ID.

        Args:
            paper_id: Paper ID or URL slug.

        Returns:
            Paper data or None if not found.
        """
        cache_key = f"pwc:paper:{paper_id}"
        if self.cache:
            cached = await self.cache.get_json(cache_key)
            if cached:
                return PWCPaper.from_api_response(cached)

        data = await self._request(f"/papers/{paper_id}/")

        if data:
            if self.cache:
                await self.cache.set_json(cache_key, data, expire=3600)
            return PWCPaper.from_api_response(data)

        return None

    async def get_paper_by_arxiv_id(self, arxiv_id: str) -> PWCPaper | None:
        """Get paper by arXiv ID.

        Args:
            arxiv_id: arXiv paper ID.

        Returns:
            Paper data or None if not found.
        """
        # Search for paper by arXiv ID
        data = await self._request("/papers/", params={"arxiv_id": arxiv_id})

        if data and data.get("results"):
            paper_data = data["results"][0]
            return PWCPaper.from_api_response(paper_data)

        return None

    async def get_paper_methods(self, paper_id: str) -> list[PWCMethod]:
        """Get methods used in a paper.

        Args:
            paper_id: Paper ID.

        Returns:
            List of methods.
        """
        data = await self._request(f"/papers/{paper_id}/methods/")

        if not data or not data.get("results"):
            return []

        return [PWCMethod.from_api_response(m) for m in data["results"]]

    async def get_method(self, method_id: str) -> PWCMethod | None:
        """Get method by ID.

        Args:
            method_id: Method ID or slug.

        Returns:
            Method data or None if not found.
        """
        cache_key = f"pwc:method:{method_id}"
        if self.cache:
            cached = await self.cache.get_json(cache_key)
            if cached:
                return PWCMethod.from_api_response(cached)

        data = await self._request(f"/methods/{method_id}/")

        if data:
            if self.cache:
                await self.cache.set_json(cache_key, data, expire=3600)
            return PWCMethod.from_api_response(data)

        return None

    async def search_methods(
        self,
        query: str,
        limit: int = 100,
    ) -> list[PWCMethod]:
        """Search for methods.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of matching methods.
        """
        methods = []
        page = 1

        while len(methods) < limit:
            data = await self._request(
                "/methods/",
                params={"q": query, "page": page},
            )

            if not data or not data.get("results"):
                break

            for method_data in data["results"]:
                methods.append(PWCMethod.from_api_response(method_data))

            if not data.get("next"):
                break

            page += 1

        return methods[:limit]

    async def get_dataset(self, dataset_id: str) -> PWCDataset | None:
        """Get dataset by ID.

        Args:
            dataset_id: Dataset ID or slug.

        Returns:
            Dataset data or None if not found.
        """
        cache_key = f"pwc:dataset:{dataset_id}"
        if self.cache:
            cached = await self.cache.get_json(cache_key)
            if cached:
                return PWCDataset.from_api_response(cached)

        data = await self._request(f"/datasets/{dataset_id}/")

        if data:
            if self.cache:
                await self.cache.set_json(cache_key, data, expire=3600)
            return PWCDataset.from_api_response(data)

        return None

    async def search_datasets(
        self,
        query: str,
        limit: int = 100,
    ) -> list[PWCDataset]:
        """Search for datasets.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of matching datasets.
        """
        datasets = []
        page = 1

        while len(datasets) < limit:
            data = await self._request(
                "/datasets/",
                params={"q": query, "page": page},
            )

            if not data or not data.get("results"):
                break

            for dataset_data in data["results"]:
                datasets.append(PWCDataset.from_api_response(dataset_data))

            if not data.get("next"):
                break

            page += 1

        return datasets[:limit]

    async def get_task(self, task_id: str) -> PWCTask | None:
        """Get task by ID.

        Args:
            task_id: Task ID or slug.

        Returns:
            Task data or None if not found.
        """
        cache_key = f"pwc:task:{task_id}"
        if self.cache:
            cached = await self.cache.get_json(cache_key)
            if cached:
                return PWCTask.from_api_response(cached)

        data = await self._request(f"/tasks/{task_id}/")

        if data:
            if self.cache:
                await self.cache.set_json(cache_key, data, expire=3600)
            return PWCTask.from_api_response(data)

        return None

    async def search_tasks(
        self,
        query: str,
        limit: int = 100,
    ) -> list[PWCTask]:
        """Search for tasks.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of matching tasks.
        """
        tasks = []
        page = 1

        while len(tasks) < limit:
            data = await self._request(
                "/tasks/",
                params={"q": query, "page": page},
            )

            if not data or not data.get("results"):
                break

            for task_data in data["results"]:
                tasks.append(PWCTask.from_api_response(task_data))

            if not data.get("next"):
                break

            page += 1

        return tasks[:limit]

    async def get_papers_for_method(
        self,
        method_id: str,
        limit: int = 100,
    ) -> list[PWCPaper]:
        """Get papers that use a specific method.

        Args:
            method_id: Method ID.
            limit: Maximum papers to return.

        Returns:
            List of papers using the method.
        """
        papers = []
        page = 1

        while len(papers) < limit:
            data = await self._request(
                f"/methods/{method_id}/papers/",
                params={"page": page},
            )

            if not data or not data.get("results"):
                break

            for paper_data in data["results"]:
                papers.append(PWCPaper.from_api_response(paper_data))

            if not data.get("next"):
                break

            page += 1

        return papers[:limit]

    async def get_papers_for_dataset(
        self,
        dataset_id: str,
        limit: int = 100,
    ) -> list[PWCPaper]:
        """Get papers that use a specific dataset.

        Args:
            dataset_id: Dataset ID.
            limit: Maximum papers to return.

        Returns:
            List of papers using the dataset.
        """
        papers = []
        page = 1

        while len(papers) < limit:
            data = await self._request(
                f"/datasets/{dataset_id}/papers/",
                params={"page": page},
            )

            if not data or not data.get("results"):
                break

            for paper_data in data["results"]:
                papers.append(PWCPaper.from_api_response(paper_data))

            if not data.get("next"):
                break

            page += 1

        return papers[:limit]
