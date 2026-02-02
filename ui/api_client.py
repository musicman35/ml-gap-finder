"""API client for ML Gap Finder backend."""

from typing import Any

import httpx


class APIClient:
    """Client for interacting with the ML Gap Finder API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize API client.

        Args:
            base_url: Base URL of the API server.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = 120.0

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle API response and errors."""
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            raise APIError("Resource not found")
        elif response.status_code == 422:
            detail = response.json().get("detail", "Validation error")
            raise APIError(f"Invalid request: {detail}")
        else:
            detail = response.json().get("detail", "Unknown error")
            raise APIError(f"API error: {detail}")

    def health_check(self) -> dict[str, Any]:
        """Check API health status."""
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/health")
                return response.json()
        except httpx.ConnectError:
            return {
                "status": "unavailable",
                "version": "unknown",
                "llm_provider": "unknown",
                "databases": {
                    "neo4j": False,
                    "postgres": False,
                    "qdrant": False,
                    "redis": False,
                },
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def search_gap(
        self,
        method_a: str,
        method_b: str,
        task: str,
        min_individual_papers: int = 5,
        max_combination_papers: int = 2,
    ) -> dict[str, Any]:
        """Search for a specific research gap."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/v1/gaps/search",
                json={
                    "method_a": method_a,
                    "method_b": method_b,
                    "task": task,
                    "min_individual_papers": min_individual_papers,
                    "max_combination_papers": max_combination_papers,
                },
            )
            return self._handle_response(response)

    def discover_gaps(
        self,
        task: str,
        method_type: str | None = None,
        min_individual_papers: int = 5,
        max_combination_papers: int = 2,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Discover research gaps automatically."""
        payload = {
            "task": task,
            "min_individual_papers": min_individual_papers,
            "max_combination_papers": max_combination_papers,
            "top_k": top_k,
        }
        if method_type:
            payload["method_type"] = method_type

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/v1/gaps/discover",
                json=payload,
            )
            return self._handle_response(response)

    def get_gap(self, gap_id: str) -> dict[str, Any]:
        """Get gap details by ID."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/api/v1/gaps/{gap_id}")
            return self._handle_response(response)

    def get_evidence(
        self,
        method: str,
        task: str,
        claim_type: str = "improves",
        max_papers: int = 10,
    ) -> dict[str, Any]:
        """Get evidence for a method-task claim."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/v1/evidence",
                json={
                    "method": method,
                    "task": task,
                    "claim_type": claim_type,
                    "max_papers": max_papers,
                },
            )
            return self._handle_response(response)

    def validate_citation(
        self,
        citing_paper_id: str,
        cited_paper_id: str,
        claimed_contribution: str,
    ) -> dict[str, Any]:
        """Validate a citation claim."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/v1/evidence/validate",
                json={
                    "citing_paper_id": citing_paper_id,
                    "cited_paper_id": cited_paper_id,
                    "claimed_contribution": claimed_contribution,
                },
            )
            return self._handle_response(response)

    def generate_hypothesis(
        self,
        gap_id: str | None = None,
        method_a: str | None = None,
        method_b: str | None = None,
        task: str | None = None,
        include_evidence: bool = True,
    ) -> dict[str, Any]:
        """Generate a research hypothesis."""
        payload = {"include_evidence": include_evidence}
        if gap_id:
            payload["gap_id"] = gap_id
        else:
            payload["method_a"] = method_a
            payload["method_b"] = method_b
            payload["task"] = task

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/v1/hypotheses/generate",
                json=payload,
            )
            return self._handle_response(response)

    def get_hypothesis(self, hypothesis_id: str) -> dict[str, Any]:
        """Get hypothesis by ID."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                f"{self.base_url}/api/v1/hypotheses/{hypothesis_id}"
            )
            return self._handle_response(response)

    def evaluate_hypothesis(
        self,
        hypothesis_id: str,
        tier: int,
    ) -> dict[str, Any]:
        """Evaluate a hypothesis."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/v1/hypotheses/{hypothesis_id}/evaluate",
                json={"tier": tier},
            )
            return self._handle_response(response)

    def position_in_literature(
        self,
        approach_description: str,
        methods: list[str],
        max_similar_papers: int = 10,
    ) -> dict[str, Any]:
        """Position an approach in the literature."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/v1/literature/position",
                json={
                    "approach_description": approach_description,
                    "methods": methods,
                    "max_similar_papers": max_similar_papers,
                },
            )
            return self._handle_response(response)

    def generate_related_work(
        self,
        approach_description: str,
        max_citations: int = 20,
    ) -> dict[str, Any]:
        """Generate a related work outline."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/v1/literature/related-work",
                json={
                    "approach_description": approach_description,
                    "max_citations": max_citations,
                },
            )
            return self._handle_response(response)


class APIError(Exception):
    """Custom exception for API errors."""

    pass
