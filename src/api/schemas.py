"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ============================================================================
# Gap Detection Schemas
# ============================================================================


class MethodInfo(BaseModel):
    """Method information."""

    method_id: str
    name: str
    type: str = "unknown"
    paper_count: int = 0


class GapSearchRequest(BaseModel):
    """Request to search for a specific gap."""

    method_a: str = Field(..., description="First method name or ID")
    method_b: str = Field(..., description="Second method name or ID")
    task: str = Field(..., description="Task name")
    min_individual_papers: int = Field(
        default=5,
        ge=1,
        description="Minimum papers using each method individually",
    )
    max_combination_papers: int = Field(
        default=2,
        ge=0,
        description="Maximum papers combining both methods for this to be a gap",
    )


class GapSearchResponse(BaseModel):
    """Response from gap search."""

    gap_id: str
    is_gap: bool
    method_a: MethodInfo
    method_b: MethodInfo
    task: str
    gap_score: float
    method_a_paper_count: int
    method_b_paper_count: int
    combination_paper_count: int
    evidence_summary: dict[str, Any] = Field(default_factory=dict)


class GapDiscoverRequest(BaseModel):
    """Request to discover gaps for a task."""

    task: str = Field(..., description="Task name to find gaps for")
    method_type: str | None = Field(
        default=None,
        description="Optional filter by method type",
    )
    min_individual_papers: int = Field(default=5, ge=1)
    max_combination_papers: int = Field(default=2, ge=0)
    top_k: int = Field(default=10, ge=1, le=100)


class GapDiscoverResponse(BaseModel):
    """Response from gap discovery."""

    gaps: list[GapSearchResponse]
    total_found: int


# ============================================================================
# Evidence Schemas
# ============================================================================


class EvidenceRequest(BaseModel):
    """Request for evidence retrieval."""

    method: str = Field(..., description="Method name or ID")
    task: str = Field(..., description="Task name")
    claim_type: str = Field(
        default="improves",
        description="Type of claim: 'improves', 'outperforms', 'enables'",
    )
    max_papers: int = Field(default=10, ge=1, le=50)


class PaperEvidence(BaseModel):
    """Evidence from a paper."""

    arxiv_id: str
    title: str
    year: int
    citation_count: int
    excerpt: str
    relevance_score: float


class EvidenceResponse(BaseModel):
    """Response with evidence bundle."""

    papers: list[PaperEvidence]
    confidence: float
    claim_support_strength: str


class CitationValidateRequest(BaseModel):
    """Request to validate a citation."""

    citing_paper_id: str
    cited_paper_id: str
    claimed_contribution: str


class CitationValidateResponse(BaseModel):
    """Response from citation validation."""

    is_valid: bool
    cited_paper_exists: bool
    claim_supported: bool
    similarity_score: float
    explanation: str


# ============================================================================
# Hypothesis Schemas
# ============================================================================


class HypothesisGenerateRequest(BaseModel):
    """Request to generate a hypothesis."""

    gap_id: str | None = Field(default=None, description="Existing gap ID")
    method_a: str | None = Field(default=None, description="First method")
    method_b: str | None = Field(default=None, description="Second method")
    task: str | None = Field(default=None, description="Task name")
    include_evidence: bool = Field(
        default=True,
        description="Include supporting evidence",
    )


class Assumption(BaseModel):
    """A hypothesis assumption."""

    text: str
    evidence_paper_id: str | None = None
    evidence_excerpt: str | None = None


class EvaluationPlan(BaseModel):
    """Suggested evaluation plan."""

    datasets: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    expected_outcome: str = ""


class HypothesisResponse(BaseModel):
    """Response with generated hypothesis."""

    hypothesis_id: str
    hypothesis_text: str
    mechanism: str
    assumptions: list[Assumption]
    evaluation_plan: EvaluationPlan
    evidence_paper_ids: list[str]
    gap_description: str
    coherence_score: int | None = None
    evidence_relevance_score: int | None = None
    specificity_score: int | None = None
    created_at: datetime
    model_version: str


class HypothesisEvaluateRequest(BaseModel):
    """Request to evaluate a hypothesis."""

    tier: int = Field(..., ge=1, le=3, description="Evaluation tier (1, 2, or 3)")


class HypothesisEvaluateResponse(BaseModel):
    """Response from hypothesis evaluation."""

    hypothesis_id: str
    tier: int
    scores: dict[str, Any]
    passed: bool
    explanation: str


# ============================================================================
# Literature Positioning Schemas
# ============================================================================


class LiteraturePositionRequest(BaseModel):
    """Request to position approach in literature."""

    approach_description: str = Field(
        ...,
        min_length=50,
        description="Description of the proposed approach",
    )
    methods: list[str] = Field(..., min_length=1, description="Methods used")
    max_similar_papers: int = Field(default=10, ge=1, le=50)


class SimilarPaper(BaseModel):
    """A semantically similar paper."""

    arxiv_id: str
    title: str
    year: int
    similarity_score: float
    abstract_excerpt: str


class MethodLineage(BaseModel):
    """Lineage of a method."""

    method_name: str
    origin_paper: str | None
    evolution_papers: list[str]


class LiteraturePositionResponse(BaseModel):
    """Response with literature positioning."""

    most_similar_papers: list[SimilarPaper]
    method_lineage: list[MethodLineage]
    differentiation_points: list[str]


class RelatedWorkOutlineRequest(BaseModel):
    """Request for related work outline."""

    approach_description: str = Field(..., min_length=50)
    max_citations: int = Field(default=20, ge=5, le=50)


class RelatedWorkSection(BaseModel):
    """A section in related work outline."""

    title: str
    theme: str
    papers_to_cite: list[str]
    transition: str


class RelatedWorkOutlineResponse(BaseModel):
    """Response with related work outline."""

    sections: list[RelatedWorkSection]
    positioning_summary: str


# ============================================================================
# Health Check
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    llm_provider: str
    databases: dict[str, bool]
