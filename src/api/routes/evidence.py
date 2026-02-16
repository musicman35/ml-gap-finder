"""Evidence retrieval API routes."""

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.schemas import (
    CitationValidateRequest,
    CitationValidateResponse,
    EvidenceRequest,
    EvidenceResponse,
    PaperEvidence,
)
from src.services.evidence_retriever import EvidenceRetrieverService

router = APIRouter(prefix="/api/v1/evidence", tags=["evidence"])


def get_evidence_retriever(request: Request) -> EvidenceRetrieverService:
    """Dependency to get evidence retriever service using shared connections."""
    return EvidenceRetrieverService(
        neo4j_client=request.app.state.neo4j,
        postgres_client=request.app.state.postgres,
        qdrant_client=request.app.state.qdrant,
    )


@router.post("", response_model=EvidenceResponse)
async def get_evidence(
    request: EvidenceRequest,
    retriever: EvidenceRetrieverService = Depends(get_evidence_retriever),
) -> EvidenceResponse:
    """Retrieve evidence for a method-task claim.

    Finds papers that demonstrate the effectiveness of a method for a task.
    """
    try:
        bundle = await retriever.get_evidence(
            method=request.method,
            task=request.task,
            claim_type=request.claim_type,
            max_papers=request.max_papers,
        )

        return EvidenceResponse(
            papers=[
                PaperEvidence(
                    arxiv_id=p.arxiv_id,
                    title=p.title,
                    year=p.year,
                    citation_count=p.citation_count,
                    excerpt=p.excerpt,
                    relevance_score=p.relevance_score,
                )
                for p in bundle.papers
            ],
            confidence=bundle.confidence,
            claim_support_strength=bundle.claim_support_strength,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=CitationValidateResponse)
async def validate_citation(
    request: CitationValidateRequest,
    retriever: EvidenceRetrieverService = Depends(get_evidence_retriever),
) -> CitationValidateResponse:
    """Validate that a citation supports a claimed contribution.

    Checks if the cited paper actually supports what is being claimed.
    """
    try:
        result = await retriever.validate_citation(
            citing_paper_id=request.citing_paper_id,
            cited_paper_id=request.cited_paper_id,
            claimed_contribution=request.claimed_contribution,
        )

        return CitationValidateResponse(
            is_valid=result.is_valid,
            cited_paper_exists=result.cited_paper_exists,
            claim_supported=result.claim_supported,
            similarity_score=result.similarity_score,
            explanation=result.explanation,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
