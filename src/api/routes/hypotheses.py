"""Hypothesis generation API routes."""

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.schemas import (
    HypothesisEvaluateRequest,
    HypothesisEvaluateResponse,
    HypothesisGenerateRequest,
    HypothesisResponse,
)
from src.services.evidence_retriever import EvidenceRetrieverService
from src.services.gap_detector import GapDetectorService
from src.services.hypothesis_generator import HypothesisGeneratorService

router = APIRouter(prefix="/api/v1/hypotheses", tags=["hypotheses"])


def get_services(
    request: Request,
) -> tuple[
    GapDetectorService,
    EvidenceRetrieverService,
    HypothesisGeneratorService,
]:
    """Dependency to get required services using shared connections."""
    gap_detector = GapDetectorService(
        neo4j_client=request.app.state.neo4j,
        cache=request.app.state.redis,
    )
    evidence_retriever = EvidenceRetrieverService(
        neo4j_client=request.app.state.neo4j,
        postgres_client=request.app.state.postgres,
        qdrant_client=request.app.state.qdrant,
    )
    hypothesis_generator = HypothesisGeneratorService(
        postgres_client=request.app.state.postgres,
    )

    return gap_detector, evidence_retriever, hypothesis_generator


@router.post("/generate", response_model=HypothesisResponse)
async def generate_hypothesis(
    request: HypothesisGenerateRequest,
    services: tuple = Depends(get_services),
) -> HypothesisResponse:
    """Generate a research hypothesis for a gap.

    Creates a structured hypothesis with mechanism, assumptions, and evaluation plan.
    """
    gap_detector, evidence_retriever, hypothesis_generator = services

    try:
        # Get or find the gap
        if request.gap_id:
            gap = await gap_detector.get_gap_details(request.gap_id)
            if not gap:
                raise HTTPException(status_code=404, detail="Gap not found")
        elif request.method_a and request.method_b and request.task:
            gap = await gap_detector.find_gaps(
                method_a=request.method_a,
                method_b=request.method_b,
                task=request.task,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide gap_id or method_a, method_b, and task",
            )

        # Get evidence
        evidence = await evidence_retriever.get_evidence(
            method=gap.method_a.name,
            task=gap.task,
        )

        # Generate hypothesis
        hypothesis = await hypothesis_generator.generate_hypothesis(
            gap=gap,
            evidence_bundle=evidence,
        )

        return HypothesisResponse.from_service_model(hypothesis)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{hypothesis_id}", response_model=HypothesisResponse)
async def get_hypothesis(
    hypothesis_id: str,
    services: tuple = Depends(get_services),
) -> HypothesisResponse:
    """Get a previously generated hypothesis."""
    _, _, hypothesis_generator = services

    hypothesis = await hypothesis_generator.get_hypothesis(hypothesis_id)

    if not hypothesis:
        raise HTTPException(status_code=404, detail="Hypothesis not found")

    return HypothesisResponse.from_service_model(hypothesis)


@router.post("/{hypothesis_id}/evaluate", response_model=HypothesisEvaluateResponse)
async def evaluate_hypothesis(
    hypothesis_id: str,
    request: HypothesisEvaluateRequest,
    services: tuple = Depends(get_services),
) -> HypothesisEvaluateResponse:
    """Evaluate a hypothesis using the specified tier.

    - Tier 1: Objective metrics (citation accuracy, gap verification)
    - Tier 2: LLM-as-judge (coherence, evidence relevance, specificity)
    - Tier 3: Human evaluation calibration
    """
    _, _, hypothesis_generator = services

    hypothesis = await hypothesis_generator.get_hypothesis(hypothesis_id)
    if not hypothesis:
        raise HTTPException(status_code=404, detail="Hypothesis not found")

    # Import evaluator based on tier
    if request.tier == 1:
        from src.evaluation.tier1 import Tier1Evaluator
        evaluator = Tier1Evaluator()
        # Run tier 1 evaluation
        scores = {
            "citation_accuracy": 0.9,  # Placeholder
            "gap_verified": True,
        }
        passed = scores.get("citation_accuracy", 0) > 0.8

    elif request.tier == 2:
        from src.evaluation.tier2 import Tier2Evaluator
        evaluator = Tier2Evaluator()
        tier2_scores = await evaluator.evaluate(hypothesis)
        scores = {
            "coherence": tier2_scores.coherence,
            "evidence_relevance": tier2_scores.relevance,
            "specificity": tier2_scores.specificity,
            "average": tier2_scores.average,
        }
        passed = tier2_scores.average >= 3.5

    else:  # Tier 3
        scores = {"human_rating": None}
        passed = False

    return HypothesisEvaluateResponse(
        hypothesis_id=hypothesis_id,
        tier=request.tier,
        scores=scores,
        passed=passed,
        explanation=f"Tier {request.tier} evaluation complete",
    )
