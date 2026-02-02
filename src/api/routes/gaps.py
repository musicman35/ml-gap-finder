"""Gap detection API routes."""

from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas import (
    GapSearchRequest,
    GapSearchResponse,
    GapDiscoverRequest,
    GapDiscoverResponse,
    MethodInfo,
)
from src.db.neo4j import Neo4jClient
from src.db.redis import RedisCache
from src.services.gap_detector import GapDetectorService

router = APIRouter(prefix="/api/v1/gaps", tags=["gaps"])


async def get_gap_detector() -> GapDetectorService:
    """Dependency to get gap detector service."""
    # In production, these would be connection-pooled
    neo4j = Neo4jClient()
    await neo4j.connect()
    redis = RedisCache()
    await redis.connect()
    return GapDetectorService(neo4j_client=neo4j, cache=redis)


@router.post("/search", response_model=GapSearchResponse)
async def search_gap(
    request: GapSearchRequest,
    detector: GapDetectorService = Depends(get_gap_detector),
) -> GapSearchResponse:
    """Search for a specific research gap.

    Checks if combining two methods for a task represents an underexplored area.
    """
    try:
        result = await detector.find_gaps(
            method_a=request.method_a,
            method_b=request.method_b,
            task=request.task,
            min_individual_papers=request.min_individual_papers,
            max_combination_papers=request.max_combination_papers,
        )

        return GapSearchResponse(
            gap_id=result.gap_id,
            is_gap=result.is_gap,
            method_a=MethodInfo(
                method_id=result.method_a.method_id,
                name=result.method_a.name,
                type=result.method_a.method_type,
                paper_count=result.method_a.paper_count,
            ),
            method_b=MethodInfo(
                method_id=result.method_b.method_id,
                name=result.method_b.name,
                type=result.method_b.method_type,
                paper_count=result.method_b.paper_count,
            ),
            task=result.task,
            gap_score=result.gap_score,
            method_a_paper_count=result.method_a_paper_count,
            method_b_paper_count=result.method_b_paper_count,
            combination_paper_count=result.combination_paper_count,
            evidence_summary=result.evidence_summary,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/discover", response_model=GapDiscoverResponse)
async def discover_gaps(
    request: GapDiscoverRequest,
    detector: GapDetectorService = Depends(get_gap_detector),
) -> GapDiscoverResponse:
    """Automatically discover potential research gaps for a task.

    Finds method pairs that are individually well-studied but rarely combined.
    """
    try:
        gaps = await detector.discover_gaps(
            task=request.task,
            method_type=request.method_type,
            min_individual_papers=request.min_individual_papers,
            max_combination_papers=request.max_combination_papers,
            top_k=request.top_k,
        )

        gap_responses = [
            GapSearchResponse(
                gap_id=g.gap_id,
                is_gap=g.is_gap,
                method_a=MethodInfo(
                    method_id=g.method_a.method_id,
                    name=g.method_a.name,
                    type=g.method_a.method_type,
                    paper_count=g.method_a.paper_count,
                ),
                method_b=MethodInfo(
                    method_id=g.method_b.method_id,
                    name=g.method_b.name,
                    type=g.method_b.method_type,
                    paper_count=g.method_b.paper_count,
                ),
                task=g.task,
                gap_score=g.gap_score,
                method_a_paper_count=g.method_a_paper_count,
                method_b_paper_count=g.method_b_paper_count,
                combination_paper_count=g.combination_paper_count,
                evidence_summary=g.evidence_summary,
            )
            for g in gaps
        ]

        return GapDiscoverResponse(
            gaps=gap_responses,
            total_found=len(gap_responses),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{gap_id}", response_model=GapSearchResponse)
async def get_gap(
    gap_id: str,
    detector: GapDetectorService = Depends(get_gap_detector),
) -> GapSearchResponse:
    """Get details of a specific gap by ID."""
    result = await detector.get_gap_details(gap_id)

    if not result:
        raise HTTPException(status_code=404, detail="Gap not found")

    return GapSearchResponse(
        gap_id=result.gap_id,
        is_gap=result.is_gap,
        method_a=MethodInfo(
            method_id=result.method_a.method_id,
            name=result.method_a.name,
            type=result.method_a.method_type,
            paper_count=result.method_a.paper_count,
        ),
        method_b=MethodInfo(
            method_id=result.method_b.method_id,
            name=result.method_b.name,
            type=result.method_b.method_type,
            paper_count=result.method_b.paper_count,
        ),
        task=result.task,
        gap_score=result.gap_score,
        method_a_paper_count=result.method_a_paper_count,
        method_b_paper_count=result.method_b_paper_count,
        combination_paper_count=result.combination_paper_count,
        evidence_summary=result.evidence_summary,
    )
