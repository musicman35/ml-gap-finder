"""Gap detection API routes."""

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.schemas import (
    GapDiscoverRequest,
    GapDiscoverResponse,
    GapSearchRequest,
    GapSearchResponse,
)
from src.services.gap_detector import GapDetectorService

router = APIRouter(prefix="/api/v1/gaps", tags=["gaps"])


def get_gap_detector(request: Request) -> GapDetectorService:
    """Dependency to get gap detector service using shared connections."""
    return GapDetectorService(
        neo4j_client=request.app.state.neo4j,
        cache=request.app.state.redis,
    )


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
        return GapSearchResponse.from_service_model(result)

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

        return GapDiscoverResponse(
            gaps=[GapSearchResponse.from_service_model(g) for g in gaps],
            total_found=len(gaps),
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

    return GapSearchResponse.from_service_model(result)
