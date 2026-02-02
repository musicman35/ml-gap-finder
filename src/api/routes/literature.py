"""Literature positioning API routes."""

from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas import (
    LiteraturePositionRequest,
    LiteraturePositionResponse,
    SimilarPaper,
    MethodLineage,
    RelatedWorkOutlineRequest,
    RelatedWorkOutlineResponse,
    RelatedWorkSection,
)
from src.db.neo4j import Neo4jClient
from src.db.postgres import PostgresClient
from src.db.qdrant import QdrantVectorStore
from src.services.literature_positioner import LiteraturePositionerService

router = APIRouter(prefix="/api/v1/literature", tags=["literature"])


async def get_literature_positioner() -> LiteraturePositionerService:
    """Dependency to get literature positioner service."""
    neo4j = Neo4jClient()
    await neo4j.connect()
    postgres = PostgresClient()
    await postgres.connect()
    qdrant = QdrantVectorStore()

    return LiteraturePositionerService(
        neo4j_client=neo4j,
        postgres_client=postgres,
        qdrant_client=qdrant,
    )


@router.post("/position", response_model=LiteraturePositionResponse)
async def position_approach(
    request: LiteraturePositionRequest,
    positioner: LiteraturePositionerService = Depends(get_literature_positioner),
) -> LiteraturePositionResponse:
    """Position an approach within existing literature.

    Finds similar papers, method lineage, and differentiation points.
    """
    try:
        result = await positioner.position_approach(
            approach_description=request.approach_description,
            methods=request.methods,
            top_k=request.max_similar_papers,
        )

        return LiteraturePositionResponse(
            most_similar_papers=[
                SimilarPaper(
                    arxiv_id=p.arxiv_id,
                    title=p.title,
                    year=p.year,
                    similarity_score=p.similarity_score,
                    abstract_excerpt=p.abstract_excerpt,
                )
                for p in result.most_similar_papers
            ],
            method_lineage=[
                MethodLineage(
                    method_name=m.method_name,
                    origin_paper=m.origin_paper,
                    evolution_papers=m.evolution_papers,
                )
                for m in result.method_lineage
            ],
            differentiation_points=result.differentiation_points,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/related-work", response_model=RelatedWorkOutlineResponse)
async def generate_related_work_outline(
    request: RelatedWorkOutlineRequest,
    positioner: LiteraturePositionerService = Depends(get_literature_positioner),
) -> RelatedWorkOutlineResponse:
    """Generate a structured related work outline.

    Creates a section-by-section outline with papers to cite.
    """
    try:
        result = await positioner.generate_related_work_outline(
            approach_description=request.approach_description,
            max_citations=request.max_citations,
        )

        return RelatedWorkOutlineResponse(
            sections=[
                RelatedWorkSection(
                    title=s.title,
                    theme=s.theme,
                    papers_to_cite=s.papers_to_cite,
                    transition=s.transition,
                )
                for s in result.sections
            ],
            positioning_summary=result.positioning_summary,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
