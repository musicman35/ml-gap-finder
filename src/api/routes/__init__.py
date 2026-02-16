"""API routes module."""

from src.api.routes.evidence import router as evidence_router
from src.api.routes.gaps import router as gaps_router
from src.api.routes.hypotheses import router as hypotheses_router
from src.api.routes.literature import router as literature_router

__all__ = [
    "gaps_router",
    "evidence_router",
    "hypotheses_router",
    "literature_router",
]
