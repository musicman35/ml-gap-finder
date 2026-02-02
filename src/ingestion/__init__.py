"""Data ingestion modules for ML Gap Finder."""

from src.ingestion.arxiv import ArxivHarvester
from src.ingestion.semantic_scholar import SemanticScholarClient
from src.ingestion.papers_with_code import PapersWithCodeClient
from src.ingestion.method_extractor import MethodExtractor

__all__ = [
    "ArxivHarvester",
    "SemanticScholarClient",
    "PapersWithCodeClient",
    "MethodExtractor",
]
