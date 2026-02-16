"""Pytest fixtures for ML Gap Finder tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_postgres():
    """Mock PostgreSQL client."""
    client = AsyncMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.execute = AsyncMock()
    client.fetch_one = AsyncMock(return_value=None)
    client.fetch_all = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_neo4j():
    """Mock Neo4j client."""
    client = AsyncMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.run_query = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    client = AsyncMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock()
    client.get_json = AsyncMock(return_value=None)
    client.set_json = AsyncMock()
    client.get_gap_result = AsyncMock(return_value=None)
    client.cache_gap_result = AsyncMock()
    client.get_paper_metadata = AsyncMock(return_value=None)
    client.cache_paper_metadata = AsyncMock()
    return client


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client."""
    client = MagicMock()
    client.search = MagicMock(return_value=[])
    client.upsert_vectors = MagicMock()
    return client


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = AsyncMock()
    client.generate = AsyncMock(return_value="Test response")
    return client


@pytest.fixture
def sample_paper():
    """Sample paper data."""
    return {
        "arxiv_id": "2301.00001",
        "title": "Test Paper on Contrastive Learning",
        "abstract": "We propose a novel approach using contrastive learning and transformers...",
        "authors": ["Author One", "Author Two"],
        "year": 2023,
        "categories": ["cs.LG", "cs.CL"],
        "citation_count": 100,
    }


@pytest.fixture
def sample_gap():
    """Sample gap result."""
    from src.services.gap_detector import GapResult, MethodInfo

    return GapResult(
        gap_id="gap_test123",
        is_gap=True,
        method_a=MethodInfo(
            method_id="contrastive_learning",
            name="Contrastive Learning",
            method_type="technique",
            paper_count=50,
        ),
        method_b=MethodInfo(
            method_id="gnn",
            name="Graph Neural Networks",
            method_type="architecture",
            paper_count=80,
        ),
        task="recommendation",
        gap_score=200.0,
        method_a_paper_count=50,
        method_b_paper_count=80,
        combination_paper_count=2,
    )


@pytest.fixture
def sample_hypothesis():
    """Sample hypothesis."""
    from src.services.hypothesis_generator import Assumption, EvaluationPlan, Hypothesis

    return Hypothesis(
        hypothesis_id="hyp_test123",
        hypothesis_text="Combining contrastive learning with GNNs will improve recommendation accuracy.",
        mechanism="Contrastive learning provides robust representations while GNNs capture collaborative signals.",
        assumptions=[
            Assumption(text="Contrastive objectives work on graphs"),
            Assumption(text="GNN aggregation preserves semantics"),
        ],
        evaluation_plan=EvaluationPlan(
            datasets=["Amazon-Book", "Yelp2018"],
            baselines=["LightGCN", "SimGCL"],
            metrics=["Recall@20", "NDCG@20"],
            expected_outcome="5-10% improvement",
        ),
        evidence_paper_ids=["2301.00001", "2302.00002"],
        gap_description="Gap in combining CL and GNN for recommendations",
    )
