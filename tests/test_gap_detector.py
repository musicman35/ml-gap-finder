"""Tests for gap detector service."""

import pytest
from unittest.mock import AsyncMock

from src.services.gap_detector import GapDetectorService, GapResult, MethodInfo


@pytest.fixture
def gap_detector(mock_neo4j, mock_redis):
    """Create gap detector with mocked dependencies."""
    return GapDetectorService(neo4j_client=mock_neo4j, cache=mock_redis)


class TestGapDetectorService:
    """Tests for GapDetectorService."""

    @pytest.mark.asyncio
    async def test_find_gaps_returns_gap_when_combination_rare(self, gap_detector, mock_neo4j):
        """Test that a gap is detected when method combination is rare."""
        # Mock individual method paper counts
        mock_neo4j.run_query.side_effect = [
            [{"method_id": "m1", "name": "Method A", "type": "technique", "paper_count": 50}],
            [{"method_id": "m2", "name": "Method B", "type": "architecture", "paper_count": 80}],
            [{"combined_papers": 1}],
        ]

        result = await gap_detector.find_gaps(
            method_a="Method A",
            method_b="Method B",
            task="classification",
        )

        assert result.is_gap is True
        assert result.gap_score > 0
        assert result.method_a_paper_count == 50
        assert result.method_b_paper_count == 80
        assert result.combination_paper_count == 1

    @pytest.mark.asyncio
    async def test_find_gaps_returns_no_gap_when_combination_common(self, gap_detector, mock_neo4j):
        """Test that no gap is detected when methods are commonly combined."""
        mock_neo4j.run_query.side_effect = [
            [{"method_id": "m1", "name": "Method A", "type": "technique", "paper_count": 50}],
            [{"method_id": "m2", "name": "Method B", "type": "architecture", "paper_count": 80}],
            [{"combined_papers": 20}],  # Many combined papers
        ]

        result = await gap_detector.find_gaps(
            method_a="Method A",
            method_b="Method B",
            task="classification",
            max_combination_papers=2,
        )

        assert result.is_gap is False
        assert result.gap_score == 0

    @pytest.mark.asyncio
    async def test_find_gaps_uses_cache(self, gap_detector, mock_redis):
        """Test that cached results are returned when available."""
        cached_result = {
            "gap_id": "cached_gap",
            "is_gap": True,
            "method_a": {"method_id": "m1", "name": "Method A", "method_type": "technique", "paper_count": 50},
            "method_b": {"method_id": "m2", "name": "Method B", "method_type": "architecture", "paper_count": 80},
            "task": "classification",
            "gap_score": 200.0,
            "method_a_paper_count": 50,
            "method_b_paper_count": 80,
            "combination_paper_count": 2,
            "evidence_summary": {},
        }
        mock_redis.get_gap_result.return_value = cached_result

        result = await gap_detector.find_gaps(
            method_a="Method A",
            method_b="Method B",
            task="classification",
        )

        assert result.gap_id == "cached_gap"
        mock_redis.get_gap_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_discover_gaps(self, gap_detector, mock_neo4j):
        """Test gap discovery for a task."""
        mock_neo4j.find_method_gaps.return_value = [
            {
                "method_1": "Transformer",
                "method_1_id": "transformer",
                "method_2": "CNN",
                "method_2_id": "cnn",
                "task": "classification",
                "papers_m1": 100,
                "papers_m2": 80,
                "combined_papers": 1,
                "gap_score": 4000.0,
            },
        ]

        gaps = await gap_detector.discover_gaps(task="classification", top_k=5)

        assert len(gaps) == 1
        assert gaps[0].method_a.name == "Transformer"
        assert gaps[0].method_b.name == "CNN"
        assert gaps[0].is_gap is True

    def test_gap_id_generation_is_consistent(self, gap_detector):
        """Test that gap IDs are consistently generated."""
        gap_id_1 = gap_detector._generate_gap_id("method_a", "method_b", "task")
        gap_id_2 = gap_detector._generate_gap_id("method_a", "method_b", "task")
        gap_id_reversed = gap_detector._generate_gap_id("method_b", "method_a", "task")

        assert gap_id_1 == gap_id_2
        assert gap_id_1 == gap_id_reversed  # Order shouldn't matter

    def test_gap_result_to_dict(self, sample_gap):
        """Test GapResult serialization."""
        result_dict = sample_gap.to_dict()

        assert result_dict["gap_id"] == "gap_test123"
        assert result_dict["is_gap"] is True
        assert result_dict["method_a"]["name"] == "Contrastive Learning"
        assert result_dict["method_b"]["name"] == "Graph Neural Networks"
        assert result_dict["task"] == "recommendation"
