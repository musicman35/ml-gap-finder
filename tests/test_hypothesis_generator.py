"""Tests for hypothesis generator service."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.services.hypothesis_generator import (
    HypothesisGeneratorService,
    Hypothesis,
    Assumption,
    EvaluationPlan,
)
from src.services.gap_detector import GapResult, MethodInfo
from src.services.evidence_retriever import EvidenceBundle, PaperEvidence


@pytest.fixture
def hypothesis_generator(mock_llm_client, mock_postgres):
    """Create hypothesis generator with mocked dependencies."""
    generator = HypothesisGeneratorService(
        llm_client=mock_llm_client,
        postgres_client=mock_postgres,
    )
    return generator


@pytest.fixture
def sample_evidence_bundle():
    """Sample evidence bundle."""
    return EvidenceBundle(
        papers=[
            PaperEvidence(
                arxiv_id="2301.00001",
                title="Paper on Contrastive Learning",
                year=2023,
                citation_count=100,
                excerpt="We show that contrastive learning improves...",
                relevance_score=0.9,
            ),
            PaperEvidence(
                arxiv_id="2302.00002",
                title="Paper on GNNs",
                year=2023,
                citation_count=80,
                excerpt="Graph neural networks excel at...",
                relevance_score=0.85,
            ),
        ],
        confidence=0.8,
        claim_support_strength="strong",
    )


class TestHypothesisGeneratorService:
    """Tests for HypothesisGeneratorService."""

    @pytest.mark.asyncio
    async def test_generate_hypothesis_returns_structured_output(
        self,
        hypothesis_generator,
        sample_gap,
        sample_evidence_bundle,
        mock_llm_client,
    ):
        """Test that generated hypothesis has correct structure."""
        # Mock LLM response
        mock_llm_client.generate.return_value = """
### Hypothesis
Combining contrastive learning with GNNs will improve recommendation accuracy.

### Proposed Mechanism
Contrastive learning provides robust representations. GNNs capture graph structure.

### Key Assumptions
1. Contrastive objectives work on graphs - Evidence: Paper 2301.00001
2. GNN aggregation preserves semantics - Evidence: Paper 2302.00002

### Evaluation Plan
- Dataset: Amazon-Book, Yelp2018
- Baselines: LightGCN, SimGCL
- Metrics: Recall@20, NDCG@20
- Expected outcome: 5-10% improvement
"""

        result = await hypothesis_generator.generate_hypothesis(
            gap=sample_gap,
            evidence_bundle=sample_evidence_bundle,
        )

        assert isinstance(result, Hypothesis)
        assert "contrastive" in result.hypothesis_text.lower() or "gnn" in result.hypothesis_text.lower()
        assert len(result.assumptions) > 0
        assert result.hypothesis_id is not None

    @pytest.mark.asyncio
    async def test_generate_hypothesis_stores_in_database(
        self,
        hypothesis_generator,
        sample_gap,
        sample_evidence_bundle,
        mock_llm_client,
        mock_postgres,
    ):
        """Test that hypothesis is stored in database."""
        mock_llm_client.generate.return_value = "### Hypothesis\nTest hypothesis"
        mock_postgres.insert_hypothesis.return_value = 1

        await hypothesis_generator.generate_hypothesis(
            gap=sample_gap,
            evidence_bundle=sample_evidence_bundle,
        )

        mock_postgres.insert_hypothesis.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_hypothesis_handles_llm_errors(
        self,
        hypothesis_generator,
        sample_gap,
        sample_evidence_bundle,
        mock_llm_client,
    ):
        """Test graceful handling of LLM errors."""
        mock_llm_client.generate.side_effect = Exception("LLM error")

        with pytest.raises(Exception, match="LLM error"):
            await hypothesis_generator.generate_hypothesis(
                gap=sample_gap,
                evidence_bundle=sample_evidence_bundle,
            )

    def test_format_evidence(self, hypothesis_generator):
        """Test evidence formatting."""
        papers = [
            MagicMock(title="Test Paper", arxiv_id="2301.00001", excerpt="Test excerpt..."),
        ]

        result = hypothesis_generator._format_evidence(papers)

        assert "Test Paper" in result
        assert "2301.00001" in result

    def test_hypothesis_to_dict(self, sample_hypothesis):
        """Test Hypothesis serialization."""
        result = sample_hypothesis.to_dict()

        assert result["hypothesis_id"] == "hyp_test123"
        assert "contrastive" in result["hypothesis_text"].lower()
        assert len(result["assumptions"]) == 2
        assert result["evaluation_plan"]["datasets"] == ["Amazon-Book", "Yelp2018"]


class TestHypothesisParsing:
    """Tests for hypothesis response parsing."""

    def test_parse_hypothesis_extracts_sections(self, hypothesis_generator, sample_gap, sample_evidence_bundle):
        """Test that sections are correctly extracted from LLM response."""
        response = """
### Hypothesis
Test hypothesis statement.

### Proposed Mechanism
This is the mechanism explanation.

### Key Assumptions
1. First assumption
2. Second assumption

### Evaluation Plan
- Dataset: TestDataset
- Baselines: Baseline1
- Metrics: Accuracy
- Expected outcome: Better results
"""

        result = hypothesis_generator._parse_hypothesis_response(
            response=response,
            gap=sample_gap,
            evidence_bundle=sample_evidence_bundle,
        )

        assert "Test hypothesis" in result.hypothesis_text
        assert "mechanism" in result.mechanism.lower()
        assert len(result.assumptions) >= 2

    def test_parse_hypothesis_handles_malformed_response(
        self,
        hypothesis_generator,
        sample_gap,
        sample_evidence_bundle,
    ):
        """Test handling of malformed LLM responses."""
        response = "Just a simple response without sections"

        result = hypothesis_generator._parse_hypothesis_response(
            response=response,
            gap=sample_gap,
            evidence_bundle=sample_evidence_bundle,
        )

        # Should still return a valid hypothesis with the response text
        assert result.hypothesis_text == response
        assert result.hypothesis_id is not None
