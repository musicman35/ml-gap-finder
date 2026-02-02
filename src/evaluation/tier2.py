"""Tier 2 evaluation: LLM-as-judge proxy metrics."""

import re
from dataclasses import dataclass
from typing import Any

import structlog

from src.llm.client import BaseLLMClient, get_llm_client
from src.llm.prompts import PromptTemplates
from src.services.hypothesis_generator import Hypothesis

logger = structlog.get_logger()


@dataclass
class Tier2Scores:
    """Tier 2 evaluation scores."""

    coherence: int
    relevance: int
    specificity: int

    @property
    def average(self) -> float:
        """Calculate average score."""
        return (self.coherence + self.relevance + self.specificity) / 3

    def passes_threshold(self, threshold: float = 3.5) -> bool:
        """Check if average passes threshold."""
        return self.average >= threshold

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "coherence": self.coherence,
            "relevance": self.relevance,
            "specificity": self.specificity,
            "average": self.average,
        }


class Tier2Evaluator:
    """Tier 2: LLM-as-judge evaluation for hypothesis quality.

    Works with both Anthropic and Ollama backends via the LLM client abstraction.
    """

    def __init__(self, llm_client: BaseLLMClient | None = None):
        """Initialize Tier 2 evaluator.

        Args:
            llm_client: Optional LLM client. If None, uses configured provider.
        """
        self._llm_client = llm_client

    @property
    def llm_client(self) -> BaseLLMClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    async def _rate_dimension(
        self,
        prompt: str,
        dimension: str,
    ) -> int:
        """Rate a single dimension using LLM.

        Args:
            prompt: Formatted prompt for rating.
            dimension: Name of the dimension being rated.

        Returns:
            Rating from 1-5.
        """
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                system="You are an expert research evaluator. Be objective and critical. Always start your response with a number from 1-5.",
                temperature=0.3,  # Lower temperature for consistent ratings
            )

            # Extract numeric rating from response
            # Look for patterns like "4 - ", "4.", "[4]", etc.
            match = re.search(r"^[^\d]*(\d)[^\d]", response.strip())
            if match:
                rating = int(match.group(1))
                return max(1, min(5, rating))  # Clamp to 1-5

            # Try to find any digit in the first 20 characters
            match = re.search(r"(\d)", response[:20])
            if match:
                rating = int(match.group(1))
                return max(1, min(5, rating))

            logger.warning(
                "Could not parse rating",
                dimension=dimension,
                response_start=response[:50],
            )
            return 3  # Default to middle if parsing fails

        except Exception as e:
            logger.error(
                "Failed to rate dimension",
                dimension=dimension,
                error=str(e),
            )
            return 3

    async def rate_coherence(self, hypothesis: Hypothesis) -> int:
        """Rate the coherence of a hypothesis.

        Args:
            hypothesis: Hypothesis to evaluate.

        Returns:
            Coherence rating (1-5).
        """
        assumptions_text = "\n".join([a.text for a in hypothesis.assumptions])
        prompt = PromptTemplates.format_coherence_prompt(
            hypothesis=hypothesis.hypothesis_text,
            mechanism=hypothesis.mechanism,
            assumptions=assumptions_text,
        )
        return await self._rate_dimension(prompt, "coherence")

    async def rate_evidence_relevance(self, hypothesis: Hypothesis) -> int:
        """Rate the evidence relevance of a hypothesis.

        Args:
            hypothesis: Hypothesis to evaluate.

        Returns:
            Evidence relevance rating (1-5).
        """
        # Format evidence summaries
        evidence_text = "\n".join([
            f"- Paper {paper_id}" for paper_id in hypothesis.evidence_paper_ids[:5]
        ])
        if not evidence_text:
            evidence_text = "No evidence papers cited."

        prompt = PromptTemplates.format_evidence_relevance_prompt(
            hypothesis=hypothesis.hypothesis_text,
            evidence_summaries=evidence_text,
        )
        return await self._rate_dimension(prompt, "relevance")

    async def rate_specificity(self, hypothesis: Hypothesis) -> int:
        """Rate the specificity/actionability of a hypothesis.

        Args:
            hypothesis: Hypothesis to evaluate.

        Returns:
            Specificity rating (1-5).
        """
        eval_plan_text = (
            f"Datasets: {', '.join(hypothesis.evaluation_plan.datasets)}\n"
            f"Baselines: {', '.join(hypothesis.evaluation_plan.baselines)}\n"
            f"Metrics: {', '.join(hypothesis.evaluation_plan.metrics)}\n"
            f"Expected outcome: {hypothesis.evaluation_plan.expected_outcome}"
        )
        prompt = PromptTemplates.format_specificity_prompt(
            hypothesis=hypothesis.hypothesis_text,
            evaluation_plan=eval_plan_text,
        )
        return await self._rate_dimension(prompt, "specificity")

    async def evaluate(self, hypothesis: Hypothesis) -> Tier2Scores:
        """Evaluate hypothesis across all dimensions.

        Args:
            hypothesis: Hypothesis to evaluate.

        Returns:
            Tier2Scores with ratings for all dimensions.
        """
        logger.info("Running Tier 2 evaluation", hypothesis_id=hypothesis.hypothesis_id)

        # Rate all dimensions
        coherence = await self.rate_coherence(hypothesis)
        relevance = await self.rate_evidence_relevance(hypothesis)
        specificity = await self.rate_specificity(hypothesis)

        scores = Tier2Scores(
            coherence=coherence,
            relevance=relevance,
            specificity=specificity,
        )

        logger.info(
            "Tier 2 evaluation complete",
            hypothesis_id=hypothesis.hypothesis_id,
            scores=scores.to_dict(),
            passes=scores.passes_threshold(),
        )

        return scores

    async def evaluate_batch(
        self,
        hypotheses: list[Hypothesis],
    ) -> list[tuple[Hypothesis, Tier2Scores]]:
        """Evaluate a batch of hypotheses.

        Args:
            hypotheses: List of hypotheses to evaluate.

        Returns:
            List of (hypothesis, scores) tuples.
        """
        results = []
        for hypothesis in hypotheses:
            scores = await self.evaluate(hypothesis)
            results.append((hypothesis, scores))
        return results

    async def get_detailed_feedback(
        self,
        hypothesis: Hypothesis,
    ) -> dict[str, Any]:
        """Get detailed feedback for a hypothesis.

        Args:
            hypothesis: Hypothesis to evaluate.

        Returns:
            Dictionary with scores and detailed feedback.
        """
        scores = await self.evaluate(hypothesis)

        prompt = f"""Provide a brief (2-3 sentence) critique of this research hypothesis:

Hypothesis: {hypothesis.hypothesis_text}
Mechanism: {hypothesis.mechanism}

Focus on:
1. What's strong about this hypothesis
2. What could be improved
3. Key assumptions that need validation
"""

        try:
            feedback = await self.llm_client.generate(
                prompt=prompt,
                system="You are a senior ML researcher reviewing hypothesis quality.",
                temperature=0.5,
            )
        except Exception as e:
            feedback = f"Unable to generate feedback: {e}"

        return {
            "scores": scores.to_dict(),
            "passes_threshold": scores.passes_threshold(),
            "feedback": feedback,
        }
