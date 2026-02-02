"""Hypothesis generation service for research gaps."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from src.db.postgres import PostgresClient
from src.llm.client import BaseLLMClient, get_llm_client
from src.llm.prompts import PromptTemplates
from src.services.gap_detector import GapResult
from src.services.evidence_retriever import EvidenceBundle

logger = structlog.get_logger()


@dataclass
class Assumption:
    """A hypothesis assumption with evidence."""

    text: str
    evidence_paper_id: str | None = None
    evidence_excerpt: str | None = None


@dataclass
class EvaluationPlan:
    """Suggested evaluation plan for a hypothesis."""

    datasets: list[str] = field(default_factory=list)
    baselines: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    expected_outcome: str = ""


@dataclass
class Hypothesis:
    """A generated research hypothesis."""

    hypothesis_id: str
    hypothesis_text: str
    mechanism: str
    assumptions: list[Assumption]
    evaluation_plan: EvaluationPlan
    evidence_paper_ids: list[str]
    gap_description: str

    # Tier 2 scores (filled in during evaluation)
    coherence_score: int | None = None
    evidence_relevance_score: int | None = None
    specificity_score: int | None = None

    created_at: datetime = field(default_factory=datetime.now)
    model_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_text": self.hypothesis_text,
            "mechanism": self.mechanism,
            "assumptions": [
                {
                    "text": a.text,
                    "evidence_paper_id": a.evidence_paper_id,
                    "evidence_excerpt": a.evidence_excerpt,
                }
                for a in self.assumptions
            ],
            "evaluation_plan": {
                "datasets": self.evaluation_plan.datasets,
                "baselines": self.evaluation_plan.baselines,
                "metrics": self.evaluation_plan.metrics,
                "expected_outcome": self.evaluation_plan.expected_outcome,
            },
            "evidence_paper_ids": self.evidence_paper_ids,
            "gap_description": self.gap_description,
            "coherence_score": self.coherence_score,
            "evidence_relevance_score": self.evidence_relevance_score,
            "specificity_score": self.specificity_score,
            "created_at": self.created_at.isoformat(),
            "model_version": self.model_version,
        }


class HypothesisGeneratorService:
    """Service for generating evidence-grounded research hypotheses."""

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        postgres_client: PostgresClient | None = None,
    ):
        """Initialize hypothesis generator.

        Args:
            llm_client: LLM client for generation.
            postgres_client: PostgreSQL client for storing hypotheses.
        """
        self._llm_client = llm_client
        self.postgres = postgres_client

    @property
    def llm_client(self) -> BaseLLMClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    async def generate_hypothesis(
        self,
        gap: GapResult,
        evidence_bundle: EvidenceBundle,
        task_sota: str = "",
    ) -> Hypothesis:
        """Generate a hypothesis for a given gap with supporting evidence.

        Args:
            gap: Detected research gap.
            evidence_bundle: Evidence for the methods.
            task_sota: Current state-of-the-art description.

        Returns:
            Generated hypothesis.
        """
        logger.info(
            "Generating hypothesis",
            gap_id=gap.gap_id,
            method_a=gap.method_a.name,
            method_b=gap.method_b.name,
        )

        # Format evidence for the prompt
        method_a_evidence = self._format_evidence(
            [p for p in evidence_bundle.papers[:3]]
        )
        method_b_evidence = self._format_evidence(
            [p for p in evidence_bundle.papers[3:6]]
        )

        gap_description = (
            f"Combining {gap.method_a.name} and {gap.method_b.name} for {gap.task}. "
            f"{gap.method_a.name} has been used in {gap.method_a_paper_count} papers, "
            f"{gap.method_b.name} in {gap.method_b_paper_count} papers, "
            f"but only {gap.combination_paper_count} papers combine both."
        )

        # Generate hypothesis using LLM
        prompt = PromptTemplates.format_hypothesis_prompt(
            gap_description=gap_description,
            method_a_name=gap.method_a.name,
            method_a_evidence=method_a_evidence,
            method_b_name=gap.method_b.name,
            method_b_evidence=method_b_evidence,
            task_name=gap.task,
            task_sota=task_sota or "Various approaches have been proposed.",
        )

        response = await self.llm_client.generate(
            prompt=prompt,
            system="You are an expert ML researcher generating research hypotheses. Be specific and cite evidence.",
            temperature=0.7,
        )

        # Parse the response
        hypothesis = self._parse_hypothesis_response(
            response=response,
            gap=gap,
            evidence_bundle=evidence_bundle,
        )

        # Store in database if available
        if self.postgres:
            try:
                await self.postgres.insert_hypothesis({
                    "gap_description": hypothesis.gap_description,
                    "hypothesis_text": hypothesis.hypothesis_text,
                    "mechanism": hypothesis.mechanism,
                    "assumptions": [a.text for a in hypothesis.assumptions],
                    "evidence_paper_ids": hypothesis.evidence_paper_ids,
                    "model_version": hypothesis.model_version,
                })
            except Exception as e:
                logger.warning("Failed to store hypothesis", error=str(e))

        logger.info("Hypothesis generated", hypothesis_id=hypothesis.hypothesis_id)
        return hypothesis

    def _format_evidence(self, papers) -> str:
        """Format papers as evidence text.

        Args:
            papers: List of paper evidence.

        Returns:
            Formatted evidence string.
        """
        if not papers:
            return "No specific evidence available."

        lines = []
        for p in papers:
            lines.append(f"- {p.title} ({p.arxiv_id}): {p.excerpt[:200]}...")
        return "\n".join(lines)

    def _parse_hypothesis_response(
        self,
        response: str,
        gap: GapResult,
        evidence_bundle: EvidenceBundle,
    ) -> Hypothesis:
        """Parse LLM response into structured hypothesis.

        Args:
            response: LLM response text.
            gap: Original gap.
            evidence_bundle: Evidence bundle.

        Returns:
            Parsed hypothesis.
        """
        import uuid

        # Extract sections using regex
        hypothesis_match = re.search(
            r"### Hypothesis\n(.*?)(?=###|\Z)",
            response,
            re.DOTALL,
        )
        mechanism_match = re.search(
            r"### Proposed Mechanism\n(.*?)(?=###|\Z)",
            response,
            re.DOTALL,
        )
        assumptions_match = re.search(
            r"### Key Assumptions\n(.*?)(?=###|\Z)",
            response,
            re.DOTALL,
        )
        eval_match = re.search(
            r"### Evaluation Plan\n(.*?)(?=###|\Z)",
            response,
            re.DOTALL,
        )

        hypothesis_text = (
            hypothesis_match.group(1).strip() if hypothesis_match else response[:500]
        )
        mechanism = mechanism_match.group(1).strip() if mechanism_match else ""

        # Parse assumptions
        assumptions = []
        if assumptions_match:
            assumption_lines = assumptions_match.group(1).strip().split("\n")
            for line in assumption_lines:
                if line.strip().startswith(("1.", "2.", "3.", "-")):
                    text = re.sub(r"^\d+\.\s*|-\s*", "", line).strip()
                    if text:
                        assumptions.append(Assumption(text=text))

        # Parse evaluation plan
        evaluation_plan = EvaluationPlan()
        if eval_match:
            eval_text = eval_match.group(1)
            dataset_match = re.search(r"Dataset[s]?:\s*(.+)", eval_text, re.IGNORECASE)
            baselines_match = re.search(r"Baseline[s]?:\s*(.+)", eval_text, re.IGNORECASE)
            metrics_match = re.search(r"Metric[s]?:\s*(.+)", eval_text, re.IGNORECASE)
            outcome_match = re.search(r"Expected outcome:\s*(.+)", eval_text, re.IGNORECASE)

            if dataset_match:
                evaluation_plan.datasets = [
                    d.strip() for d in dataset_match.group(1).split(",")
                ]
            if baselines_match:
                evaluation_plan.baselines = [
                    b.strip() for b in baselines_match.group(1).split(",")
                ]
            if metrics_match:
                evaluation_plan.metrics = [
                    m.strip() for m in metrics_match.group(1).split(",")
                ]
            if outcome_match:
                evaluation_plan.expected_outcome = outcome_match.group(1).strip()

        # Collect evidence paper IDs
        evidence_paper_ids = [p.arxiv_id for p in evidence_bundle.papers]

        return Hypothesis(
            hypothesis_id=str(uuid.uuid4())[:12],
            hypothesis_text=hypothesis_text,
            mechanism=mechanism,
            assumptions=assumptions if assumptions else [Assumption(text="See full response")],
            evaluation_plan=evaluation_plan,
            evidence_paper_ids=evidence_paper_ids,
            gap_description=f"Gap: {gap.method_a.name} + {gap.method_b.name} for {gap.task}",
            model_version=f"{self.llm_client.__class__.__name__}",
        )

    async def get_hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        """Get a previously generated hypothesis.

        Args:
            hypothesis_id: Hypothesis ID.

        Returns:
            Hypothesis or None if not found.
        """
        if not self.postgres:
            return None

        result = await self.postgres.fetch_one(
            "SELECT * FROM hypotheses WHERE id = %s",
            (hypothesis_id,),
        )

        if not result:
            return None

        return Hypothesis(
            hypothesis_id=str(result["id"]),
            hypothesis_text=result["hypothesis_text"],
            mechanism=result.get("mechanism", ""),
            assumptions=[
                Assumption(text=a) for a in result.get("assumptions", [])
            ],
            evaluation_plan=EvaluationPlan(),
            evidence_paper_ids=result.get("evidence_paper_ids", []),
            gap_description=result["gap_description"],
            coherence_score=result.get("coherence_score"),
            evidence_relevance_score=result.get("evidence_relevance_score"),
            specificity_score=result.get("specificity_score"),
            model_version=result.get("model_version", ""),
        )
