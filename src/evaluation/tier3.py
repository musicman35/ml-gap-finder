"""Tier 3 evaluation: Human calibration and correlation analysis."""

from dataclasses import dataclass
from typing import Any

import structlog

from src.db.postgres import PostgresClient
from src.evaluation.tier2 import Tier2Scores

logger = structlog.get_logger()


@dataclass
class CorrelationResult:
    """Correlation analysis result."""

    spearman_rho: float
    p_value: float
    interpretation: str
    sample_size: int


@dataclass
class CalibrationSummary:
    """Summary of Tier 2/3 calibration."""

    coherence_correlation: CorrelationResult
    relevance_correlation: CorrelationResult
    specificity_correlation: CorrelationResult
    overall_status: str  # "validated", "moderate", "needs_iteration"


class Tier3Calibrator:
    """Tier 3: Human-in-the-loop calibration for Tier 2 scores."""

    def __init__(self, postgres_client: PostgresClient | None = None):
        """Initialize Tier 3 calibrator.

        Args:
            postgres_client: PostgreSQL client for storing annotations.
        """
        self.postgres = postgres_client

    def _interpret_rho(self, rho: float) -> str:
        """Interpret Spearman correlation coefficient.

        Args:
            rho: Spearman's rho value.

        Returns:
            Interpretation string.
        """
        abs_rho = abs(rho)
        if abs_rho > 0.7:
            return "Strong agreement - LLM scores validated"
        elif abs_rho >= 0.5:
            return "Moderate agreement - report with caveats"
        else:
            return "Weak agreement - iterate on prompts"

    def calculate_correlation(
        self,
        llm_scores: list[Tier2Scores],
        human_scores: list[int],
    ) -> dict[str, CorrelationResult]:
        """Calculate Spearman correlation between LLM and human ratings.

        Args:
            llm_scores: List of Tier 2 LLM scores.
            human_scores: Corresponding human ratings (1-5).

        Returns:
            Correlation results for each dimension.
        """
        try:
            from scipy.stats import spearmanr
        except ImportError:
            logger.error("scipy not installed, using placeholder correlation")
            return self._placeholder_correlation(len(llm_scores))

        if len(llm_scores) != len(human_scores):
            raise ValueError("LLM scores and human scores must have same length")

        if len(llm_scores) < 3:
            logger.warning("Too few samples for reliable correlation")
            return self._placeholder_correlation(len(llm_scores))

        correlations = {}

        for dimension in ['coherence', 'relevance', 'specificity']:
            llm_dim_scores = [getattr(s, dimension) for s in llm_scores]

            rho, p_value = spearmanr(llm_dim_scores, human_scores)

            correlations[dimension] = CorrelationResult(
                spearman_rho=round(rho, 4),
                p_value=round(p_value, 4),
                interpretation=self._interpret_rho(rho),
                sample_size=len(llm_scores),
            )

        return correlations

    def _placeholder_correlation(self, sample_size: int) -> dict[str, CorrelationResult]:
        """Return placeholder correlation when calculation not possible.

        Args:
            sample_size: Number of samples.

        Returns:
            Placeholder correlation results.
        """
        placeholder = CorrelationResult(
            spearman_rho=0.0,
            p_value=1.0,
            interpretation="Insufficient data for correlation",
            sample_size=sample_size,
        )
        return {
            "coherence": placeholder,
            "relevance": placeholder,
            "specificity": placeholder,
        }

    def get_calibration_summary(
        self,
        correlations: dict[str, CorrelationResult],
    ) -> CalibrationSummary:
        """Get overall calibration summary.

        Args:
            correlations: Correlation results by dimension.

        Returns:
            Calibration summary with overall status.
        """
        avg_rho = sum(c.spearman_rho for c in correlations.values()) / len(correlations)

        if avg_rho > 0.7:
            overall_status = "validated"
        elif avg_rho >= 0.5:
            overall_status = "moderate"
        else:
            overall_status = "needs_iteration"

        return CalibrationSummary(
            coherence_correlation=correlations["coherence"],
            relevance_correlation=correlations["relevance"],
            specificity_correlation=correlations["specificity"],
            overall_status=overall_status,
        )

    async def record_human_annotation(
        self,
        hypothesis_id: str,
        human_rating: int,
        rater_id: str,
        notes: str = "",
    ) -> None:
        """Record a human annotation for a hypothesis.

        Args:
            hypothesis_id: ID of the hypothesis.
            human_rating: Human rating (1-5).
            rater_id: Identifier for the human rater.
            notes: Optional notes from the rater.
        """
        if not self.postgres:
            logger.warning("No PostgreSQL client, annotation not stored")
            return

        await self.postgres.execute(
            """
            UPDATE hypotheses
            SET human_rating = %s, human_rater_id = %s
            WHERE id = %s
            """,
            (human_rating, rater_id, hypothesis_id),
        )

        logger.info(
            "Recorded human annotation",
            hypothesis_id=hypothesis_id,
            rating=human_rating,
            rater=rater_id,
        )

    async def get_hypotheses_for_annotation(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get hypotheses that need human annotation.

        Args:
            limit: Maximum number to return.

        Returns:
            List of hypotheses needing annotation.
        """
        if not self.postgres:
            logger.warning("No PostgreSQL client")
            return []

        results = await self.postgres.fetch_all(
            """
            SELECT id, hypothesis_text, mechanism, gap_description,
                   coherence_score, evidence_relevance_score, specificity_score
            FROM hypotheses
            WHERE human_rating IS NULL
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )

        return results

    async def run_calibration_analysis(self) -> CalibrationSummary | None:
        """Run full calibration analysis on annotated hypotheses.

        Returns:
            Calibration summary or None if insufficient data.
        """
        if not self.postgres:
            logger.warning("No PostgreSQL client")
            return None

        # Get hypotheses with both LLM and human scores
        results = await self.postgres.fetch_all(
            """
            SELECT coherence_score, evidence_relevance_score, specificity_score,
                   human_rating
            FROM hypotheses
            WHERE human_rating IS NOT NULL
              AND coherence_score IS NOT NULL
              AND evidence_relevance_score IS NOT NULL
              AND specificity_score IS NOT NULL
            """
        )

        if len(results) < 5:
            logger.warning("Insufficient annotated hypotheses for calibration")
            return None

        llm_scores = [
            Tier2Scores(
                coherence=r["coherence_score"],
                relevance=r["evidence_relevance_score"],
                specificity=r["specificity_score"],
            )
            for r in results
        ]
        human_scores = [r["human_rating"] for r in results]

        correlations = self.calculate_correlation(llm_scores, human_scores)
        summary = self.get_calibration_summary(correlations)

        logger.info(
            "Calibration analysis complete",
            sample_size=len(results),
            overall_status=summary.overall_status,
        )

        return summary

    def generate_annotation_interface_data(
        self,
        hypothesis: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate data for annotation interface.

        Args:
            hypothesis: Hypothesis data from database.

        Returns:
            Formatted data for annotation interface.
        """
        return {
            "id": hypothesis["id"],
            "hypothesis": hypothesis["hypothesis_text"],
            "mechanism": hypothesis.get("mechanism", ""),
            "gap": hypothesis.get("gap_description", ""),
            "llm_scores": {
                "coherence": hypothesis.get("coherence_score"),
                "evidence_relevance": hypothesis.get("evidence_relevance_score"),
                "specificity": hypothesis.get("specificity_score"),
            },
            "instructions": """
Please rate this hypothesis on a scale of 1-5:
1 = Poor - Incoherent or unsupported
2 = Below Average - Major issues
3 = Average - Some merit but significant gaps
4 = Good - Solid hypothesis with minor issues
5 = Excellent - Well-formed, actionable, well-supported

Consider:
- Is the hypothesis coherent and logical?
- Is there evidence supporting the claims?
- Is the hypothesis specific and actionable?
""",
        }
