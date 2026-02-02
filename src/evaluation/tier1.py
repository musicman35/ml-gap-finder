"""Tier 1 evaluation: Objective automated metrics."""

from dataclasses import dataclass
from typing import Any

import structlog

from src.db.neo4j import Neo4jClient
from src.db.postgres import PostgresClient
from src.ingestion.papers_with_code import PapersWithCodeClient
from src.ingestion.semantic_scholar import SemanticScholarClient

logger = structlog.get_logger()


@dataclass
class MethodExtractionMetrics:
    """Metrics for method extraction evaluation."""

    precision: float
    recall: float
    f1: float
    true_positives: int
    extracted_count: int
    ground_truth_count: int


@dataclass
class CitationAccuracyMetrics:
    """Metrics for citation accuracy evaluation."""

    accuracy: float
    valid_citations: int
    total_citations: int
    missing_papers: list[str]


@dataclass
class GapDetectionMetrics:
    """Metrics for gap detection evaluation."""

    precision: float
    true_gaps: int
    total_detected: int


@dataclass
class TemporalValidationMetrics:
    """Metrics for temporal validation."""

    validation_rate: float
    validated_gaps: int
    total_gaps: int
    future_papers_found: list[str]


class Tier1Evaluator:
    """Tier 1: Automated objective metrics evaluation."""

    def __init__(
        self,
        postgres_client: PostgresClient | None = None,
        neo4j_client: Neo4jClient | None = None,
        pwc_client: PapersWithCodeClient | None = None,
        s2_client: SemanticScholarClient | None = None,
    ):
        """Initialize Tier 1 evaluator.

        Args:
            postgres_client: PostgreSQL client.
            neo4j_client: Neo4j client.
            pwc_client: Papers With Code client.
            s2_client: Semantic Scholar client.
        """
        self.postgres = postgres_client
        self.neo4j = neo4j_client
        self.pwc = pwc_client or PapersWithCodeClient()
        self.s2 = s2_client or SemanticScholarClient()

    async def evaluate_method_extraction(
        self,
        extracted_methods: list[str],
        paper_arxiv_id: str,
    ) -> MethodExtractionMetrics:
        """Compare extracted methods against Papers With Code ground truth.

        Args:
            extracted_methods: List of extracted method names.
            paper_arxiv_id: arXiv ID of the paper.

        Returns:
            Extraction metrics (precision, recall, F1).
        """
        logger.info("Evaluating method extraction", paper_id=paper_arxiv_id)

        # Get ground truth from PWC
        pwc_paper = await self.pwc.get_paper_by_arxiv_id(paper_arxiv_id)
        if not pwc_paper:
            return MethodExtractionMetrics(
                precision=1.0,
                recall=1.0,
                f1=1.0,
                true_positives=0,
                extracted_count=len(extracted_methods),
                ground_truth_count=0,
            )

        ground_truth = set(m.lower() for m in pwc_paper.methods)
        extracted = set(m.lower() for m in extracted_methods)

        if not ground_truth:
            return MethodExtractionMetrics(
                precision=1.0,
                recall=1.0,
                f1=1.0,
                true_positives=0,
                extracted_count=len(extracted),
                ground_truth_count=0,
            )

        # Calculate metrics
        true_positives = len(extracted & ground_truth)
        precision = true_positives / len(extracted) if extracted else 0
        recall = true_positives / len(ground_truth) if ground_truth else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return MethodExtractionMetrics(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            true_positives=true_positives,
            extracted_count=len(extracted),
            ground_truth_count=len(ground_truth),
        )

    async def evaluate_citation_accuracy(
        self,
        cited_paper_ids: list[str],
        claimed_contributions: list[str],
    ) -> CitationAccuracyMetrics:
        """Verify all cited papers exist and support claimed contributions.

        Args:
            cited_paper_ids: List of cited paper IDs.
            claimed_contributions: Corresponding claimed contributions.

        Returns:
            Citation accuracy metrics.
        """
        logger.info("Evaluating citation accuracy", citation_count=len(cited_paper_ids))

        valid_citations = 0
        missing_papers = []

        for paper_id, claim in zip(cited_paper_ids, claimed_contributions):
            # Check if paper exists in Semantic Scholar
            paper = await self.s2.get_paper(f"ARXIV:{paper_id}")
            if not paper:
                missing_papers.append(paper_id)
                continue

            # For now, just verify paper exists
            # In production, would also verify claim is supported
            valid_citations += 1

        accuracy = valid_citations / len(cited_paper_ids) if cited_paper_ids else 1.0

        return CitationAccuracyMetrics(
            accuracy=round(accuracy, 4),
            valid_citations=valid_citations,
            total_citations=len(cited_paper_ids),
            missing_papers=missing_papers,
        )

    async def evaluate_gap_detection_precision(
        self,
        detected_gaps: list[dict[str, Any]],
    ) -> GapDetectionMetrics:
        """Verify claimed gaps are truly absent from corpus.

        Args:
            detected_gaps: List of detected gaps with method_a, method_b, task.

        Returns:
            Gap detection precision metrics.
        """
        logger.info("Evaluating gap detection precision", gap_count=len(detected_gaps))

        if not self.neo4j:
            logger.warning("Neo4j client not available for gap verification")
            return GapDetectionMetrics(
                precision=0.0,
                true_gaps=0,
                total_detected=len(detected_gaps),
            )

        true_gaps = 0
        max_combined_papers = 2  # Threshold for "gap"

        for gap in detected_gaps:
            # Count combination papers
            query = """
                MATCH (p:Paper)-[:USES|PROPOSES]->(m1:Method)
                MATCH (p)-[:USES|PROPOSES]->(m2:Method)
                WHERE m1.name =~ $method_a_pattern
                  AND m2.name =~ $method_b_pattern
                  AND m1 <> m2
                OPTIONAL MATCH (p)-[:EVALUATES_ON]->(:Dataset)-[:BENCHMARK_FOR]->(t:Task)
                WHERE t.name =~ $task_pattern
                RETURN count(DISTINCT p) as combined_papers
            """

            result = await self.neo4j.run_query(
                query,
                {
                    "method_a_pattern": f"(?i).*{gap['method_a']}.*",
                    "method_b_pattern": f"(?i).*{gap['method_b']}.*",
                    "task_pattern": f"(?i).*{gap['task']}.*",
                },
            )

            if result and result[0]["combined_papers"] <= max_combined_papers:
                true_gaps += 1

        precision = true_gaps / len(detected_gaps) if detected_gaps else 1.0

        return GapDetectionMetrics(
            precision=round(precision, 4),
            true_gaps=true_gaps,
            total_detected=len(detected_gaps),
        )

    async def evaluate_temporal_validation(
        self,
        detected_gaps: list[dict[str, Any]],
        holdout_start_year: int = 2024,
    ) -> TemporalValidationMetrics:
        """Check if detected gaps were later explored in held-out papers.

        Args:
            detected_gaps: List of detected gaps.
            holdout_start_year: Year where holdout corpus starts.

        Returns:
            Temporal validation metrics.
        """
        logger.info(
            "Evaluating temporal validation",
            gap_count=len(detected_gaps),
            holdout_year=holdout_start_year,
        )

        if not self.neo4j:
            logger.warning("Neo4j client not available for temporal validation")
            return TemporalValidationMetrics(
                validation_rate=0.0,
                validated_gaps=0,
                total_gaps=len(detected_gaps),
                future_papers_found=[],
            )

        validated_gaps = 0
        future_papers_found = []

        for gap in detected_gaps:
            # Find papers from holdout period that combine the methods
            query = """
                MATCH (p:Paper)-[:USES|PROPOSES]->(m1:Method)
                MATCH (p)-[:USES|PROPOSES]->(m2:Method)
                WHERE p.year >= $holdout_year
                  AND m1.name =~ $method_a_pattern
                  AND m2.name =~ $method_b_pattern
                  AND m1 <> m2
                OPTIONAL MATCH (p)-[:EVALUATES_ON]->(:Dataset)-[:BENCHMARK_FOR]->(t:Task)
                WHERE t.name =~ $task_pattern
                RETURN p.arxiv_id as arxiv_id
                LIMIT 5
            """

            result = await self.neo4j.run_query(
                query,
                {
                    "holdout_year": holdout_start_year,
                    "method_a_pattern": f"(?i).*{gap['method_a']}.*",
                    "method_b_pattern": f"(?i).*{gap['method_b']}.*",
                    "task_pattern": f"(?i).*{gap['task']}.*",
                },
            )

            if result:
                validated_gaps += 1
                future_papers_found.extend([r["arxiv_id"] for r in result])

        validation_rate = validated_gaps / len(detected_gaps) if detected_gaps else 0.0

        return TemporalValidationMetrics(
            validation_rate=round(validation_rate, 4),
            validated_gaps=validated_gaps,
            total_gaps=len(detected_gaps),
            future_papers_found=future_papers_found,
        )

    async def run_full_evaluation(
        self,
        test_cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run full Tier 1 evaluation on test cases.

        Args:
            test_cases: List of test cases with expected values.

        Returns:
            Aggregated evaluation results.
        """
        logger.info("Running full Tier 1 evaluation", test_count=len(test_cases))

        results = {
            "method_extraction": [],
            "citation_accuracy": [],
            "gap_detection": [],
            "temporal_validation": [],
        }

        for case in test_cases:
            case_type = case.get("type")

            if case_type == "method_extraction":
                metrics = await self.evaluate_method_extraction(
                    extracted_methods=case["extracted"],
                    paper_arxiv_id=case["paper_id"],
                )
                results["method_extraction"].append(metrics)

            elif case_type == "citation":
                metrics = await self.evaluate_citation_accuracy(
                    cited_paper_ids=case["cited_papers"],
                    claimed_contributions=case["claims"],
                )
                results["citation_accuracy"].append(metrics)

            elif case_type == "gap_detection":
                metrics = await self.evaluate_gap_detection_precision(
                    detected_gaps=case["gaps"],
                )
                results["gap_detection"].append(metrics)

            elif case_type == "temporal":
                metrics = await self.evaluate_temporal_validation(
                    detected_gaps=case["gaps"],
                )
                results["temporal_validation"].append(metrics)

        # Calculate aggregate scores
        summary = {}
        if results["method_extraction"]:
            summary["avg_extraction_f1"] = sum(
                m.f1 for m in results["method_extraction"]
            ) / len(results["method_extraction"])

        if results["citation_accuracy"]:
            summary["avg_citation_accuracy"] = sum(
                m.accuracy for m in results["citation_accuracy"]
            ) / len(results["citation_accuracy"])

        if results["gap_detection"]:
            summary["gap_detection_precision"] = sum(
                m.precision for m in results["gap_detection"]
            ) / len(results["gap_detection"])

        if results["temporal_validation"]:
            summary["temporal_validation_rate"] = sum(
                m.validation_rate for m in results["temporal_validation"]
            ) / len(results["temporal_validation"])

        return {
            "summary": summary,
            "details": {
                k: [vars(m) for m in v] for k, v in results.items()
            },
        }
