#!/usr/bin/env python3
"""Evaluation runner for ML Gap Finder.

Runs evaluation tiers and generates reports.

Usage:
    # Run Tier 1 (objective metrics)
    uv run python scripts/run_evaluation.py --tier=1

    # Run Tier 2 (LLM-as-judge)
    uv run python scripts/run_evaluation.py --tier=2

    # Run all tiers
    uv run python scripts/run_evaluation.py --tier=all

    # Run with specific test set
    uv run python scripts/run_evaluation.py --tier=1 --test-set=data/test_set/config.yaml
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from config.settings import settings
from src.db.postgres import PostgresClient
from src.db.neo4j import Neo4jClient
from src.evaluation.tier1 import Tier1Evaluator
from src.evaluation.tier2 import Tier2Evaluator, Tier2Scores
from src.evaluation.tier3 import Tier3Calibrator
from src.services.hypothesis_generator import Hypothesis, Assumption, EvaluationPlan

logger = structlog.get_logger()


# Sample test cases for demonstration
SAMPLE_TEST_CASES = {
    "tier1": [
        {
            "type": "method_extraction",
            "paper_id": "2301.00001",
            "extracted": ["transformer", "attention mechanism", "BERT"],
        },
        {
            "type": "gap_detection",
            "gaps": [
                {"method_a": "contrastive learning", "method_b": "GNN", "task": "recommendation"},
                {"method_a": "transformer", "method_b": "CNN", "task": "image classification"},
            ],
        },
    ],
    "tier2": [
        {
            "hypothesis_text": "Combining contrastive learning with graph neural networks will improve recommendation accuracy by leveraging both collaborative filtering signals and representation learning.",
            "mechanism": "Contrastive learning provides robust representations by maximizing agreement between augmented views, while GNNs capture high-order collaborative signals.",
            "assumptions": ["Contrastive objectives can be applied to graph-structured data", "GNN message passing preserves collaborative filtering signals"],
            "evaluation_plan": {
                "datasets": ["Amazon-Book", "Yelp2018"],
                "baselines": ["LightGCN", "SimGCL"],
                "metrics": ["Recall@20", "NDCG@20"],
                "expected_outcome": "5-10% relative improvement",
            },
        },
    ],
}


async def run_tier1_evaluation(
    test_cases: list | None = None,
) -> dict:
    """Run Tier 1 objective metrics evaluation.

    Args:
        test_cases: Optional test cases. Uses samples if None.

    Returns:
        Evaluation results.
    """
    logger.info("Running Tier 1 evaluation")

    async with PostgresClient() as postgres:
        async with Neo4jClient() as neo4j:
            evaluator = Tier1Evaluator(
                postgres_client=postgres,
                neo4j_client=neo4j,
            )

            cases = test_cases or SAMPLE_TEST_CASES["tier1"]
            results = await evaluator.run_full_evaluation(cases)

            logger.info("Tier 1 evaluation complete", summary=results["summary"])
            return results


async def run_tier2_evaluation(
    test_cases: list | None = None,
) -> dict:
    """Run Tier 2 LLM-as-judge evaluation.

    Args:
        test_cases: Optional test cases. Uses samples if None.

    Returns:
        Evaluation results.
    """
    logger.info("Running Tier 2 evaluation", llm_provider=settings.llm.provider.value)

    evaluator = Tier2Evaluator()
    cases = test_cases or SAMPLE_TEST_CASES["tier2"]

    results = {
        "hypotheses": [],
        "summary": {},
    }

    all_scores = []
    for case in cases:
        # Create hypothesis object
        hypothesis = Hypothesis(
            hypothesis_id="test",
            hypothesis_text=case["hypothesis_text"],
            mechanism=case["mechanism"],
            assumptions=[Assumption(text=a) for a in case.get("assumptions", [])],
            evaluation_plan=EvaluationPlan(**case.get("evaluation_plan", {})),
            evidence_paper_ids=[],
            gap_description="Test gap",
        )

        scores = await evaluator.evaluate(hypothesis)
        all_scores.append(scores)

        results["hypotheses"].append({
            "hypothesis": case["hypothesis_text"][:100] + "...",
            "scores": scores.to_dict(),
            "passes": scores.passes_threshold(),
        })

    # Calculate summary
    if all_scores:
        results["summary"] = {
            "avg_coherence": sum(s.coherence for s in all_scores) / len(all_scores),
            "avg_relevance": sum(s.relevance for s in all_scores) / len(all_scores),
            "avg_specificity": sum(s.specificity for s in all_scores) / len(all_scores),
            "avg_overall": sum(s.average for s in all_scores) / len(all_scores),
            "pass_rate": sum(1 for s in all_scores if s.passes_threshold()) / len(all_scores),
        }

    logger.info("Tier 2 evaluation complete", summary=results["summary"])
    return results


async def run_tier3_calibration() -> dict:
    """Run Tier 3 calibration analysis.

    Returns:
        Calibration results.
    """
    logger.info("Running Tier 3 calibration")

    async with PostgresClient() as postgres:
        calibrator = Tier3Calibrator(postgres_client=postgres)

        summary = await calibrator.run_calibration_analysis()

        if summary:
            return {
                "status": summary.overall_status,
                "correlations": {
                    "coherence": {
                        "rho": summary.coherence_correlation.spearman_rho,
                        "interpretation": summary.coherence_correlation.interpretation,
                    },
                    "relevance": {
                        "rho": summary.relevance_correlation.spearman_rho,
                        "interpretation": summary.relevance_correlation.interpretation,
                    },
                    "specificity": {
                        "rho": summary.specificity_correlation.spearman_rho,
                        "interpretation": summary.specificity_correlation.interpretation,
                    },
                },
            }
        else:
            return {
                "status": "insufficient_data",
                "message": "Need more human annotations for calibration",
            }


async def log_evaluation_run(
    tier: str,
    results: dict,
) -> None:
    """Log evaluation run to database.

    Args:
        tier: Evaluation tier.
        results: Evaluation results.
    """
    try:
        async with PostgresClient() as postgres:
            summary = results.get("summary", {})
            for metric_name, metric_value in summary.items():
                await postgres.execute(
                    """
                    INSERT INTO evaluation_runs
                    (test_category, metric_name, metric_value, model_version)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (f"tier{tier}", metric_name, metric_value, settings.llm.provider.value),
                )
    except Exception as e:
        logger.warning("Failed to log evaluation run", error=str(e))


def generate_report(
    results: dict,
    output_path: str | None = None,
) -> str:
    """Generate evaluation report.

    Args:
        results: Evaluation results.
        output_path: Optional path to save report.

    Returns:
        Report as string.
    """
    report_lines = [
        "=" * 60,
        "ML Gap Finder Evaluation Report",
        f"Generated: {datetime.now().isoformat()}",
        f"LLM Provider: {settings.llm.provider.value}",
        "=" * 60,
        "",
    ]

    for tier, tier_results in results.items():
        report_lines.append(f"\n## {tier.upper()}\n")

        if "summary" in tier_results:
            report_lines.append("### Summary")
            for key, value in tier_results["summary"].items():
                if isinstance(value, float):
                    report_lines.append(f"  {key}: {value:.4f}")
                else:
                    report_lines.append(f"  {key}: {value}")

        if "details" in tier_results:
            report_lines.append("\n### Details")
            report_lines.append(json.dumps(tier_results["details"], indent=2))

    report = "\n".join(report_lines)

    if output_path:
        Path(output_path).write_text(report)
        logger.info("Report saved", path=output_path)

    return report


async def main():
    """Run evaluation pipeline."""
    parser = argparse.ArgumentParser(description="ML Gap Finder Evaluation Runner")
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Evaluation tier to run",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        help="Path to test set configuration YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for report",
    )
    args = parser.parse_args()

    results = {}

    try:
        if args.tier in ["1", "all"]:
            results["tier1"] = await run_tier1_evaluation()
            await log_evaluation_run("1", results["tier1"])

        if args.tier in ["2", "all"]:
            results["tier2"] = await run_tier2_evaluation()
            await log_evaluation_run("2", results["tier2"])

        if args.tier in ["3", "all"]:
            results["tier3"] = await run_tier3_calibration()

        # Generate report
        report = generate_report(results, args.output)
        print(report)

        # Check success criteria
        success = True
        if "tier1" in results:
            summary = results["tier1"].get("summary", {})
            if summary.get("avg_extraction_f1", 0) < 0.75:
                logger.warning("Tier 1: Method extraction F1 below threshold")
                success = False

        if "tier2" in results:
            summary = results["tier2"].get("summary", {})
            if summary.get("avg_overall", 0) < 3.5:
                logger.warning("Tier 2: Average score below threshold")
                success = False

        if not success:
            logger.warning("Some evaluation criteria not met")
            sys.exit(1)

    except Exception as e:
        logger.error("Evaluation failed", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())
