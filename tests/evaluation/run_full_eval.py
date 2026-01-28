"""
Full RAG Evaluation Pipeline Orchestrator.

Runs both retrieval and generation evaluation in sequence.

EVALUATION STRATEGY:
1. Retrieval Eval: Recall@10, MRR, Hit Rate, NDCG
2. Generation Eval: RAGAS metrics (Faithfulness, Answer Relevancy)
3. Combined report with pass/fail status

TARGETS:
- Retrieval: Recall@10 > 80%, Hit Rate@10 > 90%, NDCG > 0.75
- Generation: Faithfulness > 0.85, Answer Relevancy > 0.80
"""

import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FullEvalResult:
    """Combined evaluation result."""

    # Retrieval metrics
    recall_at_10: float
    hit_rate_at_10: float
    mrr: float
    ndcg_at_10: float

    # Generation metrics
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    # Status
    retrieval_passed: bool
    generation_passed: bool
    overall_passed: bool

    # Metadata
    timestamp: str
    retrieval_queries: int
    generation_samples: int

    def to_dict(self) -> dict:
        result = asdict(self)
        result["targets"] = {
            "retrieval": {
                "recall_10": 0.80,
                "hit_rate_10": 0.90,
                "ndcg_10": 0.75,
            },
            "generation": {
                "faithfulness": 0.85,
                "answer_relevancy": 0.80,
            },
        }
        return result


def run_retrieval_evaluation(
    dataset_path: Optional[str] = None,
    results_path: Optional[str] = None,
) -> dict:
    """Run retrieval evaluation."""
    from tests.evaluation.run_retrieval_eval import (
        load_golden_queries,
        run_retrieval_eval,
        create_sample_results,
        RECALL_10_TARGET,
        HIT_RATE_10_TARGET,
        NDCG_10_TARGET,
    )

    # Load dataset
    if dataset_path:
        dataset = load_golden_queries(dataset_path)
    else:
        # Use default golden queries
        default_path = project_root / "backend" / "evaluation" / "test_sets" / "golden_queries.json"
        if default_path.exists():
            dataset = load_golden_queries(str(default_path))
        else:
            from backend.evaluation.retrieval_eval import create_synthetic_dataset
            dataset = create_synthetic_dataset(num_queries=50)

    logger.info(f"Retrieval eval: {len(dataset.queries)} queries")

    # Load or generate results
    if results_path:
        with open(results_path) as f:
            results_data = json.load(f)
        results = results_data.get("results", results_data)
    else:
        # Generate synthetic results for demo
        logger.warning("No retrieval results provided. Using synthetic results.")
        results = create_sample_results(dataset, noise_ratio=0.2)

    result = run_retrieval_eval(results, dataset)

    return {
        "recall_at_10": result.recall_at_10,
        "hit_rate_at_10": result.hit_rate_at_10,
        "mrr": result.mrr,
        "ndcg_at_10": result.ndcg_at_10,
        "passed": result.passed,
        "num_queries": len(dataset.queries),
    }


def run_generation_evaluation(
    dataset_path: Optional[str] = None,
) -> dict:
    """Run generation evaluation with RAGAS."""
    from tests.evaluation.run_ragas import (
        run_ragas,
        load_eval_data,
        create_sample_data,
        FAITHFULNESS_TARGET,
        ANSWER_RELEVANCY_TARGET,
    )

    # Load dataset
    if dataset_path and Path(dataset_path).exists():
        dataset = load_eval_data(dataset_path)
    else:
        logger.warning("No generation eval data provided. Using sample data.")
        dataset = create_sample_data()

    logger.info(f"Generation eval: {len(dataset)} samples")

    result = run_ragas(dataset)

    return {
        "faithfulness": result.faithfulness,
        "answer_relevancy": result.answer_relevancy,
        "context_precision": result.context_precision,
        "context_recall": result.context_recall,
        "passed": result.passed,
        "num_samples": len(dataset),
    }


def run_full_evaluation(
    retrieval_dataset: Optional[str] = None,
    retrieval_results: Optional[str] = None,
    generation_dataset: Optional[str] = None,
    output_path: Optional[str] = None,
    skip_retrieval: bool = False,
    skip_generation: bool = False,
) -> FullEvalResult:
    """
    Run full RAG evaluation pipeline.

    Args:
        retrieval_dataset: Path to retrieval eval dataset (golden_queries.json)
        retrieval_results: Path to pre-computed retrieval results
        generation_dataset: Path to generation eval dataset (RAGAS format)
        output_path: Path to save combined results
        skip_retrieval: Skip retrieval evaluation
        skip_generation: Skip generation evaluation

    Returns:
        FullEvalResult with all metrics
    """
    print("\n" + "=" * 70)
    print("FULL RAG EVALUATION PIPELINE")
    print("=" * 70)

    # Initialize defaults
    retrieval_metrics = {
        "recall_at_10": 0.0,
        "hit_rate_at_10": 0.0,
        "mrr": 0.0,
        "ndcg_at_10": 0.0,
        "passed": False,
        "num_queries": 0,
    }

    generation_metrics = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "passed": False,
        "num_samples": 0,
    }

    # Run retrieval evaluation
    if not skip_retrieval:
        print("\n[1/2] RETRIEVAL EVALUATION")
        print("-" * 40)
        try:
            retrieval_metrics = run_retrieval_evaluation(
                retrieval_dataset, retrieval_results
            )
            print(f"  Recall@10:    {retrieval_metrics['recall_at_10']:.4f}")
            print(f"  Hit Rate@10:  {retrieval_metrics['hit_rate_at_10']:.4f}")
            print(f"  MRR:          {retrieval_metrics['mrr']:.4f}")
            print(f"  NDCG@10:      {retrieval_metrics['ndcg_at_10']:.4f}")
            print(f"  Status:       {'PASS' if retrieval_metrics['passed'] else 'FAIL'}")
        except Exception as e:
            logger.error(f"Retrieval evaluation failed: {e}")
            print(f"  ERROR: {e}")
    else:
        print("\n[1/2] RETRIEVAL EVALUATION - SKIPPED")

    # Run generation evaluation
    if not skip_generation:
        print("\n[2/2] GENERATION EVALUATION (RAGAS)")
        print("-" * 40)
        try:
            generation_metrics = run_generation_evaluation(generation_dataset)
            print(f"  Faithfulness:      {generation_metrics['faithfulness']:.4f}")
            print(f"  Answer Relevancy:  {generation_metrics['answer_relevancy']:.4f}")
            print(f"  Context Precision: {generation_metrics['context_precision']:.4f}")
            print(f"  Context Recall:    {generation_metrics['context_recall']:.4f}")
            print(f"  Status:            {'PASS' if generation_metrics['passed'] else 'FAIL'}")
        except Exception as e:
            logger.error(f"Generation evaluation failed: {e}")
            print(f"  ERROR: {e}")
    else:
        print("\n[2/2] GENERATION EVALUATION - SKIPPED")

    # Combine results
    overall_passed = retrieval_metrics["passed"] and generation_metrics["passed"]

    result = FullEvalResult(
        recall_at_10=retrieval_metrics["recall_at_10"],
        hit_rate_at_10=retrieval_metrics["hit_rate_at_10"],
        mrr=retrieval_metrics["mrr"],
        ndcg_at_10=retrieval_metrics["ndcg_at_10"],
        faithfulness=generation_metrics["faithfulness"],
        answer_relevancy=generation_metrics["answer_relevancy"],
        context_precision=generation_metrics["context_precision"],
        context_recall=generation_metrics["context_recall"],
        retrieval_passed=retrieval_metrics["passed"],
        generation_passed=generation_metrics["passed"],
        overall_passed=overall_passed,
        timestamp=datetime.now().isoformat(),
        retrieval_queries=retrieval_metrics["num_queries"],
        generation_samples=generation_metrics["num_samples"],
    )

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\nRetrieval:  {'PASS' if result.retrieval_passed else 'FAIL'}")
    print(f"Generation: {'PASS' if result.generation_passed else 'FAIL'}")
    print(f"\nOVERALL:    {'PASS' if result.overall_passed else 'FAIL'}")
    print("=" * 70)

    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return result


def compare_results(current_path: str, baseline_path: str) -> dict:
    """
    Compare current evaluation results with baseline for regression detection.

    Args:
        current_path: Path to current evaluation results
        baseline_path: Path to baseline results

    Returns:
        Comparison report with deltas and regression flags
    """
    with open(current_path) as f:
        current = json.load(f)
    with open(baseline_path) as f:
        baseline = json.load(f)

    metrics_to_compare = [
        "recall_at_10",
        "hit_rate_at_10",
        "mrr",
        "ndcg_at_10",
        "faithfulness",
        "answer_relevancy",
    ]

    comparison = {"metrics": {}, "regressions": [], "improvements": []}

    for metric in metrics_to_compare:
        curr_val = current.get(metric, 0)
        base_val = baseline.get(metric, 0)
        delta = curr_val - base_val
        pct_change = (delta / base_val * 100) if base_val > 0 else 0

        comparison["metrics"][metric] = {
            "current": curr_val,
            "baseline": base_val,
            "delta": delta,
            "pct_change": pct_change,
        }

        # Flag significant regressions (> 5% drop)
        if pct_change < -5:
            comparison["regressions"].append(metric)
        elif pct_change > 5:
            comparison["improvements"].append(metric)

    comparison["has_regression"] = len(comparison["regressions"]) > 0

    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full RAG evaluation")
    parser.add_argument(
        "--retrieval-dataset",
        "-rd",
        help="Path to retrieval eval dataset",
    )
    parser.add_argument(
        "--retrieval-results",
        "-rr",
        help="Path to pre-computed retrieval results",
    )
    parser.add_argument(
        "--generation-dataset",
        "-gd",
        help="Path to generation eval dataset (RAGAS format)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save combined results",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Skip retrieval evaluation",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation evaluation",
    )
    parser.add_argument(
        "--compare",
        "-c",
        help="Path to baseline results for comparison",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run with sample/synthetic data",
    )

    args = parser.parse_args()

    if args.sample:
        # Run with sample data
        result = run_full_evaluation(
            output_path=args.output,
            skip_retrieval=args.skip_retrieval,
            skip_generation=args.skip_generation,
        )
    else:
        result = run_full_evaluation(
            retrieval_dataset=args.retrieval_dataset,
            retrieval_results=args.retrieval_results,
            generation_dataset=args.generation_dataset,
            output_path=args.output,
            skip_retrieval=args.skip_retrieval,
            skip_generation=args.skip_generation,
        )

    # Compare with baseline if provided
    if args.compare and args.output:
        print("\n" + "=" * 70)
        print("REGRESSION CHECK")
        print("=" * 70)
        comparison = compare_results(args.output, args.compare)

        for metric, data in comparison["metrics"].items():
            delta_str = f"+{data['delta']:.4f}" if data["delta"] >= 0 else f"{data['delta']:.4f}"
            print(f"  {metric}: {data['current']:.4f} ({delta_str}, {data['pct_change']:+.1f}%)")

        if comparison["regressions"]:
            print(f"\n  REGRESSIONS: {', '.join(comparison['regressions'])}")
        if comparison["improvements"]:
            print(f"  IMPROVEMENTS: {', '.join(comparison['improvements'])}")

        print("=" * 70)

    # Exit with non-zero code if evaluation failed
    sys.exit(0 if result.overall_passed else 1)
