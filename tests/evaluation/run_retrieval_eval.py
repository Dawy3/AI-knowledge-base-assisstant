"""
Run Retrieval evaluation for RAG pipeline.

FOCUS: Recall@10, MRR, Hit Rate, NDCG
MUST: Run before production
TARGET: Recall@10 > 80%, Hit Rate@10 > 90%, NDCG@10 > 0.75
"""

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.evaluation.retrieval_eval import (
    EvalDataset,
    EvalQuery,
    RetrievalEvaluator,
    RetrievalMetrics,
    print_report,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Production targets
RECALL_10_TARGET = 0.80
HIT_RATE_10_TARGET = 0.90
NDCG_10_TARGET = 0.75
MRR_TARGET = 0.60


@dataclass
class RetrievalResult:
    """Retrieval evaluation result."""

    recall_at_10: float
    hit_rate_at_10: float
    mrr: float
    ndcg_at_10: float
    precision_at_10: float
    avg_latency_ms: float
    p95_latency_ms: float
    queries_per_second: float
    passed: bool
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "recall_at_10": self.recall_at_10,
            "hit_rate_at_10": self.hit_rate_at_10,
            "mrr": self.mrr,
            "ndcg_at_10": self.ndcg_at_10,
            "precision_at_10": self.precision_at_10,
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "queries_per_second": self.queries_per_second,
            "passed": self.passed,
            "timestamp": self.timestamp,
            "targets": {
                "recall_10": RECALL_10_TARGET,
                "hit_rate_10": HIT_RATE_10_TARGET,
                "ndcg_10": NDCG_10_TARGET,
                "mrr": MRR_TARGET,
            },
        }


def load_golden_queries(path: str) -> EvalDataset:
    """
    Load evaluation data from golden queries JSON format.

    Expected format:
    {
        "queries": [
            {
                "query_id": "gq_001",
                "query": "...",
                "relevant_ids": ["doc_001", ...],
                "ground_truth": "..."  # optional
            }
        ]
    }
    """
    with open(path) as f:
        data = json.load(f)

    queries = []
    for q in data.get("queries", []):
        queries.append(
            EvalQuery(
                query_id=q["query_id"],
                query_text=q.get("query", q.get("query_text", "")),
                relevant_ids=q["relevant_ids"],
                relevance_scores=q.get("relevance_scores"),
            )
        )

    return EvalDataset(name=data.get("description", Path(path).stem), queries=queries)


def load_edge_cases(path: str) -> EvalDataset:
    """
    Load edge cases dataset.

    Expected format:
    {
        "test_cases": [
            {
                "id": "ec_001",
                "query": "...",
                "expected": ["doc_001", ...]
            }
        ]
    }
    """
    with open(path) as f:
        data = json.load(f)

    queries = []
    for tc in data.get("test_cases", []):
        queries.append(
            EvalQuery(
                query_id=tc["id"],
                query_text=tc["query"],
                relevant_ids=tc.get("expected", tc.get("relevant_ids", [])),
            )
        )

    return EvalDataset(name="edge_cases", queries=queries)


def run_retrieval_eval(
    results: list[dict],
    dataset: EvalDataset,
    k_values: Optional[list[int]] = None,
) -> RetrievalResult:
    """
    Run retrieval evaluation on pre-computed results.

    Args:
        results: List of {"query_id": str, "retrieved_ids": list[str]}
        dataset: Evaluation dataset with ground truth
        k_values: K values for metrics (default: [1, 5, 10])

    Returns:
        RetrievalResult with scores and pass/fail status
    """
    k_values = k_values or [1, 5, 10]
    evaluator = RetrievalEvaluator(k_values=k_values)

    metrics = evaluator.evaluate_from_results(results, dataset)

    # Check targets
    passed = (
        metrics.recall_at_k.get(10, 0) >= RECALL_10_TARGET
        and metrics.hit_rate_at_k.get(10, 0) >= HIT_RATE_10_TARGET
        and metrics.ndcg_at_k.get(10, 0) >= NDCG_10_TARGET
        and metrics.mrr >= MRR_TARGET
    )

    return RetrievalResult(
        recall_at_10=metrics.recall_at_k.get(10, 0),
        hit_rate_at_10=metrics.hit_rate_at_k.get(10, 0),
        mrr=metrics.mrr,
        ndcg_at_10=metrics.ndcg_at_k.get(10, 0),
        precision_at_10=metrics.precision_at_k.get(10, 0),
        avg_latency_ms=metrics.avg_latency_ms,
        p95_latency_ms=metrics.p95_latency_ms,
        queries_per_second=metrics.queries_per_second,
        passed=passed,
        timestamp=datetime.now().isoformat(),
    )


def run_with_retriever(
    retriever,
    dataset: EvalDataset,
    embedding_fn=None,
    k_values: Optional[list[int]] = None,
) -> tuple[RetrievalResult, RetrievalMetrics]:
    """
    Run retrieval evaluation with a live retriever.

    Args:
        retriever: Object with search() method or callable
        dataset: Evaluation dataset
        embedding_fn: Optional embedding function
        k_values: K values for metrics

    Returns:
        Tuple of (RetrievalResult, full RetrievalMetrics)
    """
    k_values = k_values or [1, 5, 10]
    evaluator = RetrievalEvaluator(k_values=k_values)

    metrics = evaluator.evaluate(retriever, dataset, embedding_fn)

    passed = (
        metrics.recall_at_k.get(10, 0) >= RECALL_10_TARGET
        and metrics.hit_rate_at_k.get(10, 0) >= HIT_RATE_10_TARGET
        and metrics.ndcg_at_k.get(10, 0) >= NDCG_10_TARGET
        and metrics.mrr >= MRR_TARGET
    )

    result = RetrievalResult(
        recall_at_10=metrics.recall_at_k.get(10, 0),
        hit_rate_at_10=metrics.hit_rate_at_k.get(10, 0),
        mrr=metrics.mrr,
        ndcg_at_10=metrics.ndcg_at_k.get(10, 0),
        precision_at_10=metrics.precision_at_k.get(10, 0),
        avg_latency_ms=metrics.avg_latency_ms,
        p95_latency_ms=metrics.p95_latency_ms,
        queries_per_second=metrics.queries_per_second,
        passed=passed,
        timestamp=datetime.now().isoformat(),
    )

    return result, metrics


def run_from_file(
    eval_path: str,
    results_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> RetrievalResult:
    """
    Run evaluation from file(s) and optionally save results.

    Args:
        eval_path: Path to evaluation dataset (golden_queries.json format)
        results_path: Optional path to pre-computed results JSON
        output_path: Optional path to save evaluation results

    Returns:
        RetrievalResult
    """
    # Load dataset
    if "edge_case" in eval_path.lower():
        dataset = load_edge_cases(eval_path)
    else:
        dataset = load_golden_queries(eval_path)

    logger.info(f"Loaded {len(dataset.queries)} queries from {eval_path}")

    if results_path:
        # Evaluate from pre-computed results
        with open(results_path) as f:
            results_data = json.load(f)
        results = results_data.get("results", results_data)
        result = run_retrieval_eval(results, dataset)
    else:
        # Generate synthetic results for testing (replace with real retriever)
        logger.warning("No results file provided. Using synthetic results for demo.")
        synthetic_results = []
        for q in dataset.queries:
            # Simulate: return all relevant + some noise
            retrieved = q.relevant_ids[:10] + [f"noise_{i}" for i in range(5)]
            synthetic_results.append({
                "query_id": q.query_id,
                "retrieved_ids": retrieved[:10],
            })
        result = run_retrieval_eval(synthetic_results, dataset)

    # Print report
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 60)
    print(f"\nRecall@10:     {result.recall_at_10:.4f} (target: {RECALL_10_TARGET})")
    print(f"Hit Rate@10:   {result.hit_rate_at_10:.4f} (target: {HIT_RATE_10_TARGET})")
    print(f"MRR:           {result.mrr:.4f} (target: {MRR_TARGET})")
    print(f"NDCG@10:       {result.ndcg_at_10:.4f} (target: {NDCG_10_TARGET})")
    print(f"Precision@10:  {result.precision_at_10:.4f}")
    print(f"\nLatency:       {result.avg_latency_ms:.2f}ms avg, {result.p95_latency_ms:.2f}ms p95")
    print(f"Throughput:    {result.queries_per_second:.2f} QPS")
    print(f"\nPASSED: {result.passed}")
    print("=" * 60)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return result


def create_sample_results(dataset: EvalDataset, noise_ratio: float = 0.3) -> list[dict]:
    """
    Create sample retrieval results for testing.

    Args:
        dataset: Evaluation dataset
        noise_ratio: Fraction of results that are noise

    Returns:
        List of result dictionaries
    """
    import random

    results = []
    for q in dataset.queries:
        # Mix relevant docs with noise
        relevant = q.relevant_ids.copy()
        noise_count = int(10 * noise_ratio)
        noise_docs = [f"noise_{random.randint(1, 1000)}" for _ in range(noise_count)]

        # Shuffle and take top 10
        retrieved = relevant + noise_docs
        random.shuffle(retrieved)
        retrieved = retrieved[:10]

        results.append({
            "query_id": q.query_id,
            "retrieved_ids": retrieved,
        })

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Retrieval evaluation")
    parser.add_argument("--input", "-i", help="Path to evaluation dataset JSON")
    parser.add_argument("--results", "-r", help="Path to pre-computed results JSON")
    parser.add_argument("--output", "-o", help="Path to save evaluation results")
    parser.add_argument("--sample", action="store_true", help="Run with sample data")
    parser.add_argument(
        "--golden",
        action="store_true",
        help="Use golden_queries.json from test_sets",
    )

    args = parser.parse_args()

    if args.sample:
        # Create and evaluate with synthetic data
        logger.info("Running with synthetic sample data...")
        from backend.evaluation.retrieval_eval import create_synthetic_dataset

        dataset = create_synthetic_dataset(num_queries=50)
        results = create_sample_results(dataset, noise_ratio=0.2)
        result = run_retrieval_eval(results, dataset)

        print("\n" + "=" * 60)
        print("RETRIEVAL EVALUATION REPORT (Sample Data)")
        print("=" * 60)
        print(f"\nRecall@10:     {result.recall_at_10:.4f}")
        print(f"Hit Rate@10:   {result.hit_rate_at_10:.4f}")
        print(f"MRR:           {result.mrr:.4f}")
        print(f"NDCG@10:       {result.ndcg_at_10:.4f}")
        print(f"\nPASSED: {result.passed}")
        print("=" * 60)

    elif args.golden:
        # Use golden queries from test_sets
        test_sets_path = project_root / "backend" / "evaluation" / "test_sets"
        golden_path = test_sets_path / "golden_queries.json"
        run_from_file(str(golden_path), args.results, args.output)

    elif args.input:
        run_from_file(args.input, args.results, args.output)

    else:
        print("Usage:")
        print("  python run_retrieval_eval.py --input eval_data.json [--results results.json] [--output out.json]")
        print("  python run_retrieval_eval.py --golden [--results results.json]")
        print("  python run_retrieval_eval.py --sample")
