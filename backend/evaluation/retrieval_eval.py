"""
Retrieval Evaluation using VectorDBBench patterns.

FOCUS: Recall@10, Precision@K, MRR, NDCG
MUST: Run before production
TARGET: Recall@10 > 80%, NDCG > 0.75
"""

import json
import logging
import math
import time
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Standard retrieval metrics following VectorDBBench."""

    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    queries_per_second: float = 0.0

    def to_dict(self) -> dict:
        return {
            "recall": self.recall_at_k,
            "precision": self.precision_at_k,
            "mrr": self.mrr,
            "ndcg": self.ndcg_at_k,
            "latency": {"avg_ms": self.avg_latency_ms, "p95_ms": self.p95_latency_ms, "p99_ms": self.p99_latency_ms},
            "qps": self.queries_per_second,
        }

    def check_targets(self, recall_10_target: float = 0.80, ndcg_10_target: float = 0.75) -> dict[str, bool]:
        """Check if metrics meet production targets."""
        return {
            "recall_10": self.recall_at_k.get(10, 0) >= recall_10_target,
            "ndcg_10": self.ndcg_at_k.get(10, 0) >= ndcg_10_target,
        }


@dataclass
class EvalQuery:
    """Single evaluation query with ground truth."""

    query_id: str
    query_text: str
    relevant_ids: list[str]
    relevance_scores: Optional[dict[str, int]] = None


@dataclass
class EvalDataset:
    """Evaluation dataset following VectorDBBench format."""

    name: str
    queries: list[EvalQuery]

    @classmethod
    def from_json(cls, path: str) -> "EvalDataset":
        with open(path) as f:
            data = json.load(f)
        queries = [
            EvalQuery(
                query_id=q["query_id"],
                query_text=q["query_text"],
                relevant_ids=q["relevant_ids"],
                relevance_scores=q.get("relevance_scores"),
            )
            for q in data["queries"]
        ]
        return cls(name=data.get("name", Path(path).stem), queries=queries)

    @classmethod
    def from_qrels(cls, queries_path: str, qrels_path: str, name: str = "custom") -> "EvalDataset":
        """Load from TREC-style qrels format."""
        queries_map = {}
        with open(queries_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    queries_map[parts[0]] = parts[1]

        qrels = {}
        with open(qrels_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][doc_id] = rel

        eval_queries = []
        for qid, text in queries_map.items():
            if qid in qrels:
                relevant_ids = [did for did, rel in qrels[qid].items() if rel > 0]
                eval_queries.append(EvalQuery(
                    query_id=qid, query_text=text, relevant_ids=relevant_ids, relevance_scores=qrels[qid]
                ))
        return cls(name=name, queries=eval_queries)


class RetrievalEvaluator:
    """Evaluator for retrieval quality using VectorDBBench metrics."""

    def __init__(self, k_values: Optional[list[int]] = None):
        self.k_values = k_values or [1, 5, 10]

    def evaluate(self, retriever, dataset: EvalDataset, embedding_fn=None) -> RetrievalMetrics:
        """Run full evaluation on dataset."""

        all_results = []
        latencies = []

        for query in dataset.queries:
            if embedding_fn:
                query_embedding = embedding_fn(query.query_text)
            else:
                query_embedding = asyncio.run(retriever.embed_query(query.query_text))

            start = time.perf_counter()
            max_k = max(self.k_values)

            if hasattr(retriever, "search"):
                response = asyncio.run(retriever.search(query_embedding, top_k=max_k))
                retrieved_ids = [r.chunk_id for r in response.results]
            else:
                retrieved_ids = retriever(query_embedding, max_k)

            latencies.append((time.perf_counter() - start) * 1000)
            all_results.append({"query": query, "retrieved_ids": retrieved_ids})

        return self._calculate_metrics(all_results, latencies)

    def evaluate_from_results(self, results: list[dict], dataset: EvalDataset) -> RetrievalMetrics:
        """Evaluate from pre-computed results."""
        query_map = {q.query_id: q for q in dataset.queries}
        all_results = []
        for r in results:
            query = query_map.get(r["query_id"])
            if query:
                all_results.append({"query": query, "retrieved_ids": r["retrieved_ids"]})
        return self._calculate_metrics(all_results, latencies=[])

    def _calculate_metrics(self, results: list[dict], latencies: list[float]) -> RetrievalMetrics:
        recall_at_k = {k: [] for k in self.k_values}
        precision_at_k = {k: [] for k in self.k_values}
        ndcg_at_k = {k: [] for k in self.k_values}
        reciprocal_ranks = []

        for item in results:
            query = item["query"]
            retrieved = item["retrieved_ids"]
            relevant_set = set(query.relevant_ids)

            for k in self.k_values:
                retrieved_k = retrieved[:k]
                hits = len(set(retrieved_k) & relevant_set)
                recall_at_k[k].append(hits / len(relevant_set) if relevant_set else 0.0)
                precision_at_k[k].append(hits / k)

            rr = 0.0
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant_set:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            for k in self.k_values:
                ndcg = self._calculate_ndcg(
                    retrieved[:k], query.relevance_scores or {did: 1 for did in query.relevant_ids}, k
                )
                ndcg_at_k[k].append(ndcg)

        metrics = RetrievalMetrics(
            recall_at_k={k: float(np.mean(v)) for k, v in recall_at_k.items()},
            precision_at_k={k: float(np.mean(v)) for k, v in precision_at_k.items()},
            mrr=float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
            ndcg_at_k={k: float(np.mean(v)) for k, v in ndcg_at_k.items()},
        )

        if latencies:
            metrics.avg_latency_ms = float(np.mean(latencies))
            metrics.p95_latency_ms = float(np.percentile(latencies, 95))
            metrics.p99_latency_ms = float(np.percentile(latencies, 99))
            total_time = sum(latencies) / 1000
            metrics.queries_per_second = len(latencies) / total_time if total_time > 0 else 0

        return metrics

    def _calculate_ndcg(self, retrieved: list[str], relevance_scores: dict[str, int], k: int) -> float:
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            rel = relevance_scores.get(doc_id, 0)
            dcg += (2 ** rel - 1) / math.log2(i + 2)

        ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0.0


def create_synthetic_dataset(num_queries: int = 100, num_relevant_per_query: int = 5, seed: int = 42) -> EvalDataset:
    """Create synthetic dataset for testing."""
    np.random.seed(seed)
    queries = [
        EvalQuery(query_id=f"q_{i}", query_text=f"Sample query {i}", relevant_ids=[f"doc_{i}_{j}" for j in range(num_relevant_per_query)])
        for i in range(num_queries)
    ]
    return EvalDataset(name="synthetic", queries=queries)


def run_evaluation(
    retriever, dataset: EvalDataset, embedding_fn=None, k_values: Optional[list[int]] = None,
    recall_target: float = 0.80, ndcg_target: float = 0.75
) -> tuple[RetrievalMetrics, bool]:
    """Run evaluation and check against targets."""
    evaluator = RetrievalEvaluator(k_values=k_values or [1, 5, 10])
    metrics = evaluator.evaluate(retriever, dataset, embedding_fn)
    targets = metrics.check_targets(recall_target, ndcg_target)
    passed = all(targets.values())

    logger.info("=" * 50)
    logger.info(f"Evaluation Results for {dataset.name}")
    logger.info(f"Recall@10:    {metrics.recall_at_k.get(10, 0):.4f} (target: {recall_target})")
    logger.info(f"NDCG@10:      {metrics.ndcg_at_k.get(10, 0):.4f} (target: {ndcg_target})")
    logger.info(f"MRR:          {metrics.mrr:.4f}")
    logger.info(f"Avg Latency:  {metrics.avg_latency_ms:.2f}ms")
    logger.info(f"PASSED: {passed}")
    return metrics, passed


def print_report(metrics: RetrievalMetrics, dataset_name: str = "dataset") -> None:
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print(f"RETRIEVAL EVALUATION REPORT: {dataset_name}")
    print("=" * 60)

    print("\n[Quality Metrics]")
    for k in sorted(metrics.recall_at_k.keys()):
        print(f"  Recall@{k}:    {metrics.recall_at_k[k]:.4f}")
    print()
    for k in sorted(metrics.precision_at_k.keys()):
        print(f"  Precision@{k}: {metrics.precision_at_k[k]:.4f}")
    print()
    for k in sorted(metrics.ndcg_at_k.keys()):
        print(f"  NDCG@{k}:      {metrics.ndcg_at_k[k]:.4f}")
    print(f"\n  MRR:          {metrics.mrr:.4f}")

    print("\n[Performance Metrics]")
    print(f"  Avg Latency:  {metrics.avg_latency_ms:.2f} ms")
    print(f"  P95 Latency:  {metrics.p95_latency_ms:.2f} ms")
    print(f"  P99 Latency:  {metrics.p99_latency_ms:.2f} ms")
    print(f"  Throughput:   {metrics.queries_per_second:.2f} QPS")

    print("\n[Target Check]")
    for name, passed in metrics.check_targets().items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset = create_synthetic_dataset(num_queries=50)

    evaluator = RetrievalEvaluator(k_values=[1, 5, 10])
    metrics = evaluator.evaluate_from_results(
        results=[{"query_id": q.query_id, "retrieved_ids": q.relevant_ids[:10] + ["noise"] * 5} for q in dataset.queries],
        dataset=dataset,
    )
    print_report(metrics, dataset.name)
