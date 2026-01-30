"""
Evaluation module for RAG system.

Universal retrieval benchmark with:
- 250 queries with ground truth answers
- 150 relevant documents (corpus)
"""

from .retrieval_eval import (
    RetrievalEvaluator,
    BenchmarkQuery,
    BenchmarkDocument,
    EvaluationResult,
    load_benchmark_queries,
    load_benchmark_documents,
)

__all__ = [
    "RetrievalEvaluator",
    "BenchmarkQuery",
    "BenchmarkDocument",
    "EvaluationResult",
    "load_benchmark_queries",
    "load_benchmark_documents",
]
