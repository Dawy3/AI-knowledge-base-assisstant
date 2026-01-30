"""
Retrieval Evaluation - Universal RAG Benchmark.

Supports TWO evaluation modes:
1. Vector-only: Basic vector search (baseline)
2. Full Pipeline: Hybrid Search + Reranker + Relevance Filter (production)

Usage:
    # Vector-only evaluation (baseline)
    python -m backend.evaluation.retrieval_eval --evaluate

    # Full pipeline evaluation (recommended)
    python -m backend.evaluation.retrieval_eval --evaluate --full-pipeline

    # Compare both
    python -m backend.evaluation.retrieval_eval --compare
"""

import argparse
import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..core.config import settings
from ..core.embedding.generator import EmbeddingGenerator
from ..services.vector_store.qdrant import QdrantStore

# Full pipeline imports
from ..core.retrieval.bm25_search import BM25Search, BM25Tokenizer, BM25SearchResult
from ..core.retrieval.hybrid_search import HybridSearch, HybridSearchConfig
from ..core.retrieval.reranker import Reranker, RerankerConfig
from ..core.retrieval.vector_search import VectorSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to test sets
TEST_SETS_DIR = Path(__file__).parent / "test_sets"
BENCHMARK_QUERIES_PATH = TEST_SETS_DIR / "benchmark_queries.json"
BENCHMARK_DOCUMENTS_PATH = TEST_SETS_DIR / "benchmark_documents.json"


@dataclass
class BenchmarkQuery:
    """Represents a benchmark query with ground truth."""
    id: str
    query: str
    category: str
    difficulty: str
    relevant_doc_ids: list[str]
    ground_truth: str
    expected_behavior: Optional[str] = None


@dataclass
class BenchmarkDocument:
    """Represents a document in the benchmark corpus."""
    id: str
    title: str
    category: str
    content: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Results from retrieval evaluation."""
    total_queries: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    hits_at_5: int
    hits_at_10: int
    by_category: dict
    by_difficulty: dict
    failed_queries: list
    mode: str = "vector_only"  # "vector_only" or "full_pipeline"


def load_benchmark_queries() -> list[BenchmarkQuery]:
    """Load benchmark queries from JSON file."""
    with open(BENCHMARK_QUERIES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = []
    for q in data["queries"]:
        queries.append(BenchmarkQuery(
            id=q["id"],
            query=q["query"],
            category=q["category"],
            difficulty=q.get("difficulty", "medium"),
            relevant_doc_ids=q.get("relevant_doc_ids", []),
            ground_truth=q.get("ground_truth", ""),
            expected_behavior=q.get("expected_behavior"),
        ))

    logger.info(f"Loaded {len(queries)} benchmark queries")
    return queries


def load_benchmark_documents() -> list[BenchmarkDocument]:
    """Load benchmark documents (corpus) from JSON file."""
    with open(BENCHMARK_DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for doc in data["documents"]:
        documents.append(BenchmarkDocument(
            id=doc["id"],
            title=doc["title"],
            category=doc["category"],
            content=doc["content"],
            keywords=doc.get("keywords", []),
        ))

    logger.info(f"Loaded {len(documents)} benchmark documents")
    return documents


def calculate_ndcg(relevance_scores: list[int], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain."""
    def dcg(scores: list[int]) -> float:
        return sum(
            (2 ** rel - 1) / math.log2(i + 2)
            for i, rel in enumerate(scores[:k])
        )

    actual_dcg = dcg(relevance_scores)
    ideal_dcg = dcg(sorted(relevance_scores, reverse=True))

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


class RetrievalEvaluator:
    """
    Universal Retrieval Benchmark Evaluator.

    Supports two modes:
    1. Vector-only: Basic vector search (baseline)
    2. Full Pipeline: Hybrid Search + Reranker + Filter (production)
    """

    def __init__(
        self,
        corpus_collection: str = "benchmark_corpus",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        self.corpus_collection = corpus_collection
        self.qdrant_url = qdrant_url or settings.database.qdrant_url
        self.qdrant_api_key = qdrant_api_key or settings.database.qdrant_api_key

        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.corpus_store: Optional[QdrantStore] = None

        # Full pipeline components
        self.bm25_search: Optional[BM25Search] = None
        self.hybrid_search: Optional[HybridSearch] = None
        self.reranker: Optional[Reranker] = None
        self._documents_cache: Optional[list[BenchmarkDocument]] = None

    async def initialize(self):
        """Initialize embedding generator and vector store."""
        logger.info("Initializing retrieval evaluator...")

        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()

        # Initialize corpus vector store
        self.corpus_store = QdrantStore(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            collection=self.corpus_collection,
            dimension=self.embedding_generator.model.dimensions,
        )
        await self.corpus_store.connect()

        logger.info(f"Initialized with corpus collection: {self.corpus_collection}")

    async def initialize_full_pipeline(self):
        """Initialize full pipeline components (BM25 + Hybrid + Reranker)."""
        if not self.embedding_generator:
            await self.initialize()

        logger.info("Initializing full retrieval pipeline...")

        # Load documents for BM25 index
        documents = load_benchmark_documents()
        self._documents_cache = documents

        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25_search = BM25Search()

        # Prepare documents for BM25
        bm25_docs = []
        for doc in documents:
            bm25_docs.append({
                "chunk_id": doc.id,
                "content": f"{doc.title}\n\n{doc.content}",
                "metadata": {
                    "title": doc.title,
                    "category": doc.category,
                    "keywords": doc.keywords,
                },
                "document_id": doc.id,
            })

        self.bm25_search.index(bm25_docs)
        logger.info(f"BM25 index built with {len(bm25_docs)} documents")

        # Initialize reranker
        logger.info("Initializing reranker...")
        reranker_config = RerankerConfig(
            top_k_input=100,
            top_k_output=10,
            device="cpu",  # Use CPU for compatibility
        )
        self.reranker = Reranker(config=reranker_config)

        logger.info("Full pipeline initialized")

    async def close(self):
        """Close connections."""
        if self.corpus_store:
            await self.corpus_store.close()

    async def ingest_corpus(self) -> dict:
        """Ingest benchmark documents as the retrieval corpus."""
        if not self.embedding_generator or not self.corpus_store:
            await self.initialize()

        documents = load_benchmark_documents()

        if not documents:
            logger.warning("No documents to ingest")
            return {"total": 0}

        # Create document texts (title + content for better retrieval)
        doc_texts = [f"{doc.title}\n\n{doc.content}" for doc in documents]
        doc_ids = [doc.id for doc in documents]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(doc_texts)} documents...")
        result = await self.embedding_generator.embed_texts(doc_texts)

        # Prepare metadata
        metadata = []
        for doc in documents:
            metadata.append({
                "title": doc.title,
                "category": doc.category,
                "content": doc.content,
                "keywords": doc.keywords,
            })

        # Upsert to vector store
        logger.info(f"Upserting {len(doc_ids)} vectors to corpus '{self.corpus_collection}'...")
        count = await self.corpus_store.upsert(
            ids=doc_ids,
            embeddings=result.embeddings,
            metadata=metadata,
        )

        stats = self.corpus_store.stats()

        logger.info(f"Corpus ingestion complete: {count} documents")

        return {
            "total_documents": count,
            "embedding_model": result.model_id,
            "dimensions": result.dimensions,
            "collection": self.corpus_collection,
            "collection_stats": stats,
        }

    async def _search_vector_only(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[str]:
        """Vector-only search (baseline)."""
        search_results = await self.corpus_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
        return [r["id"] for r in search_results]

    async def _search_full_pipeline(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[str]:
        """
        Full pipeline search:
        1. Vector Search (top-100)
        2. BM25 Search (top-100)
        3. Hybrid Fusion: Score = (Vector×5) + (BM25×3)
        4. Reranker (100 → 10)
        5. Return final results
        """
        # Step 1: Vector Search
        vector_results = await self.corpus_store.search(
            query_embedding=query_embedding,
            top_k=100,
        )

        # Step 2: BM25 Search
        bm25_response = self.bm25_search.search(query=query, top_k=100)

        # Step 3: Hybrid Fusion
        # Build score maps
        vector_scores = {r["id"]: r["score"] for r in vector_results}
        bm25_scores = {r.chunk_id: r.score for r in bm25_response.results}

        # Normalize scores
        def normalize(scores: dict) -> dict:
            if not scores:
                return {}
            max_s = max(scores.values())
            min_s = min(scores.values())
            if max_s == min_s:
                return {k: 1.0 for k in scores}
            return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}

        norm_vector = normalize(vector_scores)
        norm_bm25 = normalize(bm25_scores)

        # Combine all unique IDs
        all_ids = set(norm_vector.keys()) | set(norm_bm25.keys())

        # Calculate fused scores: (Vector×5) + (BM25×3)
        fused_scores = {}
        for doc_id in all_ids:
            v_score = norm_vector.get(doc_id, 0.0)
            b_score = norm_bm25.get(doc_id, 0.0)
            fused_scores[doc_id] = (v_score * 5.0) + (b_score * 3.0)

        # Sort by fused score and take top-100 for reranking
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:100]

        # Step 4: Prepare candidates for reranking
        # Build content map from cached documents
        content_map = {}
        if self._documents_cache:
            for doc in self._documents_cache:
                content_map[doc.id] = f"{doc.title}\n\n{doc.content}"

        candidates = []
        for doc_id, score in sorted_results:
            candidates.append({
                "chunk_id": doc_id,
                "content": content_map.get(doc_id, ""),
                "score": score,
            })

        # Step 5: Rerank
        if self.reranker and candidates:
            reranked = await self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
            return [r.chunk_id for r in reranked.results]

        # Fallback: return top-k from fused results
        return [doc_id for doc_id, _ in sorted_results[:top_k]]

    async def evaluate(
        self,
        top_k: int = 10,
        include_edge_cases: bool = True,
        use_full_pipeline: bool = False,
    ) -> EvaluationResult:
        """
        Run retrieval evaluation.

        Args:
            top_k: Maximum K for retrieval
            include_edge_cases: Include edge case queries
            use_full_pipeline: Use full pipeline (Hybrid + Reranker) vs vector-only

        Returns:
            EvaluationResult with detailed metrics
        """
        if use_full_pipeline:
            await self.initialize_full_pipeline()
            mode = "full_pipeline"
            logger.info("Evaluating with FULL PIPELINE (Hybrid + Reranker)")
        else:
            if not self.embedding_generator or not self.corpus_store:
                await self.initialize()
            mode = "vector_only"
            logger.info("Evaluating with VECTOR-ONLY search")

        queries = load_benchmark_queries()

        # Filter out edge cases if requested
        if not include_edge_cases:
            queries = [q for q in queries if q.category != "edge_case"]

        results = []
        failed_queries = []

        logger.info(f"Evaluating {len(queries)} queries...")

        for i, query in enumerate(queries):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(queries)} queries")

            # Skip queries without relevant docs
            if not query.relevant_doc_ids:
                continue

            try:
                # Generate query embedding
                query_embedding = await self.embedding_generator.embed_query(query.query)

                # Search using selected mode
                if use_full_pipeline:
                    retrieved_ids = await self._search_full_pipeline(
                        query=query.query,
                        query_embedding=query_embedding,
                        top_k=top_k,
                    )
                else:
                    retrieved_ids = await self._search_vector_only(
                        query=query.query,
                        query_embedding=query_embedding,
                        top_k=top_k,
                    )

                # Calculate metrics for this query
                hit_at_5 = any(doc_id in retrieved_ids[:5] for doc_id in query.relevant_doc_ids)
                hit_at_10 = any(doc_id in retrieved_ids[:10] for doc_id in query.relevant_doc_ids)

                # Reciprocal Rank
                rr = 0.0
                for doc_id in query.relevant_doc_ids:
                    if doc_id in retrieved_ids:
                        rank = retrieved_ids.index(doc_id) + 1
                        rr = max(rr, 1.0 / rank)
                        break

                # Relevance scores for NDCG
                relevance_scores = [
                    1 if doc_id in query.relevant_doc_ids else 0
                    for doc_id in retrieved_ids
                ]

                results.append({
                    "query_id": query.id,
                    "query": query.query,
                    "category": query.category,
                    "difficulty": query.difficulty,
                    "hit_at_5": hit_at_5,
                    "hit_at_10": hit_at_10,
                    "reciprocal_rank": rr,
                    "relevance_scores": relevance_scores,
                })

            except Exception as e:
                logger.warning(f"Failed to evaluate query {query.id}: {e}")
                failed_queries.append({
                    "query_id": query.id,
                    "error": str(e),
                })

        # Calculate aggregate metrics
        n = len(results)
        if n == 0:
            return EvaluationResult(
                total_queries=0,
                recall_at_5=0, recall_at_10=0, mrr=0, ndcg_at_10=0,
                hits_at_5=0, hits_at_10=0,
                by_category={}, by_difficulty={},
                failed_queries=failed_queries,
                mode=mode,
            )

        recall_at_5 = sum(1 for r in results if r["hit_at_5"]) / n
        recall_at_10 = sum(1 for r in results if r["hit_at_10"]) / n
        mrr = sum(r["reciprocal_rank"] for r in results) / n
        ndcg_at_10 = sum(calculate_ndcg(r["relevance_scores"], 10) for r in results) / n

        # Breakdown by category
        by_category = {}
        for r in results:
            cat = r["category"]
            if cat not in by_category:
                by_category[cat] = {"total": 0, "hits_5": 0, "hits_10": 0, "rr_sum": 0}
            by_category[cat]["total"] += 1
            if r["hit_at_5"]:
                by_category[cat]["hits_5"] += 1
            if r["hit_at_10"]:
                by_category[cat]["hits_10"] += 1
            by_category[cat]["rr_sum"] += r["reciprocal_rank"]

        for cat in by_category:
            t = by_category[cat]["total"]
            by_category[cat]["recall_at_5"] = by_category[cat]["hits_5"] / t
            by_category[cat]["recall_at_10"] = by_category[cat]["hits_10"] / t
            by_category[cat]["mrr"] = by_category[cat]["rr_sum"] / t

        # Breakdown by difficulty
        by_difficulty = {}
        for r in results:
            diff = r["difficulty"]
            if diff not in by_difficulty:
                by_difficulty[diff] = {"total": 0, "hits_5": 0, "hits_10": 0, "rr_sum": 0}
            by_difficulty[diff]["total"] += 1
            if r["hit_at_5"]:
                by_difficulty[diff]["hits_5"] += 1
            if r["hit_at_10"]:
                by_difficulty[diff]["hits_10"] += 1
            by_difficulty[diff]["rr_sum"] += r["reciprocal_rank"]

        for diff in by_difficulty:
            t = by_difficulty[diff]["total"]
            by_difficulty[diff]["recall_at_5"] = by_difficulty[diff]["hits_5"] / t
            by_difficulty[diff]["recall_at_10"] = by_difficulty[diff]["hits_10"] / t
            by_difficulty[diff]["mrr"] = by_difficulty[diff]["rr_sum"] / t

        return EvaluationResult(
            total_queries=n,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            mrr=mrr,
            ndcg_at_10=ndcg_at_10,
            hits_at_5=sum(1 for r in results if r["hit_at_5"]),
            hits_at_10=sum(1 for r in results if r["hit_at_10"]),
            by_category=by_category,
            by_difficulty=by_difficulty,
            failed_queries=failed_queries,
            mode=mode,
        )

    async def compare_modes(self, top_k: int = 10) -> dict:
        """
        Compare vector-only vs full pipeline.

        Returns comparison metrics showing improvement from full pipeline.
        """
        logger.info("=" * 60)
        logger.info("COMPARING: Vector-Only vs Full Pipeline")
        logger.info("=" * 60)

        # Evaluate vector-only
        logger.info("\n[1/2] Evaluating Vector-Only...")
        vector_result = await self.evaluate(top_k=top_k, use_full_pipeline=False)

        # Evaluate full pipeline
        logger.info("\n[2/2] Evaluating Full Pipeline...")
        pipeline_result = await self.evaluate(top_k=top_k, use_full_pipeline=True)

        # Calculate improvements
        improvements = {
            "recall_at_5": pipeline_result.recall_at_5 - vector_result.recall_at_5,
            "recall_at_10": pipeline_result.recall_at_10 - vector_result.recall_at_10,
            "mrr": pipeline_result.mrr - vector_result.mrr,
            "ndcg_at_10": pipeline_result.ndcg_at_10 - vector_result.ndcg_at_10,
        }

        return {
            "vector_only": {
                "recall_at_5": vector_result.recall_at_5,
                "recall_at_10": vector_result.recall_at_10,
                "mrr": vector_result.mrr,
                "ndcg_at_10": vector_result.ndcg_at_10,
            },
            "full_pipeline": {
                "recall_at_5": pipeline_result.recall_at_5,
                "recall_at_10": pipeline_result.recall_at_10,
                "mrr": pipeline_result.mrr,
                "ndcg_at_10": pipeline_result.ndcg_at_10,
            },
            "improvements": improvements,
            "improvement_percentages": {
                k: f"+{v*100:.1f}%" if v > 0 else f"{v*100:.1f}%"
                for k, v in improvements.items()
            }
        }

    def export_vectordbbench_format(self, output_dir: str = "vectordbbench_data") -> dict:
        """Export benchmark data in vectordbbench-compatible format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        documents = load_benchmark_documents()
        queries = load_benchmark_queries()

        # Export documents
        docs_data = [{"id": doc.id, "text": f"{doc.title}\n\n{doc.content}"} for doc in documents]
        docs_file = output_path / "documents.json"
        with open(docs_file, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, indent=2)

        # Export queries
        queries_data = [{"id": q.id, "text": q.query} for q in queries]
        queries_file = output_path / "queries.json"
        with open(queries_file, "w", encoding="utf-8") as f:
            json.dump(queries_data, f, indent=2)

        # Export qrels
        qrels = {q.id: {doc_id: 1 for doc_id in q.relevant_doc_ids} for q in queries if q.relevant_doc_ids}
        qrels_file = output_path / "qrels.json"
        with open(qrels_file, "w", encoding="utf-8") as f:
            json.dump(qrels, f, indent=2)

        return {"output_dir": str(output_path), "total_documents": len(documents), "total_queries": len(queries)}


def format_evaluation_report(result: EvaluationResult) -> str:
    """Format evaluation results as a readable report."""
    mode_label = "FULL PIPELINE (Hybrid + Reranker)" if result.mode == "full_pipeline" else "VECTOR-ONLY"

    lines = [
        "=" * 60,
        f"RETRIEVAL EVALUATION REPORT - {mode_label}",
        "=" * 60,
        "",
        "OVERALL METRICS",
        "-" * 40,
        f"Total Queries:     {result.total_queries}",
        f"Recall@5:          {result.recall_at_5:.2%}",
        f"Recall@10:         {result.recall_at_10:.2%}",
        f"MRR:               {result.mrr:.4f}",
        f"NDCG@10:           {result.ndcg_at_10:.4f}",
        f"Hits@5:            {result.hits_at_5}/{result.total_queries}",
        f"Hits@10:           {result.hits_at_10}/{result.total_queries}",
        "",
        "BY CATEGORY",
        "-" * 40,
    ]

    for cat, metrics in sorted(result.by_category.items()):
        lines.append(
            f"{cat:20} R@5={metrics['recall_at_5']:.2%} "
            f"R@10={metrics['recall_at_10']:.2%} "
            f"MRR={metrics['mrr']:.3f} (n={metrics['total']})"
        )

    lines.extend(["", "BY DIFFICULTY", "-" * 40])

    for diff, metrics in sorted(result.by_difficulty.items()):
        lines.append(
            f"{diff:20} R@5={metrics['recall_at_5']:.2%} "
            f"R@10={metrics['recall_at_10']:.2%} "
            f"MRR={metrics['mrr']:.3f} (n={metrics['total']})"
        )

    if result.failed_queries:
        lines.extend(["", "FAILED QUERIES", "-" * 40])
        for fq in result.failed_queries[:5]:
            lines.append(f"  {fq['query_id']}: {fq['error']}")

    lines.extend(["", "=" * 60])
    return "\n".join(lines)


def format_comparison_report(comparison: dict) -> str:
    """Format comparison results."""
    lines = [
        "=" * 60,
        "COMPARISON: Vector-Only vs Full Pipeline",
        "=" * 60,
        "",
        f"{'Metric':<20} {'Vector-Only':>15} {'Full Pipeline':>15} {'Improvement':>15}",
        "-" * 65,
    ]

    metrics = ["recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]
    labels = ["Recall@5", "Recall@10", "MRR", "NDCG@10"]

    for metric, label in zip(metrics, labels):
        v = comparison["vector_only"][metric]
        p = comparison["full_pipeline"][metric]
        imp = comparison["improvement_percentages"][metric]
        lines.append(f"{label:<20} {v:>14.2%} {p:>14.2%} {imp:>15}")

    lines.extend(["", "=" * 60])
    return "\n".join(lines)


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Universal RAG Retrieval Benchmark")
    parser.add_argument("--ingest-corpus", action="store_true", help="Ingest benchmark documents")
    parser.add_argument("--evaluate", action="store_true", help="Run retrieval evaluation")
    parser.add_argument("--full-pipeline", action="store_true", help="Use full pipeline (Hybrid + Reranker)")
    parser.add_argument("--compare", action="store_true", help="Compare vector-only vs full pipeline")
    parser.add_argument("--export-vectordbbench", action="store_true", help="Export for vectordbbench")
    parser.add_argument("--collection", type=str, default="benchmark_corpus", help="Collection name")
    parser.add_argument("--qdrant-url", type=str, help="Qdrant URL")
    parser.add_argument("--top-k", type=int, default=10, help="Top K for evaluation")
    parser.add_argument("--output-dir", type=str, default="vectordbbench_data", help="Output directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    evaluator = RetrievalEvaluator(
        corpus_collection=args.collection,
        qdrant_url=args.qdrant_url,
    )

    try:
        if args.ingest_corpus:
            await evaluator.initialize()
            stats = await evaluator.ingest_corpus()
            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                print(f"\nCorpus ingested: {stats['total_documents']} documents")

        if args.compare:
            comparison = await evaluator.compare_modes(top_k=args.top_k)
            if args.json:
                print(json.dumps(comparison, indent=2))
            else:
                print(format_comparison_report(comparison))

        elif args.evaluate:
            result = await evaluator.evaluate(
                top_k=args.top_k,
                use_full_pipeline=args.full_pipeline,
            )
            if args.json:
                print(json.dumps({
                    "mode": result.mode,
                    "total_queries": result.total_queries,
                    "recall_at_5": result.recall_at_5,
                    "recall_at_10": result.recall_at_10,
                    "mrr": result.mrr,
                    "ndcg_at_10": result.ndcg_at_10,
                }, indent=2))
            else:
                print(format_evaluation_report(result))

        if args.export_vectordbbench:
            result = evaluator.export_vectordbbench_format(args.output_dir)
            print(f"\nExported to: {result['output_dir']}")

        if not any([args.ingest_corpus, args.evaluate, args.compare, args.export_vectordbbench]):
            parser.print_help()

    finally:
        await evaluator.close()


if __name__ == "__main__":
    asyncio.run(main())
