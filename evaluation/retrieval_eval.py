"""
Production Retrieval Evaluation
CRITICAL: 90% of RAG failures = no evaluation

Metrics:
- Recall@K: % of relevant docs retrieved
- Precision@K: % of retrieved docs that are relevant
- MRR (Mean Reciprocal Rank): Quality of first relevant result
- NDCG@K (Normalized Discounted Cumulative Gain): Ranking quality

Targets:
- Recall@10 > 80%
- NDCG@10 > 0.75
- MRR > 0.70 
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime
from collections import defaultdict

from main import RAGSystem


logger = logging.getLogger(__name__)

@dataclass
class RetrievalMetrics:
    """Container for retrieval metrics"""
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    precision_at_10: float
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_5: float
    ndcg_at_10: float
    hit_rate_at_10: float
    num_queries: int
    timestamp: str
    
    def pass_production_threshold(self) -> bool:
        """Check if metrics meet prodcution requiremetns"""
        return(
            self.recall_at_10 >= 0.80 and 
            self.ndcg_at_10 >= 0.75 and
            self.mrr >= 0.70
        )
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    

class RetrievalEvaluator:
    """
    Evaluate retrieval quality using standard IR metrics
    MUST run before production deployment
    """
    def __init__(self, test_set_path: str = "evaluation/test_sets/golden_queries.json"):
        self.test_set_path = Path(test_set_path)
        self.test_queries = self._load_test_set()
        logger.info(f"Loaded {len(self.test_queries)} test queries")
        
    def _load_test_set(self) -> List[Dict]:
        """Load golden test set with ground truth"""
        if not self.test_set_path.exists():
            logger.warning(f"Test set not found at {self.test_set_path}")
            return []
        
        with open(self.test_set_path, 'r') as f:
            data = json.load(f)
        
        # Validate structure
        required_fields = ["query", "expected_doc_ids"]
        for idx, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    raise ValueError(
                        f"Test query {idx} missing required field: {field}"
                    )
        
        return data

    def evaluate_retrieval(
        self,
        rag_system,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval performance across all test queries
        
        Args:
            rag_system: RAG system instance
            k_values: K values to evaluate
            
        Returns:
            RetrievalMetrics with all scores
        """
        logger.info("Starting retrieval evaluation...")
        
        if not self.test_queries:
            raise ValueError("No test queries loaded. Cannot evaluate.")
        
        all_results = {
            "recall": defaultdict(list),
            "precision": defaultdict(list),
            "reciprocal_ranks": [],
            "ndcg": defaultdict(list),
            "hit_rate": []
        }
        
        max_k = max(k_values)
        
        for idx, test_item in enumerate(self.test_queries):
            query = test_item["query"]
            expected_doc_ids = set(test_item["expected_doc_ids"])
            
            if not expected_doc_ids:
                logger.warning(f"Query {idx} has no expected docs. Skipping.")
                continue
                
            # Retrieve documents
            try:
                retrieval_result = rag_system.retriever.retrieve(
                    query=query,
                    top_k=max_k,
                    use_multi_query=False  # Disable for fair evaluation
                )
                
                if retrieval_result["status"] != "success":
                    logger.warning(
                        f"Query {idx} failed: {retrieval_result.get('reason', 'unknown')}"
                    )
                    # Add zeros for this query
                    for k in k_values:
                        all_results["recall"][k].append(0.0)
                        all_results["precision"][k].append(0.0)
                        all_results["ndcg"][k].append(0.0)
                    all_results["reciprocal_ranks"].append(0.0)
                    all_results["hit_rate"].append(0.0)
                    continue
                
                retrieved_results = retrieval_result["results"]
                retrieved_doc_ids = [r["id"] for r in retrieved_results]
                
            except Exception as e:
                logger.error(f"Error retrieving for query {idx}: {e}")
                # Add zeros
                for k in k_values:
                    all_results["recall"][k].append(0.0)
                    all_results["precision"][k].append(0.0)
                    all_results["ndcg"][k].append(0.0)
                all_results["reciprocal_ranks"].append(0.0)
                all_results["hit_rate"].append(0.0)
                continue
            # Calculate metrics for each K
            for k in k_values:
                retrieved_at_k = set(retrieved_doc_ids[:k])
                
                # Recall@K
                recall = len(retrieved_at_k & expected_doc_ids) / len(expected_doc_ids)
                all_results["recall"][k].append(recall)
                
                # Precision@K
                if len(retrieved_at_k) > 0:
                    precision = len(retrieved_at_k & expected_doc_ids) / len(retrieved_at_k)
                else:
                    precision = 0.0
                all_results["precision"][k].append(precision)
                
                # NDCG@K
                ndcg = self._calculate_ndcg(
                    retrieved_doc_ids[:k],
                    expected_doc_ids
                )
                all_results["ndcg"][k].append(ndcg)
            
            # MRR - find rank of first relevant document
            reciprocal_rank = 0.0
            for rank, doc_id in enumerate(retrieved_doc_ids, 1):
                if doc_id in expected_doc_ids:
                    reciprocal_rank = 1.0 / rank
                    break
            all_results["reciprocal_ranks"].append(reciprocal_rank)
            
            # Hit Rate@10 - at least one relevant doc in top 10
            hit = 1.0 if any(doc_id in expected_doc_ids for doc_id in retrieved_doc_ids[:10]) else 0.0
            all_results["hit_rate"].append(hit)
            
            if (idx + 1) % 50 == 0:
                logger.info(f"Evaluated {idx + 1}/{len(self.test_queries)} queries")
        
        # Aggregate metrics
        metrics = RetrievalMetrics(
            recall_at_1=np.mean(all_results["recall"][1]),
            recall_at_3=np.mean(all_results["recall"][3]),
            recall_at_5=np.mean(all_results["recall"][5]),
            recall_at_10=np.mean(all_results["recall"][10]),
            precision_at_1=np.mean(all_results["precision"][1]),
            precision_at_3=np.mean(all_results["precision"][3]),
            precision_at_5=np.mean(all_results["precision"][5]),
            precision_at_10=np.mean(all_results["precision"][10]),
            mrr=np.mean(all_results["reciprocal_ranks"]),
            ndcg_at_5=np.mean(all_results["ndcg"][5]),
            ndcg_at_10=np.mean(all_results["ndcg"][10]),
            hit_rate_at_10=np.mean(all_results["hit_rate"]),
            num_queries=len(self.test_queries),
            timestamp=datetime.now().isoformat()
        )
        
        logger.info("Evaluation complete!")
        self._log_results(metrics)
        
        return metrics
    
    def _calculate_ndcg(
        self,
        retrieved_ids: List[int],
        relevant_ids: Set[int]
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain
        
        DCG = sum(rel_i / log2(i + 1))
        NDCG = DCG / IDCG (ideal DCG)
        """
        if not retrieved_ids or not relevant_ids:
            return 0.0
        
        # DCG - actual ranking
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            relevance = 1.0 if doc_id in relevant_ids else 0.0
            dcg += relevance / np.log2(rank + 1)
        
        # IDCG - ideal ranking (all relevant docs first)
        ideal_retrieved = list(relevant_ids)[:len(retrieved_ids)]
        idcg = 0.0
        for rank in range(1, len(ideal_retrieved) + 1):
            idcg += 1.0 / np.log2(rank + 1)
        
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg
    
    def _log_results(self, metrics: RetrievalMetrics):
        """Log evaluation results"""
        logger.info("=" * 80)
        logger.info("RETRIEVAL EVALUATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Test Queries: {metrics.num_queries}")
        logger.info("")
        logger.info("RECALL METRICS:")
        logger.info(f"  Recall@1:  {metrics.recall_at_1:.4f}")
        logger.info(f"  Recall@3:  {metrics.recall_at_3:.4f}")
        logger.info(f"  Recall@5:  {metrics.recall_at_5:.4f}")
        logger.info(f"  Recall@10: {metrics.recall_at_10:.4f} {'✅' if metrics.recall_at_10 >= 0.80 else '❌ BELOW TARGET'}")
        logger.info("")
        logger.info("PRECISION METRICS:")
        logger.info(f"  Precision@1:  {metrics.precision_at_1:.4f}")
        logger.info(f"  Precision@3:  {metrics.precision_at_3:.4f}")
        logger.info(f"  Precision@5:  {metrics.precision_at_5:.4f}")
        logger.info(f"  Precision@10: {metrics.precision_at_10:.4f}")
        logger.info("")
        logger.info("RANKING QUALITY:")
        logger.info(f"  MRR:       {metrics.mrr:.4f} {'✅' if metrics.mrr >= 0.70 else '❌ BELOW TARGET'}")
        logger.info(f"  NDCG@5:    {metrics.ndcg_at_5:.4f}")
        logger.info(f"  NDCG@10:   {metrics.ndcg_at_10:.4f} {'✅' if metrics.ndcg_at_10 >= 0.75 else '❌ BELOW TARGET'}")
        logger.info(f"  Hit Rate@10: {metrics.hit_rate_at_10:.4f}")
        logger.info("")
        
        if metrics.pass_production_threshold():
            logger.info("✅ PASSED: Ready for production deployment")
        else:
            logger.error("❌ FAILED: Does NOT meet production requirements")
            logger.error("   Required: Recall@10>=0.80, NDCG@10>=0.75, MRR>=0.70")
        logger.info("=" * 80)
    
    def save_results(self, metrics: RetrievalMetrics, output_path: str):
        """Save evaluation results to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def run_ablation_study(self, rag_system) -> Dict:
        """
        Test impact of different components
        - Vector only vs Hybrid
        - With/without reranking
        - With/without multi-query
        """
        logger.info("Running ablation study...")
        
        configurations = [
            {
                "name": "baseline_vector_only",
                "hybrid_alpha": 1.0,
                "use_reranking": False,
                "use_multi_query": False
            },
            {
                "name": "hybrid_no_rerank",
                "hybrid_alpha": 0.5,
                "use_reranking": False,
                "use_multi_query": False
            },
            {
                "name": "hybrid_with_rerank",
                "hybrid_alpha": 0.5,
                "use_reranking": True,
                "use_multi_query": False
            },
            {
                "name": "full_pipeline",
                "hybrid_alpha": 0.5,
                "use_reranking": True,
                "use_multi_query": True
            }
        ]
        
        results = {}
        
        for config in configurations:
            logger.info(f"Testing configuration: {config['name']}")
            
            # Temporarily modify system settings
            original_settings = {
                "use_reranking": rag_system.retriever.enable_reranking
            }
            
            rag_system.retriever.enable_reranking = config["use_reranking"]
            
            # Run evaluation
            metrics = self.evaluate_retrieval(rag_system)
            results[config["name"]] = metrics.to_dict()
            
            # Restore settings
            rag_system.retriever.enable_reranking = original_settings["use_reranking"]
        
        logger.info("Ablation study complete")
        return results
    

if __name__ == "__main__":
    # Example usage
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize system
    rag = RAGSystem()
    
    # Run evaluation
    evaluator = RetrievalEvaluator()
    metrics = evaluator.evaluate_retrieval(rag)
    
    # Save results
    evaluator.save_results(
        metrics,
        f"evaluation/results/retrieval_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )