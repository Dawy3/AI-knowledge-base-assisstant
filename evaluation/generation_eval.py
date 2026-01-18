# ============================================================================
# evaluation/generation_eval.py - Generation Quality Evaluation with RAGAS
# ============================================================================

"""
Production Generation Evaluation using RAGAS framework

Metrics:
- Faithfulness: Answer is grounded in retrieved context (no hallucinations)
- Answer Relevancy: Answer addresses the question
- Context Precision: Relevant contexts ranked higher
- Context Recall: All relevant info from ground truth is in context

Targets:
- Faithfulness > 90%
- Answer Relevancy > 85%
- Context Precision > 80%
- Context Recall > 80%
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy
)
from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """Container for generation evaluation metrics"""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    context_relevancy: float
    num_queries: int
    timestamp: str
    
    def passes_production_threshold(self) -> bool:
        """Check if metrics meet production requirements"""
        return (
            self.faithfulness >= 0.90 and
            self.answer_relevancy >= 0.85 and
            self.context_precision >= 0.80 and
            self.context_recall >= 0.80
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class GenerationEvaluator:
    """
    Evaluate generation quality using RAGAS framework
    MUST verify before production deployment
    """
    
    def __init__(
        self,
        test_set_path: str = "evaluation/test_sets/golden_queries.json"
    ):
        self.test_set_path = Path(test_set_path)
        self.test_queries = self._load_test_set()
        logger.info(f"Loaded {len(self.test_queries)} test queries for generation eval")
        
    def _load_test_set(self) -> List[Dict]:
        """Load test set with expected answers"""
        if not self.test_set_path.exists():
            logger.warning(f"Test set not found at {self.test_set_path}")
            return []
        
        with open(self.test_set_path, 'r') as f:
            data = json.load(f)
        
        # Validate structure - need expected_answer for generation eval
        required_fields = ["query", "expected_answer"]
        for idx, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    logger.warning(
                        f"Test query {idx} missing '{field}'. May not be suitable for generation eval."
                    )
        
        return data
    
    def evaluate_generation(
        self,
        rag_system,
        sample_size: int = None
    ) -> GenerationMetrics:
        """
        Evaluate generation quality using RAGAS
        
        Args:
            rag_system: RAG system instance
            sample_size: Number of queries to evaluate (None = all)
            
        Returns:
            GenerationMetrics with all scores
        """
        logger.info("Starting generation evaluation with RAGAS...")
        
        if not self.test_queries:
            raise ValueError("No test queries loaded. Cannot evaluate.")
        
        # Sample if requested
        test_queries = self.test_queries
        if sample_size and sample_size < len(test_queries):
            import random
            test_queries = random.sample(test_queries, sample_size)
            logger.info(f"Sampling {sample_size} queries for evaluation")
        
        # Collect data for RAGAS
        evaluation_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for idx, test_item in enumerate(test_queries):
            query = test_item["query"]
            expected_answer = test_item.get("expected_answer", "")
            
            if not expected_answer:
                logger.warning(f"Query {idx} has no expected answer. Skipping.")
                continue
            
            try:
                # Get RAG response
                response = rag_system.query(query, top_k=5, use_cache=False)
                
                if response["status"] != "success":
                    logger.warning(f"Query {idx} failed: {response.get('reason', 'unknown')}")
                    continue
                
                # Extract retrieved contexts
                contexts = []
                for source in response.get("sources", []):
                    chunk_id = source["chunk_id"]
                    content = rag_system.retriever._get_chunk_content(chunk_id)
                    if content:
                        contexts.append(content)
                
                if not contexts:
                    logger.warning(f"Query {idx} has no contexts. Skipping.")
                    continue
                
                # Add to evaluation dataset
                evaluation_data["question"].append(query)
                evaluation_data["answer"].append(response["answer"])
                evaluation_data["contexts"].append(contexts)
                evaluation_data["ground_truth"].append(expected_answer)
                
            except Exception as e:
                logger.error(f"Error processing query {idx}: {e}")
                continue
            
            if (idx + 1) % 20 == 0:
                logger.info(f"Processed {idx + 1}/{len(test_queries)} queries")
        
        if len(evaluation_data["question"]) == 0:
            raise ValueError("No valid evaluation data collected")
        
        logger.info(f"Collected {len(evaluation_data['question'])} valid examples")
        
        # Create HuggingFace Dataset
        dataset = Dataset.from_dict(evaluation_data)
        
        # Run RAGAS evaluation
        logger.info("Running RAGAS evaluation (this may take a while)...")
        
        try:
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    context_relevancy
                ]
            )
            
            # Extract metrics
            metrics = GenerationMetrics(
                faithfulness=result["faithfulness"],
                answer_relevancy=result["answer_relevancy"],
                context_precision=result["context_precision"],
                context_recall=result["context_recall"],
                context_relevancy=result["context_relevancy"],
                num_queries=len(evaluation_data["question"]),
                timestamp=datetime.now().isoformat()
            )
            
            logger.info("RAGAS evaluation complete!")
            self._log_results(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            raise
    
    def _log_results(self, metrics: GenerationMetrics):
        """Log evaluation results"""
        logger.info("=" * 80)
        logger.info("GENERATION EVALUATION RESULTS (RAGAS)")
        logger.info("=" * 80)
        logger.info(f"Test Queries: {metrics.num_queries}")
        logger.info("")
        logger.info("ANSWER QUALITY:")
        logger.info(
            f"  Faithfulness:      {metrics.faithfulness:.4f} "
            f"{'✅' if metrics.faithfulness >= 0.90 else '❌ BELOW TARGET (0.90)'}"
        )
        logger.info(
            f"  Answer Relevancy:  {metrics.answer_relevancy:.4f} "
            f"{'✅' if metrics.answer_relevancy >= 0.85 else '❌ BELOW TARGET (0.85)'}"
        )
        logger.info("")
        logger.info("CONTEXT QUALITY:")
        logger.info(
            f"  Context Precision: {metrics.context_precision:.4f} "
            f"{'✅' if metrics.context_precision >= 0.80 else '❌ BELOW TARGET (0.80)'}"
        )
        logger.info(
            f"  Context Recall:    {metrics.context_recall:.4f} "
            f"{'✅' if metrics.context_recall >= 0.80 else '❌ BELOW TARGET (0.80)'}"
        )
        logger.info(
            f"  Context Relevancy: {metrics.context_relevancy:.4f}"
        )
        logger.info("")
        
        if metrics.passes_production_threshold():
            logger.info("✅ PASSED: Ready for production deployment")
        else:
            logger.error("❌ FAILED: Does NOT meet production requirements")
            logger.error("   Required: Faithfulness>=0.90, Relevancy>=0.85, Precision>=0.80, Recall>=0.80")
        logger.info("=" * 80)
    
    def save_results(self, metrics: GenerationMetrics, output_path: str):
        """Save evaluation results to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def analyze_failures(
        self,
        rag_system,
        faithfulness_threshold: float = 0.7,
        relevancy_threshold: float = 0.7
    ) -> Dict:
        """
        Identify queries with low scores for debugging
        
        Returns:
            Dict with low-scoring examples for analysis
        """
        logger.info("Analyzing failure cases...")
        
        failures = {
            "low_faithfulness": [],
            "low_relevancy": [],
            "low_context_precision": []
        }
        
        for test_item in self.test_queries[:50]:  # Analyze subset
            query = test_item["query"]
            
            try:
                response = rag_system.query(query, top_k=5, use_cache=False)
                
                if response["status"] != "success":
                    continue
                
                # Prepare single-item dataset for RAGAS
                contexts = []
                for source in response.get("sources", []):
                    chunk_id = source["chunk_id"]
                    content = rag_system.retriever._get_chunk_content(chunk_id)
                    if content:
                        contexts.append(content)
                
                if not contexts:
                    continue
                
                eval_data = {
                    "question": [query],
                    "answer": [response["answer"]],
                    "contexts": [contexts],
                    "ground_truth": [test_item.get("expected_answer", "")]
                }
                
                dataset = Dataset.from_dict(eval_data)
                result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
                
                # Check for failures
                if result["faithfulness"] < faithfulness_threshold:
                    failures["low_faithfulness"].append({
                        "query": query,
                        "answer": response["answer"],
                        "score": result["faithfulness"],
                        "contexts": contexts
                    })
                
                if result["answer_relevancy"] < relevancy_threshold:
                    failures["low_relevancy"].append({
                        "query": query,
                        "answer": response["answer"],
                        "score": result["answer_relevancy"]
                    })
                
                if result["context_precision"] < 0.6:
                    failures["low_context_precision"].append({
                        "query": query,
                        "score": result["context_precision"],
                        "num_contexts": len(contexts)
                    })
                
            except Exception as e:
                logger.error(f"Error analyzing query '{query}': {e}")
                continue
        
        # Log summary
        logger.info(f"Found {len(failures['low_faithfulness'])} low faithfulness cases")
        logger.info(f"Found {len(failures['low_relevancy'])} low relevancy cases")
        logger.info(f"Found {len(failures['low_context_precision'])} low context precision cases")
        
        return failures


class RAGASBenchmark:
    """Run comprehensive RAGAS benchmark suite"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.evaluator = GenerationEvaluator()
    
    def run_full_benchmark(self) -> Dict:
        """Run complete evaluation suite"""
        logger.info("Starting full RAGAS benchmark...")
        
        results = {}
        
        # 1. Generation quality
        gen_metrics = self.evaluator.evaluate_generation(self.rag_system)
        results["generation"] = gen_metrics.to_dict()
        
        # 2. Failure analysis
        failures = self.evaluator.analyze_failures(self.rag_system)
        results["failures"] = {
            "low_faithfulness_count": len(failures["low_faithfulness"]),
            "low_relevancy_count": len(failures["low_relevancy"]),
            "low_context_precision_count": len(failures["low_context_precision"])
        }
        
        # 3. Overall assessment
        results["production_ready"] = gen_metrics.passes_production_threshold()
        results["timestamp"] = datetime.now().isoformat()
        
        return results


if __name__ == "__main__":
    # Example usage
    from main import RAGSystem
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize system
    rag = RAGSystem()
    
    # Run evaluation
    evaluator = GenerationEvaluator()
    metrics = evaluator.evaluate_generation(rag, sample_size=50)
    
    # Save results
    evaluator.save_results(
        metrics,
        f"evaluation/results/generation_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )