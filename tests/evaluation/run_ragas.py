"""
Run Ragas evaluation for RAG pipeline.

FOCUS: Faithfulness, Answer Relevancy, Context Precision/Recall
MUST: Run before production
TARGET: Faithfulness > 0.85, Answer Relevancy > 0.80
"""

# Load environment variables FIRST before any other imports
import os
from dotenv import load_dotenv
load_dotenv()

# Configure OpenAI client to use OpenRouter if available
# This must be done BEFORE importing openai or ragas
openrouter_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if openrouter_key and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = openrouter_key
    os.environ["OPENAI_BASE_URL"] = openrouter_base

# Patch openai to use OpenRouter base URL
import openai
if openrouter_key:
    openai.api_key = openrouter_key
    openai.base_url = openrouter_base

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Import config-based LLM/embeddings helpers
try:
    from backend.evaluation.generation_eval import get_ragas_llm, get_ragas_embeddings
except ImportError:
    # Fallback if running standalone
    get_ragas_llm = lambda: None
    get_ragas_embeddings = lambda: None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Targets
FAITHFULNESS_TARGET = 0.85
ANSWER_RELEVANCY_TARGET = 0.80


@dataclass
class RagasResult:
    """Ragas evaluation result."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    passed: bool
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "passed": self.passed,
            "timestamp": self.timestamp,
        }


def load_eval_data(path: str) -> Dataset:
    """
    Load evaluation data from JSON.

    Expected format:
    {
        "samples": [
            {
                "question": "...",
                "answer": "...",
                "contexts": ["...", "..."],
                "ground_truth": "..."  # optional
            }
        ]
    }
    """
    with open(path) as f:
        data = json.load(f)

    samples = data.get("samples", data.get("data", []))

    dataset_dict = {
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
    }

    if any("ground_truth" in s for s in samples):
        dataset_dict["ground_truth"] = [s.get("ground_truth", "") for s in samples]

    return Dataset.from_dict(dataset_dict)


def run_ragas(
    dataset: Dataset,
    metrics: Optional[list] = None,
    llm=None,
    embeddings=None,
    use_config: bool = True,
) -> RagasResult:
    """
    Run Ragas evaluation.

    Args:
        dataset: HuggingFace Dataset with question, answer, contexts
        metrics: Optional list of metrics (default: all)
        llm: Optional LangChain LLM for evaluation
        embeddings: Optional LangChain embeddings for evaluation
        use_config: If True and llm/embeddings not provided, use config settings

    Returns:
        RagasResult with scores

    Note:
        For free evaluation without OpenAI, set in .env:
        EMBEDDING__MODEL_PROVIDER=huggingface
        EMBEDDING__MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
    """
    if metrics is None:
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    # Use config-based LLM/embeddings if not provided
    if use_config:
        if llm is None:
            llm = get_ragas_llm()
            if llm:
                logger.info("Using config-based LLM for Ragas evaluation")
        if embeddings is None:
            embeddings = get_ragas_embeddings()
            if embeddings:
                logger.info("Using config-based embeddings for Ragas evaluation")

    logger.info(f"Running Ragas evaluation on {len(dataset)} samples...")

    # Build kwargs
    kwargs = {}
    if llm:
        kwargs["llm"] = llm
    if embeddings:
        kwargs["embeddings"] = embeddings

    result = evaluate(dataset, metrics=metrics, **kwargs)

    # Helper to extract mean score from result
    def get_mean_score(result_obj, key, default=0.0):
        try:
            val = result_obj[key]
            if isinstance(val, list):
                # Filter out None values and calculate mean
                valid_vals = [v for v in val if v is not None]
                return sum(valid_vals) / len(valid_vals) if valid_vals else default
            return float(val) if val is not None else default
        except (KeyError, TypeError):
            return default

    # Extract mean scores
    faith_score = get_mean_score(result, "faithfulness")
    answer_rel_score = get_mean_score(result, "answer_relevancy")
    ctx_prec_score = get_mean_score(result, "context_precision")
    ctx_recall_score = get_mean_score(result, "context_recall")

    logger.info(f"Scores - Faithfulness: {faith_score:.4f}, Answer Relevancy: {answer_rel_score:.4f}")

    scores = RagasResult(
        faithfulness=faith_score,
        answer_relevancy=answer_rel_score,
        context_precision=ctx_prec_score,
        context_recall=ctx_recall_score,
        passed=(
            faith_score >= FAITHFULNESS_TARGET
            and answer_rel_score >= ANSWER_RELEVANCY_TARGET
        ),
        timestamp=datetime.now().isoformat(),
    )

    return scores


def run_from_file(eval_path: str, output_path: Optional[str] = None) -> RagasResult:
    """Run evaluation from file and optionally save results."""
    dataset = load_eval_data(eval_path)
    result = run_ragas(dataset)

    # Print report
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION REPORT")
    print("=" * 60)
    print(f"\nFaithfulness:       {result.faithfulness:.4f} (target: {FAITHFULNESS_TARGET})")
    print(f"Answer Relevancy:   {result.answer_relevancy:.4f} (target: {ANSWER_RELEVANCY_TARGET})")
    print(f"Context Precision:  {result.context_precision:.4f}")
    print(f"Context Recall:     {result.context_recall:.4f}")
    print(f"\nPASSED: {result.passed}")
    print("=" * 60)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return result


def create_sample_data() -> Dataset:
    """Create sample evaluation data for testing."""
    samples = {
        "question": [
            "What is Python?",
            "How does machine learning work?",
            "What is a neural network?",
        ],
        "answer": [
            "Python is a high-level programming language known for its readability and versatility.",
            "Machine learning works by training algorithms on data to recognize patterns and make predictions.",
            "A neural network is a computing system inspired by biological neural networks in the brain.",
        ],
        "contexts": [
            ["Python is a high-level, interpreted programming language.", "Python emphasizes code readability."],
            ["Machine learning is a subset of AI.", "ML algorithms learn patterns from training data."],
            ["Neural networks consist of layers of interconnected nodes.", "They are inspired by the human brain."],
        ],
        "ground_truth": [
            "Python is a programming language.",
            "Machine learning uses data to train models.",
            "Neural networks are brain-inspired computing systems.",
        ],
    }
    return Dataset.from_dict(samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Ragas evaluation")
    parser.add_argument("--input", "-i", help="Path to evaluation data JSON")
    parser.add_argument("--output", "-o", help="Path to save results")
    parser.add_argument("--sample", action="store_true", help="Run with sample data")

    args = parser.parse_args()

    if args.sample:
        logger.info("Running with sample data...")
        dataset = create_sample_data()
        result = run_ragas(dataset)

        print("\n" + "=" * 60)
        print("RAGAS EVALUATION REPORT (Sample Data)")
        print("=" * 60)
        print(f"\nFaithfulness:       {result.faithfulness:.4f}")
        print(f"Answer Relevancy:   {result.answer_relevancy:.4f}")
        print(f"Context Precision:  {result.context_precision:.4f}")
        print(f"Context Recall:     {result.context_recall:.4f}")
        print(f"\nPASSED: {result.passed}")
        print("=" * 60)

    elif args.input:
        run_from_file(args.input, args.output)

    else:
        print("Usage: python run_ragas.py --input eval_data.json [--output results.json]")
        print("       python run_ragas.py --sample")
