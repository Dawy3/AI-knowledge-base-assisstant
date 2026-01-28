"""
Generation Evaluation using Ragas.

FOCUS: Faithfulness, Answer Relevancy, Context Precision/Recall
MUST: Run before production
TARGET: Faithfulness > 0.85, Answer Relevancy > 0.80
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # Load .env file

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

from backend.core.config import settings

logger = logging.getLogger(__name__)

# Conditional imports for optional dependencies
try:
    from langchain_openai import ChatOpenAI
    HAS_LANGCHAIN_OPENAI = True
except ImportError:
    HAS_LANGCHAIN_OPENAI = False
    logger.warning("langchain-openai not installed")

try:
    from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
    HAS_LANGCHAIN_HF = True
except ImportError:
    HAS_LANGCHAIN_HF = False
    logger.debug("langchain-huggingface not installed (optional)")

try:
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def get_ragas_llm():
    """
    Get LLM for Ragas evaluation based on config settings.

    Priority:
    1. OpenRouter (if OPENROUTER_API_KEY set) - FREE models available
    2. OpenAI (if OPENAI_API_KEY set)
    3. HuggingFace local (if available)

    Supports:
    - OpenRouter (free models like llama, gemma, etc.)
    - OpenAI
    - HuggingFace (local, free)
    """
    if not HAS_LANGCHAIN_OPENAI:
        logger.error("langchain-openai required. Run: pip install langchain-openai")
        return None

    # Check for OpenRouter first (free models available!)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if openrouter_key:
        try:
            logger.info(f"Using OpenRouter LLM: {settings.llm.openai_model}")
            return ChatOpenAI(
                model=settings.llm.openai_model,
                api_key=openrouter_key,
                base_url=openrouter_base,
            )
        except Exception as e:
            logger.warning(f"OpenRouter LLM setup failed: {e}")

    # Check for OpenAI
    openai_key = settings.llm.openai_api_key or os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "sk-your-openai-api-key":
        try:
            logger.info(f"Using OpenAI LLM: {settings.llm.openai_model}")
            return ChatOpenAI(
                model=settings.llm.openai_model,
                api_key=openai_key,
            )
        except Exception as e:
            logger.warning(f"OpenAI LLM setup failed: {e}")

    # Fallback to HuggingFace local models (completely free, runs locally)
    if HAS_LANGCHAIN_HF and HAS_TRANSFORMERS:
        try:
            model_name = os.getenv("RAGAS_LLM_MODEL", "google/flan-t5-small")
            logger.info(f"Using HuggingFace local LLM: {model_name}")
            pipe = hf_pipeline("text2text-generation", model=model_name, max_length=512)
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            logger.warning(f"HuggingFace LLM setup failed: {e}")

    logger.error("No LLM available. Set OPENROUTER_API_KEY (free) or OPENAI_API_KEY")
    return None


def get_ragas_embeddings():
    """
    Get embeddings for Ragas evaluation based on config settings.

    Priority:
    1. HuggingFace (free, runs locally)
    2. OpenAI (if api key available)
    """
    provider = settings.embedding.model_provider.lower()

    # For HuggingFace/free models
    if provider in ("huggingface", "local") and HAS_LANGCHAIN_HF:
        try:
            logger.info(f"Using HuggingFace embeddings: {settings.embedding.model_name}")
            return HuggingFaceEmbeddings(
                model_name=settings.embedding.model_name,
            )
        except Exception as e:
            logger.warning(f"HuggingFace embeddings setup failed: {e}")

    # For OpenAI
    openai_key = settings.llm.openai_api_key or os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "sk-your-openai-api-key" and HAS_LANGCHAIN_OPENAI:
        try:
            from langchain_openai import OpenAIEmbeddings
            logger.info(f"Using OpenAI embeddings: {settings.embedding.model_name}")
            return OpenAIEmbeddings(
                model=settings.embedding.model_name,
                api_key=openai_key,
            )
        except Exception as e:
            logger.warning(f"OpenAI embeddings setup failed: {e}")

    logger.warning("No embeddings available, Ragas will use defaults")
    return None

    return None


@dataclass
class GenerationMetrics:
    """Ragas generation metrics."""

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }

    def check_targets(
        self,
        faithfulness_target: float = 0.85,
        relevancy_target: float = 0.80,
    ) -> dict[str, bool]:
        """Check if metrics meet production targets."""
        return {
            "faithfulness": self.faithfulness >= faithfulness_target,
            "answer_relevancy": self.answer_relevancy >= relevancy_target,
        }


@dataclass
class EvalSample:
    """Single evaluation sample for Ragas."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: Optional[str] = None


@dataclass
class GenerationDataset:
    """Dataset for generation evaluation."""

    name: str
    samples: list[EvalSample]

    @classmethod
    def from_json(cls, path: str) -> "GenerationDataset":
        with open(path) as f:
            data = json.load(f)
        samples = [
            EvalSample(
                question=s["question"],
                answer=s["answer"],
                contexts=s["contexts"],
                ground_truth=s.get("ground_truth"),
            )
            for s in data["samples"]
        ]
        return cls(name=data.get("name", Path(path).stem), samples=samples)

    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset for Ragas."""
        data = {
            "question": [s.question for s in self.samples],
            "answer": [s.answer for s in self.samples],
            "contexts": [s.contexts for s in self.samples],
        }
        if any(s.ground_truth for s in self.samples):
            data["ground_truth"] = [s.ground_truth or "" for s in self.samples]
        return Dataset.from_dict(data)


class GenerationEvaluator:
    """Evaluator for generation quality using Ragas."""

    def __init__(self, metrics: Optional[list] = None, use_config: bool = True):
        """
        Initialize evaluator.

        Args:
            metrics: Ragas metrics to evaluate
            use_config: If True, use config settings for LLM/embeddings
        """
        self.metrics = metrics or [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        self.use_config = use_config

    def evaluate(self, dataset: GenerationDataset, llm=None, embeddings=None) -> GenerationMetrics:
        """
        Run Ragas evaluation.

        If llm/embeddings not provided and use_config=True, will use config settings.
        For free evaluation, set EMBEDDING__MODEL_PROVIDER=huggingface in .env
        """
        hf_dataset = dataset.to_hf_dataset()

        # Use config-based LLM/embeddings if not provided
        if self.use_config:
            if llm is None:
                llm = get_ragas_llm()
            if embeddings is None:
                embeddings = get_ragas_embeddings()

        # Run evaluation
        kwargs = {}
        if llm:
            kwargs["llm"] = llm
        if embeddings:
            kwargs["embeddings"] = embeddings

        result = evaluate(hf_dataset, metrics=self.metrics, **kwargs)

        return GenerationMetrics(
            faithfulness=result.get("faithfulness", 0.0),
            answer_relevancy=result.get("answer_relevancy", 0.0),
            context_precision=result.get("context_precision", 0.0),
            context_recall=result.get("context_recall", 0.0),
        )

    def evaluate_from_results(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: Optional[list[str]] = None,
        llm=None,
        embeddings=None,
    ) -> GenerationMetrics:
        """Evaluate from raw results."""
        samples = [
            EvalSample(
                question=q,
                answer=a,
                contexts=c,
                ground_truth=g if ground_truths else None,
            )
            for q, a, c, g in zip(
                questions,
                answers,
                contexts,
                ground_truths or [None] * len(questions),
            )
        ]
        dataset = GenerationDataset(name="inline", samples=samples)
        return self.evaluate(dataset, llm, embeddings)


def create_synthetic_dataset(num_samples: int = 20) -> GenerationDataset:
    """Create synthetic dataset for testing."""
    samples = [
        EvalSample(
            question=f"What is topic {i}?",
            answer=f"Topic {i} is about subject {i}. It covers key aspects of area {i}.",
            contexts=[f"Topic {i} covers subject {i}.", f"Area {i} is related to topic {i}."],
            ground_truth=f"Topic {i} is about subject {i}.",
        )
        for i in range(num_samples)
    ]
    return GenerationDataset(name="synthetic", samples=samples)


def run_evaluation(
    dataset: GenerationDataset,
    llm=None,
    embeddings=None,
    faithfulness_target: float = 0.85,
    relevancy_target: float = 0.80,
) -> tuple[GenerationMetrics, bool]:
    """Run evaluation and check against targets."""
    evaluator = GenerationEvaluator()
    metrics = evaluator.evaluate(dataset, llm, embeddings)

    targets = metrics.check_targets(faithfulness_target, relevancy_target)
    passed = all(targets.values())

    logger.info("=" * 50)
    logger.info(f"Generation Evaluation Results for {dataset.name}")
    logger.info(f"Faithfulness:       {metrics.faithfulness:.4f} (target: {faithfulness_target})")
    logger.info(f"Answer Relevancy:   {metrics.answer_relevancy:.4f} (target: {relevancy_target})")
    logger.info(f"Context Precision:  {metrics.context_precision:.4f}")
    logger.info(f"Context Recall:     {metrics.context_recall:.4f}")
    logger.info(f"PASSED: {passed}")

    return metrics, passed


def print_report(metrics: GenerationMetrics, dataset_name: str = "dataset") -> None:
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print(f"GENERATION EVALUATION REPORT: {dataset_name}")
    print("=" * 60)

    print("\n[Ragas Metrics]")
    print(f"  Faithfulness:       {metrics.faithfulness:.4f}")
    print(f"  Answer Relevancy:   {metrics.answer_relevancy:.4f}")
    print(f"  Context Precision:  {metrics.context_precision:.4f}")
    print(f"  Context Recall:     {metrics.context_recall:.4f}")

    print("\n[Target Check]")
    for name, passed in metrics.check_targets().items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo without actual LLM calls
    print("Generation Evaluation Module")
    print("Usage: evaluator.evaluate(dataset, llm=your_llm)")

    dataset = create_synthetic_dataset(num_samples=5)
    print(f"\nSynthetic dataset created with {len(dataset.samples)} samples")
    print("Sample question:", dataset.samples[0].question)
    print("Sample answer:", dataset.samples[0].answer)
