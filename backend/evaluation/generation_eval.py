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

import warnings

# Suppress ragas deprecation warnings (metrics will be moved in v1.0)
warnings.filterwarnings("ignore", message=".*Importing.*from 'ragas.metrics'.*deprecated.*")

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

    Uses LLM__PROVIDER and LLM__MODEL from config:
    - openrouter: Uses OPENROUTER_API_KEY, model format "provider/model"
    - openai: Uses OPENAI_API_KEY, model format "gpt-4o-mini"
    - local: Uses LOCAL_LLM_URL, no API key needed

    Fallback to HuggingFace local if no provider configured.

    .env example:
        LLM__PROVIDER=openrouter
        LLM__MODEL=openai/gpt-4o-mini
        OPENROUTER_API_KEY=sk-or-xxx
    """
    if not HAS_LANGCHAIN_OPENAI:
        logger.error("langchain-openai required. Run: pip install langchain-openai")
        return None

    provider = settings.llm.provider
    model = settings.llm.model
    api_key = settings.llm.api_key
    base_url = settings.llm.base_url

    # Use configured provider
    if provider in ("openrouter", "openai") and api_key:
        try:
            logger.info(f"Using {provider} LLM: {model}")
            return ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url if provider == "openrouter" else None,
            )
        except Exception as e:
            logger.warning(f"{provider} LLM setup failed: {e}")

    # Local provider (OpenAI-compatible API)
    if provider == "local":
        try:
            logger.info(f"Using local LLM: {model} at {base_url}")
            return ChatOpenAI(
                model=model,
                api_key="not-needed",  # Local servers often don't need API key
                base_url=base_url,
            )
        except Exception as e:
            logger.warning(f"Local LLM setup failed: {e}")

    # Fallback to HuggingFace local models (completely free, runs locally)
    if HAS_LANGCHAIN_HF and HAS_TRANSFORMERS:
        try:
            model_name = os.getenv("RAGAS_LLM_MODEL", "google/flan-t5-small")
            logger.info(f"Using HuggingFace local LLM: {model_name}")
            pipe = hf_pipeline("text2text-generation", model=model_name, max_length=512)
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            logger.warning(f"HuggingFace LLM setup failed: {e}")

    logger.error(
        "No LLM available. Configure in .env:\n"
        "  LLM__PROVIDER=openrouter\n"
        "  LLM__MODEL=openai/gpt-4o-mini\n"
        "  OPENROUTER_API_KEY=sk-or-xxx"
    )
    return None


def get_ragas_embeddings():
    """
    Get embeddings for Ragas evaluation based on config settings.

    Uses EMBEDDING__MODEL_PROVIDER from config:
    - huggingface/local: Free, runs locally
    - openai: Uses OPENAI_API_KEY

    .env example:
        EMBEDDING__MODEL_PROVIDER=huggingface
        EMBEDDING__MODEL_NAME=e5-large-v2
    """
    provider = settings.embedding.model_provider.lower()

    # For HuggingFace/free models (recommended for evaluation)
    if provider in ("huggingface", "local") and HAS_LANGCHAIN_HF:
        try:
            logger.info(f"Using HuggingFace embeddings: {settings.embedding.model_name}")
            return HuggingFaceEmbeddings(
                model_name=settings.embedding.model_name,
            )
        except Exception as e:
            logger.warning(f"HuggingFace embeddings setup failed: {e}")

    # For OpenAI embeddings
    openai_key = settings.llm.openai_api_key or os.getenv("OPENAI_API_KEY")
    if provider == "openai" and openai_key and HAS_LANGCHAIN_OPENAI:
        try:
            from langchain_openai import OpenAIEmbeddings
            logger.info(f"Using OpenAI embeddings: {settings.embedding.model_name}")
            return OpenAIEmbeddings(
                model=settings.embedding.model_name,
                api_key=openai_key,
            )
        except Exception as e:
            logger.warning(f"OpenAI embeddings setup failed: {e}")

    # Fallback: try HuggingFace anyway
    if HAS_LANGCHAIN_HF:
        try:
            fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Using fallback HuggingFace embeddings: {fallback_model}")
            return HuggingFaceEmbeddings(model_name=fallback_model)
        except Exception as e:
            logger.warning(f"Fallback embeddings setup failed: {e}")

    logger.warning("No embeddings available, Ragas will use defaults")
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
        """Check if metrics meet production targets. NaN values are considered failures."""
        import math
        return {
            "faithfulness": not math.isnan(self.faithfulness) and self.faithfulness >= faithfulness_target,
            "answer_relevancy": not math.isnan(self.answer_relevancy) and self.answer_relevancy >= relevancy_target,
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

        # Convert EvaluationResult to pandas DataFrame and get mean scores
        df = result.to_pandas()
        return GenerationMetrics(
            faithfulness=df["faithfulness"].mean() if "faithfulness" in df.columns else 0.0,
            answer_relevancy=df["answer_relevancy"].mean() if "answer_relevancy" in df.columns else 0.0,
            context_precision=df["context_precision"].mean() if "context_precision" in df.columns else 0.0,
            context_recall=df["context_recall"].mean() if "context_recall" in df.columns else 0.0,
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


def _format_metric(value: float) -> str:
    """Format metric value, handling NaN."""
    import math
    if math.isnan(value):
        return "N/A (model couldn't compute)"
    return f"{value:.4f}"


def print_report(metrics: GenerationMetrics, dataset_name: str = "dataset") -> None:
    """Print formatted evaluation report."""
    import math

    print("\n" + "=" * 60)
    print(f"GENERATION EVALUATION REPORT: {dataset_name}")
    print("=" * 60)

    print("\n[Ragas Metrics]")
    print(f"  Faithfulness:       {_format_metric(metrics.faithfulness)}")
    print(f"  Answer Relevancy:   {_format_metric(metrics.answer_relevancy)}")
    print(f"  Context Precision:  {_format_metric(metrics.context_precision)}")
    print(f"  Context Recall:     {_format_metric(metrics.context_recall)}")

    # Check if any metrics are NaN
    has_nan = any(math.isnan(v) for v in [
        metrics.faithfulness, metrics.answer_relevancy,
        metrics.context_precision, metrics.context_recall
    ])
    if has_nan:
        print("\n[Warning] Some metrics returned N/A.")
        print("  This typically happens with free/small LLMs that don't")
        print("  follow RAGAS's required response format precisely.")
        print("  For accurate results, use gpt-4o-mini or similar.")

    print("\n[Target Check]")
    for name, passed in metrics.check_targets().items():
        value = getattr(metrics, name)
        if math.isnan(value):
            print(f"  {name}: N/A")
        else:
            print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Generation Evaluation Module")
    print("Running RAGAS evaluation...\n")

    # Create synthetic dataset
    dataset = create_synthetic_dataset(num_samples=5)
    print(f"Dataset: {len(dataset.samples)} samples")

    # Run actual evaluation
    metrics, passed = run_evaluation(dataset)

    # Print formatted report
    print_report(metrics, dataset.name)
