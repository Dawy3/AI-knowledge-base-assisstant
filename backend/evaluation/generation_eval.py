"""
Generation Evaluation using Ragas.

FOCUS: Faithfulness, Answer Relevancy, Context Precision/Recall
MUST: Run before production
TARGET: Faithfulness > 0.85, Answer Relevancy > 0.80
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

logger = logging.getLogger(__name__)


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

    def __init__(self, metrics: Optional[list] = None):
        self.metrics = metrics or [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    def evaluate(self, dataset: GenerationDataset, llm=None, embeddings=None) -> GenerationMetrics:
        """Run Ragas evaluation."""
        hf_dataset = dataset.to_hf_dataset()

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
