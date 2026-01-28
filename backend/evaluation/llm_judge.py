"""
LLM-as-Judge Evaluation Module.

Uses GPT-4/Claude to rate RAG response quality on a 1-5 scale.

EVALUATION DIMENSIONS:
- Relevance: Does the answer address the question?
- Faithfulness: Is the answer grounded in the provided context?
- Completeness: Does the answer fully address the question?
- Conciseness: Is the answer appropriately concise?

COST: ~$0.03-0.05 per evaluation (GPT-4)
RECOMMENDATION: Use for 50-100 samples to calibrate with human eval
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class JudgeModel(Enum):
    """Available judge models."""

    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    CLAUDE_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_HAIKU = "claude-3-5-haiku-20241022"


@dataclass
class JudgmentScore:
    """Single dimension judgment."""

    dimension: str
    score: int  # 1-5
    explanation: str


@dataclass
class JudgmentResult:
    """Complete judgment for one sample."""

    query: str
    answer: str
    contexts: list[str]
    scores: list[JudgmentScore]
    overall_score: float
    passed: bool
    model: str
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "contexts": self.contexts,
            "scores": [
                {"dimension": s.dimension, "score": s.score, "explanation": s.explanation}
                for s in self.scores
            ],
            "overall_score": self.overall_score,
            "passed": self.passed,
            "model": self.model,
            "timestamp": self.timestamp,
        }


@dataclass
class JudgeMetrics:
    """Aggregated judgment metrics."""

    relevance_avg: float = 0.0
    faithfulness_avg: float = 0.0
    completeness_avg: float = 0.0
    conciseness_avg: float = 0.0
    overall_avg: float = 0.0
    pass_rate: float = 0.0
    num_samples: int = 0
    scores_by_dimension: dict[str, list[int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "relevance_avg": self.relevance_avg,
            "faithfulness_avg": self.faithfulness_avg,
            "completeness_avg": self.completeness_avg,
            "conciseness_avg": self.conciseness_avg,
            "overall_avg": self.overall_avg,
            "pass_rate": self.pass_rate,
            "num_samples": self.num_samples,
        }


# Evaluation prompt template
JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system.

Evaluate the following response based on the query and retrieved context.

## Query
{query}

## Retrieved Context
{context}

## Generated Answer
{answer}

## Evaluation Criteria
Rate each dimension from 1-5:

1. **Relevance** (Does the answer address the question?)
   - 5: Directly and fully addresses the question
   - 3: Partially addresses the question
   - 1: Does not address the question at all

2. **Faithfulness** (Is the answer grounded in the provided context?)
   - 5: All claims are supported by the context
   - 3: Some claims lack context support
   - 1: Contains hallucinations or contradicts context

3. **Completeness** (Does the answer fully address the question?)
   - 5: Comprehensive answer with all necessary details
   - 3: Addresses main points but misses some details
   - 1: Incomplete or superficial answer

4. **Conciseness** (Is the answer appropriately concise?)
   - 5: Clear and concise, no unnecessary content
   - 3: Somewhat verbose or could be clearer
   - 1: Excessively verbose or poorly structured

## Response Format
Respond with valid JSON only:
{{
    "relevance": {{"score": <1-5>, "explanation": "<brief explanation>"}},
    "faithfulness": {{"score": <1-5>, "explanation": "<brief explanation>"}},
    "completeness": {{"score": <1-5>, "explanation": "<brief explanation>"}},
    "conciseness": {{"score": <1-5>, "explanation": "<brief explanation>"}}
}}"""


class LLMJudge:
    """LLM-as-Judge evaluator for RAG responses."""

    def __init__(
        self,
        model: JudgeModel = JudgeModel.GPT4O_MINI,
        pass_threshold: float = 3.5,
        api_key: Optional[str] = None,
    ):
        """
        Initialize LLM Judge.

        Args:
            model: Judge model to use
            pass_threshold: Minimum average score to pass (1-5)
            api_key: Optional API key (defaults to env var)
        """
        self.model = model
        self.pass_threshold = pass_threshold
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Get or create API client."""
        if self._client is not None:
            return self._client

        model_name = self.model.value

        if model_name.startswith("gpt"):
            try:
                from openai import OpenAI

                api_key = self.api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                self._client = OpenAI(api_key=api_key)
                self._client_type = "openai"
            except ImportError:
                raise ImportError("openai package required: pip install openai")

        elif model_name.startswith("claude"):
            try:
                from anthropic import Anthropic

                api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not set")
                self._client = Anthropic(api_key=api_key)
                self._client_type = "anthropic"
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")

        return self._client

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt."""
        client = self._get_client()

        if self._client_type == "openai":
            response = client.chat.completions.create(
                model=self.model.value,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        elif self._client_type == "anthropic":
            response = client.messages.create(
                model=self.model.value,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

    def judge_single(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> JudgmentResult:
        """
        Judge a single RAG response.

        Args:
            query: User query
            answer: Generated answer
            contexts: Retrieved context chunks

        Returns:
            JudgmentResult with scores
        """
        # Format context
        context_str = "\n\n".join(
            f"[Context {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )

        # Build prompt
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=query,
            context=context_str,
            answer=answer,
        )

        # Call LLM
        try:
            response = self._call_llm(prompt)
            result = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse judge response: {e}")
            # Return default low scores
            result = {
                dim: {"score": 1, "explanation": "Parse error"}
                for dim in ["relevance", "faithfulness", "completeness", "conciseness"]
            }
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

        # Extract scores
        scores = []
        for dim in ["relevance", "faithfulness", "completeness", "conciseness"]:
            dim_result = result.get(dim, {"score": 1, "explanation": "Missing"})
            scores.append(
                JudgmentScore(
                    dimension=dim,
                    score=dim_result.get("score", 1),
                    explanation=dim_result.get("explanation", ""),
                )
            )

        # Calculate overall
        overall = sum(s.score for s in scores) / len(scores)
        passed = overall >= self.pass_threshold

        return JudgmentResult(
            query=query,
            answer=answer,
            contexts=contexts,
            scores=scores,
            overall_score=overall,
            passed=passed,
            model=self.model.value,
            timestamp=datetime.now().isoformat(),
        )

    def judge_batch(
        self,
        samples: list[dict],
        verbose: bool = True,
    ) -> tuple[list[JudgmentResult], JudgeMetrics]:
        """
        Judge a batch of samples.

        Args:
            samples: List of {"query": str, "answer": str, "contexts": list[str]}
            verbose: Print progress

        Returns:
            Tuple of (results list, aggregated metrics)
        """
        results = []
        dimension_scores = {
            "relevance": [],
            "faithfulness": [],
            "completeness": [],
            "conciseness": [],
        }

        for i, sample in enumerate(samples):
            if verbose:
                print(f"Judging sample {i + 1}/{len(samples)}...", end="\r")

            try:
                result = self.judge_single(
                    query=sample["query"],
                    answer=sample["answer"],
                    contexts=sample["contexts"],
                )
                results.append(result)

                for score in result.scores:
                    dimension_scores[score.dimension].append(score.score)

            except Exception as e:
                logger.error(f"Failed to judge sample {i}: {e}")
                continue

        if verbose:
            print()

        # Calculate metrics
        num_samples = len(results)
        if num_samples == 0:
            return results, JudgeMetrics()

        metrics = JudgeMetrics(
            relevance_avg=sum(dimension_scores["relevance"]) / num_samples,
            faithfulness_avg=sum(dimension_scores["faithfulness"]) / num_samples,
            completeness_avg=sum(dimension_scores["completeness"]) / num_samples,
            conciseness_avg=sum(dimension_scores["conciseness"]) / num_samples,
            overall_avg=sum(r.overall_score for r in results) / num_samples,
            pass_rate=sum(1 for r in results if r.passed) / num_samples,
            num_samples=num_samples,
            scores_by_dimension=dimension_scores,
        )

        return results, metrics

    def judge_from_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> tuple[list[JudgmentResult], JudgeMetrics]:
        """
        Judge samples from JSON file.

        Expected format:
        {
            "samples": [
                {"query": "...", "answer": "...", "contexts": ["...", ...]}
            ]
        }
        """
        with open(input_path) as f:
            data = json.load(f)

        samples = data.get("samples", data.get("data", []))
        results, metrics = self.judge_batch(samples)

        if output_path:
            output_data = {
                "metrics": metrics.to_dict(),
                "results": [r.to_dict() for r in results],
                "model": self.model.value,
                "timestamp": datetime.now().isoformat(),
            }
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Results saved to {output_path}")

        return results, metrics


def print_judgment_report(metrics: JudgeMetrics, results: list[JudgmentResult]) -> None:
    """Print formatted judgment report."""
    print("\n" + "=" * 60)
    print("LLM-AS-JUDGE EVALUATION REPORT")
    print("=" * 60)

    print("\n[Average Scores (1-5)]")
    print(f"  Relevance:     {metrics.relevance_avg:.2f}")
    print(f"  Faithfulness:  {metrics.faithfulness_avg:.2f}")
    print(f"  Completeness:  {metrics.completeness_avg:.2f}")
    print(f"  Conciseness:   {metrics.conciseness_avg:.2f}")
    print(f"\n  Overall:       {metrics.overall_avg:.2f}")
    print(f"  Pass Rate:     {metrics.pass_rate * 100:.1f}%")
    print(f"  Samples:       {metrics.num_samples}")

    # Show worst performers
    if results:
        sorted_results = sorted(results, key=lambda r: r.overall_score)
        print("\n[Lowest Scoring Samples]")
        for r in sorted_results[:3]:
            print(f"  Score {r.overall_score:.2f}: {r.query[:50]}...")

    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM-as-Judge evaluation")
    parser.add_argument("--input", "-i", required=True, help="Path to samples JSON")
    parser.add_argument("--output", "-o", help="Path to save results")
    parser.add_argument(
        "--model",
        "-m",
        default="gpt-4o-mini",
        choices=[m.value for m in JudgeModel],
        help="Judge model to use",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=3.5,
        help="Pass threshold (1-5)",
    )

    args = parser.parse_args()

    # Find model enum
    model = next(m for m in JudgeModel if m.value == args.model)

    judge = LLMJudge(model=model, pass_threshold=args.threshold)
    results, metrics = judge.judge_from_file(args.input, args.output)

    print_judgment_report(metrics, results)
