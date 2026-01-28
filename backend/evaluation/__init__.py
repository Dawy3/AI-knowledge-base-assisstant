"""
RAG Evaluation Modules.

Provides:
- Retrieval evaluation (Recall@K, MRR, Hit Rate, NDCG)
- Generation evaluation (RAGAS metrics)
- LLM-as-Judge evaluation
"""

from backend.evaluation.retrieval_eval import (
    RetrievalMetrics,
    RetrievalEvaluator,
    EvalQuery,
    EvalDataset,
    run_evaluation,
    print_report,
)

from backend.evaluation.generation_eval import (
    GenerationEvaluator,
    GenerationMetrics,
    EvalSample,
    GenerationDataset,
)

from backend.evaluation.llm_judge import (
    LLMJudge,
    JudgeModel,
    JudgmentResult,
    JudgeMetrics,
    print_judgment_report,
)

__all__ = [
    # Retrieval
    "RetrievalMetrics",
    "RetrievalEvaluator",
    "EvalQuery",
    "EvalDataset",
    "run_evaluation",
    "print_report",
    # Generation
    "GenerationEvaluator",
    "GenerationMetrics",
    "EvalSample",
    "GenerationDataset",
    # LLM Judge
    "LLMJudge",
    "JudgeModel",
    "JudgmentResult",
    "JudgeMetrics",
    "print_judgment_report",
]
