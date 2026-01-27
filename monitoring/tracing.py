"""
Distributed Tracing using LangSmith.

FOCUS: Track latency per component
MUST: Trace retrieval, generation, full pipeline

Setup:
    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_API_KEY=your-api-key
    export LANGCHAIN_PROJECT=rag-backend
"""

import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional

from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree

# Check if LangSmith is configured
LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"


class LangSmithTracer:
    """
    LangSmith tracing wrapper.
    
    Usage:
        tracer = LangSmithTracer()
        
        @tracer.trace("retrieval")
        async def retrieve(query):
            ...
        
        # Or manual
        with tracer.span("custom_operation"):
            ...
    """
    
    def __init__(self, project_name: Optional[str] = None):
        self.project = project_name or os.getenv("LANGCHAIN_PROJECT", "rag-backend")
        self.enabled = LANGSMITH_ENABLED
        
        if self.enabled:
            self.client = Client()
        else:
            self.client = None
    
    def trace(
        self,
        name: str,
        run_type: str = "chain",
        metadata: Optional[dict] = None,
    ) -> Callable:
        """
        Decorator to trace a function.
        
        Args:
            name: Name for the trace
            run_type: Type (chain, llm, retriever, tool)
            metadata: Additional metadata
        """
        def decorator(func: Callable) -> Callable:
            if not self.enabled:
                return func
            
            @wraps(func)
            @traceable(name=name, run_type=run_type, metadata=metadata)
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            
            @wraps(func)
            @traceable(name=name, run_type=run_type, metadata=metadata)
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            # Return appropriate wrapper
            if _is_async(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    @contextmanager
    def span(self, name: str, metadata: Optional[dict] = None):
        """
        Context manager for manual tracing.
        
        Usage:
            with tracer.span("my_operation"):
                # do work
                pass
        """
        if not self.enabled:
            yield
            return
        
        # Use LangSmith's traceable as context
        run_tree = get_current_run_tree()
        if run_tree:
            child = run_tree.create_child(name=name, run_type="chain")
            child.post()
            try:
                yield child
                child.end()
            except Exception as e:
                child.end(error=str(e))
                raise
        else:
            yield None
    
    def log_feedback(
        self,
        run_id: str,
        score: float,
        feedback_type: str = "user",
        comment: Optional[str] = None,
    ) -> None:
        """
        Log feedback for a run.
        
        Args:
            run_id: The run ID to attach feedback to
            score: Score (0-1)
            feedback_type: Type of feedback
            comment: Optional comment
        """
        if not self.enabled or not self.client:
            return
        
        self.client.create_feedback(
            run_id=run_id,
            key=feedback_type,
            score=score,
            comment=comment,
        )
    
    def get_run_url(self, run_id: str) -> Optional[str]:
        """Get URL to view run in LangSmith."""
        if not self.enabled:
            return None
        
        return f"https://smith.langchain.com/runs/{run_id}"


def _is_async(func: Callable) -> bool:
    """Check if function is async."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


# Convenience decorators
def trace_retrieval(func: Callable) -> Callable:
    """Trace retrieval operations."""
    return traceable(name="retrieval", run_type="retriever")(func) if LANGSMITH_ENABLED else func


def trace_generation(func: Callable) -> Callable:
    """Trace generation operations."""
    return traceable(name="generation", run_type="llm")(func) if LANGSMITH_ENABLED else func


def trace_chain(name: str) -> Callable:
    """Trace chain operations."""
    def decorator(func: Callable) -> Callable:
        return traceable(name=name, run_type="chain")(func) if LANGSMITH_ENABLED else func
    return decorator