"""
Structured Logging for RAG Pipeline.

FOCUS: Log EVERY query + retrieval + response
MUST: Include query_id, user_id, timestamps
PURPOSE: Build evaluation dataset
"""

import json
import logging
import sys
import time 
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional

# Context variable for request tracking
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
) -> None:
    """
    Setup structured logging.
    
    Args:
        level: Log level
        json_format: Use JSON format (recommended for production)

    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers = [handler]


class JsonFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(""),
        }
        
        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data)


@dataclass
class QueryLog:
    """Structured log for a query."""

    query_id: str
    query: str
    user_id: Optional[str] = None
    query_type: Optional[str] = None

    # Retrieval
    chunks_retrieved: int = 0
    retrieval_latency_ms: float = 0
    search_type: str = "hybrid"
    
    # Generation
    response: str = ""
    model: str = ""
    generation_latency_ms: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    # Cache
    cache_hit: bool = False
    cache_layer: Optional[str] = None
    
    # Timing
    total_latency_ms: float = 0
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


class QueryLogger:
    """
    Logger for RAG queries.
    
    FOCUS: Log EVERY query for evaluation dataset building.
    
    Usage:
        logger = QueryLogger()
        
        log = logger.start_query("What is RAG?", user_id="123")
        log.chunks_retrieved = 5
        log.retrieval_latency_ms = 50
        log.response = "RAG is..."
        logger.end_query(log)
    """
    
    def __init__(self, logger_name: str = "rag.queries"):
        self.logger = logging.getLogger(logger_name)
    
    def start_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        query_id: Optional[str] = None,
    ) -> QueryLog:
        """Start tracking a query."""
        query_id = query_id or str(uuid.uuid4())
        
        # Set request context
        request_id_var.set(query_id)
        
        log = QueryLog(
            query_id=query_id,
            query=query,
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat(),
        )
        
        self.logger.info(f"Query started: {query[:100]}...")
        return log
    
    def end_query(self, log: QueryLog) -> None:
        """Complete query logging."""
        self.logger.info(
            "Query completed",
            extra={"extra_data": log.to_dict()},
        )
    
    def log_retrieval(
        self,
        log: QueryLog,
        chunks: int,
        latency_ms: float,
        search_type: str = "hybrid",
    ) -> None:
        """Log retrieval step."""
        log.chunks_retrieved = chunks
        log.retrieval_latency_ms = latency_ms
        log.search_type = search_type
        
        self.logger.debug(
            f"Retrieved {chunks} chunks in {latency_ms:.1f}ms"
        )
    
    def log_generation(
        self,
        log: QueryLog,
        response: str,
        model: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Log generation step."""
        log.response = response
        log.model = model
        log.generation_latency_ms = latency_ms
        log.prompt_tokens = prompt_tokens
        log.completion_tokens = completion_tokens
        
        self.logger.debug(
            f"Generated {completion_tokens} tokens in {latency_ms:.1f}ms"
        )
    
    def log_cache_hit(self, log: QueryLog, layer: Optional[str] = None) -> None:
        """Log cache hit."""
        log.cache_hit = True
        log.cache_layer = layer
        if layer:
            self.logger.debug(f"Cache hit (layer: {layer})")
        else:
            self.logger.debug("Cache hit")
    
    def log_error(self, log: QueryLog, error: str) -> None:
        """Log error."""
        self.logger.error(
            f"Query failed: {error}",
            extra={"extra_data": {"query_id": log.query_id, "error": error}},
        )