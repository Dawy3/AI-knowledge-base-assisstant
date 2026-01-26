"""
Context Builder for RAG Pipeline.

FOCUS: Dynamic context loading by query type
MUST: Simple queries=800 tokens, complex=full
EXPECTED: 40% average token reduction
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Context sizing configuration."""
    simple_max_tokens: int = 800
    complex_max_tokens: int = 4000
    default_max_tokens: int = 2000
    max_chunks: int = 10
    chars_per_token: int = 4  # Rough estimate


class ContextBuilder:
    """
    Builds context dynamically based on query complexity.
    
    FOCUS: Load less context for simple queries, full for complex.
    
    Usage:
        builder = ContextBuilder()
        context = builder.build(chunks, query_type="simple")
    """
    
    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()
    
    def build(
        self,
        chunks: list[dict],
        query_type: str = "default",
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        """
        Build context from chunks based on query type.
        
        Args:
            chunks: List of {content, score, ...}
            query_type: "simple", "complex", or "default"
            max_tokens: Override max tokens
            
        Returns:
            List of context strings fitting within token limit
        """
        if not chunks:
            return []
        
        # Determine token limit
        if max_tokens:
            limit = max_tokens
        elif query_type == "simple":
            limit = self.config.simple_max_tokens
        elif query_type == "complex":
            limit = self.config.complex_max_tokens
        else:
            limit = self.config.default_max_tokens
        
        # Build context within limit
        context = []
        total_chars = 0
        max_chars = limit * self.config.chars_per_token
        
        for chunk in chunks[:self.config.max_chunks]:
            content = chunk.get("content", "")
            chunk_chars = len(content)
            
            if total_chars + chunk_chars > max_chars:
                # Truncate last chunk if needed
                remaining = max_chars - total_chars
                if remaining > 100:  # Worth including partial
                    context.append(content[:remaining] + "...")
                break
            
            context.append(content)
            total_chars += chunk_chars
        
        logger.debug(
            f"Built context: {len(context)} chunks, "
            f"~{total_chars // self.config.chars_per_token} tokens "
            f"(limit: {limit}, type: {query_type})"
        )
        
        return context
    
    def build_with_sources(
        self,
        chunks: list[dict],
        query_type: str = "default",
    ) -> tuple[list[str], list[dict]]:
        """
        Build context with source tracking.
        
        Returns:
            (context_strings, source_metadata)
        """
        context = self.build(chunks, query_type)
        
        sources = []
        for i, chunk in enumerate(chunks[:len(context)]):
            sources.append({
                "index": i,
                "document_id": chunk.get("document_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "score": chunk.get("score", 0),
            })
        
        return context, sources
    
    def format_context(
        self,
        chunks: list[dict],
        query_type: str = "default",
        include_sources: bool = False,
    ) -> str:
        """
        Build and format context as single string.
        
        Args:
            chunks: Retrieved chunks
            query_type: Query complexity type
            include_sources: Add source references
            
        Returns:
            Formatted context string
        """
        context = self.build(chunks, query_type)
        
        if not context:
            return ""
        
        if include_sources:
            formatted = []
            for i, text in enumerate(context):
                source_id = chunks[i].get("document_id", f"source_{i}")
                formatted.append(f"[{i+1}] {text}")
            return "\n\n".join(formatted)
        
        return "\n\n---\n\n".join(context)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // self.config.chars_per_token