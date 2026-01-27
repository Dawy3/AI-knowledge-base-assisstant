"""
Prompt Manager for RAG Pipeline.

FOCUS: Compress prompts (2800→900 tokens)
MUST: Remove redundant instructions
MUST: Move edge cases to conditional prompts
EXPECTED: 60-70% token reduction
"""

import logging
from dataclasses import dataclass
from string import Template
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """Prompt configuration."""
    max_context_tokens: int = 2000
    max_history_tokens: int = 500
    include_sources: bool = True


class PromptManager:
    """
    Manages prompt templates with compression.
    
    FOCUS: Minimize tokens while maintaining quality.
    
    Usage:
        pm = PromptManager()
        prompt = pm.build_rag_prompt(query, contexts, history)
    """
    
    # Base system prompt - kept minimal
    SYSTEM_PROMPT = """You are a helpful assistant. Answer based on the provided context. If unsure, say so."""
    
    # RAG prompt template - compressed
    RAG_TEMPLATE = """Context:
$context

Question: $query

Answer based on the context above. Be concise."""
    
    # With history - only when needed
    RAG_WITH_HISTORY_TEMPLATE = """Context:
$context

Previous conversation:
$history

Question: $query

Answer based on context. Be concise."""
    
    # Edge case templates - loaded conditionally
    EDGE_CASE_TEMPLATES = {
        "no_context": """No relevant information found for: $query

Explain that you don't have information on this topic.""",
        
        "out_of_scope": """The question "$query" is outside the knowledge base scope.

Politely explain this is out of scope.""",
        
        "clarification": """The question "$query" is ambiguous.

Ask for clarification about: $ambiguous_parts""",
    }
    
    def __init__(self, config: Optional[PromptConfig] = None):
        self.config = config or PromptConfig()
        self._templates = {
            "rag": Template(self.RAG_TEMPLATE),
            "rag_history": Template(self.RAG_WITH_HISTORY_TEMPLATE),
            **{k: Template(v) for k, v in self.EDGE_CASE_TEMPLATES.items()},
        }
    
    def build_rag_prompt(
        self,
        query: str,
        contexts: list[str],
        history: Optional[list[dict]] = None,
        query_type: str = "normal",
    ) -> tuple[str, str]:
        """
        Build RAG prompt.
        
        Returns: (system_prompt, user_prompt)
        
        FOCUS: Compress prompts - only include what's needed.
        """
        # Handle edge cases with minimal prompts
        if query_type == "out_of_scope":
            return self.SYSTEM_PROMPT, self._templates["out_of_scope"].substitute(query=query)
        
        if query_type == "clarification":
            return self.SYSTEM_PROMPT, self._templates["clarification"].substitute(
                query=query,
                ambiguous_parts="the specific aspect you're asking about",
            )
        
        if not contexts:
            return self.SYSTEM_PROMPT, self._templates["no_context"].substitute(query=query)
        
        # Build context string - compressed
        context_str = self._build_context(contexts)
        
        # Choose template based on history presence
        if history:
            history_str = self._build_history(history)
            prompt = self._templates["rag_history"].substitute(
                context=context_str,
                history=history_str,
                query=query,
            )
        else:
            prompt = self._templates["rag"].substitute(
                context=context_str,
                query=query,
            )
        
        return self.SYSTEM_PROMPT, prompt
    
    def _build_context(self, contexts: list[str]) -> str:
        """
        Build context string with compression.
        
        MUST: Stay within max_context_tokens.
        """
        if not contexts:
            return "No context available."
        
        # Simple concatenation with separators
        # In production, use tiktoken to count and truncate
        combined = "\n---\n".join(contexts)
        
        # Rough truncation (4 chars ≈ 1 token)
        max_chars = self.config.max_context_tokens * 4
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "..."
        
        return combined
    
    def _build_history(self, history: list[dict]) -> str:
        """
        Build conversation history - compressed.
        
        FOCUS: Last 3 full, older ones summarized.
        """
        if not history:
            return ""
        
        lines = []
        for msg in history[-3:]:  # Only last 3
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def get_system_prompt(self, variant: str = "default") -> str:
        """Get system prompt variant."""
        variants = {
            "default": self.SYSTEM_PROMPT,
            "concise": "Answer concisely based on context.",
            "detailed": "Provide detailed answers with explanations based on context.",
        }
        return variants.get(variant, self.SYSTEM_PROMPT)
    
    def add_template(self, name: str, template: str) -> None:
        """Add custom template."""
        self._templates[name] = Template(template)
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate (use tiktoken in production)."""
        return len(text) // 4