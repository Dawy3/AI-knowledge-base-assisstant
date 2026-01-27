"""
Conversation History Manager.

FOCUS: Sliding window (last 3 full, rest summarized)
MUST: Last 3 messages full, messages 4+ = 150 tokens max
EXPECTED: 60-70% history token reduction
AVOID: Sending full history every time
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Single conversation message."""
    role: str  # "user" or "assistant"
    content: str
    token_count: int = 0
    
@dataclass
class ConversationConfig:
    """Configuration for conversation memory."""
    full_messages: int = 3  # Keep last N messages in full
    summary_max_tokens: int = 150  # Max tokens for older messages
    max_total_tokens: int = 2000  # Max total history tokens
    chars_per_token: int = 4  # Rough estimate
    
class ConversationMemory:
    """
    Conversation history with sliding window compression.
    
    FOCUS: Last 3 full, older summarized
    EXPECTED: 60-70% token reduction
    
    Usage:
        memory = ConversationMemory(conversation_id="123")
        memory.add("user", "What is RAG?")
        memory.add("assistant", "RAG is...")
        
        history = memory.get_history()  # Compressed history
    """
    def __init__(
        self,
        conversation_id: str,
        config: Optional[ConversationConfig] = None,
    ):
        self.conversation_id = conversation_id
        self.config = config or ConversationConfig()
        self.messages: list[Message] = []
        self._summary: Optional[str] = None
        
    def add(self, role:str, content: str) -> None:
        """Add message to history."""
        token_count = len(content) // self.config.chars_per_token
        self.messages.append(Message(
            role=role,
            content=content,
            token_count=token_count,
        ))
    
    def get_history(self) -> list[dict]:
        """
        Get compressed conversation history.
        
        Returns last N messages full, older ones summarized.
        """
        if not self.messages:
            return []

        full_count = self.config.full_messages
        
        # If few messages, return all
        if len(self.messages) <= full_count:
            return [{"role": m.role, "content": m.content} for m in self.messages]
        
        # Split: older messages + recent full messages
        older = self.messages[:-full_count]
        recent = self.messages[-full_count:]
        
        result = []
        
        # Add compressed older messages
        if older:
            summary = self._compress_messages(older)
            if summary:
                result.append({
                    "role": "system",
                    "content": f"Previous conversation summary: {summary}",
                })
        
        # Add recent messages in full
        for m in recent:
            result.append({"role": m.role, "content": m.content})
        
        return result
    
    def _compress_messages(self, messages: list[Message]) -> str:
        """
        Compress older messages into summary.
        
        Simple approach: Truncate each message, keep key points.
        For production: Use LLM to summarize.
        """
        max_per_message = self.config.summary_max_tokens // max(len(messages), 1)
        max_chars = max_per_message * self.config.chars_per_token
        
        summaries = []
        for m in messages:
            # Truncate long messages
            content = m.content[:max_chars]
            if len(m.content) > max_chars:
                content += "..."
            summaries.append(f"{m.role}: {content}")
        
        return " | ".join(summaries)
    
    def get_last_n(self, n: int) -> list[dict]:
        """Get last N messages without compression."""
        recent = self.messages[-n:] if n > 0 else []
        return [{"role": m.role, "content": m.content} for m in recent]
    
    def get_token_count(self) -> int:
        """Get total token count of current history."""
        history = self.get_history()
        total = sum(len(m["content"]) for m in history)
        return total // self.config.chars_per_token
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        self._summary = None
    
    def to_dict(self) -> dict:
        """Serialize conversation."""
        return {
            "conversation_id": self.conversation_id,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in self.messages
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMemory":
        """Deserialize conversation."""
        memory = cls(conversation_id=data["conversation_id"])
        for m in data.get("messages", []):
            memory.add(m["role"], m["content"])
        return memory

        