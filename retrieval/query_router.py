"""
Query Router: Classify queries to determine if they should be answered
"""
from enum import Enum
from typing import Dict
import logging
from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query classification types"""
    ANSWERABLE = "answerable"
    REJECTION = "rejection"
    CLARIFICATION = "clarification"


class QueryRouter:
    """
    Route queries to determine if they should be answered, rejected, or need clarification
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        if use_llm:
            self.client = OpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL
            )
    
    def classify_query(self, query: str) -> Dict:
        """
        Classify a query to determine if it should be answered
        
        Args:
            query: User query
            
        Returns:
            Dict with 'type' and 'reasoning'
        """
        if not self.use_llm:
            # Simple heuristic-based routing
            return self._heuristic_routing(query)
        
        # LLM-based routing
        prompt = f"""Classify the following query into one of these categories:
1. ANSWERABLE - The query can be answered based on knowledge base content
2. REJECTION - The query is inappropriate, harmful, or cannot be answered
3. CLARIFICATION - The query is ambiguous and needs clarification

Query: {query}

Respond in JSON format:
{{
    "type": "ANSWERABLE|REJECTION|CLARIFICATION",
    "reasoning": "brief explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=settings.ROUTING_MODEL,
                messages=[
                    {"role": "system", "content": "You are a query classification assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Map to QueryType enum
            query_type_str = result.get("type", "ANSWERABLE").upper()
            if query_type_str == "REJECTION":
                return {"type": QueryType.REJECTION, "reasoning": result.get("reasoning", "")}
            elif query_type_str == "CLARIFICATION":
                return {"type": QueryType.CLARIFICATION, "reasoning": result.get("reasoning", "")}
            else:
                return {"type": QueryType.ANSWERABLE, "reasoning": result.get("reasoning", "")}
                
        except Exception as e:
            logger.warning(f"Query routing error: {e}, defaulting to answerable")
            return {"type": QueryType.ANSWERABLE, "reasoning": "Routing failed, defaulting to answerable"}
    
    def _heuristic_routing(self, query: str) -> Dict:
        """Simple heuristic-based routing"""
        query_lower = query.lower()
        
        # Rejection patterns
        rejection_patterns = ["hack", "exploit", "illegal", "harmful"]
        if any(pattern in query_lower for pattern in rejection_patterns):
            return {"type": QueryType.REJECTION, "reasoning": "Query contains inappropriate content"}
        
        # Default to answerable
        return {"type": QueryType.ANSWERABLE, "reasoning": "Query appears answerable"}

