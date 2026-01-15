"""
Query Classification & Routing
"""
from enum import Enum
import json
from typing import Dict
import anthropic
from config.settings import settings

class QueryType(Enum):
    RETRIEVAL = "retrieval"  # Factual questions answerable from docs
    GENERATION = "generation"  # Creative, summarization, analysis
    CLARIFICATION = "clarification"  # Ambiguous queries
    REJECTION = "rejection"  # Out of scope
    
    
class QueryRouter:
    """
    Routes queries to appropriate handlers:
    - Not every query should go to RAG
    - Classifies intent before expensive retrieval
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        if use_llm:
            self.client = anthropic.Anthropic()
            
    def classify_query(self, query:str) -> Dict:
        """
        Classify query type and confidence
        
        Returns:
            {"type": QueryType, "Confidence": float, "reasoning": str}
        """
        if self.use_llm:
            return self._llm_classify(query)  
        else:
            return self._rule_based_classify(query)
        
    def _rule_based_classify(self, query: str) -> Dict:
        """Fast rule-based classification"""
        query_lower= query.lower()
        
        # Rejection patterns (out of scope)
        rejection_keywords = ["weather", "stock price", "current news", "today's"]
        if any(kw in query_lower for kw in rejection_keywords):
            return {
                "type": QueryType.REJECTION,
                "confidence": 0.85,
                "reasoning": "Query appears to be about real-time data"
            }
        
        # Clarification (too short or ambiguous)
        if len(query.split()) < 3:
            return {
                "type" : QueryType.CLARIFICATION,
                "confidence" : 0.80,
                "reasoning" : "Query is too brief"
            }
            
        # Generate patterns
        generation_keywrod = {"create", "write", "generate", "summarize", "analyze"}
        if any(kw in query_lower for kw in generation_keywrod):
            return {
                "type" : QueryType.GENERATION,
                "confidence": 0.75,
                "reasoning": "Query requests generation/analysis"
            }
        
        # Default to retrieval
        return {
            "type" : QueryType.RETRIEVAL,
            "confidence" : 0.70,
            "reasoning" : "Factual question pattern detected"
        }    
        
    def _llm_classify(self, query:str) -> Dict:
        """LLM based classification for better accuracy"""
        prompt = f"""Classify this user query into one of these categories:
1. RETRIEVAL: Factual questions answerable from documents
2. GENERATION: Creative writing, summarization, or analysis tasks
3. CLARIFICATION: Too ambiguous or needs more context
4. REJECTION: Out of scope (real-time data, external services)

Query: "{query}"

Respond in JSON format:
{{"type": "RETRIEVAL|GENERATION|CLARIFICATION|REJECTION", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

        try:
            response = self.client.messages.create(
                model= settings.ROUTING_MODEL,
                max_tokens=200,
                messages={{"role":"user", "content": prompt}}
            )
            
            result = json.loads(response.content[0].text)
            result["type"] = QueryRouter[result["type"]]
            return result
        
        except Exception as e:
            # Fallback to rule-based
            return self._rule_based_classify(query)

            
    