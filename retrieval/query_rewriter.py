"""
Query Rewriter: Generate multiple query variants for better retrieval
"""
from typing import List
import logging
from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Rewrite queries to generate multiple variants for better retrieval coverage
    """
    
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL
        )
    
    def rewrite_query(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Generate query variants
        
        Args:
            query: Original query
            num_variants: Number of variants to generate
            
        Returns:
            List of query variants (including original)
        """
        prompt = f"""Generate {num_variants} different ways to ask the following question. 
Each variant should:
- Maintain the same intent
- Use different wording/phrasing
- Be suitable for document retrieval

Original query: {query}

Respond with only the query variants, one per line, no numbering or bullets."""

        try:
            response = self.client.chat.completions.create(
                model=settings.LLM,
                messages=[
                    {"role": "system", "content": "You are a query rewriting assistant. Generate query variants."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            variants_text = response.choices[0].message.content.strip()
            variants = [v.strip() for v in variants_text.split("\n") if v.strip()]
            
            # Include original query and variants
            all_queries = [query] + variants[:num_variants]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in all_queries:
                if q.lower() not in seen:
                    seen.add(q.lower())
                    unique_queries.append(q)
            
            return unique_queries[:num_variants + 1]
            
        except Exception as e:
            logger.warning(f"Query rewriting error: {e}, using original query only")
            return [query]

