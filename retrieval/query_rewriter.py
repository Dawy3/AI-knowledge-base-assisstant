"""
Multi-Query Generation: Generate 3-5 query variations  -> (highest ROI)
"""
from typing import List
import anthropic

class QueryRewriter:
    """
    Pre-retrieval optimization: Generate multiple query variations
    to improve recall
    """
    
    def __init__(self):
        self.client = anthropic.Anthropic()
        
    def rewrite_query(self, original_query: str, num_variants: int = 3) -> List[str]:
        """
        Generate query variations for better retrieval
        
        Args:
            original_query: Original User query
            Num_variants : Number of variants to generate
        Returns:
            List of query variants (include original)
        """
        prompt = f"""Generate {num_variants - 1} alternative phrasing of this query to improve document retrieval.
Each variant should:
- Capture the same intent
- Use different keywords/phrases
- Be suitable for semantic search

Original query: "{original_query}"

Return ONLY the alternative queries, one per line, without numbers or explanations."""

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                messages=[{"role":"user", "content": prompt}]
            )
        
            variants = [original_query] # Always include original
            generated = response.content[0].text.strip().split('\n')
            variants.extend([v.strip() for v in generated if v.strip()])
            
            return variants[:num_variants]
        except Exception as e:
            # Fallback: simple variants
            return [
                original_query,
                f"{original_query} explanation",
                f"what is {original_query}"
                
            ][:num_variants]