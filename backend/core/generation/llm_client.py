"""
LLM Client for RAG Pipeline.

FOCUS: Support streaming + batching
MUST: Implement retries, timeouts, fallbacks

MODEL ROUTING PATTERN - 3-Tier Architecture:
- Tier 1 (70% queries): Simple model (GPT-3.5) for straightforward Q&A
- Tier 2 (25% queries): Medium model (GPT-4o-mini) for moderate complexity
- Tier 3 (5% queries): Best model (GPT-4) for complex reasoning

Supports:
- OpenAI API (GPT-4, GPT-4o-mini, GPT-3.5)
- Automatic fallback on failure
- Tiered model routing based on query complexity
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: dict  # {prompt_tokens, completion_tokens, total_tokens}
    latency_ms: float
    is_fallback: bool = False
    
class ModelTier(str):
    """Model tiers for routing pattern."""
    TIER_1 = "tier_1"  # Simple model (70% of queries) - straightforward Q&A
    TIER_2 = "tier_2"  # Medium model (25% of queries) - moderate complexity
    TIER_3 = "tier_3"  # Best model (5% of queries) - complex reasoning

# Default models for each tier
TIER_MODELS = {
    ModelTier.TIER_1: "gpt-3.5-turbo",      # Simple queries (70%)
    ModelTier.TIER_2: "gpt-4o-mini",         # Moderate queries (25%)
    ModelTier.TIER_3: "gpt-4-turbo-preview", # Complex queries (5%)
}


class LLMClient:
    """
    Unified LLM client with retries and fallbacks.
    
    MODEL ROUTING PATTERN - 3-Tier Architecture:
    - Tier 1 (70%): Simple model (GPT-3.5) for straightforward Q&A
    - Tier 2 (25%): Medium model (GPT-4o-mini) for moderate complexity
    - Tier 3 (5%): Best model (GPT-4) for complex reasoning

    Usage:
        client = LLMClient(api_key="...")

        # Non-streaming - uses tier routing
        response = await client.generate(prompt, system_prompt, tier=ModelTier.TIER_1)

        # Streaming
        async for chunk in client.stream(prompt, system_prompt):
            print(chunk, end="")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        tier1_model: str = "gpt-3.5-turbo",
        tier2_model: str = "gpt-4o-mini",
        tier3_model: str = "gpt-4-turbo-preview",
        fallback_model: str = "gpt-3.5-turbo",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Args:
            api_key: OpenAI API key
            tier1_model: Tier 1 model for simple queries (70%)
            tier2_model: Tier 2 model for moderate queries (25%)
            tier3_model: Tier 3 model for complex queries (5%)
            fallback_model: Fallback on primary failure
            timeout: Request timeout in seconds
            max_retries: Max retry attempts
        """
        self.api_key = api_key
        self.tier1_model = tier1_model
        self.tier2_model = tier2_model
        self.tier3_model = tier3_model
        self.fallback_model = fallback_model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Tier model mapping
        self._tier_models = {
            ModelTier.TIER_1: tier1_model,
            ModelTier.TIER_2: tier2_model,
            ModelTier.TIER_3: tier3_model,
        }

        self._openai_base = "https://api.openai.com/v1"
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if  self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    def get_model_for_tier(self, tier: str) -> str:
        """Get the model name for a given tier."""
        return self._tier_models.get(tier, self.tier1_model)
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        tier: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Generate response (non-streaming).

        3-Tier Model Routing:
        - Tier 1: Simple model for straightforward Q&A
        - Tier 2: Medium model for moderate complexity
        - Tier 3: Best model for complex reasoning

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Specific model override (if None, uses tier)
            tier: Model tier (TIER_1, TIER_2, TIER_3). Defaults to TIER_3 if not specified.
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        MUST: Retries with exponential backoff, fallback on failure.
        """
        # Determine model from tier or use explicit model
        if model is None:
            tier = tier or ModelTier.TIER_3     # Default to best model if not specified
            model = self.get_model_for_tier(tier)
            
        start = time.perf_counter()
        
        try:
            response = await self._call_with_retry(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return LLMResponse(
                content=response["content"],
                model=model,
                usage=response["usage"],
                latency_ms=(time.perf_counter() - start) * 1000,
                is_fallback=False,
            )
        except Exception as e:
            logger.warning(f"Primary model {model} failed: {e}, trying fallback")

            # Try fallback (Tier 1 model is the fallback)
            try:
                response = await self._call_with_retry(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=self.fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return LLMResponse(
                    content=response["content"],
                    model=self.fallback_model,
                    usage=response["usage"],
                    latency_ms=(time.perf_counter() - start) * 1000,
                    is_fallback=True,
                )
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _call_with_retry(
        self,
        prompt:str,
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Make API call with retry logic."""
        client = await self._get_client()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Use OpenAI API endpoint
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        response = await client.post(
            f"{self._openai_base}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens" : max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "content" : data["choices"][0]["message"]["content"],
            "usage" : data.get("usage", {}),
        }

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        tier: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """
        Stream response chunks.

        3-Tier Model Routing:
        - Tier 1: Simple model for straightforward Q&A
        - Tier 2: Medium model for moderate complexity
        - Tier 3: Best model for complex reasoning

        Usage:
            async for chunk in client.stream(prompt, tier=ModelTier.TIER_1):
                print(chunk, end="", flush=True)
        """
        # Determine model from tier or use explicit model
        if model is None:
            tier = tier or ModelTier.TIER_3   # Default to best model if not specified
            model = self.get_model_for_tier(tier)
        
        client = await self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role":"system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        async with client.stream(
            "POST",
            f"{self._openai_base}/chat/completions",
            headers=headers,
            json={
                "model" : model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            
            async for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    
                    try:
                        import json
                        chunk = json.loads(data)
                        content = chunk["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except:
                        continue
                    
    async def generate_batch(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        tier: Optional[str] = None,
        max_concurrent: int = 5,
    ) -> list[LLMResponse]:
        """
        Generate responses for multiple prompts concurrently.

        Args:
            prompts: List of prompts
            system_prompt: Optional system prompt
            model: Specific model override
            tier: Model tier (TIER_1, TIER_2, TIER_3)
            max_concurrent: Max concurrent requests
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_one(prompt: str) -> LLMResponse:
            async with semaphore:
                return await self.generate(prompt, system_prompt, model, tier)
        
        return await asyncio.gather(*[generate_one(p) for p in prompts])
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    # Convenience methods for tier-based generation
    async def generate_simple(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate using Tier 1 (simple) model for straightforward Q&A."""
        return await self.generate(prompt, system_prompt, tier=ModelTier.TIER_1, **kwargs)

    async def generate_moderate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate using Tier 2 (moderate) model for moderate complexity."""
        return await self.generate(prompt, system_prompt, tier=ModelTier.TIER_2, **kwargs)

    async def generate_complex(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate using Tier 3 (best) model for complex reasoning."""
        return await self.generate(prompt, system_prompt, tier=ModelTier.TIER_3, **kwargs)