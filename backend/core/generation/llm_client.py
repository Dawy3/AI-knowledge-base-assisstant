"""
LLM Client for RAG Pipeline.

FOCUS: Support streaming + batching
MUST: Implement retries, timeouts, fallbacks

Execution layer for LLM API calls with tier-based routing.
Tier decisions come from core.query.router.QueryRouter.

Supports:
- Multiple providers: OpenRouter, OpenAI, Local (via LLM__PROVIDER env)
- Automatic fallback on failure
- Streaming and batching
- Retry logic with exponential backoff

Configuration via .env:
    LLM__PROVIDER=openrouter   # or: openai, local
    LLM__MODEL=openai/gpt-4o-mini
    OPENROUTER_API_KEY=sk-or-xxx
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings
from ..query.router import ModelTier, TIER_TO_MODEL

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: dict  # {prompt_tokens, completion_tokens, total_tokens}
    latency_ms: float
    is_fallback: bool = False


class LLMClient:
    """
    Unified LLM client with retries and fallbacks.

    Execution layer for LLM API calls. Tier routing decisions should come from
    core.query.router.QueryRouter.

    Usage:
        from core.query.router import QueryRouter
        from core.generation.llm_client import LLMClient

        # Initialize
        router = QueryRouter()
        client = LLMClient()

        # Route query intelligently
        routing_decision = router.route(query)

        # Execute with routed tier
        response = await client.generate(
            prompt=query,
            tier=routing_decision.tier
        )

        # Streaming
        async for chunk in client.stream(prompt, tier=routing_decision.tier):
            print(chunk, end="")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        tier1_model: Optional[str] = None,
        tier2_model: Optional[str] = None,
        tier3_model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Initialize LLM client.

        Tier routing is only applied for OpenAI provider.
        For OpenRouter/local, the single LLM__MODEL is used for all requests.

        Args:
            api_key: API key (defaults to config based on provider)
            tier1_model: Tier 1 model for simple queries (OpenAI only)
            tier2_model: Tier 2 model for moderate queries (OpenAI only)
            tier3_model: Tier 3 model for complex queries (OpenAI only)
            fallback_model: Fallback on primary failure
            timeout: Request timeout in seconds (defaults to config)
            max_retries: Max retry attempts (defaults to config)
        """
        # Get provider from config
        self.provider = settings.llm.provider
        self.use_tier_routing = self.provider == "openai"

        # Use config defaults based on provider
        self.api_key = api_key or settings.llm.api_key
        self._base_url = settings.llm.base_url

        # Model configuration depends on provider
        if self.use_tier_routing:
            # OpenAI: Use tier-based routing
            self.tier1_model = tier1_model or settings.llm.effective_tier1_model or "gpt-3.5-turbo"
            self.tier2_model = tier2_model or settings.llm.effective_tier2_model or "gpt-4o-mini"
            self.tier3_model = tier3_model or settings.llm.effective_tier3_model or "gpt-4"
            self.fallback_model = fallback_model or self.tier1_model
        else:
            # OpenRouter/Local: Use single model for all tiers
            single_model = settings.llm.model
            self.tier1_model = single_model
            self.tier2_model = single_model
            self.tier3_model = single_model
            self.fallback_model = fallback_model or single_model

        self.timeout = timeout if timeout is not None else settings.llm.request_timeout
        self.max_retries = max_retries if max_retries is not None else settings.llm.max_retries

        # Tier to model mapping (execution layer only)
        self._tier_models = {
            ModelTier.TIER_1: self.tier1_model,
            ModelTier.TIER_2: self.tier2_model,
            ModelTier.TIER_3: self.tier3_model,
        }

        self._client: Optional[httpx.AsyncClient] = None

        if self.use_tier_routing:
            logger.info(
                f"Initialized LLMClient [OpenAI tier routing]: "
                f"Tier1={self.tier1_model}, "
                f"Tier2={self.tier2_model}, "
                f"Tier3={self.tier3_model}"
            )
        else:
            logger.info(
                f"Initialized LLMClient [{self.provider}]: "
                f"Model={settings.llm.model}, "
                f"BaseURL={self._base_url}"
            )

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

        Use QueryRouter to determine tier intelligently:
            routing_decision = router.route(query)
            response = await client.generate(prompt, tier=routing_decision.tier)

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Specific model override (if None, uses tier)
            tier: Model tier from QueryRouter (TIER_1, TIER_2, TIER_3)
                  Defaults to TIER_3 if not specified
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with content, model, usage, and latency

        MUST: Retries with exponential backoff, fallback on failure.
        """
        # Determine model from tier or use explicit model
        if model is None:
            tier = tier or ModelTier.TIER_3  # Default to best model if not specified
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
        
        # Use provider's API endpoint
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # OpenRouter requires additional headers
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/your-app"  # Optional
            headers["X-Title"] = "RAG Knowledge Assistant"  # Optional

        response = await client.post(
            f"{self._base_url}/chat/completions",
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

        Use QueryRouter to determine tier intelligently:
            routing_decision = router.route(query)
            async for chunk in client.stream(prompt, tier=routing_decision.tier):
                print(chunk, end="", flush=True)

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Specific model override (if None, uses tier)
            tier: Model tier from QueryRouter (TIER_1, TIER_2, TIER_3)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Response chunks as they arrive
        """
        # Determine model from tier or use explicit model
        if model is None:
            tier = tier or ModelTier.TIER_3  # Default to best model if not specified
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

        # OpenRouter requires additional headers
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/your-app"
            headers["X-Title"] = "RAG Knowledge Assistant"

        async with client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
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