"""
OpenAI-compatible API Client.

Supports:
- OpenAI (api.openai.com)
- OpenRouter (openrouter.ai) - same API format
- Any OpenAI-compatible endpoint (local models, Azure, etc.)

OpenRouter free models for testing:
- meta-llama/llama-3.2-3b-instruct:free
- google/gemma-2-9b-it:free  
- mistralai/mistral-7b-instruct:free
- qwen/qwen-2-7b-instruct:free
"""

import logging
import os
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

from backend.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: dict
    finish_reason: str = "stop"

# Preset configurations
PRESETS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4-turbo-preview",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.2-3b-instruct:free",
    },
    "local": {
        "base_url": "http://localhost:8000/v1",
        "default_model": "local-model",
    },
}


# OpenRouter free models for testing
OPENROUTER_FREE_MODELS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
    "huggingfaceh4/zephyr-7b-beta:free",
]

class OpenAIClient:
    """
    OpenAI-compatible API client.

    Works with OpenAI, OpenRouter, and any compatible endpoint.
    Uses config settings by default if api_key not provided.

    Usage:
        # Use config settings (recommended)
        client = OpenAIClient()

        # OpenAI
        client = OpenAIClient(api_key="sk-...", provider="openai")

        # OpenRouter (for free testing)
        client = OpenAIClient(api_key="sk-or-...", provider="openrouter")

        # Custom endpoint
        client = OpenAIClient(api_key="...", base_url="http://localhost:8000/v1")

        response = await client.chat("What is RAG?")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        # Use config settings if not provided
        if provider is None:
            provider = settings.llm.provider  # From LLM__PROVIDER

        if api_key is None:
            api_key = settings.llm.api_key  # Auto-selects based on provider

        if base_url is None:
            base_url = settings.llm.base_url  # Auto-selects based on provider

        if default_model is None:
            default_model = settings.llm.model  # From LLM__MODEL

        if timeout is None:
            timeout = settings.llm.request_timeout

        # Fallback to preset if config not set
        preset = PRESETS.get(provider, PRESETS["openai"])
        self.base_url = base_url or preset["base_url"]
        self.default_model = default_model or preset["default_model"]
        self.provider = provider

        self.client = AsyncOpenAI(
            api_key=api_key or "not-needed",  # Local may not need key
            base_url=self.base_url,
            timeout=timeout,
        )

        logger.info(f"OpenAIClient initialized: provider={provider}, model={self.default_model}")


    async def chat(
        self,
        prompt:str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Send chat completion request.
        
        Args:
            prompt: User message
            system_prompt: Optional system message
            model: Model to use (defaults to default_model)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
        """
        model = model or self.default_model
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        choice = response.choices[0]
        
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            finish_reason=choice.finish_reason or "stop",
        )
        
    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """
        Stream chat completion.
        
        Usage:
            async for chunk in client.stream("Hello"):
                print(chunk, end="", flush=True)
        """
        model = model or self.default_model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    async def embed(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small",
    ) -> list[list[float]]:
        """
        Generate embeddings.
        
        Note: OpenRouter doesn't support embeddings, use OpenAI directly.
        """
        if self.provider == "openrouter":
            raise NotImplementedError("OpenRouter doesn't support embeddings. Use OpenAI directly.")
        
        response = await self.client.embeddings.create(
            model=model,
            input=texts,
        )
        
        return [item.embedding for item in response.data]
    
    def list_free_models(self) -> list[str]:
        """List available free models (OpenRouter)."""
        if self.provider == "openrouter":
            return OPENROUTER_FREE_MODELS
        return []
    
    
        

        