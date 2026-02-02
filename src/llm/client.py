"""LLM client implementations for Anthropic and Ollama."""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from anthropic import AsyncAnthropic
from ollama import AsyncClient as OllamaAsyncClient

from config.settings import settings, LLMProvider


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion for the given prompt.

        Args:
            prompt: User prompt.
            system: Optional system prompt.

        Yields:
            Generated text chunks.
        """
        pass


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client implementation."""

    def __init__(self):
        """Initialize Anthropic client."""
        self.client = AsyncAnthropic(api_key=settings.llm.anthropic_api_key)
        self.model = settings.llm.anthropic_model
        self.max_tokens = settings.llm.anthropic_max_tokens

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a completion using Claude.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text.
        """
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else settings.llm.temperature,
            system=system or "You are a helpful research assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    async def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion using Claude.

        Args:
            prompt: User prompt.
            system: Optional system prompt.

        Yields:
            Generated text chunks.
        """
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or "You are a helpful research assistant.",
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text


class OllamaLLMClient(BaseLLMClient):
    """Ollama local LLM client implementation."""

    def __init__(self):
        """Initialize Ollama client."""
        self.client = OllamaAsyncClient(host=settings.llm.ollama_base_url)
        self.model = settings.llm.ollama_model

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a completion using Ollama.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature if temperature is not None else settings.llm.temperature,
                "num_predict": max_tokens or 4096,
            },
        )
        return response["message"]["content"]

    async def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion using Ollama.

        Args:
            prompt: User prompt.
            system: Optional system prompt.

        Yields:
            Generated text chunks.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        stream = await self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            if chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]


def get_llm_client() -> BaseLLMClient:
    """Factory function to get the configured LLM client.

    Returns:
        LLM client based on configuration.

    Raises:
        ValueError: If unknown provider is configured.
    """
    if settings.llm.provider == LLMProvider.ANTHROPIC:
        return AnthropicClient()
    elif settings.llm.provider == LLMProvider.OLLAMA:
        return OllamaLLMClient()
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm.provider}")


async def get_llm() -> BaseLLMClient:
    """FastAPI dependency for LLM client.

    Returns:
        LLM client instance.
    """
    return get_llm_client()
