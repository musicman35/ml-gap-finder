"""Tests for LLM client implementations."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from config.settings import LLMProvider
from src.llm.client import (
    BaseLLMClient,
    AnthropicClient,
    OllamaLLMClient,
    get_llm_client,
)


class TestBaseLLMClient:
    """Tests for base LLM client interface."""

    def test_base_client_is_abstract(self):
        """Test that BaseLLMClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMClient()


class TestAnthropicClient:
    """Tests for Anthropic client."""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic SDK."""
        with patch("src.llm.client.AsyncAnthropic") as mock:
            instance = MagicMock()
            mock.return_value = instance
            yield instance

    @pytest.mark.asyncio
    async def test_generate_returns_text(self, mock_anthropic):
        """Test that generate returns the message text."""
        # Mock the response structure
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Generated response")]
        mock_anthropic.messages.create = AsyncMock(return_value=mock_message)

        with patch("src.llm.client.settings") as mock_settings:
            mock_settings.llm.anthropic_api_key = "test-key"
            mock_settings.llm.anthropic_model = "test-model"
            mock_settings.llm.anthropic_max_tokens = 4096
            mock_settings.llm.temperature = 0.7

            client = AnthropicClient()
            client.client = mock_anthropic

            result = await client.generate("Test prompt")

            assert result == "Generated response"

    @pytest.mark.asyncio
    async def test_generate_uses_custom_parameters(self, mock_anthropic):
        """Test that custom parameters are passed correctly."""
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Response")]
        mock_anthropic.messages.create = AsyncMock(return_value=mock_message)

        with patch("src.llm.client.settings") as mock_settings:
            mock_settings.llm.anthropic_api_key = "test-key"
            mock_settings.llm.anthropic_model = "test-model"
            mock_settings.llm.anthropic_max_tokens = 4096
            mock_settings.llm.temperature = 0.7

            client = AnthropicClient()
            client.client = mock_anthropic

            await client.generate(
                prompt="Test",
                system="Custom system",
                max_tokens=100,
                temperature=0.5,
            )

            mock_anthropic.messages.create.assert_called_once()
            call_kwargs = mock_anthropic.messages.create.call_args.kwargs
            assert call_kwargs["system"] == "Custom system"
            assert call_kwargs["max_tokens"] == 100
            assert call_kwargs["temperature"] == 0.5


class TestOllamaClient:
    """Tests for Ollama client."""

    @pytest.fixture
    def mock_ollama(self):
        """Mock Ollama SDK."""
        with patch("src.llm.client.OllamaAsyncClient") as mock:
            instance = AsyncMock()
            mock.return_value = instance
            yield instance

    @pytest.mark.asyncio
    async def test_generate_returns_text(self, mock_ollama):
        """Test that generate returns the message content."""
        mock_ollama.chat = AsyncMock(
            return_value={"message": {"content": "Ollama response"}}
        )

        with patch("src.llm.client.settings") as mock_settings:
            mock_settings.llm.ollama_base_url = "http://localhost:11434"
            mock_settings.llm.ollama_model = "llama3"
            mock_settings.llm.temperature = 0.7

            client = OllamaLLMClient()
            client.client = mock_ollama

            result = await client.generate("Test prompt")

            assert result == "Ollama response"

    @pytest.mark.asyncio
    async def test_generate_includes_system_message(self, mock_ollama):
        """Test that system message is included when provided."""
        mock_ollama.chat = AsyncMock(
            return_value={"message": {"content": "Response"}}
        )

        with patch("src.llm.client.settings") as mock_settings:
            mock_settings.llm.ollama_base_url = "http://localhost:11434"
            mock_settings.llm.ollama_model = "llama3"
            mock_settings.llm.temperature = 0.7

            client = OllamaLLMClient()
            client.client = mock_ollama

            await client.generate(
                prompt="Test",
                system="Be helpful",
            )

            call_kwargs = mock_ollama.chat.call_args.kwargs
            messages = call_kwargs["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "Be helpful"


class TestGetLLMClient:
    """Tests for LLM client factory."""

    def test_get_anthropic_client(self):
        """Test that Anthropic client is returned for anthropic provider."""
        with patch("src.llm.client.settings") as mock_settings:
            mock_settings.llm.provider = LLMProvider.ANTHROPIC
            mock_settings.llm.anthropic_api_key = "test-key"
            mock_settings.llm.anthropic_model = "test-model"
            mock_settings.llm.anthropic_max_tokens = 4096

            client = get_llm_client()
            assert isinstance(client, AnthropicClient)

    def test_get_ollama_client(self):
        """Test that Ollama client is returned for ollama provider."""
        with patch("src.llm.client.settings") as mock_settings:
            mock_settings.llm.provider = LLMProvider.OLLAMA
            mock_settings.llm.ollama_base_url = "http://localhost:11434"
            mock_settings.llm.ollama_model = "llama3"

            client = get_llm_client()
            assert isinstance(client, OllamaLLMClient)

    def test_get_unknown_provider_raises(self):
        """Test that unknown provider raises ValueError."""
        with patch("src.llm.client.settings") as mock_settings:
            mock_settings.llm.provider = "unknown"

            with pytest.raises(ValueError, match="Unknown LLM provider"):
                get_llm_client()
