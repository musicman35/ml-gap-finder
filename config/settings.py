"""Application settings with Pydantic."""

from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class LLMSettings(BaseSettings):
    """LLM provider configuration with switchable backends."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        extra="ignore",
    )

    # Provider selection
    provider: LLMProvider = Field(
        default=LLMProvider.ANTHROPIC,
        description="LLM provider: 'anthropic' or 'ollama'",
    )

    # Anthropic settings
    anthropic_api_key: str = Field(
        default="",
        validation_alias="ANTHROPIC_API_KEY",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model ID",
    )
    anthropic_max_tokens: int = Field(default=4096)

    # Ollama settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(
        default="llama3.1:70b",
        description="Ollama model name",
    )

    # Shared settings
    temperature: float = Field(default=0.7)
    timeout: int = Field(default=120, description="Request timeout in seconds")


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # LLM configuration (nested)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # Neo4j Graph Database
    neo4j_uri: str = Field(default="bolt://localhost:7687", validation_alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", validation_alias="NEO4J_USER")
    neo4j_password: str = Field(default="", validation_alias="NEO4J_PASSWORD")

    # PostgreSQL Database
    database_url: str = Field(
        default="postgresql://localhost:5432/mlgapfinder",
        validation_alias="DATABASE_URL",
    )

    # Qdrant Vector Store
    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")

    # Redis Cache
    redis_url: str = Field(default="redis://localhost:6379", validation_alias="REDIS_URL")

    # Semantic Scholar API (optional)
    semantic_scholar_api_key: str = Field(
        default="",
        validation_alias="SEMANTIC_SCHOLAR_API_KEY",
    )

    # Application settings
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Global settings instance
settings = get_settings()
