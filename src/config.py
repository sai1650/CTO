"""Centralized application configuration."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENROUTER_FALLBACK_MODELS = (
    "google/gemma-4-31b-it:free,"
    "google/gemma-4-26b-a4b-it:free,"
    "liquid/lfm-2.5-2.6b:free"
)
load_dotenv(PROJECT_ROOT / ".env")


def resolve_project_path(value: str | Path) -> Path:
    """Resolve configured paths relative to this project's root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", extra="ignore"
    )

    embedding_model: str = Field(
        default="BAAI/bge-m3", alias="EMBEDDING_MODEL"
    )
    chroma_persist_directory: Path = Field(
        default=PROJECT_ROOT / "chroma_db", alias="CHROMA_PERSIST_DIRECTORY"
    )
    chroma_collection_name: str = Field(
        default="hindi_document", alias="CHROMA_COLLECTION_NAME"
    )
    top_k: int = Field(default=4, alias="TOP_K", ge=1, le=20)
    relevance_threshold: float = Field(
        default=0.25, alias="RELEVANCE_THRESHOLD", ge=0.0, le=1.0
    )
    chunk_size: int = Field(default=400, alias="CHUNK_SIZE", ge=100, le=2000)
    chunk_overlap: int = Field(
        default=60, alias="CHUNK_OVERLAP", ge=0, le=1000
    )
    llm_provider: str = Field(default="openrouter", alias="LLM_PROVIDER")
    llm_model: str | None = Field(
        default="openrouter/free", alias="LLM_MODEL"
    )
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openrouter_api_key: str | None = Field(
        default=None, alias="OPENROUTER_API_KEY"
    )
    openrouter_base_url: str | None = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    openrouter_fallback_models: str = Field(
        default=DEFAULT_OPENROUTER_FALLBACK_MODELS,
        alias="OPENROUTER_FALLBACK_MODELS",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one cached settings object for the process."""
    settings = Settings()
    settings.chroma_persist_directory = resolve_project_path(
        settings.chroma_persist_directory
    )
    return settings
