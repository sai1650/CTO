"""Provider-agnostic LLM interfaces and provider factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import re
import time
from typing import Any

from .errors import (
    LLMConfigurationError,
    LLMProviderUnavailableError,
    UnsupportedLLMProviderError,
)
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _safe_error_text(error: Exception) -> str:
    message = str(error)
    message = re.sub(
        r"(?i)(authorization\s*['\"]?\s*:\s*['\"]?\s*"
        r"bearer\s+)[^\s,;\"']+",
        r"\1[REDACTED]",
        message,
    )
    message = re.sub(
        r"(?i)(bearer\s+)[^\s,;\"']+",
        r"\1[REDACTED]",
        message,
    )
    return re.sub(
        r"(?i)((?:api[_ -]?key|token)\s*['\"]?\s*[:=]\s*"
        r"['\"]?\s*)[^\s,;\"']+",
        r"\1[REDACTED]",
        message,
    )


try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency guarded at runtime
    OpenAI = None


class LLMProvider(ABC):
    """Generic interface consumed by the RAG pipeline."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a grounded response for a prompt."""
        raise NotImplementedError


class OpenAICompatibleLLM(LLMProvider):
    """Chat provider for OpenAI and OpenAI-compatible APIs."""

    def __init__(
        self,
        model: str | None,
        api_key: str | None,
        base_url: str | None = None,
    ) -> None:
        if not model or not model.strip():
            raise LLMConfigurationError(
                "LLM model is missing. Add LLM_MODEL to your local .env file."
            )
        if not api_key or not api_key.strip():
            raise LLMConfigurationError(
                "LLM provider is not configured. Add the required API key "
                "to your local .env file and restart the application."
            )

        if OpenAI is None:
            logger.exception("OpenAI provider dependency is unavailable")
            raise LLMProviderUnavailableError(
                "The OpenAI provider is unavailable. Install the openai "
                "package and restart the application."
            )

        client_options: dict[str, Any] = {"api_key": api_key}
        client_options["timeout"] = 60.0
        if base_url and base_url.strip():
            client_options["base_url"] = base_url.strip()
        self.client = OpenAI(**client_options)
        self.model = model.strip()

    def generate(self, prompt: str) -> str:
        """Generate text through the configured OpenAI-compatible endpoint."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=messages,
            )
        except Exception as error:
            logger.warning(
                "LLM generation failed: provider=compatible error_type=%s "
                "error_message=%s",
                type(error).__name__,
                _safe_error_text(error),
            )
            raise LLMProviderUnavailableError(
                "The LLM provider could not generate an answer. "
                "Check the provider configuration and availability."
            ) from error
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as error:
            raise LLMProviderUnavailableError(
                "The LLM provider returned a malformed response."
            ) from error
        if not content or not content.strip():
            raise LLMProviderUnavailableError(
                "The LLM provider returned an empty response."
            )
        return content.strip()


class OpenRouterLLM(OpenAICompatibleLLM):
    """OpenRouter-specific LLM with ordered free-model fallback support."""

    def __init__(
        self,
        model: str | None,
        api_key: str | None,
        base_url: str | None = None,
        fallback_models: list[str] | None = None,
    ) -> None:
        super().__init__(model, api_key, base_url)
        self.fallback_models = self._normalize_fallback_models(
            fallback_models or []
        )

    @staticmethod
    def _normalize_fallback_models(fallback_models: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in fallback_models:
            value = (item or "").strip()
            if value and value not in normalized:
                normalized.append(value)
        return normalized

    def _ordered_models(self) -> list[str]:
        ordered: list[str] = []
        for candidate in [self.model] + self.fallback_models:
            if candidate and candidate not in ordered:
                ordered.append(candidate)
        return ordered

    def _extract_status_code(self, error: Exception) -> int | None:
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        response = getattr(error, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
            if isinstance(status, int):
                return status
        return None

    @staticmethod
    def _safe_error_message(error: Exception) -> str:
        return _safe_error_text(error)

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def _request(self, messages: list[dict[str, str]]) -> Any:
        model_name = self.model
        request_models = self._ordered_models()
        logger.info(
            "OpenRouter request",
            extra={
                "provider": "openrouter",
                "requested_model": model_name,
                "fallback_models": self.fallback_models,
                "ordered_models": request_models,
            },
        )
        return self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            extra_body={"models": request_models},
        )

    def generate(self, prompt: str) -> str:
        """Generate a grounded answer through the OpenRouter fallback list."""
        messages = self._build_messages(prompt)
        started_at = time.perf_counter()
        try:
            response = self._request(messages)
            elapsed = time.perf_counter() - started_at
            logger.info(
                "OpenRouter response",
                extra={
                    "provider": "openrouter",
                    "requested_model": self.model,
                    "fallback_models": self.fallback_models,
                    "ordered_models": self._ordered_models(),
                    "response_model": getattr(response, "model", self.model),
                    "elapsed_seconds": round(elapsed, 4),
                },
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise LLMProviderUnavailableError(
                    "The LLM provider returned an empty response."
                )
            return content.strip()
        except LLMProviderUnavailableError:
            raise
        except Exception as error:  # pragma: no cover
            status_code = self._extract_status_code(error)
            elapsed = time.perf_counter() - started_at
            safe_error_message = self._safe_error_message(error)
            logger.warning(
                "OpenRouter request failed: provider=%s requested_model=%s "
                "fallback_models=%s http_status=%s error_type=%s "
                "error_message=%s elapsed_seconds=%.4f",
                "openrouter",
                self.model,
                self.fallback_models,
                status_code,
                type(error).__name__,
                safe_error_message,
                elapsed,
                extra={
                    "provider": "openrouter",
                    "requested_model": self.model,
                    "fallback_models": self.fallback_models,
                    "ordered_models": self._ordered_models(),
                    "http_status": status_code,
                    "error_type": type(error).__name__,
                    "error_message": safe_error_message,
                    "elapsed_seconds": round(elapsed, 4),
                },
            )
            if status_code == 401:
                raise LLMProviderUnavailableError(
                    "OpenRouter API key is invalid or missing."
                ) from error
            if status_code == 403:
                raise LLMProviderUnavailableError(
                    "OpenRouter rejected the request. Check model access "
                    "and account permissions."
                ) from error
            if status_code == 404:
                raise LLMProviderUnavailableError(
                    "The configured OpenRouter model was not found."
                ) from error
            if status_code == 429:
                raise LLMProviderUnavailableError(
                    "OpenRouter is temporarily rate-limited. Please try "
                    "again shortly."
                ) from error
            timeout_text = str(error).lower()
            if "timed out" in timeout_text or "timeout" in timeout_text:
                raise LLMProviderUnavailableError(
                    "The LLM request timed out. Please try again."
                ) from error
            raise LLMProviderUnavailableError(
                "All configured free LLM providers are currently "
                "unavailable. Please try again shortly."
            ) from error


# Backward-compatible name for existing imports and tests.
OpenAILLM = OpenAICompatibleLLM


def _resolve_llm_credentials(settings):
    """Return the API key and base URL for the selected provider."""
    provider_name = (settings.llm_provider or "").strip().lower()

    if provider_name == "openrouter":
        api_key = settings.openrouter_api_key
        base_url = (
            settings.openrouter_base_url or "https://openrouter.ai/api/v1"
        )
        if not api_key or not api_key.strip():
            raise LLMConfigurationError(
                "OpenRouter API key is missing. Add OPENROUTER_API_KEY to "
                "your local .env file and restart the application."
            )
        return api_key, base_url

    if provider_name == "openai":
        api_key = settings.openai_api_key
        base_url = settings.openai_base_url
        if not api_key or not api_key.strip():
            raise LLMConfigurationError(
                "LLM provider is not configured. Add the required API key "
                "to your local .env file and restart the application."
            )
        return api_key, base_url

    raise UnsupportedLLMProviderError(
        f"Unsupported LLM provider: {provider_name or '(missing)'}. "
        "Set LLM_PROVIDER in your local .env file."
    )


def build_llm(settings) -> LLMProvider:
    """Build the configured provider without exposing credentials."""
    provider_name = (settings.llm_provider or "").strip().lower()
    providers = {"openai": OpenAILLM, "openrouter": OpenRouterLLM}
    provider_class = providers.get(provider_name)
    if provider_class is None:
        raise UnsupportedLLMProviderError(
            f"Unsupported LLM provider: {provider_name or '(missing)'}. "
            "Set LLM_PROVIDER in your local .env file."
        )

    api_key, base_url = _resolve_llm_credentials(settings)
    if provider_name == "openrouter":
        return provider_class(
            settings.llm_model,
            api_key,
            base_url,
            fallback_models=_parse_openrouter_fallbacks(
                settings.openrouter_fallback_models, settings.llm_model
            ),
        )
    return provider_class(settings.llm_model, api_key, base_url)


def _parse_openrouter_fallbacks(
    raw_value: str | None, primary_model: str | None
) -> list[str]:
    """Parse a comma-separated list of fallback free model IDs safely."""
    models: list[str] = []
    if not raw_value:
        return models
    for item in raw_value.split(","):
        model = item.strip()
        if not model:
            continue
        if primary_model and model == primary_model:
            continue
        if model not in models:
            models.append(model)
    return models
