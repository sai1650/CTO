import pytest

from src.config import Settings
from src.errors import (
    LLMConfigurationError,
    LLMProviderUnavailableError,
    UnsupportedLLMProviderError,
)
from src.llm import LLMProvider, OpenAICompatibleLLM, build_llm


def test_missing_api_key_is_a_configuration_error():
    with pytest.raises(LLMConfigurationError, match="required API key"):
        OpenAICompatibleLLM("test-model", None)


def test_missing_model_is_a_configuration_error():
    with pytest.raises(LLMConfigurationError, match="LLM model is missing"):
        OpenAICompatibleLLM(None, "test-key")


def test_valid_configuration_is_forwarded_without_exposing_key(monkeypatch):
    received = {}

    class FakeProvider(LLMProvider):
        def __init__(self, model, api_key, base_url=None):
            received.update(model=model, api_key=api_key, base_url=base_url)

        def generate(self, prompt):
            return "answer"

    monkeypatch.setattr("src.llm.OpenAILLM", FakeProvider)
    settings = Settings(
        LLM_PROVIDER="openai",
        LLM_MODEL="test-model",
        OPENAI_API_KEY="test-key",
        OPENAI_BASE_URL="https://example.test/v1",
    )

    provider = build_llm(settings)

    assert isinstance(provider, FakeProvider)
    assert received == {
        "model": "test-model",
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
    }


def test_invalid_provider_is_rejected():
    settings = Settings(
        LLM_PROVIDER="unknown",
        LLM_MODEL="test-model",
        OPENAI_API_KEY="test-key",
    )

    with pytest.raises(UnsupportedLLMProviderError, match="Unsupported"):
        build_llm(settings)


def test_openrouter_provider_uses_openrouter_settings(monkeypatch):
    class FakeProvider(LLMProvider):
        def __init__(
            self, model, api_key, base_url=None, fallback_models=None
        ):
            self.model = model
            self.api_key = api_key
            self.base_url = base_url
            self.fallback_models = fallback_models or []

        def generate(self, prompt):
            return "answer"

    monkeypatch.setattr("src.llm.OpenRouterLLM", FakeProvider)
    settings = Settings(
        LLM_PROVIDER="openrouter",
        LLM_MODEL="google/gemma-4-31b-it:free",
        OPENROUTER_API_KEY="test-openrouter-key",
        OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
        OPENROUTER_FALLBACK_MODELS=(
            "openai/gpt-4o-mini:free,"
            "meta-llama/llama-3.1-8b-instruct:free"
        ),
    )

    provider = build_llm(settings)

    assert isinstance(provider, FakeProvider)
    assert provider.model == "google/gemma-4-31b-it:free"
    assert provider.api_key == "test-openrouter-key"
    assert provider.base_url == "https://openrouter.ai/api/v1"
    assert provider.fallback_models == [
        "openai/gpt-4o-mini:free",
        "meta-llama/llama-3.1-8b-instruct:free",
    ]


def test_provider_selection_uses_registered_provider(monkeypatch):
    class FakeProvider(LLMProvider):
        def __init__(self, model, api_key, base_url=None):
            self.model = model

        def generate(self, prompt):
            return "answer"

    monkeypatch.setattr("src.llm.OpenAILLM", FakeProvider)
    settings = Settings(
        LLM_PROVIDER="OPENAI",
        LLM_MODEL="test-model",
        OPENAI_API_KEY="test-key",
    )

    provider = build_llm(settings)

    assert isinstance(provider, FakeProvider)


def test_empty_provider_response_is_user_safe(monkeypatch):
    class FakeClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    return type(
                        "Response",
                        (),
                        {
                            "choices": [
                                type(
                                    "Choice",
                                    (),
                                    {
                                        "message": type(
                                            "Message", (), {"content": ""}
                                        )()
                                    },
                                )()
                            ]
                        },
                    )()

            completions = Completions()

        chat = Chat()

    monkeypatch.setattr("src.llm.OpenAI", lambda **kwargs: FakeClient())
    provider = OpenAICompatibleLLM("test-model", "test-key")
    with pytest.raises(LLMProviderUnavailableError, match="empty response"):
        provider.generate("question")
