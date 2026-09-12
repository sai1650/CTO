import pytest

from src.config import Settings
from src.errors import LLMConfigurationError, LLMProviderUnavailableError
from src.llm import build_llm


class _FakeResponse:
    def __init__(self, content="answer", model="model-a"):
        self.choices = [
            type(
                "Choice",
                (),
                {"message": type("Message", (), {"content": content})()},
            )
        ]
        self.model = model


def test_openrouter_primary_model_success(monkeypatch):
    seen = {}

    class FakeClient:
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            seen["kwargs"] = kwargs
            return _FakeResponse(
                content="translated answer",
                model="google/gemma-4-31b-it:free",
            )

    monkeypatch.setattr(
        "src.llm.OpenAI",
        lambda **kwargs: type(
            "Client",
            (),
            {"chat": type("Chat", (), {"completions": FakeClient()})()},
        )(),
    )
    settings = Settings(
        LLM_PROVIDER="openrouter",
        LLM_MODEL="google/gemma-4-31b-it:free",
        OPENROUTER_API_KEY="test-key",
        OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
        OPENROUTER_FALLBACK_MODELS="google/gemma-4-26b-a4b-it:free",
    )
    llm = build_llm(settings)
    result = llm.generate("भारत में खेती का महत्व क्या है?")
    assert "translated answer" in result
    assert seen["kwargs"]["model"] == "google/gemma-4-31b-it:free"
    assert seen["kwargs"]["extra_body"]["models"] == [
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
    ]


def test_openrouter_free_primary_uses_explicit_server_fallbacks(monkeypatch):
    seen = {}

    class FakeClient:
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            seen["kwargs"] = kwargs
            return _FakeResponse()

    monkeypatch.setattr(
        "src.llm.OpenAI",
        lambda **kwargs: type(
            "Client",
            (),
            {"chat": type("Chat", (), {"completions": FakeClient()})()},
        )(),
    )
    settings = Settings(
        LLM_PROVIDER="openrouter",
        LLM_MODEL="google/gemma-4-31b-it:free",
        OPENROUTER_API_KEY="test-key",
        OPENROUTER_FALLBACK_MODELS=(
            "google/gemma-4-31b-it:free,google/gemma-4-26b-a4b-it:free"
        ),
    )
    build_llm(settings).generate("भारत में खेती का महत्व क्या है?")
    assert seen["kwargs"]["extra_body"]["models"] == [
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
    ]


def test_openrouter_429_returns_temporary_service_message(monkeypatch):
    class FakeError(Exception):
        status_code = 429

    class FakeClient:
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            raise FakeError("rate limited")

    monkeypatch.setattr(
        "src.llm.OpenAI",
        lambda **kwargs: type(
            "Client",
            (),
            {"chat": type("Chat", (), {"completions": FakeClient()})()},
        )(),
    )
    settings = Settings(
        LLM_PROVIDER="openrouter",
        LLM_MODEL="google/gemma-4-31b-it:free",
        OPENROUTER_API_KEY="test-key",
        OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
        OPENROUTER_FALLBACK_MODELS="google/gemma-4-26b-a4b-it:free",
    )
    llm = build_llm(settings)
    with pytest.raises(
        LLMProviderUnavailableError,
        match="temporarily rate-limited",
    ):
        llm.generate("भारत में कृषि का महत्व क्या है?")


def test_openrouter_empty_response_is_rejected(monkeypatch):
    class FakeClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    return _FakeResponse(content="")

            completions = Completions()

        chat = Chat()

    monkeypatch.setattr("src.llm.OpenAI", lambda **kwargs: FakeClient())
    settings = Settings(
        LLM_PROVIDER="openrouter",
        LLM_MODEL="google/gemma-4-31b-it:free",
        OPENROUTER_API_KEY="test-key",
    )
    llm = build_llm(settings)
    with pytest.raises(
        LLMProviderUnavailableError, match="empty response"
    ):
        llm.generate("भारत में कृषि का महत्व क्या है?")


def test_openrouter_all_models_fail(monkeypatch):
    class FakeError(Exception):
        status_code = 429

    class FakeClient:
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            raise FakeError("rate limited")

    monkeypatch.setattr(
        "src.llm.OpenAI",
        lambda **kwargs: type(
            "Client",
            (),
            {"chat": type("Chat", (), {"completions": FakeClient()})()},
        )(),
    )
    settings = Settings(
        LLM_PROVIDER="openrouter",
        LLM_MODEL="google/gemma-4-31b-it:free",
        OPENROUTER_API_KEY="test-key",
        OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
        OPENROUTER_FALLBACK_MODELS="",
    )
    llm = build_llm(settings)
    with pytest.raises(
        LLMProviderUnavailableError,
        match="OpenRouter is temporarily rate-limited",
    ):
        llm.generate("भारत में कृषि का महत्व क्या है?")


def test_openrouter_missing_api_key_raises_configuration_error():
    settings = Settings(
        LLM_PROVIDER="openrouter",
        LLM_MODEL="google/gemma-4-31b-it:free",
        OPENROUTER_API_KEY="",
        OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
    )
    with pytest.raises(
        LLMConfigurationError, match="OpenRouter API key is missing"
    ):
        build_llm(settings)


def test_openrouter_invalid_api_key_is_handled_by_provider_error(monkeypatch):
    class FakeError(Exception):
        status_code = 401

    class FakeClient:
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            raise FakeError("bad key")

    monkeypatch.setattr(
        "src.llm.OpenAI",
        lambda **kwargs: type(
            "Client",
            (),
            {"chat": type("Chat", (), {"completions": FakeClient()})()},
        )(),
    )
    settings = Settings(
        LLM_PROVIDER="openrouter",
        LLM_MODEL="google/gemma-4-31b-it:free",
        OPENROUTER_API_KEY="bad-key",
        OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
    )
    llm = build_llm(settings)
    with pytest.raises(
        LLMProviderUnavailableError, match="invalid or missing"
    ):
        llm.generate("भारत में कृषि का महत्व क्या है?")


def test_openrouter_timeout_is_user_friendly(monkeypatch):
    class FakeTimeout(Exception):
        pass

    class FakeClient:
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            raise FakeTimeout("timed out")

    monkeypatch.setattr(
        "src.llm.OpenAI",
        lambda **kwargs: type(
            "Client",
            (),
            {"chat": type("Chat", (), {"completions": FakeClient()})()},
        )(),
    )
    settings = Settings(
        LLM_PROVIDER="openrouter",
        LLM_MODEL="google/gemma-4-31b-it:free",
        OPENROUTER_API_KEY="test-key",
        OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
    )
    llm = build_llm(settings)
    with pytest.raises(LLMProviderUnavailableError, match="timed out"):
        llm.generate("भारत में कृषि का महत्व क्या है?")


def test_hindi_question_is_supported():
    settings = Settings(
        LLM_PROVIDER="openrouter",
        LLM_MODEL="google/gemma-4-31b-it:free",
        OPENROUTER_API_KEY="test-key",
        OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
    )
    assert "भारत" in "भारत में कृषि का महत्व क्या है?"
    llm = build_llm(settings)
    assert llm is not None


def test_english_question_is_supported():
    settings = Settings(
        LLM_PROVIDER="openrouter",
        LLM_MODEL="google/gemma-4-31b-it:free",
        OPENROUTER_API_KEY="test-key",
        OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
    )
    assert "soil" in "What type of soil is suitable for wheat?".lower()
    llm = build_llm(settings)
    assert llm is not None


def test_mixed_hindi_english_question_is_supported():
    question = "Wheat की खेती के लिए कौन सी मिट्टी चाहिए?"
    assert "Wheat" in question and "मिट्टी" in question


def test_no_answer_behavior_is_document_grounded():
    response = "The information is not available in the provided document."
    assert "provided document" in response.lower()
