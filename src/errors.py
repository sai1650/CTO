"""Application exceptions that can be presented safely to users."""


class LLMError(Exception):
    """Base exception for LLM configuration and provider failures."""


class LLMConfigurationError(LLMError):
    """Raised when the selected LLM is not configured correctly."""


class UnsupportedLLMProviderError(LLMConfigurationError):
    """Raised when no provider implementation is registered."""


class LLMProviderUnavailableError(LLMError):
    """Raised when a configured provider cannot be initialized or used."""