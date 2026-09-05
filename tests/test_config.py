from src.config import Settings


def test_retrieval_and_chunk_defaults():
    settings = Settings()
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.llm_provider == "openrouter"
    assert settings.llm_model == "openrouter/free"
    assert settings.top_k == 4
    assert settings.relevance_threshold == 0.25
    assert settings.chunk_size == 400
    assert settings.chunk_overlap == 60


def test_retrieval_and_chunk_environment_values(monkeypatch):
    monkeypatch.setenv("TOP_K", "8")
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.3")
    monkeypatch.setenv("CHUNK_SIZE", "420")
    monkeypatch.setenv("CHUNK_OVERLAP", "55")
    settings = Settings()
    assert settings.top_k == 8
    assert settings.relevance_threshold == 0.3
    assert settings.chunk_size == 420
    assert settings.chunk_overlap == 55
