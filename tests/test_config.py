from src.config import PROJECT_ROOT, Settings, resolve_project_path


def test_chroma_path_is_resolved_from_project_root():
    expected_path = (PROJECT_ROOT / "chroma_db").resolve()
    assert resolve_project_path("chroma_db") == expected_path


def test_retrieval_and_chunk_defaults(monkeypatch):
    monkeypatch.delenv("TOP_K", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("RELEVANCE_THRESHOLD", raising=False)
    settings = Settings(_env_file=None)
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.llm_provider == "openrouter"
    assert settings.llm_model == "google/gemma-4-31b-it:free"
    assert settings.openrouter_fallback_models == (
        "google/gemma-4-26b-a4b-it:free"
    )
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
