from src.rag_pipeline import RAGPipeline


class FakeRetriever:
    def __init__(self, contexts):
        self.contexts = contexts

    def retrieve(self, question):
        return self.contexts


class FakeLLM:
    def generate(self, prompt):
        assert "Question" in prompt
        assert "Answer in English" in prompt
        return "Sustainable agriculture balances production and conservation."


def test_pipeline_is_grounded_through_llm():
    contexts = [
        {
            "text": "संतुलन",
            "metadata": {"chunk_id": "chunk-0000"},
            "similarity": 0.9,
        }
    ]
    result = RAGPipeline(FakeRetriever(contexts), FakeLLM()).ask(
        "लक्ष्य क्या है?"
    )
    assert result.has_context is True
    assert result.sources == contexts
    assert result.answer.startswith("Sustainable agriculture")


def test_pipeline_missing_context():
    result = RAGPipeline(FakeRetriever([]), FakeLLM()).ask("अज्ञात")
    assert result.has_context is False
    assert result.sources == []
    assert result.answer == (
        "The provided document does not contain sufficient information "
        "to answer this question."
    )
