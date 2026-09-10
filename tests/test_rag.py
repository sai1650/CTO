from src.rag_pipeline import RAGPipeline


class FakeRetriever:
    def __init__(self, contexts):
        self.contexts = contexts

    def retrieve(self, question):
        return self.contexts


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        assert "Question" in prompt
        assert "Answer in Hindi" in prompt
        return "सतत कृषि उत्पादन और संरक्षण के बीच संतुलन बनाती है।"


def test_pipeline_is_grounded_through_llm():
    contexts = [
        {
            "text": "संतुलन",
            "metadata": {"chunk_id": "chunk-0000"},
            "similarity": 0.9,
        }
    ]
    llm = FakeLLM()
    result = RAGPipeline(FakeRetriever(contexts), llm).ask(
        "गेहूं की खेती के लिए कौन सी मिट्टी उपयुक्त है?"
    )
    assert result.has_context is True
    assert result.sources == contexts
    assert result.answer.startswith("सतत कृषि")
    assert llm.calls == 1


def test_pipeline_missing_context():
    llm = FakeLLM()
    result = RAGPipeline(FakeRetriever([]), llm).ask("How are you?")
    assert result.has_context is False
    assert result.sources == []
    assert result.answer == (
        "दिए गए दस्तावेज़ में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी "
        "उपलब्ध नहीं है।"
    )
    assert llm.calls == 0


def test_pipeline_does_not_answer_unsupported_question():
    llm = FakeLLM()
    result = RAGPipeline(FakeRetriever([]), llm).ask("Tell me a joke")

    assert result.answer == (
        "दिए गए दस्तावेज़ में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी "
        "उपलब्ध नहीं है।"
    )
    assert result.has_context is False
    assert llm.calls == 0
