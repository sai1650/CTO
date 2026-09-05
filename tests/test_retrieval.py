from src.retriever import Retriever


class FakeStore:
    def search(self, query, top_k, relevance_threshold):
        if query == "missing":
            return []
        return [{"text": "संदर्भ", "metadata": {}, "similarity": 0.8}]


def test_retriever_returns_context_and_handles_missing():
    retriever = Retriever(FakeStore(), top_k=2, relevance_threshold=0.3)
    assert retriever.retrieve("जल")
    assert retriever.retrieve("missing") == []
