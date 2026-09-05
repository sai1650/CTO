from src.embeddings import EmbeddingModel


class FakeModel:
    def encode(self, values, **kwargs):
        return [[float(len(value))] for value in values]


def test_embedding_adapter_supports_batch_and_unicode():
    adapter = EmbeddingModel("fake")
    adapter._model = FakeModel()
    assert adapter.encode(["भारत", "English"]) == [[4.0], [7.0]]
