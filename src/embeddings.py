"""Multilingual embedding model adapter."""

from typing import Sequence


class EmbeddingModel:
    """Lazy Sentence Transformers adapter for BAAI/bge-m3 or another model."""

    def __init__(
        self, model_name: str = "BAAI/bge-m3"
    ) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import logging

            logging.getLogger(__name__).info(
                "Embedding model loaded: %s", self.model_name
            )
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: str | Sequence[str]) -> list[list[float]]:
        """Encode Hindi or English text into normalized vectors."""
        values = [texts] if isinstance(texts, str) else list(texts)
        if not values:
            return []
        vectors = self.model.encode(
            values,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist() if hasattr(vectors, "tolist") else vectors
