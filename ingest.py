"""Build the local chunk artifact and ChromaDB collection."""

from pathlib import Path
import logging

from src.config import get_settings
from src.embeddings import EmbeddingModel
from src.indexing import ensure_index, load_chunks
from src.text_processor import save_chunks
from src.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    root = Path(__file__).resolve().parent
    source_path = root / "data" / "raw" / "hindi_document.txt"
    chunks_path = root / "data" / "processed" / "chunks.json"
    chunks = load_chunks(settings, source_path)
    logger.info("Chunks created: %d", len(chunks))
    logger.info("Chunk IDs: %s", [chunk.chunk_id for chunk in chunks])
    save_chunks(chunks, chunks_path)
    embeddings = EmbeddingModel(settings.embedding_model)
    store = ChromaVectorStore(
        settings.chroma_persist_directory,
        settings.chroma_collection_name,
        embeddings,
    )
    logger.info("Embedding model: %s", embeddings.model_name)
    logger.info("ChromaDB path: %s", store.persist_directory)
    logger.info("Chroma collection: %s", settings.chroma_collection_name)
    document_count = ensure_index(settings, store, source_path)
    logger.info("Chroma document count: %d", document_count)
    print(
        f"Ingestion completed: {sum(len(chunk.text) for chunk in chunks)} "
        f"document characters, "
        f"{len(chunks)} chunks, {document_count} Chroma documents."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
