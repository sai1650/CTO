"""Simple retrieval evaluation harness."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.config import get_settings
    from src.embeddings import EmbeddingModel
    from src.retriever import Retriever
    from src.vector_store import ChromaVectorStore

    root = Path(__file__).resolve().parents[1]
    questions_path = root / "evaluation" / "questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    settings = get_settings()
    store = ChromaVectorStore(
        settings.chroma_persist_directory,
        settings.chroma_collection_name,
        EmbeddingModel(settings.embedding_model),
    )
    retriever = Retriever(store, settings.top_k, settings.relevance_threshold)
    answerable_hits = 0
    unsupported_with_no_evidence = 0
    for item in questions:
        results = retriever.retrieve(item["question"])
        context = " ".join(result["text"] for result in results)
        terms = item.get("evidence_terms", [])
        evidence_found = any(term in context for term in terms)
        if item["answer_present"]:
            answerable_hits += int(evidence_found)
        else:
            unsupported_with_no_evidence += int(not evidence_found)
        print(
            f"{item['question']} -> results={len(results)} "
            f"evidence={evidence_found}"
        )
    answerable = sum(item["answer_present"] for item in questions)
    unanswerable = len(questions) - answerable
    print(f"Answerable retrieval hit rate: {answerable_hits / answerable:.2%}")
    print(
        "Unsupported questions without labeled evidence: "
        f"{unsupported_with_no_evidence}/{unanswerable}"
    )


if __name__ == "__main__":
    main()
