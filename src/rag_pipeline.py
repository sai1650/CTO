"""End-to-end retrieval augmented generation pipeline."""

from dataclasses import dataclass

from .prompts import build_rag_prompt


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    has_context: bool


class RAGPipeline:
    def __init__(self, retriever, llm) -> None:
        self.retriever = retriever
        self.llm = llm

    def ask(self, question: str) -> RAGResponse:
        contexts = self.retriever.retrieve(question)
        if not contexts:
            return RAGResponse(
                "The provided document does not contain sufficient "
                "information to answer this question.",
                [],
                False,
            )
        answer = self.llm.generate(build_rag_prompt(question, contexts))
        return RAGResponse(answer=answer, sources=contexts, has_context=True)
