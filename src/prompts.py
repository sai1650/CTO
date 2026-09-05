"""Grounded prompt construction."""

SYSTEM_PROMPT = """You answer questions about a supplied Hindi agriculture
document.
Use only the supplied context. Never use outside knowledge or invent facts.
If the answer is not supported by the context, say:
The provided document does not contain sufficient information to answer this
question.
Always answer in concise English, even when the question or context is Hindi.
Do not include chunk IDs or other internal references in the answer.
"""


def build_rag_prompt(question: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[{item['metadata'].get('chunk_id', 'unknown')}] {item['text']}"
        for item in contexts
    )
    return (
        "Answer only from the context below. If it does not contain the "
        "answer, use the required no-answer response.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\n"
        "Answer in English:"
    )
