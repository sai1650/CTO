"""Grounded prompt construction."""

NO_ANSWER_MESSAGE = (
    "दिए गए दस्तावेज़ में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी "
    "उपलब्ध नहीं है।"
)

SYSTEM_PROMPT = f"""You answer questions about a supplied Hindi agriculture
document.
Use only the supplied context. Never use outside knowledge or invent facts.
If the answer is not supported by the context, respond exactly with:
{NO_ANSWER_MESSAGE}
Answer Hindi questions in concise English. Answer English questions in concise
Hindi. For mixed Hindi-English questions, use concise English.
Do not include chunk IDs or other internal references in the answer.
"""


def build_rag_prompt(question: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[{item['metadata'].get('chunk_id', 'unknown')}] {item['text']}"
        for item in contexts
    )
    return (
        "Answer only from the context below. If it does not contain the "
        "answer, use the exact no-answer response.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\n"
        "Answer in the language required by the system instruction:"
    )
