"""
The end-to-end answer path, with the governance wrapped around the model call:

    input guard -> retrieve + rerank -> grounding check -> cited answer -> redact

Every answer either cites the sources it used ([S1], [S2], ...) or explicitly
refuses. There's no third option where it makes something up - that's the whole
point of a governed RAG.
"""
from config import MIN_RELEVANCE
from rag import guards
from rag.retriever import retrieve
from rag.llm import ask

SYSTEM = (
    "You answer questions about corporate financial filings using ONLY the "
    "numbered context passages provided. Rules:\n"
    "- Cite the passages you use inline as [S1], [S2], etc.\n"
    "- If the passages don't contain the answer, say you can't find it in the "
    "provided filings. Do not use outside knowledge.\n"
    "- Be concise and precise with figures; quote numbers exactly as written."
)


def _format_context(chunks):
    # number the passages so the model has something concrete to cite
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[S{i}] (source: {c['source']})\n{c['text']}")
    return "\n\n".join(lines)


def answer(query: str) -> dict:
    result = {
        "query": query,
        "answer": "",
        "citations": [],
        "contexts": [],
        "refused": False,
        "reason": "",
    }

    # 1. input guard - stop injection / off-topic before we spend a retrieval
    ok, reason = guards.input_guard(query)
    if not ok:
        result["refused"] = True
        result["reason"] = reason
        result["answer"] = reason
        return result

    # 2. retrieve + rerank
    chunks = retrieve(query)
    result["contexts"] = chunks

    # 3. grounding check - if nothing cleared the bar, refuse instead of guessing
    top = chunks[0]["score"] if chunks else 0.0
    if not guards.is_grounded(top, MIN_RELEVANCE):
        result["refused"] = True
        result["reason"] = "No sufficiently relevant passage found."
        result["answer"] = ("I can't find that in the provided filings. "
                            "Try rephrasing, or it may not be covered.")
        return result

    # 4. grounded generation with inline citations
    context = _format_context(chunks)
    raw = ask(SYSTEM, f"Context:\n{context}\n\nQuestion: {query}")

    # 5. scrub any PII the model might have surfaced before returning
    result["answer"] = guards.redact(raw)
    result["citations"] = [c["source"] for c in chunks]
    return result
