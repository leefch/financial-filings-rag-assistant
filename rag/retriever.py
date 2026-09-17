"""
Retrieval in two stages:
  1. vector search over Chroma pulls a wide-ish net of candidate chunks (TOP_K)
  2. a cross-encoder reranks those candidates and we keep the best few (KEEP_K)

The rerank matters: bi-encoder embeddings are cheap but blunt, so they over-
retrieve. The cross-encoder actually reads (query, chunk) together and scores
relevance properly. It's slower, which is exactly why we only run it on the
dozen candidates instead of the whole corpus.
"""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import INDEX_DIR, EMBED_MODEL, RERANK_MODEL, TOP_K, KEEP_K


@lru_cache(maxsize=1)
def _vectorstore():
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            "no index found - run `python ingest/build_index.py` first"
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=str(INDEX_DIR), embedding_function=embeddings)


@lru_cache(maxsize=1)
def _reranker():
    # imported lazily so the eval/tests that don't rerank don't pay for it
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


def retrieve(query: str):
    """Return a list of {text, source, score} sorted best-first, already cut to
    KEEP_K. score is the cross-encoder relevance, roughly 0..1 after sigmoid."""
    vs = _vectorstore()
    candidates = vs.similarity_search(query, k=TOP_K)
    if not candidates:
        return []

    pairs = [(query, d.page_content) for d in candidates]
    scores = _reranker().predict(pairs, apply_softmax=False)

    # cross-encoder logits -> 0..1 so the threshold in config is interpretable
    import numpy as np
    probs = 1.0 / (1.0 + np.exp(-np.asarray(scores)))

    ranked = sorted(zip(candidates, probs), key=lambda t: t[1], reverse=True)
    out = []
    for doc, score in ranked[:KEEP_K]:
        out.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": float(score),
        })
    return out
