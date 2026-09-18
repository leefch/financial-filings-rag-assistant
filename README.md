Governed Financial-Filings RAG Assistant

A retrieval-augmented Q&A system over financial filings (SEC 10-Ks) that answers with citations, refuses when it can't ground an answer, and ships with an automated evaluation gate — a RAG built the way an enterprise would need it: reliable, explainable, and governed.

Built with Python, LangChain, Claude, RAG, ChromaDB, and RAGAS.

What it does
Ask a question about a set of filings and the system retrieves the relevant passages, reranks them, and generates an answer grounded only in the retrieved text, with inline citations. If nothing sufficiently relevant is found, it says so instead of guessing. Prompt-injection and out-of-scope inputs are blocked, and PII is redacted from output.

Answer quality isn't left to chance: a RAGAS evaluation suite scores the pipeline on faithfulness, answer relevancy, and context precision/recall, and runs as a pass/fail gate in CI.
Ingest (offline): filings → chunk → embed → Chroma vector index.

Query pipeline: input guard → retrieve → rerank → grounding check → generate (cited) → PII redact.

Evaluation: labeled Q&A → RAGAS (4 metrics) → pytest quality gate.

Key features
Citation-grounded answers — every answer traces to source passages
Two-stage retrieval — embedding search followed by a cross-encoder reranking model for precision
Refusal over hallucination — answers only when retrieval clears a relevance threshold
Guardrails — prompt-injection blocking, out-of-scope refusal, and PII redaction
Automated evaluation — RAGAS metrics enforced as thresholds in a pytest CI gate
Local, enterprise-friendly models — embeddings and reranking run locally (no data leaves the box for retrieval)
Streamlit UI — ask questions, see the cited answer and the retrieved passages with scores
Tech stack

Python · LangChain · Claude (Anthropic) · ChromaDB · Hugging Face (embeddings + cross-encoder) · RAGAS · pytest · Streamlit

Setup
Requires Python 3.12 and an Anthropic API key. The first index build downloads two small local models (embedding + reranker).runbook agentic-aml-investigator-RUN.txt
