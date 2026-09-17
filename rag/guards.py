"""
Guardrails. These are deliberately plain, dependency-free heuristics so they're
fast, testable, and easy to explain in a review. In production you'd likely back
the input check with something like Llama Guard or NeMo Guardrails, but the
control points stay the same: guard the input, ground the output, redact PII.

Three jobs:
  1. input_guard   - reject prompt-injection / obviously out-of-scope questions
  2. redact        - scrub anything that looks like PII out of the final answer
  3. is_grounded   - decide, from the retrieval score, whether we're allowed to
                     answer at all (refuse rather than hallucinate)
"""
import re

# phrases that show up in prompt-injection attempts. not exhaustive - a real
# deployment would layer a model-based detector on top - but catches the obvious.
INJECTION_PATTERNS = [
    r"ignore (all|any|previous|the above)",
    r"disregard (all|any|previous|the above)",
    r"forget (your|all|the) (instructions|rules)",
    r"you are now",
    r"system prompt",
    r"reveal your (prompt|instructions|system)",
    r"act as (if|though) you",
]

# very rough finance-domain check. if a question has none of these signals it's
# probably off-topic for a filings assistant. kept loose to avoid false refusals.
SCOPE_HINTS = [
    "revenue", "income", "margin", "ebitda", "debt", "dividend", "segment",
    "risk", "cash", "guidance", "earnings", "expense", "growth", "sales",
    "company", "fiscal", "quarter", "report", "filing", "10-k", "10k",
    "share", "profit", "loss", "assets", "liabilit", "r&d", "capital",
]

# PII patterns to scrub from any generated answer before it's shown
PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN REDACTED]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL REDACTED]"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE REDACTED]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CARD REDACTED]"),
]


def input_guard(query: str):
    """Return (ok, reason). ok=False means don't even retrieve."""
    q = query.lower().strip()

    if len(q) < 3:
        return False, "Question is too short to answer."

    for pat in INJECTION_PATTERNS:
        if re.search(pat, q):
            # note WHAT tripped it, for the audit log - but we don't echo the
            # user's payload back
            return False, "Request looks like a prompt-injection attempt and was blocked."

    if not any(h in q for h in SCOPE_HINTS):
        return False, ("Question doesn't appear to be about the filings this "
                       "assistant covers. Try asking about revenue, margins, "
                       "risk factors, and similar.")

    return True, ""


def redact(text: str) -> str:
    for pat, repl in PII_PATTERNS:
        text = pat.sub(repl, text)
    return text


def is_grounded(top_score: float, threshold: float) -> bool:
    """We only answer if the best retrieved chunk clears the bar. Below it, the
    honest move is to say we can't find it."""
    return top_score >= threshold
