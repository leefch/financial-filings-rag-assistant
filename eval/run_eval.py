"""
Run the RAGAS evaluation over the labeled test set and print a metrics table.

    python eval/run_eval.py

Metrics (the standard four):
  faithfulness       - is the answer supported by the retrieved context (no hallucination)
  answer_relevancy   - does the answer actually address the question
  context_precision  - were the retrieved chunks on-point
  context_recall     - did retrieval pull everything needed for the ground truth

RAGAS itself uses an LLM as judge, so this needs your LLM key set. The metric
API has moved around between RAGAS versions - this targets the modern
`evaluate(dataset, metrics=...)` interface; pin the version in requirements.txt
if an upgrade breaks it.

run() returns a plain dict of metric -> score so the pytest gate can assert on it.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from eval.testset import TESTSET
from rag.pipeline import answer

# the bar each metric must clear. deliberately not 0.9 - these are LLM-judged and
# a bit noisy; the point is to catch real regressions, not chase a perfect score.
THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.70,
    "context_precision": 0.70,
    "context_recall": 0.70,
}


def _collect():
    """Run every test question through the real pipeline and gather what RAGAS
    needs: the question, our answer, the contexts we used, and the truth."""
    questions, answers, contexts, truths = [], [], [], []
    for row in TESTSET:
        out = answer(row["question"])
        questions.append(row["question"])
        answers.append(out["answer"])
        contexts.append([c["text"] for c in out["contexts"]] or ["(none)"])
        truths.append(row["ground_truth"])
    return questions, answers, contexts, truths


def run() -> dict:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision, context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_anthropic import ChatAnthropic
    from langchain_huggingface import HuggingFaceEmbeddings
    from config import ANTHROPIC_MODEL, EMBED_MODEL

    # RAGAS's own judge + embeddings. Without these it silently defaults to OpenAI.
    judge = LangchainLLMWrapper(ChatAnthropic(model="claude-haiku-4-5", temperature=None))
    embedder = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL))

    q, a, c, t = _collect()
    ds = Dataset.from_dict({
        "question": q, "answer": a, "contexts": c, "ground_truth": t,
    })

    report = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,          # all four metrics now judge with Claude
        embeddings=embedder,  # answer_relevancy uses local MiniLM, no OpenAI key
    )
    # RAGAS result -> {metric: mean score}, works across versions
    df = report.to_pandas()
    metric_cols = [c for c in df.columns
               if c not in ("question", "answer", "contexts", "ground_truth",
                            "user_input", "response", "retrieved_contexts", "reference")]
    scores = {c: float(df[c].mean()) for c in metric_cols}
    return scores


def main():
    scores = run()
    print("\nRAGAS results")
    print("-" * 44)
    ok = True
    for metric, threshold in THRESHOLDS.items():
        val = scores.get(metric, float("nan"))
        mark = "PASS" if val >= threshold else "FAIL"
        if val < threshold:
            ok = False
        print(f"{metric:<20} {val:>6.3f}   (>= {threshold})  {mark}")
    print("-" * 44)
    print("OVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
