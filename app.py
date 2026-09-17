"""
Streamlit front end for the governed RAG. Ask a question about the filings and
you get either a cited answer or an explicit refusal - plus the passages it used
and their relevance scores, so you can see exactly what grounded the answer.

    streamlit run app.py
"""
import streamlit as st

from rag.pipeline import answer

st.set_page_config(page_title="Filings Q&A", layout="wide")

st.title("Governed Financial RAG")
st.caption("Citation-grounded Q&A over 10-K filings. Refuses when it can't ground an answer.")

# a few starter questions so the demo isn't a blank box
examples = [
    "What was ACME's total revenue in fiscal 2024?",
    "What is a key risk factor for Globex's margins?",
    "What was Globex's net-debt-to-EBITDA ratio at year end?",
]
with st.expander("Example questions"):
    for e in examples:
        st.write("- " + e)

query = st.text_input("Ask about the filings", "")
go = st.button("Ask", type="primary")

if go and query.strip():
    with st.spinner("Retrieving and answering..."):
        result = answer(query)

    if result["refused"]:
        # make refusals visually distinct - this is a feature, not an error
        st.warning(f"Refused: {result['reason']}")
        st.write(result["answer"])
    else:
        st.subheader("Answer")
        st.write(result["answer"])

        srcs = sorted(set(result["citations"]))
        if srcs:
            st.caption("Sources: " + ", ".join(srcs))

    # always show what was retrieved - transparency is the point of the exercise
    if result["contexts"]:
        st.subheader("Retrieved passages")
        for i, c in enumerate(result["contexts"], start=1):
            with st.expander(f"[S{i}] {c['source']}  -  relevance {c['score']:.2f}"):
                st.write(c["text"])
