"""
Chunk the filings and build the Chroma index.

    python ingest/build_index.py

Reads every .txt under data/sample_filings, splits into overlapping chunks, and
persists embeddings to ./chroma_db. Re-run it whenever the source docs change.
The index is gitignored - it's derived data, rebuild it don't commit it.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DOCS_DIR, INDEX_DIR, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def load_docs():
    """Read the .txt filings into (text, source) pairs. Kept dead simple; if you
    wire in fetch_filings.py you'd point this at the downloaded set instead."""
    docs = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        docs.append((path.read_text(encoding="utf-8"), path.name))
    if not docs:
        raise SystemExit(f"no .txt files in {DOCS_DIR}")
    return docs


def main():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # split on natural boundaries first so chunks don't cut mid-sentence
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    texts, metadatas = [], []
    for content, source in load_docs():
        for chunk in splitter.split_text(content):
            texts.append(chunk)
            metadatas.append({"source": source})

    print(f"embedding {len(texts)} chunks with {EMBED_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # from_texts persists automatically when a directory is given
    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=str(INDEX_DIR),
    )
    print(f"index written to {INDEX_DIR}")


if __name__ == "__main__":
    main()
