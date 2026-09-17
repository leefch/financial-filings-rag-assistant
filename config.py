import os
# Windows native-crash workaround: torch and onnxruntime each ship an OpenMP
# runtime; without this the process hard-aborts when models load under Streamlit.
# Must run before torch is imported, so it lives at the very top of config.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")   # load .env no matter where you launch from

DOCS_DIR = ROOT / "data" / "sample_filings"
INDEX_DIR = ROOT / "chroma_db"

# retrieval knobs
TOP_K = 12          # how many chunks the vector store returns before rerank
KEEP_K = 4          # how many survive rerank and go into the prompt
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# if the best reranked chunk scores below this, we refuse rather than guess.
# tune against your own eval set - too high and you refuse good questions.
MIN_RELEVANCE = 0.15

# local, enterprise-friendly models by default (no data leaves the box for
# embedding/reranking). swap for hosted ones if you'd rather.
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# generation LLM - provider is swappable, default claude
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
