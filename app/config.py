from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
TRACE_FILE = DATA_DIR / "traces.jsonl"
EVAL_FILE = DATA_DIR / "eval_set.jsonl"

COLLECTION_NAME = "rag_chunks"
DEFAULT_CHAT_MODEL = "qwen3:8b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_RERANK_MODEL = "qwen3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

BM25_CANDIDATES = 10
VECTOR_CANDIDATES = 10
RRF_K = 60
FINAL_TOP_K = 5
RERANK_TOP_N = 8

MIN_CONTEXT_HITS = 2
MIN_CONFIDENCE_SCORE = 0.025

SESSION_MAX_TURNS = 6
SESSION_RECENT_TURNS = 3

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
