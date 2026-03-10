from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"

COLLECTION_NAME = "rag_chunks"
DEFAULT_CHAT_MODEL = "qwen3:8b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
