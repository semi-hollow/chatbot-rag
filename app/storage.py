import json
from pathlib import Path
from uuid import uuid4

import chromadb

from app.config import CHROMA_DIR, CHUNKS_FILE, COLLECTION_NAME


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def append_chunks(chunks: list[dict]) -> None:
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_FILE.open("a", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def load_all_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        return []
    records: list[dict] = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def make_chunk_id(source_file: str, chunk_index: int) -> str:
    return f"{Path(source_file).stem}-{chunk_index}-{uuid4().hex[:8]}"
