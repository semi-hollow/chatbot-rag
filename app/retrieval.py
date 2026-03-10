from collections import defaultdict

from rank_bm25 import BM25Okapi

from app.ollama_client import embed_texts
from app.storage import get_collection


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.corpus_tokens = [c["text"].lower().split() for c in chunks]
        self.model = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        if not self.model:
            return []
        scores = self.model.get_scores(query.lower().split())
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for rank, (idx, score) in enumerate(scored, start=1):
            chunk = self.chunks[idx]
            results.append({"chunk": chunk, "score": float(score), "rank": rank, "source": "bm25"})
        return results


class VectorRetriever:
    def search(self, query: str, top_k: int = 8) -> list[dict]:
        collection = get_collection()
        query_embedding = embed_texts([query])[0]
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for rank, (chunk_id, doc, meta, distance) in enumerate(zip(ids, docs, metas, distances), start=1):
            hits.append(
                {
                    "chunk": {
                        "chunk_id": chunk_id,
                        "text": doc,
                        "source_file": meta.get("source_file", "unknown"),
                        "chunk_index": meta.get("chunk_index", -1),
                        "title": meta.get("title", ""),
                    },
                    "score": float(distance),
                    "rank": rank,
                    "source": "vector",
                }
            )
        return hits


def rrf_fusion(bm25_hits: list[dict], vector_hits: list[dict], k: int = 60, top_k: int = 8) -> list[dict]:
    fused_scores = defaultdict(float)
    chunk_map = {}

    for hit in bm25_hits:
        cid = hit["chunk"]["chunk_id"]
        chunk_map[cid] = hit["chunk"]
        fused_scores[cid] += 1.0 / (k + hit["rank"])

    for hit in vector_hits:
        cid = hit["chunk"]["chunk_id"]
        chunk_map[cid] = hit["chunk"]
        fused_scores[cid] += 1.0 / (k + hit["rank"])

    merged = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"chunk": chunk_map[cid], "rrf_score": score, "rank": rank}
        for rank, (cid, score) in enumerate(merged, start=1)
    ]
