from pathlib import Path

from app.chunking import chunk_text
from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.document_loader import extract_title, iter_document_paths, load_text
from app.ollama_client import chat, embed_texts
from app.prompts import build_prompt
from app.reranker import rerank
from app.retrieval import BM25Retriever, VectorRetriever, rrf_fusion
from app.storage import append_chunks, get_collection, load_all_chunks, make_chunk_id


def ingest_documents(path: str) -> dict:
    input_path = Path(path)
    documents = list(iter_document_paths(input_path))
    if not documents:
        return {"files": 0, "chunks": 0}

    all_new_chunks: list[dict] = []

    for doc_path in documents:
        raw_text = load_text(doc_path)
        title = extract_title(doc_path, raw_text)
        chunks = chunk_text(raw_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        for idx, text in enumerate(chunks):
            all_new_chunks.append(
                {
                    "chunk_id": make_chunk_id(str(doc_path), idx),
                    "source_file": str(doc_path),
                    "chunk_index": idx,
                    "text": text,
                    "title": title,
                }
            )

    if not all_new_chunks:
        return {"files": len(documents), "chunks": 0}

    embeddings = embed_texts([c["text"] for c in all_new_chunks])
    collection = get_collection()

    collection.add(
        ids=[c["chunk_id"] for c in all_new_chunks],
        documents=[c["text"] for c in all_new_chunks],
        metadatas=[
            {
                "source_file": c["source_file"],
                "chunk_index": c["chunk_index"],
                "title": c["title"],
            }
            for c in all_new_chunks
        ],
        embeddings=embeddings,
    )
    append_chunks(all_new_chunks)
    return {"files": len(documents), "chunks": len(all_new_chunks)}


def answer_question(question: str, language: str = "zh-CN", top_k: int = 5, enable_reranker: bool = False) -> dict:
    chunks = load_all_chunks()
    bm25_hits = BM25Retriever(chunks).search(question, top_k=top_k * 2)
    vector_hits = VectorRetriever().search(question, top_k=top_k * 2)
    fused = rrf_fusion(bm25_hits, vector_hits, top_k=top_k * 2)
    reranked = rerank(question, fused, enabled=enable_reranker)
    final_hits = reranked[:top_k]

    context = "\n\n".join(
        f"[{i+1}] {hit['chunk']['text']}"
        for i, hit in enumerate(final_hits)
    )
    prompt = build_prompt(language, question, context)
    answer = chat(prompt)

    citations = [
        {
            "index": i + 1,
            "source_file": hit["chunk"]["source_file"],
            "chunk_index": hit["chunk"]["chunk_index"],
            "chunk_id": hit["chunk"]["chunk_id"],
        }
        for i, hit in enumerate(final_hits)
    ]

    return {
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": [hit["chunk"] for hit in final_hits],
    }
