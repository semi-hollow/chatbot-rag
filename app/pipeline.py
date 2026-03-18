from pathlib import Path
from time import perf_counter

from app.chunking import chunk_text
from app.config import (
    BM25_CANDIDATES,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FINAL_TOP_K,
    MIN_CONFIDENCE_SCORE,
    MIN_CONTEXT_HITS,
    RERANK_TOP_N,
    VECTOR_CANDIDATES,
)
from app.document_loader import extract_title, iter_document_paths, load_text
from app.ollama_client import OllamaError, chat
from app.prompts import build_prompt
from app.reranker import rerank
from app.retrieval import BM25Retriever, VectorRetriever, rrf_fusion
from app.storage import append_chunks, get_collection, load_all_chunks, make_chunk_id
from app.tracing import Trace


class InsufficientEvidenceError(RuntimeError):
    pass


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

    from app.ollama_client import embed_texts

    embeddings = embed_texts([c["text"] for c in all_new_chunks])
    collection = get_collection()

    collection.add(
        ids=[c["chunk_id"] for c in all_new_chunks],
        documents=[c["text"] for c in all_new_chunks],
        metadatas=[
            {"source_file": c["source_file"], "chunk_index": c["chunk_index"], "title": c["title"]}
            for c in all_new_chunks
        ],
        embeddings=embeddings,
    )
    append_chunks(all_new_chunks)
    return {"files": len(documents), "chunks": len(all_new_chunks)}


def _build_context(final_hits: list[dict]) -> str:
    return "\n\n".join([f"[{i+1}] {hit['chunk']['text']}" for i, hit in enumerate(final_hits)])


def _evidence_gate(final_hits: list[dict]) -> None:
    if len(final_hits) < MIN_CONTEXT_HITS:
        raise InsufficientEvidenceError("Not enough retrieved evidence.")
    avg_score = sum(hit.get("rrf_score", 0.0) for hit in final_hits) / max(len(final_hits), 1)
    if avg_score < MIN_CONFIDENCE_SCORE:
        raise InsufficientEvidenceError("Retrieved evidence confidence is too low.")


def _run_retrieval(query: str, top_k: int, bm25_k: int, vector_k: int) -> dict:
    chunks = load_all_chunks()
    bm25_hits = BM25Retriever(chunks).search(query, top_k=bm25_k)
    vector_hits = VectorRetriever().search(query, top_k=vector_k)
    fused = rrf_fusion(bm25_hits, vector_hits, top_k=top_k * 2)
    return {"bm25_hits": bm25_hits, "vector_hits": vector_hits, "fused": fused}


def answer_question(
    question: str,
    language: str = "zh-CN",
    top_k: int = FINAL_TOP_K,
    enable_reranker: bool = False,
    rerank_top_n: int = RERANK_TOP_N,
    rerank_strategy: str = "hybrid",
    rewrite_question: str | None = None,
    trace_enabled: bool = False,
    debug: bool = False,
    bm25_k: int = BM25_CANDIDATES,
    vector_k: int = VECTOR_CANDIDATES,
) -> dict:
    trace = Trace(enabled=trace_enabled)
    working_question = rewrite_question or question
    trace.add("question", question)
    trace.add("retrieval_question", working_question)

    try:
        retrieval_started = perf_counter()
        retrieved = _run_retrieval(working_question, top_k=top_k, bm25_k=bm25_k, vector_k=vector_k)
        trace.mark(
            "retrieval",
            {
                "duration_ms": round((perf_counter() - retrieval_started) * 1000, 2),
                "bm25_top": [
                    {"chunk_id": h["chunk"]["chunk_id"], "score": h["score"], "rank": h["rank"]}
                    for h in retrieved["bm25_hits"][:top_k]
                ],
                "vector_top": [
                    {"chunk_id": h["chunk"]["chunk_id"], "score": h["score"], "rank": h["rank"]}
                    for h in retrieved["vector_hits"][:top_k]
                ],
            },
        )

        rerank_started = perf_counter()
        reranked = rerank(
            working_question,
            retrieved["fused"],
            enabled=enable_reranker,
            top_n=rerank_top_n,
            strategy=rerank_strategy,
        )
        trace.mark(
            "rerank",
            {
                "duration_ms": round((perf_counter() - rerank_started) * 1000, 2),
                "enabled": enable_reranker,
                "top": [
                    {
                        "chunk_id": h["chunk"]["chunk_id"],
                        "rrf_score": h.get("rrf_score", 0.0),
                        "rerank_score": h.get("rerank_score"),
                        "rank": h["rank"],
                    }
                    for h in reranked[:top_k]
                ],
            },
        )

        final_hits = reranked[:top_k]

        _evidence_gate(final_hits)
        context = _build_context(final_hits)
        prompt = build_prompt(language, question, context)
        generation_started = perf_counter()
        answer = chat(prompt)
        trace.mark("generation", {"duration_ms": round((perf_counter() - generation_started) * 1000, 2)})
        fallback_used = False
    except (InsufficientEvidenceError, OllamaError) as exc:
        retrieved = {"fused": []}
        final_hits = []
        fallback_used = True
        answer = (
            "证据不足，无法可靠回答当前问题。请补充上下文、扩大检索范围，或尝试改写问题。"
            if language == "zh-CN"
            else "Insufficient evidence to answer reliably. Please provide more context or rewrite the question."
        )
        context = ""
        trace.mark("fallback", {"reason": str(exc)})

    citations = [
        {
            "index": i + 1,
            "source_file": hit["chunk"]["source_file"],
            "chunk_index": hit["chunk"]["chunk_index"],
            "chunk_id": hit["chunk"]["chunk_id"],
            "rrf_score": hit.get("rrf_score", 0.0),
            "rerank_score": hit.get("rerank_score"),
        }
        for i, hit in enumerate(final_hits)
    ]

    trace.add("fallback_used", fallback_used)
    trace.add("answer", answer)
    trace.add("final_context", context)
    trace.save()

    result = {
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": [hit["chunk"] for hit in final_hits],
        "retrieval_question": working_question,
        "trace_id": trace.trace_id,
        "debug": trace.stages if debug else {},
    }

    if enable_reranker:
        result["rerank_before"] = [hit["chunk"]["chunk_id"] for hit in retrieved["fused"][:top_k]]
        result["rerank_after"] = [hit["chunk"]["chunk_id"] for hit in final_hits]
    return result
