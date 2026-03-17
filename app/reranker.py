"""True second-stage reranker for hybrid retrieval candidates."""

import json
import re

from app.config import DEFAULT_RERANK_MODEL
from app.ollama_client import OllamaError, chat


def _keyword_score(query: str, text: str) -> float:
    q_terms = {t for t in query.lower().split() if t}
    if not q_terms:
        return 0.0
    tokens = text.lower().split()
    overlap = sum(1 for t in tokens if t in q_terms)
    return overlap / max(len(tokens), 1)


def _llm_score(query: str, text: str) -> float:
    prompt = (
        "Score document relevance for answering the query. Return JSON only: {\"score\": 0-100}.\n"
        f"Query: {query}\n"
        f"Document: {text[:1200]}"
    )
    raw = chat(prompt, model=DEFAULT_RERANK_MODEL)
    try:
        payload = json.loads(raw)
        return float(payload.get("score", 0)) / 100.0
    except json.JSONDecodeError:
        match = re.search(r"(\d{1,3})", raw)
        if not match:
            return 0.0
        return min(float(match.group(1)), 100.0) / 100.0


def rerank(
    query: str,
    fused_hits: list[dict],
    enabled: bool = False,
    top_n: int = 8,
    strategy: str = "hybrid",
) -> list[dict]:
    if not enabled:
        return fused_hits

    candidates = fused_hits[:top_n]
    rescored: list[dict] = []

    for hit in candidates:
        text = hit["chunk"]["text"]
        lexical = _keyword_score(query, text)
        llm = 0.0
        if strategy in {"llm", "hybrid"}:
            try:
                llm = _llm_score(query, text)
            except OllamaError:
                llm = 0.0

        if strategy == "llm":
            score = llm
        elif strategy == "lexical":
            score = lexical
        else:
            score = 0.75 * llm + 0.25 * lexical

        updated = dict(hit)
        updated["rerank_score"] = score
        rescored.append(updated)

    remaining = fused_hits[top_n:]
    rescored.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
    merged = rescored + remaining

    for rank, hit in enumerate(merged, start=1):
        hit["rank"] = rank
    return merged
