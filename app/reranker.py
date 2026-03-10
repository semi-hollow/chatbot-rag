"""Optional reranker module.

This is intentionally a placeholder for phase-1. You can plug in a cross-encoder reranker
later without changing the rest of the static workflow RAG pipeline.
"""


def rerank(query: str, fused_hits: list[dict], enabled: bool = False) -> list[dict]:
    if not enabled:
        return fused_hits
    # Placeholder behavior: return as-is.
    return fused_hits
