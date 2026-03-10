from typing import Any

import requests

from app.config import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL, OLLAMA_BASE_URL


def embed_texts(texts: list[str], model: str = DEFAULT_EMBED_MODEL) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for text in texts:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        embeddings.append(payload["embedding"])
    return embeddings


def chat(prompt: str, model: str = DEFAULT_CHAT_MODEL) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    return payload["message"]["content"]
