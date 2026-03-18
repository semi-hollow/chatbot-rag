from typing import Any

import requests

from app.config import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL, OLLAMA_BASE_URL


class OllamaError(RuntimeError):
    pass


def embed_texts(texts: list[str], model: str = DEFAULT_EMBED_MODEL) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for text in texts:
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=60,
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            embeddings.append(payload["embedding"])
        except requests.RequestException as exc:
            raise OllamaError(
                "Failed to call Ollama embeddings. Ensure `ollama serve` is running and model is pulled."
            ) from exc
    return embeddings


def chat(prompt: str, model: str = DEFAULT_CHAT_MODEL) -> str:
    return chat_messages([{"role": "user", "content": prompt}], model=model)


def chat_messages(messages: list[dict[str, str]], model: str = DEFAULT_CHAT_MODEL) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload["message"]["content"]
    except requests.RequestException as exc:
        raise OllamaError(
            "Failed to call Ollama. Ensure `ollama serve` is running and model is pulled."
        ) from exc
