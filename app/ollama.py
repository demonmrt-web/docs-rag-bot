"""Клиент Ollama: векторы (embed) и ответ модели (chat)."""
import httpx

from . import config


class OllamaError(RuntimeError):
    pass


def embed(texts: list[str]) -> list[list[float]]:
    """Векторы для списка текстов. Пачкой по 16, чтобы не упереться в память."""
    out: list[list[float]] = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i + 16]
        try:
            r = httpx.post(f"{config.OLLAMA_URL}/api/embed",
                           json={"model": config.EMBED_MODEL, "input": batch}, timeout=300)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaError(f"Ollama (embed) недоступна или нет модели {config.EMBED_MODEL}: {e}") from e
        out += r.json()["embeddings"]
    return out


def chat(system: str, user: str) -> str:
    try:
        r = httpx.post(f"{config.OLLAMA_URL}/api/chat", timeout=600, json={
            "model": config.LLM_MODEL, "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 4096},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        })
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise OllamaError(f"Ollama (chat) недоступна или нет модели {config.LLM_MODEL}: {e}") from e
    return r.json()["message"]["content"].strip()
