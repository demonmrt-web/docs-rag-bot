"""Ядро RAG: загрузка документов в индекс и ответ на вопрос по найденным кускам."""
from pathlib import Path

from . import config, ollama
from .chunking import chunk_text
from .loaders import SUPPORTED, load_document

NO_ANSWER = "В загруженных документах ответа на этот вопрос нет."

SYSTEM = (
    "Ты помощник по документации. Отвечай ТОЛЬКО по приведённым фрагментам. "
    "Если во фрагментах нет ответа, ответь ровно: «" + NO_ANSWER + "». Ничего не придумывай. "
    "После утверждений ставь номер источника в квадратных скобках, например [1]. "
    "Отвечай на языке вопроса, коротко и по делу."
)


class Rag:
    def __init__(self, store, embed=ollama.embed, chat=ollama.chat):
        self.store, self._embed, self._chat = store, embed, chat

    def ingest_text(self, source: str, text: str) -> int:
        """Заменяет документ в индексе: старые куски этого источника удаляются, чтобы не было дублей."""
        chunks = chunk_text(text, config.CHUNK_CHARS, config.CHUNK_OVERLAP)
        self.store.delete_source(source)
        if not chunks:
            return 0
        vectors = self._embed(chunks)
        self.store.add([{"source": source, "chunk": i, "text": c} for i, c in enumerate(chunks)], vectors)
        return len(chunks)

    def ingest_file(self, path: str | Path, source: str | None = None) -> int:
        p = Path(path)
        return self.ingest_text(source or p.name, load_document(p))

    def ingest_folder(self, folder: str | Path) -> dict[str, int]:
        res = {}
        for p in sorted(Path(folder).rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED:
                res[p.name] = self.ingest_file(p)
        return res

    def ask(self, question: str, top_k: int | None = None) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Пустой вопрос")
        hits = self.store.search(self._embed([question])[0], top_k or config.TOP_K)
        hits = [h for h in hits if h["score"] >= config.MIN_SCORE]
        if not hits:   # нерелевантно: модель не зовём совсем, ответ не выдумывается
            return {"answer": NO_ANSWER, "sources": [], "used_llm": False}
        context = "\n\n".join(f"[{i}] (файл: {h['source']})\n{h['text']}" for i, h in enumerate(hits, 1))
        answer = self._chat(SYSTEM, f"Фрагменты:\n{context}\n\nВопрос: {question}")
        return {"answer": answer, "used_llm": True,
                "sources": [{"n": i, "source": h["source"], "chunk": h["chunk"],
                             "score": round(h["score"], 3), "text": h["text"]} for i, h in enumerate(hits, 1)]}
