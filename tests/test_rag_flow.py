"""Проверка всей цепочки без Ollama: векторы и «модель» подменены простыми функциями."""
import re

import pytest

from app import config
from app.rag import NO_ANSWER, Rag
from app.store import LocalStore

VOCAB = ["субтитры", "whisper", "бот", "биржа", "расширение", "браузер", "ключ", "режим"]


def fake_embed(texts):
    """Вектор = сколько раз встречается каждое слово словаря. Достаточно, чтобы проверить поиск."""
    return [[float(len(re.findall(w, t.lower()))) for w in VOCAB] + [0.01] for t in texts]


def fake_chat(system, user):
    return "ОТВЕТ-ЗАГЛУШКА по: " + user.split("Вопрос:")[-1].strip()


@pytest.fixture
def rag(tmp_path):
    return Rag(LocalStore(tmp_path / "idx"), embed=fake_embed, chat=fake_chat)


def test_ingest_and_ask_returns_relevant_source(rag):
    rag.ingest_text("subs.md", "Приложение делает субтитры через whisper из видео.")
    rag.ingest_text("bot.md", "Торговый бот для биржи работает в режиме симуляции.")
    res = rag.ask("Как делаются субтитры?")
    assert res["used_llm"] is True
    assert res["sources"][0]["source"] == "subs.md"
    assert "ОТВЕТ-ЗАГЛУШКА" in res["answer"]


def test_irrelevant_question_skips_llm(rag):
    rag.ingest_text("subs.md", "Приложение делает субтитры через whisper из видео.")
    res = rag.ask("Какая погода на Марсе?")
    assert res == {"answer": NO_ANSWER, "sources": [], "used_llm": False}


def test_empty_index_gives_no_answer(rag):
    assert rag.ask("субтитры")["used_llm"] is False


def test_reingest_replaces_old_chunks(rag):
    rag.ingest_text("a.md", "Старый текст про субтитры.")
    rag.ingest_text("a.md", "Новый текст про биржа и бот.")
    assert rag.store.sources() == {"a.md": 1}
    assert "биржа" in rag.store.search(fake_embed(["биржа"])[0], 1)[0]["text"].lower()


def test_delete_source(rag):
    rag.ingest_text("a.md", "текст про бот")
    assert rag.store.delete_source("a.md") == 1
    assert rag.store.sources() == {}


def test_index_survives_restart(tmp_path):
    r1 = Rag(LocalStore(tmp_path / "idx"), embed=fake_embed, chat=fake_chat)
    r1.ingest_text("a.md", "расширение для браузера")
    r2 = Rag(LocalStore(tmp_path / "idx"), embed=fake_embed, chat=fake_chat)
    assert r2.store.sources() == {"a.md": 1}
    assert r2.ask("расширение браузера")["sources"][0]["source"] == "a.md"


def test_empty_question_rejected(rag):
    with pytest.raises(ValueError):
        rag.ask("   ")
