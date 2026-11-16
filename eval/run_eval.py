"""Оценка качества на наборе вопросов (eval/questions.json). Нужны запущенная Ollama и проиндексированные docs/.
  python eval/run_eval.py
Для вопросов с ответом проверяем две вещи: нужный файл есть в источниках и в ответе встречается ключевое слово.
Для вопросов вне документов ожидаем отказ («ответа нет») без обращения к модели."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.rag import NO_ANSWER, Rag  # noqa: E402
from app.store import make_store  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rag = Rag(make_store())
    items = json.loads((Path(__file__).parent / "questions.json").read_text("utf-8"))
    rows, ok_total = [], 0
    for it in items:
        t0 = time.time()
        res = rag.ask(it["q"])
        dt = time.time() - t0
        if it["source"] is None:                      # вопрос вне документов: допустим отказ без модели или отказ самой модели
            ok = (not res["used_llm"]) or NO_ANSWER.rstrip(".") in res["answer"]
            detail = ("отказ без модели" if not res["used_llm"] else "модель сама отказала") if ok \
                else "ОШИБКА: ответил, хотя в документах этого нет"
        else:
            src_ok = any(s["source"] == it["source"] for s in res["sources"])
            kw_ok = it["keyword"].lower() in res["answer"].lower()
            ok = src_ok and kw_ok
            detail = f"источник {'да' if src_ok else 'НЕТ'}, ключевое слово {'да' if kw_ok else 'НЕТ'}"
        ok_total += ok
        rows.append({"q": it["q"], "ok": ok, "detail": detail, "seconds": round(dt, 1), "answer": res["answer"]})
        print(f"{'OK ' if ok else 'ПЛОХО'} {dt:5.1f} с  {it['q']}\n      {detail}\n      -> {res['answer'][:160]!r}")
    print(f"\nИтого: {ok_total} из {len(items)}")
    (Path(__file__).parent / "last_run.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), "utf-8")


if __name__ == "__main__":
    main()
