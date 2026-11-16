"""Консольный запуск без сервера:
  python -m app.cli ingest docs/
  python -m app.cli ask "Как запустить расширение?"
"""
import json
import sys

from . import config
from .rag import Rag
from .store import make_store


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 3 or sys.argv[1] not in ("ingest", "ask"):
        print(__doc__)
        return
    rag = Rag(make_store())
    if sys.argv[1] == "ingest":
        res = rag.ingest_folder(sys.argv[2])
        for name, n in res.items():
            print(f"{name}: {n} кусков")
        print(f"Готово. Документов: {len(res)}, всего кусков: {sum(res.values())}")
    else:
        res = rag.ask(" ".join(sys.argv[2:]))
        print(res["answer"])
        for s in res["sources"]:
            print(f"  [{s['n']}] {s['source']} (сходство {s['score']})")


if __name__ == "__main__":
    main()
