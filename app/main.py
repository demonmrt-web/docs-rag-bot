"""HTTP-API: /health, /ingest, /reindex, /ask, /sources. Запуск: uvicorn app.main:app"""
import shutil
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

from . import config
from .loaders import SUPPORTED
from .ollama import OllamaError
from .rag import Rag
from .store import make_store

app = FastAPI(title="docs-rag-bot", version="0.1.0")
rag = Rag(make_store())


def check_key(x_api_key: str = Header(default="")):
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(401, "Неверный X-API-Key")


class Question(BaseModel):
    question: str
    top_k: int | None = None


@app.get("/health")
def health():
    return {"ok": True, "llm": config.LLM_MODEL, "embed": config.EMBED_MODEL, "store": config.VECTOR_STORE}


@app.post("/ask", dependencies=[Depends(check_key)])
def ask(q: Question):
    try:
        return rag.ask(q.question, q.top_k)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except OllamaError as e:
        raise HTTPException(503, str(e))


@app.post("/ingest", dependencies=[Depends(check_key)])
def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(400, f"Формат {suffix or '?'} не поддерживается: {', '.join(sorted(SUPPORTED))}")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / Path(file.filename).name
        with open(p, "wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            n = rag.ingest_file(p, source=Path(file.filename).name)
        except OllamaError as e:
            raise HTTPException(503, str(e))
    return {"source": file.filename, "chunks": n}


@app.post("/reindex", dependencies=[Depends(check_key)])
def reindex():
    try:
        return rag.ingest_folder(config.DOCS_DIR)
    except OllamaError as e:
        raise HTTPException(503, str(e))


@app.get("/sources", dependencies=[Depends(check_key)])
def sources():
    return rag.store.sources()


@app.delete("/sources/{name}", dependencies=[Depends(check_key)])
def delete_source(name: str):
    return {"deleted_chunks": rag.store.delete_source(name)}
