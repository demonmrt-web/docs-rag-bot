"""Хранилище векторов. Два варианта с одним интерфейсом:
  LocalStore  — файлы на диске, numpy. Не требует сервера: удобно для разработки и тестов.
  QdrantStore — сервер Qdrant по REST (в docker-compose).
Выбор: переменная VECTOR_STORE=local|qdrant."""
import json
import uuid
from pathlib import Path

import httpx
import numpy as np

from . import config


class LocalStore:
    def __init__(self, path: Path):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_f, self.vec_f = self.dir / "meta.json", self.dir / "vectors.npy"
        self.meta: list[dict] = json.loads(self.meta_f.read_text("utf-8")) if self.meta_f.exists() else []
        self.vecs = np.load(self.vec_f) if self.vec_f.exists() else np.zeros((0, 0), dtype=np.float32)

    def _save(self):
        self.meta_f.write_text(json.dumps(self.meta, ensure_ascii=False), "utf-8")
        np.save(self.vec_f, self.vecs)

    @staticmethod
    def _norm(v):
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.where(n == 0, 1, n)

    def delete_source(self, source: str) -> int:
        keep = [i for i, m in enumerate(self.meta) if m["source"] != source]
        removed = len(self.meta) - len(keep)
        if removed:
            self.meta = [self.meta[i] for i in keep]
            self.vecs = self.vecs[keep] if keep else np.zeros((0, 0), dtype=np.float32)
            self._save()
        return removed

    def add(self, metas: list[dict], vectors: list[list[float]]):
        if not metas:
            return
        v = self._norm(vectors)
        self.vecs = v if self.vecs.size == 0 else np.vstack([self.vecs, v])
        self.meta += metas
        self._save()

    def search(self, vector, k: int) -> list[dict]:
        if not self.meta:
            return []
        scores = self.vecs @ self._norm(vector)
        idx = np.argsort(-scores)[:k]
        return [{**self.meta[i], "score": float(scores[i])} for i in idx]

    def sources(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.meta:
            out[m["source"]] = out.get(m["source"], 0) + 1
        return out


class QdrantStore:
    def __init__(self, url: str, collection: str):
        self.url, self.col = url.rstrip("/"), collection
        self.http = httpx.Client(timeout=60)
        self._ready = False

    def _req(self, method, path, **kw):
        r = self.http.request(method, f"{self.url}{path}", **kw)
        r.raise_for_status()
        return r.json()

    def _ensure(self, dim: int):
        if self._ready:
            return
        if self.http.get(f"{self.url}/collections/{self.col}").status_code == 404:
            self._req("PUT", f"/collections/{self.col}", json={"vectors": {"size": dim, "distance": "Cosine"}})
        self._ready = True

    def delete_source(self, source: str) -> int:
        before = self.sources().get(source, 0)
        if before:
            self._req("POST", f"/collections/{self.col}/points/delete?wait=true",
                      json={"filter": {"must": [{"key": "source", "match": {"value": source}}]}})
        return before

    def add(self, metas: list[dict], vectors: list[list[float]]):
        if not metas:
            return
        self._ensure(len(vectors[0]))
        points = [{"id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{m['source']}#{m['chunk']}")),
                   "vector": v, "payload": m} for m, v in zip(metas, vectors)]
        self._req("PUT", f"/collections/{self.col}/points?wait=true", json={"points": points})

    def search(self, vector, k: int) -> list[dict]:
        if self.http.get(f"{self.url}/collections/{self.col}").status_code == 404:
            return []
        res = self._req("POST", f"/collections/{self.col}/points/search",
                        json={"vector": list(vector), "limit": k, "with_payload": True})["result"]
        return [{**p["payload"], "score": p["score"]} for p in res]

    def sources(self) -> dict[str, int]:
        if self.http.get(f"{self.url}/collections/{self.col}").status_code == 404:
            return {}
        out: dict[str, int] = {}
        offset = None
        while True:
            body = {"limit": 256, "with_payload": ["source"], "with_vector": False}
            if offset is not None:
                body["offset"] = offset
            res = self._req("POST", f"/collections/{self.col}/points/scroll", json=body)["result"]
            for p in res["points"]:
                s = p["payload"]["source"]
                out[s] = out.get(s, 0) + 1
            offset = res.get("next_page_offset")
            if offset is None:
                return out


def make_store():
    if config.VECTOR_STORE == "qdrant":
        return QdrantStore(config.QDRANT_URL, config.QDRANT_COLLECTION)
    return LocalStore(config.DATA_DIR / "index")
