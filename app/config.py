"""Настройки сервиса. Всё переопределяется переменными окружения (см. docker-compose.yml)."""
import os
from pathlib import Path

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")          # генерация ответа
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")           # векторы для поиска (мультиязычная, хорошо знает русский)

VECTOR_STORE = os.getenv("VECTOR_STORE", "local")          # local — файл на диске, qdrant — сервер Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DOCS_DIR = Path(os.getenv("DOCS_DIR", "./docs"))

CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "800"))          # размер куска текста
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))      # перекрытие соседних кусков
TOP_K = int(os.getenv("TOP_K", "4"))                        # сколько кусков отдавать модели
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.35"))           # ниже этого сходства кусок считается нерелевантным

API_KEY = os.getenv("API_KEY", "")                          # если задан — требуется заголовок X-API-Key
