"""Нарезка текста на куски для поиска: по абзацам, с перекрытием, без разрыва слов."""
import re


def _split_long(par: str, size: int) -> list[str]:
    """Абзац длиннее size режем по предложениям, а если и они длинные — по словам."""
    sentences = re.split(r"(?<=[.!?…])\s+", par)
    out, cur = [], ""
    for s in sentences:
        while len(s) > size:                       # предложение длиннее куска
            cut = s.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            if cur:
                out.append(cur); cur = ""
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if cur and len(cur) + 1 + len(s) > size:
            out.append(cur); cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Возвращает куски длиной примерно до size символов. Соседние куски перекрываются на overlap символов,
    чтобы мысль на границе не потерялась."""
    if size <= 0:
        raise ValueError("size должен быть > 0")
    overlap = max(0, min(overlap, size // 2))
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    pieces: list[str] = []
    for par in re.split(r"\n\s*\n", text):
        par = " ".join(par.split())
        if par:
            pieces += _split_long(par, size) if len(par) > size else [par]
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 2 + len(p) > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if tail and " " in tail else tail   # начинаем с целого слова
            cur = f"{tail} {p}".strip() if tail else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks
