"""Чтение документов: .txt, .md, .pdf, .docx -> простой текст."""
import html
import re
import zipfile
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


def _read_text_file(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_docx(p: Path) -> str:
    xml = zipfile.ZipFile(p).read("word/document.xml").decode("utf-8")
    paras = []
    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        paras.append(html.unescape("".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", para))))
    return "\n\n".join(paras)


def _read_pdf(p: Path) -> str:
    from pypdf import PdfReader
    return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(p)).pages)


def load_document(path: str | Path) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Формат {ext} не поддерживается (нужны: {', '.join(sorted(SUPPORTED))})")
    if ext == ".docx":
        return _read_docx(p)
    if ext == ".pdf":
        return _read_pdf(p)
    return _read_text_file(p)
