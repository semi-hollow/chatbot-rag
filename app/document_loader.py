from pathlib import Path
from typing import Iterable

from docx import Document

SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx"}


def iter_document_paths(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
        yield path
        return

    if path.is_dir():
        for ext in SUPPORTED_EXTENSIONS:
            for doc in path.rglob(f"*{ext}"):
                if doc.is_file():
                    yield doc
        return

    raise ValueError(f"Unsupported input path: {path}")


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".docx":
        doc = Document(str(path))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())

    raise ValueError(f"Unsupported file type: {path.suffix}")


def extract_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip(" #\t")
        if stripped:
            return stripped[:120]
    return path.stem
