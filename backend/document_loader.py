"""Loads PDF, TXT, and CSV files into LangChain Document chunks."""
import csv
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

from config import settings


def _load_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_csv(path: str) -> str:
    """Convert CSV to readable text: 'col1: val1 | col2: val2 ...' per row."""
    rows_out = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return ""
        rows_out.append(" | ".join(header))
        for row in reader:
            if not row:
                continue
            paired = [f"{h}: {v}" for h, v in zip(header, row)]
            rows_out.append(" | ".join(paired))
    return "\n".join(rows_out)


def load_documents(path: str) -> List[Document]:
    """Load and chunk a file. Returns a list of LangChain Documents with metadata."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".pdf":
        text = _load_pdf(path)
    elif ext == ".txt":
        text = _load_txt(path)
    elif ext == ".csv":
        text = _load_csv(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": p.name,
                    "chunk_id": i,
                    "file_type": ext.lstrip("."),
                },
            )
        )
    return docs