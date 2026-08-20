"""
ingest.py
---------
Turns raw PDF files into a list of small, overlapping text chunks.

Why this file exists (the concept):
  Embedding models turn text into fixed-size vectors. If you embed an
  entire 10-page PDF as one vector, the vector becomes a "blurry average"
  of everything in the document -- it won't match a specific question
  well. So we split documents into small chunks *first*, embed each
  chunk *separately*, and let retrieval find the single most relevant
  paragraph instead of the whole document.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader


@dataclass
class Chunk:
    """One retrievable unit of text plus where it came from."""
    text: str
    source: str       # filename
    page: int          # 1-indexed page number
    chunk_id: str       # unique id, e.g. "report.pdf-p3-c0"


def load_pdf_text(pdf_path: str | Path) -> list[tuple[int, str]]:

    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())  # normalize whitespace
        if text.strip():
            pages.append((i, text))
    return pages


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[str]:

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size]
        if piece.strip():
            chunks.append(piece)
        start += step
    return chunks


def ingest_pdf(pdf_path: str | Path, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:

    pdf_path = Path(pdf_path)
    pages = load_pdf_text(pdf_path)

    chunks: list[Chunk] = []
    for page_num, page_text in pages:
        for i, piece in enumerate(chunk_text(page_text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    text=piece,
                    source=pdf_path.name,
                    page=page_num,
                    chunk_id=f"{pdf_path.stem}-p{page_num}-c{i}",
                )
            )
    return chunks


def ingest_folder(folder: str | Path, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Ingest every PDF in a folder."""
    folder = Path(folder)
    all_chunks: list[Chunk] = []
    for pdf_path in sorted(folder.glob("*.pdf")):
        all_chunks.extend(ingest_pdf(pdf_path, chunk_size, overlap))
    return all_chunks


if __name__ == "__main__":
    # Quick manual test: `python src/ingest.py data`
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    chunks = ingest_folder(folder)
    print(f"Produced {len(chunks)} chunks from PDFs in '{folder}'")
    if chunks:
        print("\nFirst chunk:")
        print(chunks[0])
