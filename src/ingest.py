from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str


def load_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text.strip():
            pages.append((i, text))
    return pages

def chunk_text(text, chunk_size=800, overlap=150):
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

def ingest_pdf(pdf_path, chunk_size=800, overlap=150):
    pdf_path = Path(pdf_path)
    pages = load_pdf_text(pdf_path) 

    chunks = []
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

def ingest_folder(folder, chunk_size=800, overlap=150):
    folder = Path(folder)
    all_chunks = []
    for pdf_path in sorted(folder.glob("*.pdf")):
        all_chunks.extend(ingest_pdf(pdf_path, chunk_size, overlap))
    return all_chunks


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    chunks = ingest_folder(folder)
    print(f"Produced {len(chunks)} chunks from PDFs in '{folder}'")
    if chunks:
        print("\nFirst chunk:")
        print(chunks[0])