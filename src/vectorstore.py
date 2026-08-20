"""
vectorstore.py
--------------
Wraps sentence-transformers (embeddings) + Chroma (vector database).

The concept:
  An embedding model maps a piece of text to a vector (a list of ~384
  numbers) positioned in space such that semantically similar texts end
  up close together. "Retrieval" then becomes a nearest-neighbor search:
  embed the query, ask Chroma for the k closest chunk vectors.

  This is the FAST, CHEAP, APPROXIMATE first pass. It scans thousands of
  chunks in milliseconds but can be fooled by surface-level word overlap
  rather than true relevance -- that's exactly the gap our cross-encoder
  re-ranker (reranker.py) is trained to close.
"""

from __future__ import annotations
import chromadb
from sentence_transformers import SentenceTransformer

from ingest import Chunk

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, 384-dim, good baseline


class VectorStore:
    def __init__(self, persist_dir: str = "chroma_db", collection_name: str = "docs"):
        # PersistentClient writes to disk, so your index survives restarts.
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
    collection_name,
    metadata={"hnsw:space": "cosine"},
)
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed a batch of chunks and store them in Chroma."""
        if not chunks:
            return
        texts = [c.text for c in chunks]
        # encode() runs the chunks through the embedding model in batches.
        # normalize_embeddings=True makes cosine similarity == dot product,
        # which is what Chroma's default distance metric expects.
        vectors = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
        )

    def query(self, question: str, top_k: int = 10) -> list[dict]:
        """Embed the question, return the top_k nearest chunks.

        top_k is deliberately larger than what you'd show the LLM
        (e.g. 10 here, but only 3-4 make it to the final prompt) --
        we retrieve a wider net cheaply, then let the re-ranker narrow
        it down accurately.
        """
        q_vector = self.embedder.encode([question], normalize_embeddings=True)[0]
        results = self.collection.query(
            query_embeddings=[q_vector.tolist()],
            n_results=top_k,
        )
        hits = []
        for text, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({
                "text": text,
                "source": meta["source"],
                "page": meta["page"],
                "retrieval_score": 1 - dist,  # convert distance -> similarity
            })
        return hits

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(self.collection.name)
