import chromadb
from sentence_transformers import SentenceTransformer

from ingest import Chunk

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

class VectorStore:
    def __init__(self, persist_dir="chroma_db", collection_name="docs"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

    def add_chunks(self, chunks):
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
        )

    def query(self, question, top_k=10):
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
                "retrieval_score": 1 - dist,
            })
        return hits


    def count(self):
        return self.collection.count()

    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(self.collection.name)