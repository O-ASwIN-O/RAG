"""
pipeline.py
-----------
Orchestrates the full flow: retrieve -> re-rank -> generate.
This is the file app.py (the UI) calls -- it has no UI code itself,
so you can also run it from a plain script or a notebook.
"""

from __future__ import annotations
from vectorstore import VectorStore
from reranker import Reranker
from llm import generate_answer


class DocQAPipeline:
    def __init__(
        self,
        persist_dir: str = "chroma_db",
        reranker_model_path: str | None = None,
        llm_model: str = "llama3.2",
    ):
        self.store = VectorStore(persist_dir=persist_dir)
        # If you fine-tuned a re-ranker, pass its folder path here;
        # otherwise it falls back to the pretrained MS MARCO checkpoint.
        from reranker import BASE_MODEL_NAME
        self.reranker = Reranker(reranker_model_path or BASE_MODEL_NAME)
        self.llm_model = llm_model

    def answer(
        self,
        question: str,
        retrieve_k: int = 10,
        rerank_k: int = 4,
    ) -> dict:

        # Stage 1: cheap approximate retrieval
        candidates = self.store.query(question, top_k=retrieve_k)
        if not candidates:
            return {
                "answer": "No documents have been ingested yet -- upload some PDFs first.",
                "chunks": [],
            }

        # Stage 2: accurate re-ranking (the trained model)
        top_chunks = self.reranker.rerank(question, candidates, top_k=rerank_k)

        # Stage 3: grounded generation
        answer_text = generate_answer(question, top_chunks, model=self.llm_model)

        return {"answer": answer_text, "chunks": top_chunks}
