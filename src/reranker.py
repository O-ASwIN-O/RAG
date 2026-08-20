"""
reranker.py
-----------
The re-ranker: a cross-encoder that scores (query, chunk) pairs.

Architecture (this IS the deep learning model, spelled out):
  input:  "[CLS] <query tokens> [SEP] <chunk tokens> [SEP]"
  body:   a small pretrained transformer (MiniLM, 6 layers) processes
          the whole sequence -- query and chunk attend to EACH OTHER,
          which is the key difference from the embedding retriever.
  head:   the [CLS] token's output vector -> one Linear(hidden_size, 1)
          layer -> sigmoid -> a single relevance score in [0, 1].

Why bother, if the retriever already ranked things?
  Embedding similarity (retriever) is a proxy for relevance -- fast but
  fuzzy. The cross-encoder directly predicts relevance, at the cost of
  being much slower (you can't precompute it, because the score depends
  on BOTH texts together). So the standard pattern is:
    retriever: cheap, scan thousands of chunks, keep top ~10-20
    re-ranker: expensive but accurate, re-score just those ~10-20
"""

from __future__ import annotations
from sentence_transformers import CrossEncoder

# Base model: pretrained on the MS MARCO passage-ranking dataset, so it
# already "knows" what query-passage relevance looks like in general.
# We optionally fine-tune it further on OUR documents in train_reranker.py.
BASE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


class Reranker:
    def __init__(self, model_path: str = BASE_MODEL_NAME):
        # model_path can be BASE_MODEL_NAME (off-the-shelf) or a local
        # folder produced by train_reranker.py (our fine-tuned version).
        self.model = CrossEncoder(model_path)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 4) -> list[dict]:
        """Score every candidate chunk against the query, return the best top_k.

        candidates: list of dicts with at least a "text" key (the output
        of VectorStore.query()).
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)  # one float per pair

        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k]


if __name__ == "__main__":
    # Quick sanity check that the model can tell relevant from irrelevant.
    reranker = Reranker()
    query = "What is the penalty for late payment?"
    candidates = [
        {"text": "Late payments incur a 2% monthly fee after the due date."},
        {"text": "The company was founded in 1998 in Chicago."},
        {"text": "Payment terms are net-30 from the invoice date."},
    ]
    ranked = reranker.rerank(query, candidates, top_k=3)
    for r in ranked:
        print(f"{r['rerank_score']:.3f}  {r['text']}")
