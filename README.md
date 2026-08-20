# Document Q&A Assistant with a Custom Re-ranker

A local RAG (retrieval-augmented generation) chatbot with a fine-tunable
cross-encoder re-ranker sitting between retrieval and generation.

## How it works

```
PDFs -> chunk -> embed (sentence-transformers) -> Chroma vector store
                                                          |
User question -> embed -> retrieve top-k (fast, approximate)
                                                          |
                          cross-encoder re-ranker (accurate, learned)
                                                          |
                          top few chunks -> local LLM (Ollama) -> answer
```

Two models are doing work here:
- **Embedding model** (`all-MiniLM-L6-v2`): off-the-shelf, turns text into vectors for fast approximate search.
- **Cross-encoder re-ranker** (`cross-encoder/ms-marco-MiniLM-L6-v2`): the model you can fine-tune on your own labeled data to correct the retriever's mistakes.

## Setup

### 1. Install Ollama and pull a model
Ollama runs the LLM locally, no API key needed.
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
# or download from https://ollama.com/download for Windows

ollama pull llama3.2
```
Keep the Ollama app/service running in the background -- it serves a local API on `localhost:11434` that `src/llm.py` talks to.

### 2. Install Python dependencies
```bash
cd doc-qa-reranker
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```
Upload a PDF or two in the sidebar, click "Ingest uploaded PDFs", then ask a question.

## Fine-tuning the re-ranker (the deep learning part)

1. Ask your app real questions and look at the retrieved chunks.
2. Label them: edit `training_data/labeled_pairs.jsonl`, one line per
   example: `{"query": "...", "chunk": "...", "label": 1}` (1 =
   relevant, 0 = not). Aim for 30-50+ pairs from your actual documents
   for a real improvement over the template file included here.
3. Run the fine-tuning script:
   ```bash
   python src/train_reranker.py
   ```
   This saves a fine-tuned model to `models/finetuned-reranker/`.
4. In the Streamlit sidebar, check "Use fine-tuned re-ranker" to switch
   the pipeline over to your version.

## Project structure

```
doc-qa-reranker/
  app.py                    Streamlit UI
  requirements.txt
  data/                     uploaded PDFs land here
  chroma_db/                persistent vector store (auto-created)
  training_data/
    labeled_pairs.jsonl     your (query, chunk, label) examples
  models/
    finetuned-reranker/     output of train_reranker.py (auto-created)
  src/
    ingest.py                PDF -> chunks
    vectorstore.py            embeddings + Chroma
    reranker.py                cross-encoder inference
    train_reranker.py           cross-encoder fine-tuning
    llm.py                       Ollama prompt + call
    pipeline.py                   retrieve -> rerank -> generate
```

## Things to try next (each teaches you something different)
- Change `chunk_size`/`overlap` in `ingest.py` and see how it affects retrieval quality.
- Print `retrieval_score` vs `rerank_score` in the UI for the same query before and after fine-tuning -- watch the re-ranker's judgments shift toward your labels.
- Swap `all-MiniLM-L6-v2` for a larger embedding model and compare retrieval quality vs. speed.
- Try `MarginMSE` loss instead of `BinaryCrossEntropyLoss` in `train_reranker.py` if you have graded relevance (not just 0/1).
