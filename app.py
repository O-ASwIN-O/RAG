"""
app.py
------
Streamlit UI for the Document Q&A Assistant.
Run with:  streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))  # so `import ingest` etc. works

import streamlit as st
from ingest import ingest_pdf
from pipeline import DocQAPipeline

st.set_page_config(page_title="Doc Q&A Assistant", layout="wide")
st.title("Document Q&A Assistant")
st.caption("RAG + a fine-tunable cross-encoder re-ranker, fully local via Ollama.")


@st.cache_resource
def get_pipeline(reranker_path: str | None, llm_model: str) -> DocQAPipeline:
    # @st.cache_resource keeps the models loaded in memory across
    # reruns -- without it, Streamlit would reload the embedding model
    # and re-ranker on every single interaction, which is slow.
    return DocQAPipeline(reranker_model_path=reranker_path, llm_model=llm_model)


with st.sidebar:
    st.header("Setup")
    llm_model = st.text_input("Ollama model name", value="llama3.2")
    use_finetuned = st.checkbox("Use fine-tuned re-ranker (models/finetuned-reranker)")
    reranker_path = "models/finetuned-reranker" if use_finetuned else None

    st.divider()
    st.header("Upload documents")
    uploaded = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if uploaded and st.button("Ingest uploaded PDFs"):
        pipeline = get_pipeline(reranker_path, llm_model)
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        total_chunks = 0
        for f in uploaded:
            save_path = data_dir / f.name
            save_path.write_bytes(f.read())
            chunks = ingest_pdf(save_path)
            pipeline.store.add_chunks(chunks)
            total_chunks += len(chunks)
        st.success(f"Ingested {len(uploaded)} PDF(s), {total_chunks} chunks added.")

    st.divider()
    pipeline_preview = get_pipeline(reranker_path, llm_model)
    st.metric("Chunks in store", pipeline_preview.store.count())
    if st.button("Clear vector store"):
        pipeline_preview.store.reset()
        st.rerun()

st.divider()
question = st.text_input("Ask a question about your documents")
retrieve_k = st.slider("Retrieve top-k (fast, approximate)", 4, 20, 10)
rerank_k = st.slider("Keep top-k after re-ranking (sent to the LLM)", 1, 8, 4)

if question:
    pipeline = get_pipeline(reranker_path, llm_model)
    with st.spinner("Retrieving, re-ranking, and generating..."):
        result = pipeline.answer(question, retrieve_k=retrieve_k, rerank_k=rerank_k)

    st.subheader("Answer")
    st.write(result["answer"])

    if result["chunks"]:
        st.subheader("Chunks used (after re-ranking)")
        st.caption(
            "retrieval_score = cosine similarity from the embedding model. "
            "rerank_score = the cross-encoder's learned relevance score. "
            "Watch for cases where these two disagree -- that's the re-ranker "
            "correcting the retriever's mistakes."
        )
        for c in result["chunks"]:
            with st.expander(
                f"{c['source']} p.{c['page']}  |  "
                f"retrieval={c['retrieval_score']:.3f}  rerank={c['rerank_score']:.3f}"
            ):
                st.write(c["text"])
