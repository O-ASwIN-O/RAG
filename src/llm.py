"""
llm.py
------
Sends the re-ranked chunks + the user's question to a local Ollama
model and asks it to answer strictly from that context.

Why "answer only from the context" matters:
  Without this constraint, an LLM will happily use its own pretrained
  knowledge to fill gaps -- which defeats the point of RAG (you wanted
  answers grounded in YOUR documents, not the model's general training
  data). The prompt below explicitly forbids that and asks the model to
  say so when the answer isn't in the provided chunks.
"""

from __future__ import annotations
import ollama

DEFAULT_MODEL = "llama3.2"  # any model you've pulled with `ollama pull <name>`

SYSTEM_PROMPT = """You are a document Q&A assistant. Answer the user's \
question using ONLY the context chunks provided below. If the answer \
isn't in the context, say "I couldn't find that in the provided \
documents" -- do not use outside knowledge. When you use a chunk, cite \
its source and page like this: (source.pdf, p.3)."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Assemble the context block + question into one prompt string."""
    context_block = "\n\n".join(
        f"[Chunk {i+1} - {c['source']}, p.{c['page']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer (cite sources inline):"
    )


def generate_answer(question: str, chunks: list[dict], model: str = DEFAULT_MODEL) -> str:
    """Call the local Ollama server and return the model's answer text."""
    prompt = build_prompt(question, chunks)
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]
