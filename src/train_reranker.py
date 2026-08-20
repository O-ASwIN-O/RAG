"""
train_reranker.py
------------------
Fine-tunes the cross-encoder re-ranker on YOUR labeled (query, chunk,
label) pairs, starting from the pretrained MS MARCO checkpoint.

Run it as:
    python src/train_reranker.py

How to get labeled data (this is the annotation work you do, not code):
  1. Ask your Doc Q&A app some real questions.
  2. Look at the top-10 chunks the retriever pulled back.
  3. Mark each one 1 (actually answers/supports the question) or
     0 (doesn't). Even 30-50 pairs is enough to see a difference,
     because we're fine-tuning, not training from scratch.
  4. Save them as JSONL in training_data/labeled_pairs.jsonl
     (one JSON object per line: {"query": ..., "chunk": ..., "label": ...}).

What "fine-tuning" means here, step by step:
  - Start from a model that already understands language and general
    relevance (the MS MARCO checkpoint) -- we are NOT training a
    transformer from random weights, that would need millions of examples.
  - Show it YOUR (query, chunk, label) pairs in small batches.
  - For each batch: predict a score, measure how wrong it was
    (binary cross-entropy loss), backpropagate the error through the
    network, and adjust the weights slightly (via the Adam optimizer)
    to reduce that error next time.
  - Repeat for a few passes (epochs) over the whole dataset.
  - The result: a model that's still generally good at relevance, but
    now specifically tuned to the vocabulary and relevance judgments in
    YOUR documents (e.g. what "penalty" or "approval" mean in your
    company's context).
"""

from __future__ import annotations
import json
from pathlib import Path

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from torch.utils.data import DataLoader
from sentence_transformers import InputExample

from reranker import BASE_MODEL_NAME

DATA_PATH = Path(__file__).parent.parent / "training_data" / "labeled_pairs.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "finetuned-reranker"


def load_training_examples(path: Path) -> list[InputExample]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # InputExample bundles the pair of texts with its label.
            # label is a float: 1.0 = relevant, 0.0 = not relevant.
            examples.append(
                InputExample(texts=[row["query"], row["chunk"]], label=float(row["label"]))
            )
    return examples


def train(
    data_path: Path = DATA_PATH,
    output_dir: Path = OUTPUT_DIR,
    epochs: int = 4,
    batch_size: int = 8,
):
    examples = load_training_examples(data_path)
    print(f"Loaded {len(examples)} labeled pairs from {data_path}")

    if len(examples) < 10:
        print(
            "Warning: very few examples. Fine-tuning works best with 30+ pairs. "
            "The provided file is just a template -- add your own labeled examples "
            "from your actual documents for real gains."
        )

    # DataLoader batches the examples and shuffles them each epoch --
    # shuffling prevents the model from learning spurious patterns tied
    # to the ORDER examples happen to appear in the file.
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)

    model = CrossEncoder(BASE_MODEL_NAME, num_labels=1)  # 1 = single relevance score

    # BinaryCrossEntropyLoss: compares the model's sigmoid output to the
    # 0/1 label using the formula  loss = -[y*log(p) + (1-y)*log(1-p)].
    # This is the "how wrong were you" signal that drives every weight update.
    loss_fn = BinaryCrossEntropyLoss(model)

    print(f"Fine-tuning for {epochs} epochs...")
    model.fit(
        train_dataloader=train_dataloader,
        loss_fct=loss_fn,
        epochs=epochs,
        warmup_steps=10,        # ramp the learning rate up gradually at the start
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    print(f"Fine-tuned model saved to {output_dir}")
    print("Use it in reranker.py by pointing model_path to this folder.")


if __name__ == "__main__":
    train()
