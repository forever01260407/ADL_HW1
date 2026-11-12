from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import ID2LABEL, LABEL2ID, LABELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the actual 10-label classifier")
    parser.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    ).cuda()

    batch = tokenizer(
        ["這個多少錢？", "How much is it and does it have a warranty?"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to("cuda")
    labels = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    outputs = model(**batch, labels=labels)
    print(f"logits shape: {tuple(outputs.logits.shape)}")
    print(f"loss: {outputs.loss.item():.6f}")
    assert outputs.logits.shape == (2, len(LABELS))
    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()
    print("Classifier forward/backward test passed.")
    print("The randomly initialized classification head must now be fine-tuned.")


if __name__ == "__main__":
    main()

