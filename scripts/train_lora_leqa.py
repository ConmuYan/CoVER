"""LoRA training loop for LEQA warm-up.

Trains Qwen3-4B + LoRA on synthetic per-token quality labels using
HuggingFace Trainer + peft.  Loss: standard causal-LM cross-entropy
on the JSON output template.

No audit loss in this script -- that comes during joint reasoner+LoRA
training in Block 2 (EXPERIMENT_PLAN).  Commit 2 just produces a
working warm-up checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LoRA for LEQA warm-up")
    p.add_argument("--base_model", type=str,
                   default="/data1/mq/models/Qwen3-4B-Instruct-2507")
    p.add_argument("--train_data", type=str, required=True,
                   help="Path to synthetic training JSONL")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save LoRA adapter")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_training_data(path: str, tokenizer, max_seq_len: int):
    """Load synthetic JSONL and tokenize for causal LM training."""
    from torch.utils.data import Dataset

    class LEQADataset(Dataset):
        def __init__(self, texts, tokenizer, max_len):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )

        def __len__(self):
            return self.encodings["input_ids"].shape[0]

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = item["input_ids"].clone()
            return item

    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # The record should contain a "training_text" field
            # (produced by generate_synth_token_quality.py)
            if "training_text" in record:
                texts.append(record["training_text"])
            elif "text" in record:
                texts.append(record["text"])

    if not texts:
        raise ValueError(f"No training examples found in {path}")
    logger.info("Loaded %d training examples from %s", len(texts), path)
    return LEQADataset(texts, tokenizer, max_seq_len)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Lazy imports so script can be parsed without GPU
    from transformers import TrainingArguments, Trainer
    from models.lora_qwen_loader import (
        load_base_model_and_tokenizer,
        attach_lora,
    )

    logger.info("Loading base model: %s", args.base_model)
    model, tokenizer = load_base_model_and_tokenizer(
        args.base_model, bf16=args.bf16, device_map="auto",
    )

    logger.info("Attaching LoRA (r=%d, alpha=%d)", args.rank, args.alpha)
    model = attach_lora(model, rank=args.rank, alpha=args.alpha)

    logger.info("Loading training data: %s", args.train_data)
    dataset = load_training_data(args.train_data, tokenizer, args.max_seq_len)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.0,
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="epoch",
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    logger.info("Starting LoRA training for %d epochs", args.epochs)
    trainer.train()

    # Save final adapter
    logger.info("Saving LoRA adapter to %s", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training metrics
    metrics_path = output_dir / "train_metrics.json"
    metrics = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "rank": args.rank,
        "alpha": args.alpha,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "bf16": args.bf16,
        "seed": args.seed,
        "num_examples": len(dataset),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Training complete. Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
