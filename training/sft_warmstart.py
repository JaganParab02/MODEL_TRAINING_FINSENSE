"""
Real Supervised Fine-Tuning (SFT) warm-start for FinSense.

This script trains a lightweight LoRA adapter on the JSON decision format used
throughout the project so it can be demonstrated in Colab with a small model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "./checkpoints/sft-warmstart"


def ensure_pad_token(tokenizer) -> None:
    """Set a safe pad token for causal LM training."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def load_dataset_splits(
    data_path: str,
    eval_split: float = 0.2,
    seed: int = 42,
    max_samples: int | None = None,
) -> DatasetDict:
    """Load the warm-start dataset and split it deterministically."""
    dataset = load_dataset("json", data_files=data_path, split="train")

    if max_samples is not None:
        limit = min(max_samples, len(dataset))
        dataset = dataset.select(range(limit))

    if len(dataset) < 2:
        raise ValueError("Need at least 2 examples to create train/eval splits.")

    dataset = dataset.shuffle(seed=seed)

    if eval_split <= 0:
        return DatasetDict({"train": dataset, "eval": dataset.select([])})

    eval_count = max(1, int(round(len(dataset) * eval_split)))
    eval_count = min(eval_count, len(dataset) - 1)
    train_count = len(dataset) - eval_count

    train_dataset = dataset.select(range(train_count))
    eval_dataset = dataset.select(range(train_count, len(dataset)))
    return DatasetDict({"train": train_dataset, "eval": eval_dataset})


def format_chat_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """Render chat messages into plain text for SFTTrainer."""

    def _format_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        texts = []
        for messages in batch["messages"]:
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text": texts}

    return dataset.map(_format_batch, batched=True, remove_columns=dataset.column_names)


def build_lora_config() -> LoraConfig:
    """LoRA config tuned for small instruct models in Colab."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_sft_config(
    output_dir: str,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    epochs: float,
    max_seq_length: int,
    seed: int,
) -> SFTConfig:
    """Create trainer config with Colab-friendly defaults."""
    use_cuda = torch.cuda.is_available()
    return SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        gradient_checkpointing=use_cuda,
        fp16=use_cuda,
        bf16=False,
        seed=seed,
    )


def train_sft(
    model_name: str = DEFAULT_MODEL,
    data_path: str = "warmstart_data.jsonl",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    epochs: float = 3.0,
    batch_size: int = 2,
    grad_accum: int = 2,
    learning_rate: float = 2e-5,
    max_seq_length: int = 512,
    eval_split: float = 0.2,
    seed: int = 42,
    max_samples: int | None = None,
    dataset_splits: DatasetDict | None = None,
) -> dict[str, Any]:
    """Train a LoRA adapter and save it to disk."""
    set_seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if dataset_splits is None:
        dataset_splits = load_dataset_splits(
            data_path=data_path,
            eval_split=eval_split,
            seed=seed,
            max_samples=max_samples,
        )

    train_dataset = dataset_splits["train"]
    eval_dataset = dataset_splits["eval"]

    print(f"[SFT] Model: {model_name}")
    print(f"[SFT] Train examples: {len(train_dataset)}")
    print(f"[SFT] Eval examples: {len(eval_dataset)}")
    print(f"[SFT] Output dir: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ensure_pad_token(tokenizer)

    formatted_train = format_chat_dataset(train_dataset, tokenizer)

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    model.config.use_cache = False

    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(
            output_dir=output_dir,
            batch_size=batch_size,
            grad_accum=grad_accum,
            learning_rate=learning_rate,
            epochs=epochs,
            max_seq_length=max_seq_length,
            seed=seed,
        ),
        train_dataset=formatted_train,
        peft_config=build_lora_config(),
        processing_class=tokenizer,
    )

    print("[SFT] Training started...")
    train_result = trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "model_name": model_name,
        "data_path": data_path,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "seed": seed,
        "max_samples": max_samples,
        "train_runtime_sec": train_result.metrics.get("train_runtime") if hasattr(train_result, "metrics") else None,
        "train_loss": train_result.metrics.get("train_loss") if hasattr(train_result, "metrics") else None,
    }

    metadata_path = Path(output_dir) / "sft_run_config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[SFT] Adapter saved to {output_dir}")
    print(f"[SFT] Metadata saved to {metadata_path}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA SFT warm-start for FinSense.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model name.")
    parser.add_argument("--data", default="warmstart_data.jsonl", help="Path to JSONL training data.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for the adapter.")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max training sequence length.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on total samples loaded.")
    parser.add_argument("--eval-split", type=float, default=0.2, help="Holdout ratio for deterministic split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_sft(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        eval_split=args.eval_split,
        seed=args.seed,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
