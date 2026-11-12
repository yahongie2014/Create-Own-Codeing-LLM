"""CPU-only LoRA fine-tuning for the local coding and skills datasets."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

from .skills import SkillLibrary


MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
MAX_SEQ_LENGTH = 2_048


def load_training_conversations(project_dir: Path) -> list[dict[str, Any]]:
    data_dir = project_dir / "data"
    conversations: list[dict[str, Any]] = []

    for filename in (
        "coding_dataset.jsonl",
        "coding_dataset_php.jsonl",
        "coding_dataset_advanced.jsonl",
    ):
        dataset_path = data_dir / filename
        with dataset_path.open(encoding="utf-8") as dataset_file:
            for line_number, line in enumerate(dataset_file, start=1):
                if not line.strip():
                    continue
                try:
                    conversations.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {dataset_path} at line {line_number}."
                    ) from exc

    skill_profile = json.loads(
        (data_dir / "coding_skills.json").read_text(encoding="utf-8")
    )
    conversations.extend(skill_profile["training_conversations"])

    skill_library = SkillLibrary(data_dir / "AI-Research-SKILLs")
    conversations.extend(skill_library.training_conversations())
    conversations.extend(skill_library.training_document_conversations())
    return conversations


def tokenize_conversations(
    conversations: list[dict[str, Any]], tokenizer: Any, max_seq_length: int
) -> tuple[list[dict[str, list[int]]], int]:
    if max_seq_length < 1:
        raise ValueError("Maximum sequence length must be positive.")

    training_examples = []
    split_count = 0
    for conversation in conversations:
        rendered = tokenizer.apply_chat_template(
            conversation["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        input_ids = tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        for offset in range(0, len(input_ids), max_seq_length):
            token_chunk = input_ids[offset : offset + max_seq_length]
            training_examples.append(
                {
                    "input_ids": token_chunk,
                    "attention_mask": [1] * len(token_chunk),
                }
            )
        if len(input_ids) > max_seq_length:
            split_count += 1
    return training_examples, split_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune the 0.5B coding model using CPU and system RAM."
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume training from a saved Trainer checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=1,
        help="Training epochs (default: 1 to limit CPU training time).",
    )
    args = parser.parse_args()
    if args.resume_from_checkpoint is not None and not args.resume_from_checkpoint.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {args.resume_from_checkpoint}"
        )
    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CPU training dependencies are missing. Install them with: "
            "python -m pip install -r requirements-training.txt"
        ) from exc

    project_dir = Path(__file__).resolve().parents[1]
    torch.set_num_threads(min(os.cpu_count() or 1, 8))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )

    conversations = load_training_conversations(project_dir)
    training_examples, split_count = tokenize_conversations(
        conversations, tokenizer, MAX_SEQ_LENGTH
    )

    if not training_examples:
        raise ValueError("No training conversations were loaded.")
    print(
        f"CPU-only training: {len(conversations)} conversations, "
        f"{len(training_examples)} token sequences, "
        f"{split_count} long conversations split at {MAX_SEQ_LENGTH} tokens.",
        flush=True,
    )

    def collate(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        batch = tokenizer.pad(
            [
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
                for feature in features
            ],
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

    training_dir = (
        project_dir
        / "artifacts"
        / "training"
        / f"cpu-skills-run-{datetime.now():%Y%m%d-%H%M%S}"
    )
    trainer = Trainer(
        model=model,
        train_dataset=training_examples,
        data_collator=collate,
        args=TrainingArguments(
            output_dir=str(training_dir),
            use_cpu=True,
            per_device_train_batch_size=1,
            num_train_epochs=args.epochs,
            learning_rate=1e-4,
            warmup_steps=0,
            logging_steps=5,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            fp16=False,
            bf16=False,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            report_to="none",
            seed=342,
        ),
    )
    trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        )
    )

    adapter_dir = training_dir / "final-adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    print(f"CPU fine-tuning complete. LoRA adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
