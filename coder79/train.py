"""Train a small language model from random initialization on local data."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from .skills import SkillLibrary


MAX_SEQ_LENGTH = 512
TOKENIZER_VOCAB_SIZE = 8_192
HIDDEN_SIZE = 256
NUM_LAYERS = 4
NUM_ATTENTION_HEADS = 4
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
CHAT_TOKENS = ("<|system|>", "<|user|>", "<|assistant|>")
GGUF_FILENAME = "coder79-scratch-f16.gguf"


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


def train_tokenizer(
    conversations: list[dict[str, Any]],
    output_dir: Path,
    vocab_size: int = TOKENIZER_VOCAB_SIZE,
) -> Any:
    import sentencepiece as spm
    from transformers import LlamaTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        sentence_iterator=(
            message["content"]
            for conversation in conversations
            for message in conversation["messages"]
        ),
        model_prefix=str(output_dir / "tokenizer"),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=True,
        hard_vocab_limit=False,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=PAD_TOKEN,
        unk_piece=UNK_TOKEN,
        bos_piece=BOS_TOKEN,
        eos_piece=EOS_TOKEN,
        user_defined_symbols=list(CHAT_TOKENS),
        shuffle_input_sentence=False,
        minloglevel=2,
    )
    tokenizer = LlamaTokenizer(
        vocab_file=str(output_dir / "tokenizer.model"),
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        additional_special_tokens=list(CHAT_TOKENS),
        legacy=False,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|' + message['role'] + '|>\\n' + message['content'] + eos_token }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
    )
    return tokenizer


def export_to_gguf(
    model_dir: Path, output_path: Path, llama_cpp_dir: Path
) -> Path:
    llama_cpp_dir = llama_cpp_dir.expanduser().resolve()
    output_path = output_path.resolve()
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise FileNotFoundError(
            f"llama.cpp converter not found: {converter}. "
            "Pass the llama.cpp folder with --llama-cpp-dir or set LLAMA_CPP_DIR."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(converter),
            str(model_dir.resolve()),
            "--outfile",
            str(output_path.resolve()),
            "--outtype",
            "f16",
        ],
        check=True,
        cwd=llama_cpp_dir,
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"GGUF converter did not create a model: {output_path}")
    return output_path


def default_llama_cpp_dir() -> Path | None:
    configured_dir = os.environ.get("LLAMA_CPP_DIR")
    if configured_dir:
        return Path(configured_dir)

    standard_dir = Path("C:/tools/llama.cpp")
    if os.name == "nt" and (standard_dir / "convert_hf_to_gguf.py").is_file():
        return standard_dir
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a small language model from scratch using CPU and local data."
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
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=default_llama_cpp_dir(),
        help="Path to a llama.cpp checkout (or set LLAMA_CPP_DIR) for GGUF export.",
    )
    args = parser.parse_args()
    if args.resume_from_checkpoint is not None and not args.resume_from_checkpoint.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {args.resume_from_checkpoint}"
        )
    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")
    if args.llama_cpp_dir is None:
        parser.error(
            "GGUF export requires llama.cpp. Pass --llama-cpp-dir or set LLAMA_CPP_DIR."
        )
    llama_cpp_dir = args.llama_cpp_dir.expanduser().resolve()
    if not (llama_cpp_dir / "convert_hf_to_gguf.py").is_file():
        parser.error(
            f"llama.cpp converter not found in {llama_cpp_dir}; "
            "expected convert_hf_to_gguf.py."
        )

    try:
        import torch
        from transformers import (
            LlamaConfig,
            LlamaForCausalLM,
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

    conversations = load_training_conversations(project_dir)
    if not conversations:
        raise ValueError("No training conversations were loaded.")

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    training_dir = (
        project_dir
        / "artifacts"
        / "training"
        / f"cpu-scratch-run-{run_timestamp}"
    )
    torch.manual_seed(342)
    tokenizer = train_tokenizer(conversations, training_dir)
    model_config = LlamaConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=MAX_SEQ_LENGTH,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=HIDDEN_SIZE * 4,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_ATTENTION_HEADS,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(model_config)
    model.config.use_cache = False

    training_examples, split_count = tokenize_conversations(
        conversations, tokenizer, MAX_SEQ_LENGTH
    )

    if not training_examples:
        raise ValueError("No training conversations were loaded.")
    print(
        f"CPU-only from-scratch training: {len(conversations)} conversations, "
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

    trainer = Trainer(
        model=model,
        train_dataset=training_examples,
        data_collator=collate,
        args=TrainingArguments(
            output_dir=str(training_dir),
            use_cpu=True,
            per_device_train_batch_size=1,
            num_train_epochs=args.epochs,
            learning_rate=5e-4,
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

    model_dir = training_dir / "final-model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)
    gguf_path = export_to_gguf(
        model_dir,
        project_dir / "models" / "coder79" / GGUF_FILENAME,
        llama_cpp_dir,
    )
    print(f"CPU from-scratch training complete. Full model saved to {model_dir}")
    print(f"Importable GGUF model saved to {gguf_path}")


if __name__ == "__main__":
    main()
