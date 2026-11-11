import unsloth  # MUST be the absolute first import

import torch
from importlib import import_module
from pathlib import Path
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, get_chat_template

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent

    try:
        load_dataset = import_module("datasets").load_dataset
    except ModuleNotFoundError as exc:
        if exc.name == "datasets":
            raise RuntimeError(
                "The 'datasets' package is required. Install it with: "
                "python -m pip install datasets"
            ) from exc
        raise

    # 1. Configuration Constants
    MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    MAX_SEQ_LENGTH = 4096
    DTYPE = torch.bfloat16
    LOAD_IN_4bit = False  # Must be False for CPU training

    # 2. Initialize Model and Tokenizer for CPU
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4bit,
    )

    # 3. Setup PEFT (Parameter-Efficient Fine-Tuning) via LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # 4. Apply Chat Template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",
    )

    # 5. Load and Format Dataset (using inline mapping to prevent Windows scope errors)
    dataset = load_dataset(
        "json",
        data_files=str(project_dir / "coding_dataset.jsonl"),
        split="train",
    )
    dataset = dataset.map(
        lambda x: {
            "text": [
                tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                for c in x["messages"]
            ]
        },
        batched=True,
    )

    # 6. Configure Trainer using SFTConfig
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            packing=False,
            dataset_num_proc=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=False,
            bf16=False,
            logging_steps=1,
            output_dir=str(project_dir / "outputs"),
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=342,
        ),
    )

    # 7. Execute Training
    trainer_stats = trainer.train()

    # 8. Save Locally & Export to GGUF (for Ollama)
    model.save_pretrained_gguf(
        str(project_dir / "coder79_slm_model"),
        tokenizer,
        quantization_method="q4_k_m"
    )
    print(
        "Training complete! GGUF model successfully saved to "
        f"{project_dir / 'coder79_slm_model'}"
    )