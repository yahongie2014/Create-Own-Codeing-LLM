import torch
from importlib import import_module

try:
    load_dataset = import_module("datasets").load_dataset
except ModuleNotFoundError as exc:
    if exc.name == "datasets":
        raise RuntimeError(
            "The 'datasets' package is required. Install it with: "
            "python -m pip install datasets"
        ) from exc
    raise

from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, get_chat_template

# 1. Configuration Constants
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_SEQ_LENGTH = 4096
DTYPE = torch.bfloat16
LOAD_IN_4bit = False    # Must be False for CPU training

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

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

# 5. Load and Format Dataset
dataset = load_dataset("json", data_files="coding_dataset.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 6. Configure Trainer using SFTConfig for modern TRL compatibility
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        dataset_num_proc=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        logging_steps=1,
        output_dir="outputs",
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
    "coder79_slm_model",
    tokenizer,
    quantization_method="q4_k_m"
)
print("Training complete! GGUF model successfully saved to ./coder79_slm_model")