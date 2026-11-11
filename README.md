# Custom Model Training Project

This project fine-tunes a small code-focused language model using a JSONL instruction dataset and exports the trained model to a GGUF format that can be served by Ollama.

## Overview

The project uses:

- Python
- PyTorch
- Hugging Face `datasets`
- TRL `SFTTrainer`
- Unsloth for efficient LoRA fine-tuning
- Ollama for local model serving

The main training flow is implemented in `main.py` and performs the following steps:

1. Loads the `Qwen/Qwen2.5-Coder-1.5B-Instruct` model.
2. Applies LoRA adapters for efficient fine-tuning.
3. Loads a chat-style dataset from `coding_dataset.jsonl`.
4. Formats the dataset using the Qwen chat template.
5. Trains the model using supervised fine-tuning.
6. Exports the trained model to the `coder79_slm_model` directory in GGUF format.

## Repository Structure

- `main.py` — main training script
- `coding_dataset.jsonl` — conversation dataset used for fine-tuning
- `docker-compose.yml` — runs a local Ollama service
- `ollama_data/` — persistent Ollama runtime data
- `coder79_slm_model/` — exported model output (GGUF format)

## Prerequisites

Before running the project, ensure the following are installed:

- Python 3.10+
- PyTorch
- `datasets`
- `trl`
- `unsloth`
- CUDA-enabled GPU is optional for this project, but the script is configured for CPU training in the current setup.

The script includes a guard that raises a clear error if the `datasets` package is missing:

```bash
python -m pip install datasets
```

You will likely also need the ML dependencies required by the selected model and the training stack. In a typical environment, the following packages are required:

```bash
python -m pip install torch datasets trl unsloth
```

If you are using a constrained environment, verify that the dependencies match the installed Python version and operating system.

## Dataset Format

The training data is stored in JSON Lines format (`coding_dataset.jsonl`). Each line is a JSON object containing a chat conversation structure similar to:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert full-stack engineer."},
    {"role": "user", "content": "Write a Python function to parse JSON."},
    {"role": "assistant", "content": "```python\nimport json\n...\n```"}
  ]
}
```

The script applies the Hugging Face chat template to each conversation before training.

## Training Configuration

The main training parameters are defined near the top of `main.py`:

```python
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_SEQ_LENGTH = 4096
DTYPE = torch.bfloat16
LOAD_IN_4bit = False
```

Key training settings include:

- LoRA rank: `r=16`
- Training steps: `max_steps=60`
- Batch size: `per_device_train_batch_size=2`
- Learning rate: `2e-4`
- Output directory: `outputs`
- Seed: `342`

These values are suitable for lightweight experimentation and CPU-oriented development, but they can be adjusted for larger or more specialized training runs.

## How the Script Works

The script follows a straightforward workflow:

1. `FastLanguageModel.from_pretrained(...)` loads the base model and tokenizer.
2. `FastLanguageModel.get_peft_model(...)` wraps the model in LoRA adapters.
3. `get_chat_template(...)` applies the Qwen chat template.
4. `dataset.map(...)` converts the raw message data into plain text prompt strings.
5. `SFTTrainer(...)` executes supervised fine-tuning.
6. `model.save_pretrained_gguf(...)` exports the trained model to a GGUF artifact.

At the end of training, the program prints:

```text
Training complete! GGUF model successfully saved to ./coder79_slm_model
```

## Exporting the Model

The script exports the fine-tuned model to the local folder:

```text
./coder79_slm_model
```

This folder is also mounted into the container defined in `docker-compose.yml`, enabling the Ollama service to use the model as a local runtime artifact.

## Running Ollama via Docker Compose

The repository includes a Docker Compose configuration to start Ollama with the trained model directory mounted.

Start the service:

```bash
docker compose up -d
```

This launches an Ollama container on port `11434`.

To verify the service is running:

```bash
curl http://localhost:11434
```

The service uses resource limits configured in `docker-compose.yml`:

- CPU cap: 6 cores
- Memory limit: 16 GB
- Reserved memory: 8 GB

## Notes and Limitations

- The script is configured for CPU training and sets `LOAD_IN_4bit = False` for compatibility with CPU execution.
- Fine-tuning is intentionally lightweight and may not produce production-quality results without a larger dataset or longer training run.
- The model export is done in GGUF format for compatibility with local inference tools such as Ollama.
- Running the full training pipeline requires substantial memory and a compatible ML environment.

## Suggested Next Steps

- Add more high-quality training examples to `coding_dataset.jsonl`.
- Tune the LoRA hyperparameters for better output quality.
- Increase the number of training steps for stronger convergence.
- Test the exported model in Ollama and evaluate the generated code quality.
- Add validation and evaluation scripts for automatic quality checks.

## Quick Start

```bash
python -m pip install torch datasets trl unsloth
python main.py
```

Then optionally start Ollama:

```bash
docker compose up -d
```

## Summary

This project is a compact local fine-tuning workflow for a coding-focused language model. It trains on a JSONL conversation dataset, uses parameter-efficient fine-tuning, and exports the resulting model to a GGUF format for local serving with Ollama.
