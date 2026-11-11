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

To run the exported model, install and start Docker Desktop and make sure it is running. You do not need Python or the training packages to run the existing GGUF model.

Python 3.10+ and the training dependencies are only needed if you want to train or export the model again:

```bash
python -m pip install torch datasets trl unsloth
```

Training is performed by `main.py` on the host; Docker is used to serve the exported model with Ollama.

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
- Training duration: `num_train_epochs=3`
- Batch size: `per_device_train_batch_size=1`, `gradient_accumulation_steps=1`
- Warmup steps: `0` (the current dataset is too small to benefit from warmup)
- Learning rate: `2e-4`
- Output directory: `outputs`
- Seed: `342`

The checked-in dataset currently contains only three examples. The previous checkpoint records 60 epochs over that tiny dataset, which risks memorization rather than improving general coding ability. The script now uses three epochs by default. Add diverse, high-quality examples and evaluate on held-out tasks before increasing training duration.

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
Training complete! GGUF model successfully saved to <project directory>\coder79_slm_model
```

Training paths are resolved relative to `main.py`, so the script can be launched from a different working directory.

## Exporting the Model

The script exports the fine-tuned model to the local folder:

```text
./coder79_slm_model
```

This folder is also mounted into the container defined in `docker-compose.yml`. The `Modelfile` in this folder tells Ollama which GGUF artifact to load.

## Run the Model with Docker

The exported GGUF file and `Modelfile` are mounted into the Ollama container by `docker-compose.yml`. In PowerShell, from the project root, start the container and register the model:

```powershell
docker compose up -d
docker compose exec --workdir /app/model ollama ollama create coder79_slm_model -f Modelfile
```

Open an interactive chat:

```powershell
docker compose exec -it ollama ollama run coder79_slm_model
```

The Ollama API is available at `http://localhost:11434`. To generate a response from PowerShell:

```powershell
$body = @{
    model = "coder79_slm_model"
    prompt = "Write a Python function to parse JSON safely."
    stream = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:11434/api/generate" `
    -Method Post -ContentType "application/json" -Body $body
```

Open the Ollama web interface at `http://localhost:3000`. It connects to the Ollama service automatically, and its application data is persisted in the `open-webui` Docker volume.

The Ollama model is stored in `ollama_data`, so you only need to run `ollama create` again if that data is removed or the GGUF file changes. Stop the container with:

```powershell
docker compose down
```

### Recreate or remove the Ollama model

After changing the GGUF file or settings in `coder79_slm_model/Modelfile`, recreate the registered model:

```powershell
docker compose exec ollama ollama rm coder79_slm_model
docker compose exec --workdir /app/model ollama ollama create coder79_slm_model -f Modelfile
```

To remove the model from Ollama without deleting the source GGUF file or `Modelfile`:

```powershell
docker compose exec ollama ollama rm coder79_slm_model
```

This removes the registered model from Ollama's `ollama_data`; the source files in `coder79_slm_model/` remain available to register it again.

The service uses resource limits configured in `docker-compose.yml`:

- CPU cap: 6 cores
- Memory limit: 16 GB
- Reserved memory: 8 GB

## Notes and Limitations

- The script is configured for CPU training and sets `LOAD_IN_4bit = False` for compatibility with CPU execution.
- Fine-tuning is intentionally lightweight and may not produce production-quality results without a larger dataset or longer training run.
- The model export is done in GGUF format for compatibility with local inference tools such as Ollama.
- Running the full training pipeline requires substantial memory and a compatible ML environment.
- The Ollama model is a text-generation model, not an operating-system agent. It cannot browse or edit arbitrary local files unless a separate application explicitly supplies those files or connects authorized tools. The system prompt asks it to be honest about this limitation and to ground repository guidance in provided context.
- The training examples and model prompt request concise explanations, not hidden chain-of-thought or `<think>` output.

## Suggested Next Steps

- Add more high-quality training examples to `coding_dataset.jsonl`.
- Tune the LoRA hyperparameters for better output quality.
- Increase the number of training steps for stronger convergence.
- Test the exported model in Ollama and evaluate the generated code quality.
- Add validation and evaluation scripts for automatic quality checks.

## Quick Start

Run the already-exported model from Docker:

```powershell
docker compose up -d
docker compose exec --workdir /app/model ollama ollama create coder79_slm_model -f Modelfile
docker compose exec -it ollama ollama run coder79_slm_model
```

To train/export a new model first, install the training dependencies and run `python main.py`; then repeat the Docker commands above.

## Summary

This project is a compact local fine-tuning workflow for a coding-focused language model. It trains on a JSONL conversation dataset, uses parameter-efficient fine-tuning, and exports the resulting model to a GGUF format for local serving with Ollama.
