# Custom Model Training Project

This project serves a code-focused model through Ollama and provides an optional small-model fine-tuning pipeline. The default Coder79 service uses Qwen2.5-Coder 3B with the project's system instructions; optional local training remains a separate workflow.

## Overview

The project uses:

- Python
- PyTorch
- Hugging Face `datasets`
- TRL `SFTTrainer`
- Unsloth for efficient LoRA fine-tuning
- Ollama for local model serving

The default serving flow pulls `qwen2.5-coder:3b` from Ollama, then applies Coder79's system instructions and serves it under the `coder79` name. The optional training flow in `main.py` remains separate:

1. Loads the `Qwen/Qwen2.5-Coder-1.5B-Instruct` model.
2. Applies LoRA adapters for efficient fine-tuning.
3. Loads the chat-style datasets and skill-profile examples.
4. Formats the dataset using the Qwen chat template.
5. Trains the model using supervised fine-tuning.
6. Exports the trained model to the `coder79_slm_model` directory in GGUF format. This output does not replace the default 3B serving model.

## Repository Structure

- `main.py` — main training script
- `coding_dataset.jsonl` — conversation dataset used for fine-tuning
- `docker-compose.yml` — runs a local Ollama service
- `ollama_data/` — persistent Ollama runtime data
- `coder79_slm_model/` — exported model output (GGUF format)
- `coder79_agent.py` — read-only terminal project explorer
- `coder79_ide_server.py` — MCP and OpenAI-compatible IDE integration

## Prerequisites

To run Coder79, install and start Docker Desktop and make sure it is running. Internet access is needed on first startup to download the 3B model. You do not need Python or the training packages for standard Ollama chat. Python 3.10+ is needed to run the local file-navigation agent; it uses only the standard library. Training packages are needed only for the separate optional fine-tuning workflow:

```bash
python -m pip install torch datasets trl unsloth
```

Training is performed by `main.py` on the host; Docker is used to serve the exported model with Ollama.

## Dataset Format

The training data is stored in JSON Lines files: `coding_dataset.jsonl`, `coding_dataset_php.jsonl`, and `coding_dataset_advanced.jsonl`. `coding_skills.json` contains the structured skill profile and additional training conversations. Each JSONL line is a chat object similar to:

```json
{
  "messages": [
    {"role": "system", "content": "You are Coder79, a knowledgeable full-stack software engineer."},
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

The checked-in datasets provide 31 training conversations: 9 core coding, 6 advanced coding, 3 PHP, and 13 skill-profile examples. The expanded profile emphasizes repository analysis, debugging from evidence, architecture and migrations, reliability, testing, frontend accessibility, and code review. `main.py` reads these conversations from the three JSONL files and `coding_skills.json`. This remains a small dataset; it can overfit and does not guarantee these capabilities. Evaluate on held-out tasks before increasing training duration.

### Applying and training updated skills

The serving model and its system prompt are configured by `write_coder79_modelfile.ps1`. The startup batch file regenerates the Modelfile, pulls the 3B base if needed, registers `coder79`, and starts the WebUI and IDE agent. To perform these steps manually:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\write_coder79_modelfile.ps1
docker compose up -d --build ollama-webui coder79-agent
```

The 3B base is a speed-focused compromise for CPU-only local use; it is smaller and generally responds faster than the previously configured 7B model, though with less capacity. The optional `python main.py` training pipeline still fine-tunes the 1.5B base; its output is not automatically used by the default Coder79 service.

## How the Script Works

The script follows a straightforward workflow:

1. `FastLanguageModel.from_pretrained(...)` loads the base model and tokenizer.
2. `FastLanguageModel.get_peft_model(...)` wraps the model in LoRA adapters.
3. `get_chat_template(...)` applies the Qwen chat template.
4. `dataset.map(...)` converts conversations from the JSONL files and `coding_skills.json` into plain text prompt strings.
5. `SFTTrainer(...)` executes supervised fine-tuning.
6. `model.save_pretrained_gguf(...)` exports the trained model to a GGUF artifact.

At the end of training, the program prints:

```text
Training complete! GGUF model successfully saved to <project directory>\coder79_slm_model
```

Training paths are resolved relative to `main.py`, so the script can be launched from a different working directory.
The trainer creates a new timestamped output folder for each run. To restore a completed or interrupted run, pass its checkpoint directory, for example:

```powershell
python main.py --resume-from-checkpoint outputs\skills-run-<timestamp>\checkpoint-60
```

## Exporting the Model

The script exports the fine-tuned model to the local folder:

```text
./coder79_slm_model
```

This folder is also mounted into the container defined in `docker-compose.yml`. The `Modelfile` in this folder tells Ollama which GGUF artifact to load.

## Run the Model with Docker

The exported GGUF file and `Modelfile` are mounted into the Ollama container by `docker-compose.yml`. The Docker container is named `coder79model`, and the Ollama model is named `coder79`. In PowerShell, from the project root, start the container and register the model:

```powershell
docker compose up -d
docker compose exec --workdir /app/model ollama ollama create coder79 -f Modelfile
```

Open an interactive chat:

```powershell
docker compose exec -it ollama ollama run coder79
```

The Ollama API is available at `http://localhost:11434`. To generate a response from PowerShell:

```powershell
$body = @{
    model = "coder79"
    prompt = "Write a Python function to parse JSON safely."
    stream = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:11434/api/generate" `
    -Method Post -ContentType "application/json" -Body $body
```

Open the Ollama web interface at `http://localhost:3000`. It connects to the Ollama service automatically, and its application data is persisted in the `open-webui` Docker volume.

### Use Coder79 from an IDE

The local IDE server connects an IDE to Ollama and scopes file access to one project folder you select. It uses Python's standard library; no extra packages are needed. Start Docker/Ollama first. There are two connection methods—use the one your IDE supports:

#### Configure VS Code to use Coder79

This repository includes a VS Code task and a copy-ready custom model entry:

- `.vscode/tasks.json` builds and starts the Docker Coder79 agent, scoped to the currently opened workspace.
- `.vscode/coder79-chatLanguageModels.json` is a template for registering the model in your VS Code user profile.

1. Open this `customModel` folder in VS Code and make sure Docker Desktop is running.

2. Press `Ctrl+Shift+P`, run **Tasks: Run Task**, then select **Start Coder79 for this workspace**. Docker starts Ollama, registers `coder79` from the local GGUF if it has not been registered yet, builds the agent image, mounts the selected workspace read-only, and connects the agent to Ollama over the Compose network. Leave the task terminal running; run `docker compose stop coder79-agent` to stop the IDE agent.

3. Run **Chat: Manage Language Models** from the Command Palette, select **Add Models** → **Custom Endpoint**, and choose **Chat Completions**. When VS Code opens its `chatLanguageModels.json` user-profile file, copy the contents of `.vscode/coder79-chatLanguageModels.json` into that file and save. If VS Code already has providers in the array, add the `Coder79 Local` object rather than replacing them.

4. Open the Chat view, select **Coder79** from the model picker, and use Chat/Ask mode.

VS Code stores model providers in your user profile, not in this project, so importing the template is a one-time per-profile setup. The included task mounts the currently opened folder into the agent container. For another project, run Docker Compose from this `customModel` folder and set `WORKSPACE_PATH` to the project folder first:

```powershell
$env:WORKSPACE_PATH = (Resolve-Path "C:\path\to\your\project").Path
docker compose up -d --build coder79-agent
```

The endpoint supports Chat Completions and streamed responses. Coder79 gets the mounted workspace tree and has internal read-only search/read tools. **This setup is for Chat/Ask, not autonomous Agent/Edit mode**: the model does not expose VS Code's editing tools or write files. The agent container listens on its private Docker interface, but the host port is bound to `127.0.0.1`; do not change it to a public interface.

#### MCP workspace tools

Use this when the IDE supports MCP servers. MCP makes the read-only project tools available to that IDE's selected assistant model; by itself, it does not switch that assistant to Coder79. Add this server to the IDE's MCP configuration (for example, `.vscode/mcp.json` in the project you want to open), replacing the script path with this repository's actual location:

```json
{
  "servers": {
    "coder79": {
      "type": "stdio",
      "command": "python",
      "args": [
        "C:\\path\\to\\customModel\\coder79_ide_server.py",
        "--transport",
        "mcp",
        "--workspace",
        "${workspaceFolder}"
      ]
    }
  }
}
```

Restart or reconnect MCP servers after changing the configuration. The MCP server runs locally and reads files only under the configured workspace. Both methods provide tree listing, text search, and bounded file reads; common generated folders and likely credential files are excluded. IDEs differ, so use MCP where supported, or the OpenAI-compatible endpoint where custom model providers are supported.

For terminal chat without an IDE, the simpler interactive client is still available:

```powershell
python coder79_agent.py "C:\path\to\your\project"
```

It uses the `coder79` model by default. Type `/exit` to end the chat; use `--model` or `--ollama-url` to select another model or Ollama server.

The Ollama model is stored in `ollama_data`, so you only need to run `ollama create` again if that data is removed or the GGUF file changes. Stop the container with:

```powershell
docker compose down
```

### Recreate or remove the Ollama model

After changing the GGUF file or settings in `coder79_slm_model/Modelfile`, recreate the registered model:

```powershell
docker compose exec ollama ollama rm coder79
docker compose exec --workdir /app/model ollama ollama create coder79 -f Modelfile
```

To remove the model from Ollama without deleting the source GGUF file or `Modelfile`:

```powershell
docker compose exec ollama ollama rm coder79
```

This removes the registered model from Ollama's `ollama_data`; the source files in `coder79_slm_model/` remain available to register it again.

The service uses resource limits configured in `docker-compose.yml`:

- CPU cap: 6 cores
- Memory limit: 16 GB
- Reserved memory: 8 GB

## Notes and Limitations

- The script is configured for CPU training and sets `LOAD_IN_4bit = False` for compatibility with CPU execution.
- Fine-tuning is intentionally lightweight and may not produce production-quality results without a larger dataset or longer training run.
- Updating the skill profile or datasets does not change model weights until `main.py` completes training and exports the GGUF. CPU training is supported by the installed Unsloth version but is substantially slower than GPU training.
- The model export is done in GGUF format for compatibility with local inference tools such as Ollama.
- Running the full training pipeline requires substantial memory and a compatible ML environment.
- The Ollama model is a text-generation model, not an operating-system agent. The local terminal and IDE integrations provide read-only access to one explicitly selected workspace; the model cannot inspect other folders or edit or run project files.
- The training examples and model prompt request concise explanations, not hidden chain-of-thought or `<think>` output.

## Suggested Next Steps

- Add more high-quality training examples to `coding_dataset.jsonl` and run the training/export pipeline on a compatible machine.
- Tune the LoRA hyperparameters for better output quality.
- Increase the number of training steps for stronger convergence.
- Test the exported model in Ollama and evaluate the generated code quality.
- Add validation and evaluation scripts for automatic quality checks.

## Quick Start

On Windows, start Docker Desktop and double-click `start_and_test_coder79.bat`. It checks the training inputs and GGUF; if the GGUF is missing, it creates the model folder and runs `python main.py`. Then it starts Ollama, creates the Modelfile if needed, registers or updates `coder79`, and runs an inference check. Training dependencies must already be installed for automatic training. Double-click `stop_coder79.bat` to stop the Ollama container without deleting the model or its saved data.

Run the already-exported model from Docker:

```powershell
docker compose up -d
docker compose exec --workdir /app/model ollama ollama create coder79 -f Modelfile
docker compose exec -it ollama ollama run coder79
```

To train/export a new model first, install the training dependencies and run `python main.py`; then repeat the Docker commands above.

## Summary

This project is a compact local fine-tuning workflow for a coding-focused language model. It trains on a JSONL conversation dataset, uses parameter-efficient fine-tuning, and exports the resulting model to a GGUF format for local serving with Ollama.
