# Custom Model Training Project

This project serves a code-focused model through Ollama and provides an optional small-model fine-tuning pipeline. The default Coder79 service uses Qwen2.5-Coder 3B with the project's system instructions; optional local training remains a separate workflow. Coder79's terminal and IDE agent integrations can also fetch public static web pages when given a URL.

## Repository Layout

- `coder79/` — Python package for the workspace agent, IDE server, and training entry point.
- `tests/` — unit tests for the workspace tools and IDE API.
- `data/` — training conversations and skill profile.
- `models/coder79/` — Ollama `Modelfile` and local GGUF/model export artifacts.
- `artifacts/training/` — generated fine-tuning runs and checkpoints (ignored by Git).
- `ollama_data/` — persistent Ollama runtime state (ignored by Git; do not share).
- `.continue/` and `.vscode/` — Continue and VS Code workspace configuration.
- Root files — Docker/Compose and Windows start/stop helpers.

## Overview

The project uses:

- Python
- PyTorch
- Hugging Face `datasets`
- TRL `SFTTrainer`
- Unsloth for efficient LoRA fine-tuning
- Ollama for local model serving

The default serving flow pulls `qwen2.5-coder:3b` from Ollama, then applies Coder79's system instructions and serves it under the `coder79` name. The optional training flow in `coder79/train.py` remains separate:

1. Loads the `Qwen/Qwen2.5-Coder-1.5B-Instruct` model.
2. Applies LoRA adapters for efficient fine-tuning.
3. Loads the chat-style datasets and skill-profile examples.
4. Formats the dataset using the Qwen chat template.
5. Trains the model using supervised fine-tuning.
6. Exports the trained model to `models/coder79/` in GGUF format. This output does not replace the default 3B serving model.

## Prerequisites

To run Coder79, install and start Docker Desktop and make sure it is running. Internet access is needed on first startup to download the 3B model. You do not need Python or the training packages for standard Ollama chat. Python 3.10+ is needed to run the local file-navigation agent; it uses only the standard library. Training packages are needed only for the separate optional fine-tuning workflow:

```bash
python -m pip install torch datasets trl unsloth
```

Training is performed on the host with `python -m coder79.train`; Docker is used to serve the exported model with Ollama.

## Dataset Format

The training data is stored in `data/` as JSON Lines files: `coding_dataset.jsonl`, `coding_dataset_php.jsonl`, and `coding_dataset_advanced.jsonl`. `data/coding_skills.json` contains the structured skill profile and additional training conversations. Each JSONL line is a chat object similar to:

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

The main training parameters are defined near the top of `coder79/train.py`:

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
- Output directory: `artifacts/training`
- Seed: `342`

The checked-in datasets provide 31 training conversations: 9 core coding, 6 advanced coding, 3 PHP, and 13 skill-profile examples. The expanded profile emphasizes repository analysis, debugging from evidence, architecture and migrations, reliability, testing, frontend accessibility, and code review. `coder79/train.py` reads these conversations from `data/`. This remains a small dataset; it can overfit and does not guarantee these capabilities. Evaluate on held-out tasks before increasing training duration.

### Applying and training updated skills

The serving model and its system prompt are configured by `write_coder79_modelfile.ps1`, which writes `models/coder79/Modelfile`. The startup batch file regenerates the Modelfile, pulls the 3B base if needed, registers `coder79`, and starts the WebUI and IDE agent. To perform these steps manually:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\write_coder79_modelfile.ps1
docker compose up -d --build ollama-webui coder79-agent
```

The 3B base is a speed-focused compromise for CPU-only local use; it is smaller and generally responds faster than the previously configured 7B model, though with less capacity. The optional `python -m coder79.train` pipeline still fine-tunes the 1.5B base; its output is not automatically used by the default Coder79 service.

## How the Script Works

The script follows a straightforward workflow:

1. `FastLanguageModel.from_pretrained(...)` loads the base model and tokenizer.
2. `FastLanguageModel.get_peft_model(...)` wraps the model in LoRA adapters.
3. `get_chat_template(...)` applies the Qwen chat template.
4. `dataset.map(...)` converts conversations from the JSONL files and `data/coding_skills.json` into plain text prompt strings.
5. `SFTTrainer(...)` executes supervised fine-tuning.
6. `model.save_pretrained_gguf(...)` exports the trained model to a GGUF artifact.

At the end of training, the program prints:

```text
Training complete! GGUF model successfully saved to <project directory>\models\coder79
```

Training data and export paths are resolved relative to the project root, so training can be launched from the repository root with `python -m coder79.train`. The trainer creates a new timestamped output folder under `artifacts/training/` for each run. To restore a completed or interrupted run, pass its checkpoint directory, for example:

```powershell
python -m coder79.train --resume-from-checkpoint artifacts\training\skills-run-<timestamp>\checkpoint-60
```

## Exporting the Model

The script exports the fine-tuned model to:

```text
./models/coder79
```

This folder is also mounted into the container defined in `docker-compose.yml`. The `Modelfile` in this folder tells Ollama which GGUF artifact to load.

## Run the Model with Docker

The exported GGUF file and `Modelfile` are mounted read-only into the Ollama container by `docker-compose.yml`; Ollama's writable runtime data is kept separately in `ollama_data/`. Compose health checks wait for Ollama before starting the WebUI or registering the model. In PowerShell, from the project root, start the stack and register the model:

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

Open WebUI at `http://localhost:3000`, sign in, and select the `coder79` model. Open WebUI routes model requests through the Coder79 agent, which calls Ollama and provides its read-only workspace/web tools; it does not connect directly to Ollama. The workspace is the path in `WORKSPACE_PATH` (the repository root by default). Open WebUI application data is persisted in the `open-webui` Docker volume. Ollama, Open WebUI, and the IDE endpoint publish ports on loopback only (`127.0.0.1`) by default, so they are not exposed to other network devices.

### Use Coder79 from an IDE

The local IDE server connects an IDE to Ollama and scopes file access to one project folder you select. It uses Python's standard library; no extra packages are needed. Start Docker/Ollama first. In addition to workspace tools, Coder79 can fetch a public HTTP(S) URL to inspect static page text, headings, links, and common page landmarks. It does not search the web, sign in, execute JavaScript, or access local/private network destinations. Fetches use standard web ports, validate each redirect, and enforce timeout and response-size limits. This feature requires outbound network access and a trusted TLS certificate chain from the agent process/container. If your environment uses a private certificate authority, set `SSL_CERT_FILE` to its trusted CA bundle; certificate verification is never disabled.

#### Configure Continue for Plan and Agent modes

The project includes [`.continue/config.yaml`](./.continue/config.yaml), an OpenAI-compatible Continue model configuration pointing to the same agent endpoint used by Open WebUI. Continue loads its YAML configuration from `%USERPROFILE%\.continue\config.yaml` on Windows. Copy the template there if you do not have a config yet; if one already exists, merge the model entry from the template into its `models` list instead of replacing your existing settings. The model uses the `chat` role and advertises `tool_use`, which Continue requires to make Plan and Agent modes available:

1. Install the Continue VS Code extension and open this project folder.
2. Start the server with the **Start Coder79 for this workspace** task described below. Keep its terminal running.
3. Open `%USERPROFILE%\.continue\config.yaml` from Continue's configuration selector. Create it from this template or merge the Coder79 model entry, then reload the configuration.
4. Choose **Coder79** as the model and switch between Plan and Agent modes.
5. To enable inline completions, select **Qwen2.5-Coder 1.5B Autocomplete** as Continue's autocomplete model and make sure **Enable Tab Autocomplete** and VS Code's `editor.inlineSuggest.enabled` setting are on.

Plan and Agent use Coder79's built-in, read-only workspace and public-page tools. Continue-provided editing, file-writing, and terminal tools are not executed; ask Coder79 for recommended changes rather than expecting it to modify files. Inline completions use the separate local Qwen2.5-Coder 1.5B model served by Ollama. Docker downloads this model along with the Coder79 base model during startup. The project is mounted read-only in Docker.

#### Configure VS Code to use Coder79

This repository includes a VS Code task and a copy-ready custom model entry:

- `.vscode/tasks.json` builds and starts the Docker Coder79 agent, scoped to the currently opened workspace.
- `.vscode/coder79-chatLanguageModels.json` is a template for registering the model in your VS Code user profile.

1. Open this `customModel` folder in VS Code and make sure Docker Desktop is running.

2. Press `Ctrl+Shift+P`, run **Tasks: Run Task**, then select **Start Coder79 for this workspace**. Docker starts Ollama, registers `coder79`, builds the agent image, mounts the selected workspace read-only, and connects the agent to Ollama over the Compose network. Leave the task terminal running; run `docker compose stop coder79-agent` to stop the IDE agent.

3. Run **Chat: Manage Language Models** from the Command Palette, select **Add Models** → **Custom Endpoint**, and choose **Chat Completions**. When VS Code opens its `chatLanguageModels.json` user-profile file, copy the contents of `.vscode/coder79-chatLanguageModels.json` into that file and save. If VS Code already has providers in the array, add the `Coder79 Local` object rather than replacing them.

4. Open the Chat view, select **Coder79** from the model picker, and use Chat/Ask mode.

VS Code stores model providers in your user profile, not in this project, so importing the template is a one-time per-profile setup. The included task mounts the currently opened folder into the agent container. For another project, run Docker Compose from this `customModel` folder and set `WORKSPACE_PATH` to the project folder first:

```powershell
$env:WORKSPACE_PATH = (Resolve-Path "C:\path\to\your\project").Path
docker compose up -d --build coder79-agent
```

The endpoint supports Chat Completions and streamed responses. Coder79 gets the mounted workspace tree and has internal read-only search/read tools. The VS Code custom model template is for Chat/Ask; use the Continue configuration above for its Plan and read-only Agent modes. The agent container listens on its private Docker interface, but the host port is bound to `127.0.0.1`; do not change it to a public interface.

#### MCP workspace tools

Use this when the IDE supports MCP servers. MCP makes the read-only project tools available to that IDE's selected assistant model; by itself, it does not switch that assistant to Coder79. Add this server to the IDE's MCP configuration (for example, `.vscode/mcp.json` in the project you want to open), replacing the script path with this repository's actual location:

```json
{
  "servers": {
    "coder79": {
      "type": "stdio",
      "command": "python",
      "args": [
        "C:\\path\\to\\customModel\\coder79\\ide_server.py",
        "--transport",
        "mcp",
        "--workspace",
        "${workspaceFolder}"
      ]
    }
  }
}
```

Restart or reconnect MCP servers after changing the configuration. The MCP server reads files only under the configured workspace and also offers the bounded public-page `fetch_url` tool. Both methods provide tree listing, text search, and bounded file reads; common generated folders and likely credential files are excluded. Direct `ollama run` chat does not provide these agent tools. IDEs differ, so use MCP where supported, or the OpenAI-compatible endpoint where custom model providers are supported.

For terminal chat without an IDE, the simpler interactive client is still available:

```powershell
python -m coder79.agent "C:\path\to\your\project"
```

It uses the `coder79` model by default. Type `/exit` to end the chat; use `--model` or `--ollama-url` to select another model or Ollama server.

The Ollama model is stored in `ollama_data`, so you only need to run `ollama create` again if that data is removed or the GGUF file changes. Stop all services while preserving model and UI data with:

```powershell
docker compose down
```

### Recreate or remove the Ollama model

After changing the GGUF file or settings in `models/coder79/Modelfile`, recreate the registered model:

```powershell
docker compose exec ollama ollama rm coder79
docker compose exec --workdir /app/model ollama ollama create coder79 -f Modelfile
```

To remove the model from Ollama without deleting the source GGUF file or `Modelfile`:

```powershell
docker compose exec ollama ollama rm coder79
```

This removes the registered model from Ollama's `ollama_data`; the source files in `models/coder79/` remain available to register it again.

The service uses resource limits configured in `docker-compose.yml`:

- CPU cap: 6 cores
- Memory limit: 16 GB
- Reserved memory: 8 GB

## Notes and Limitations

- The script is configured for CPU training and sets `LOAD_IN_4bit = False` for compatibility with CPU execution.
- Fine-tuning is intentionally lightweight and may not produce production-quality results without a larger dataset or longer training run.
- Updating the skill profile or datasets does not change model weights until `python -m coder79.train` completes training and exports the GGUF. CPU training is supported by the installed Unsloth version but is substantially slower than GPU training.
- The model export is done in GGUF format for compatibility with local inference tools such as Ollama.
- Running the full training pipeline requires substantial memory and a compatible ML environment.
- The Ollama model is a text-generation model, not an operating-system agent. The local terminal and IDE integrations provide read-only access to one explicitly selected workspace and a public static-page fetch tool; they cannot edit or run project files, search the web, or access private/local network URLs.
- The training examples and model prompt request concise explanations, not hidden chain-of-thought or `<think>` output.

## Suggested Next Steps

- Add more high-quality training examples to `data/coding_dataset.jsonl` and run the training/export pipeline on a compatible machine.
- Tune the LoRA hyperparameters for better output quality.
- Increase the number of training steps for stronger convergence.
- Test the exported model in Ollama and evaluate the generated code quality.
- Add validation and evaluation scripts for automatic quality checks.

## Quick Start

On Windows, start Docker Desktop and double-click `start_and_test_coder79.bat`. It builds the agent image, starts the services, waits for Ollama, then runs an inference check. Double-click `stop_coder79.bat` to stop all Compose services without deleting the model, Ollama data, or WebUI data.

Run the already-exported model from Docker:

```powershell
docker compose up -d
docker compose exec --workdir /app/model ollama ollama create coder79 -f Modelfile
docker compose exec -it ollama ollama run coder79
```

To train/export a new model first, install the training dependencies and run `python -m coder79.train`; then repeat the Docker commands above.

## Summary

This project is a compact local fine-tuning workflow for a coding-focused language model. It trains on a JSONL conversation dataset, uses parameter-efficient fine-tuning, and exports the resulting model to a GGUF format for local serving with Ollama.
