# Custom Model Training Project

This project serves a code-focused model through Ollama and provides an optional from-scratch CPU training pipeline. The default Coder79 service uses Qwen2.5-Coder 3B with the project's system instructions; local training is a separate workflow that does not use a pretrained model. Coder79's terminal and IDE agent integrations can search the public web, fetch public static web pages, and retrieve relevant AI research skill guides.

## Repository Layout

- `coder79/` — Python package for the workspace agent, IDE server, and training entry point.
- `tests/` — unit tests for the workspace tools and IDE API.
- `data/` — training conversations and skill profile.
- `data/AI-Research-SKILLs/` — MIT-licensed research skill guides loaded on demand by the agent.
- `models/coder79/` — Ollama `Modelfile`, scratch-model GGUF export, and scratch-model import template.
- `artifacts/training/` — generated from-scratch models and checkpoints (ignored by Git).
- `ollama_data/` — persistent Ollama runtime state (ignored by Git; do not share).
- `.continue/` and `.vscode/` — Continue and VS Code workspace configuration.
- Root files — Docker/Compose and Windows start/stop helpers.

## Overview

The project uses:

- Python
- PyTorch
- Hugging Face Transformers
- Tokenizers for learning a tokenizer from the local training data
- Ollama for local model serving

The default serving flow pulls `qwen2.5-coder:3b` from Ollama, then applies Coder79's system instructions and serves it under the `coder79` name. The optional training flow in `coder79/train.py` remains separate:

1. Learns a SentencePiece BPE tokenizer from the coding conversations and local skill guides.
2. Initializes a small Llama-style causal transformer with random weights.
3. Loads the coding datasets and all local AI research skill guides.
4. Formats conversations with the locally trained tokenizer and splits long sequences without dropping their content.
5. Trains the complete model using system RAM.
6. Saves a standalone model and tokenizer under `artifacts/training/`, then converts it to an F16 GGUF at `models/coder79/coder79-scratch-f16.gguf`. This output does not replace the default 3B serving model.

## Prerequisites

To run Coder79, install and start Docker Desktop and make sure it is running. Internet access is needed on first startup to download the 3B model. You do not need Python or the training packages for standard Ollama chat. Python 3.10+ is needed to run the local file-navigation agent; it uses only the standard library. Training packages are needed only for the separate optional from-scratch workflow:

```powershell
python -m pip install -r requirements-training.txt
python -m coder79.train --llama-cpp-dir C:\tools\llama.cpp
```

Training runs on the host and uses the CPU and system RAM only. It does not download or load a pretrained model. GGUF export uses the local `convert_hf_to_gguf.py` script from llama.cpp. CPU training can be slow; the resulting model is intentionally small and experimental.

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

The script uses a tokenizer trained from the included conversations and applies a local chat template before training.

### On-demand AI research skills

The agent can search the MIT-licensed AI Research Skills library by Orchestra
Research (98 Markdown guides under `data/AI-Research-SKILLs/`) and load only
relevant guides when a task involves AI/ML research, model
architecture, training, evaluation, inference serving, RAG, agents, or paper
writing. It does not stuff the entire library into every prompt or silently
change model weights. Skill files are technical references; their instructions
do not override the user's request or the agent's tool and safety limits.

The Docker agent mounts this directory read-only. When running an agent server
or terminal client outside Docker, the default is the same directory in this
project; use `--skills <folder>` to select a different local library. Preserve
the bundled library's `LICENSE` when redistributing it.

## Training Configuration

The main training parameters are defined near the top of `coder79/train.py`:

```python
MAX_SEQ_LENGTH = 512
TOKENIZER_VOCAB_SIZE = 8192
HIDDEN_SIZE = 256
NUM_LAYERS = 4
NUM_ATTENTION_HEADS = 4
```

Key training settings include:

- Model: randomly initialized Llama-style causal transformer
- Device and weights: CPU only, `torch.float32`
- Training duration: one epoch by default (`--epochs` can override it)
- Batch size: `per_device_train_batch_size=1`
- Gradient checkpointing: enabled to reduce activation memory
- Learning rate: `5e-4`
- Output directory: `artifacts/training`
- Seed: `342`

Training uses the coding/profile datasets and local skill guides. Long conversations are split into sequences without dropping their remaining tokens. The result is a complete, standalone model and an F16 GGUF file; training does not alter the separate Ollama model.

Install the CPU training requirements, clone llama.cpp, install its converter requirements, and start training from the repository root:

```powershell
python -m pip install -r requirements-training.txt
git clone --depth 1 https://github.com/ggml-org/llama.cpp C:\tools\llama.cpp
python -m pip install -r C:\tools\llama.cpp\requirements\requirements-convert_hf_to_gguf.txt
python -m coder79.train --llama-cpp-dir C:\tools\llama.cpp
```

The GGUF export is saved to `models\coder79\coder79-scratch-f16.gguf`; `LLAMA_CPP_DIR` can be set instead of passing `--llama-cpp-dir`. Training uses only local data and does not download a base model. The randomly initialized model is deliberately small; the included dataset is not enough to match a pretrained coding model, and training can still overfit. Evaluate on held-out tasks before increasing training duration.

### Applying and training updated skills

The serving model and its system prompt are configured by `write_coder79_modelfile.ps1`, which writes `models/coder79/Modelfile`. The startup batch file regenerates the Modelfile, pulls the 3B base if needed, registers `coder79`, and starts the WebUI and IDE agent. To perform these steps manually:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\write_coder79_modelfile.ps1
docker compose up -d --build ollama-webui coder79-agent
```

The 3B base is a speed-focused compromise for CPU-only local use; it is smaller and generally responds faster than the previously configured 7B model, though with less capacity. The optional `python -m coder79.train` pipeline trains a separate model from random initialization and does not replace the default Coder79 service.

The generated Ollama `Modelfile` keeps the serving system prompt compact so CPU inference spends less time processing repeated instructions. After changing it, regenerate the file and recreate the `coder79` model as described above.

## How the Script Works

The CPU training script follows this workflow:

1. The trainer combines the JSONL datasets, `data/coding_skills.json`, and every `SKILL.md` under `data/AI-Research-SKILLs`.
2. A SentencePiece BPE tokenizer is trained on the local conversation text.
3. `LlamaForCausalLM` is initialized from a new configuration, with no pretrained weights.
4. Conversations are tokenized, and long examples are split into 512-token sequences without truncation.
5. Hugging Face `Trainer` trains all model weights on CPU.
6. The complete model and tokenizer are saved under the timestamped training run.
7. llama.cpp converts that local model to an F16 GGUF in `models/coder79/`.

At the end of training, the program prints the standalone model path. It can be resumed from a saved Trainer checkpoint, for example:

```powershell
python -m coder79.train --llama-cpp-dir C:\tools\llama.cpp --resume-from-checkpoint artifacts\training\cpu-scratch-run-<timestamp>\checkpoint-100
```

## Using the CPU-trained model

The training command saves a complete Transformers model and tokenizer. Load them without a separate base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = r"artifacts\training\cpu-scratch-run-<timestamp>\final-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
```

The training command also exports `models\coder79\coder79-scratch-f16.gguf`. Open this file in LM Studio, or import it into Ollama using the included scratch Modelfile:

```powershell
docker compose exec ollama ollama create coder79-scratch -f /app/model/Modelfile-scratch
docker compose exec ollama ollama run coder79-scratch
```

The scratch model trained with 512-token sequences. Its Ollama Modelfile sets a 4096-token runtime context to leave more room for prompts; this extends beyond its training context and may reduce quality on long inputs. Re-run `ollama create` after changing the Modelfile to reload the model.

This model does not replace the default Ollama model.

## Run the Model with Docker

The exported GGUF file and `Modelfile` are mounted read-only into the Ollama container by `docker-compose.yml`; Ollama's writable runtime data is kept separately in `ollama_data/`. Compose health checks wait for Ollama before starting the WebUI or registering the model. In PowerShell, from the project root, start the stack and register the model:

```powershell
docker compose up -d
docker compose exec --workdir /app/model ollama ollama create coder79 -f Modelfile
```

For an interactive chat that can inspect public URLs, use Open WebUI at
`http://localhost:3000` or the local agent client described below. Running
`ollama run coder79` directly talks to the text model without its web-fetch
tools.

To discover websites from a topic in Open WebUI, select `coder79` and ask it
to search the web for the topic. It can then fetch relevant public result
pages. To inspect a known URL directly, ask it to fetch that URL, for example:

> Fetch https://example.com/ and summarize its purpose, key points, and any
> limitations you can infer from the page.

Web search uses Bing's public HTML results and requires outbound
network access. URL fetching supports public HTTP(S) pages served as static
HTML, XHTML, or plain text; the agent does not run JavaScript, access
sign-in-only content, or fetch PDFs. Only the content returned by a page
(capped at 12,000 characters for model context) can be analyzed.

Direct model chat remains available for text-only use:

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

The local IDE server connects an IDE to Ollama and scopes file access to one project folder you select. It uses Python's standard library; no extra packages are needed. Start Docker/Ollama first. In addition to workspace tools, Coder79 can search the public web with Bing and fetch public HTTP(S) URLs to inspect static page text, headings, links, and common page landmarks. It cannot sign in, execute JavaScript, or access local/private network destinations. Requests use standard web ports, validate each redirect, and enforce timeout and response-size limits. These features require outbound network access and a trusted TLS certificate chain from the agent process/container. If your environment uses a private certificate authority, set `SSL_CERT_FILE` to its trusted CA bundle; certificate verification is never disabled.

#### Configure Continue for Plan and Agent modes

The project includes [`.continue/config.yaml`](./.continue/config.yaml), an OpenAI-compatible Continue model configuration pointing to the same agent endpoint used by Open WebUI. Continue loads its YAML configuration from `%USERPROFILE%\.continue\config.yaml` on Windows. Copy the template there if you do not have a config yet; if one already exists, merge the model entry from the template into its `models` list instead of replacing your existing settings. The model uses the `chat` role and advertises `tool_use`. The endpoint accepts Continue function-tool schemas and returns OpenAI-compatible tool calls for Continue to execute:

1. Install the Continue VS Code extension and open this project folder.
2. Start the server with the **Start Coder79 for this workspace** task described below. Keep its terminal running.
3. Open `%USERPROFILE%\.continue\config.yaml` from Continue's configuration selector. Create it from this template or merge the Coder79 model entry, then reload the configuration.
4. Choose **Coder79** as the model and switch between Plan and Agent modes. Coder79 is configured for chat, edit, apply, and rerank. Local LLM reranking sends additional requests to the 3B model and may be slow on CPU. Continue's bundled `transformers.js` model provides local embeddings for codebase search; it does not use Coder79's chat endpoint.
5. To enable inline completions, select **Qwen2.5-Coder 1.5B Autocomplete** as Continue's autocomplete model and make sure **Enable Tab Autocomplete** and VS Code's `editor.inlineSuggest.enabled` setting are on.

Open the project you want to review in VS Code. Use Continue's Plan mode for read-only analysis and Agent mode for browsing files, editing, or running commands; normal Chat mode may not provide workspace tools. If Coder79 says it cannot see the project, verify the Coder79 config is loaded, the agent task is running, and Continue is in Agent mode. Ask it to list the project files as a quick tool-use check. Agent-mode tools run on the host and may edit files or run commands; review tool actions and workspace changes. The server's mounted workspace can differ from the open VS Code project, so Continue's IDE tools take precedence. Inline completions use the separate local Qwen2.5-Coder 1.5B model served by Ollama. Docker downloads this model along with the Coder79 base model during startup.

#### Configure VS Code to use Coder79

This repository includes a VS Code task and a copy-ready custom model entry:

- `.vscode/tasks.json` builds and starts the Docker Coder79 agent, scoped to the currently opened workspace.
- `.vscode/coder79-chatLanguageModels.json` is a template for registering the model in your VS Code user profile.

1. Open this `customModel` folder in VS Code and make sure Docker Desktop is running.

2. Press `Ctrl+Shift+P`, run **Tasks: Run Task**, then select **Start Coder79 for this workspace**. Docker starts Ollama, registers `coder79`, builds the agent image, mounts the selected workspace read-only, and connects the agent to Ollama over the Compose network. Leave the task terminal running; run `docker compose stop coder79-agent` to stop the IDE agent.

3. Run **Chat: Manage Language Models** from the Command Palette, select **Add Models** → **Custom Endpoint**, and choose **Chat Completions**. When VS Code opens its `chatLanguageModels.json` user-profile file, copy the contents of `.vscode/coder79-chatLanguageModels.json` into that file and save. If VS Code already has providers in the array, add the `Coder79 Local` object rather than replacing them.

4. Open the Chat view, select **Coder79** from the model picker, and use Chat/Ask mode. The custom model entry enables tool calling so Coder79 can use VS Code's workspace tools to inspect files in the currently open folder.

VS Code stores model providers in your user profile, not in this project, so importing the template is a one-time per-profile setup. The included task mounts the currently opened folder into the agent container. For another project, run Docker Compose from this `customModel` folder and set `WORKSPACE_PATH` to the project folder first:

```powershell
$env:WORKSPACE_PATH = (Resolve-Path "C:\path\to\your\project").Path
docker compose up -d --build coder79-agent
```

The endpoint supports Chat Completions and streamed responses. Open WebUI requests use the agent's internal read-only workspace tools, scoped to the mounted workspace. VS Code Chat/Ask requests use VS Code's workspace tools because the custom model template enables tool calling; use the Continue configuration above for Plan and Agent modes. Direct `ollama run coder79` chat does not provide file or folder access. The agent container listens on its private Docker interface, but the host port is bound to `127.0.0.1`; do not change it to a public interface.

Continue's configured `maxTokens` limit is forwarded to Ollama so unexpectedly long generations are capped instead of running past the editor's requested limit. Larger prompts and longer generated edits still take longer on CPU; select only the code needed for Edit and start a fresh chat when old context is no longer needed.

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

- The training pipeline is CPU-only and initializes every model weight from scratch; it does not require or download a base model.
- The small random-initialized model and included dataset are for experimentation, not production-quality coding assistance.
- Updating the skill profile or datasets does not change model weights until `python -m coder79.train` completes training.
- Training requires RAM for the model, optimizer, and activations; CPU training is substantially slower than GPU training.
- The training script saves a complete Transformers model. GGUF conversion for Ollama or LM Studio is a separate step.
- The selected Python environment must support the CPU PyTorch and Hugging Face packages in `requirements-training.txt`.
- The Ollama model is a text-generation model, not an operating-system agent. The local terminal and IDE integrations provide read-only access to one explicitly selected workspace, public web search, and a static-page fetch tool; they cannot edit or run project files or access private/local network URLs.
- The training examples and model prompt request concise explanations, not hidden chain-of-thought or `<think>` output.

## Suggested Next Steps

- Add more high-quality training examples to `data/coding_dataset.jsonl` and run the training/export pipeline on a compatible machine.
- Add more high-quality training examples and held-out evaluation tasks.
- Increase the model size or training duration only when you have enough compute and data to support them.
- Test the exported model in Ollama and evaluate the generated code quality.
- Add validation and evaluation scripts for automatic quality checks.

## Quick Start

On Windows, start Docker Desktop and double-click `start_and_test_coder79.bat`. It builds the agent image, starts the services, waits for Ollama, then runs an inference check. Double-click `stop_coder79.bat` to stop all Compose services without deleting the model, Ollama data, or WebUI data.

For web-enabled chat after startup, open `http://localhost:3000` and select
`coder79`, or run the local agent client as described above. To chat directly
with Ollama without workspace or web tools:

```powershell
docker compose up -d
docker compose exec --workdir /app/model ollama ollama create coder79 -f Modelfile
docker compose exec -it ollama ollama run coder79
```

CPU training saves a standalone Transformers model, not an Ollama-ready GGUF model. Convert it separately before registering it with Ollama.

## Summary

This project includes a compact CPU workflow that trains a small language model from random initialization on the coding dataset and local skill guides. It does not depend on pretrained model weights; its limited model size and training data make it experimental rather than a replacement for the default Ollama model.
