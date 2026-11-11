@echo off
setlocal
cd /d "%~dp0"

echo Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Desktop is not running or Docker is unavailable.
    echo Start Docker Desktop, wait for it to be ready, then run this file again.
    exit /b 1
)

set "GGUF_FILE=coder79_slm_model\coder79_slm_model.Q4_K_M.gguf"
set "TRAINING_DATA_MISSING="
for %%F in (main.py coding_dataset.jsonl coding_dataset_php.jsonl coding_dataset_advanced.jsonl coding_skills.json) do (
    if not exist "%%F" (
        echo ERROR: Required training file was not found: %%F
        set "TRAINING_DATA_MISSING=1"
    )
)
if defined TRAINING_DATA_MISSING exit /b 1

set "MODEL_WAS_TRAINED="
if not exist "%GGUF_FILE%" (
    echo GGUF model was not found. Preparing model folder and running training...
    if not exist "coder79_slm_model" mkdir "coder79_slm_model"
    where python >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python was not found on PATH. Install Python and training dependencies.
        exit /b 1
    )
    python main.py
    if errorlevel 1 (
        echo ERROR: python main.py failed. The Docker model will not be started.
        exit /b 1
    )
    if not exist "%GGUF_FILE%" (
        echo ERROR: Training finished without creating "%GGUF_FILE%".
        exit /b 1
    )
    set "MODEL_WAS_TRAINED=1"
)

echo Starting the coder79model container...
docker compose up -d ollama ollama-webui
if errorlevel 1 (
    echo ERROR: Could not start the Ollama Docker services.
    exit /b 1
)

echo Waiting for Ollama to become ready...
set "OLLAMA_READY="
for /L %%I in (1,1,30) do (
    docker compose exec -T ollama ollama list >nul 2>&1
    if not errorlevel 1 (
        set "OLLAMA_READY=1"
        goto :ollama_ready
    )
    timeout /t 2 /nobreak >nul
)

echo ERROR: Ollama did not become ready within 60 seconds.
docker compose logs --tail=40 ollama
exit /b 1

:ollama_ready
if not defined MODEL_WAS_TRAINED (
    docker compose exec -T ollama ollama list | findstr /B /C:"coder79:latest" >nul
    if not errorlevel 1 (
        echo Found existing coder79 model. Skipping model creation.
        goto :test_model
    )
)

if not exist "coder79_slm_model\Modelfile" (
    echo Modelfile was not found. Preparing it from the Coder79 skills prompt...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0write_coder79_modelfile.ps1"
    if errorlevel 1 (
        echo ERROR: Could not write the Coder79 Modelfile.
        exit /b 1
    )
)

if defined MODEL_WAS_TRAINED (
    echo Training completed. Registering the newly trained coder79 model...
    goto :create_model
)

:create_model
echo Registering the coder79 model...
docker compose exec -T --workdir /app/model ollama ollama create coder79 -f Modelfile
if errorlevel 1 (
    echo ERROR: Could not register the coder79 model.
    exit /b 1
)

:test_model
echo Running a quick inference test...
docker compose exec -T ollama ollama run coder79 "Reply with exactly: Coder79 test passed."
if errorlevel 1 (
    echo ERROR: The coder79 inference test failed.
    exit /b 1
)

echo.
echo Coder79 is ready. To start an interactive chat, run:
echo docker compose exec -it ollama ollama run coder79
echo Open WebUI is available at http://localhost:3000
exit /b 0
