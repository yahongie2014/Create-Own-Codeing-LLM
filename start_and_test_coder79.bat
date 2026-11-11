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

if not exist "coder79_slm_model\coder79_slm_model.Q4_K_M.gguf" (
    echo ERROR: The Coder79 GGUF model file was not found.
    echo Expected: "%~dp0coder79_slm_model\coder79_slm_model.Q4_K_M.gguf"
    exit /b 1
)

if not exist "coder79_slm_model\Modelfile" (
    echo ERROR: The Ollama Modelfile was not found.
    echo Expected: "%~dp0coder79_slm_model\Modelfile"
    exit /b 1
)

echo Starting the coder79model container...
docker compose up -d ollama
if errorlevel 1 (
    echo ERROR: Could not start the Ollama Docker service.
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
echo Registering or updating the coder79 model...
docker compose exec -T --workdir /app/model ollama ollama create coder79 -f Modelfile
if errorlevel 1 (
    echo ERROR: Could not register the coder79 model.
    exit /b 1
)

echo Running a quick inference test...
docker compose exec -T ollama ollama run coder79 "Reply with exactly: Coder79 test passed."
if errorlevel 1 (
    echo ERROR: The coder79 inference test failed.
    exit /b 1
)

echo.
echo Coder79 is ready. To start an interactive chat, run:
echo docker compose exec -it ollama ollama run coder79
exit /b 0
