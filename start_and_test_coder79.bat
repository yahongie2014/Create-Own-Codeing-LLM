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

echo Preparing the Coder79 model instructions...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0write_coder79_modelfile.ps1"
if errorlevel 1 (
    echo ERROR: Could not write the Coder79 Modelfile.
    exit /b 1
)

echo Building and starting Ollama, the WebUI, and the Coder79 agent...
docker compose up -d --build ollama ollama-webui coder79-agent
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
:test_model
echo Running a quick inference test...
docker compose exec -T ollama ollama run coder79 "Reply with exactly: Coder79 test passed."
if errorlevel 1 (
    echo ERROR: The coder79 inference test failed.
    exit /b 1
)

echo.
echo Coder79 is ready. For web-enabled chat, open:
echo http://localhost:3000
echo Or run the local agent client: python -m coder79.agent "%CD%"
echo Direct "ollama run coder79" chat does not have web tools.
echo The OpenAI-compatible IDE endpoint is available at http://127.0.0.1:8000/v1
exit /b 0
