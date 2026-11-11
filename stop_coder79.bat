@echo off
setlocal
cd /d "%~dp0"

echo Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Desktop is not running or Docker is unavailable.
    exit /b 1
)

echo Stopping the Coder79 Ollama container...
docker compose stop ollama
if errorlevel 1 (
    echo ERROR: Could not stop the Ollama service.
    exit /b 1
)

echo Coder79 is stopped. The model and Ollama data have been preserved.
exit /b 0
