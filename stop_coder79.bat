@echo off
setlocal
cd /d "%~dp0"

echo Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Desktop is not running or Docker is unavailable.
    exit /b 1
)

echo Stopping the Coder79 Docker services...
docker compose stop
if errorlevel 1 (
    echo ERROR: Could not stop the Docker services.
    exit /b 1
)

echo Coder79 is stopped. The model, Ollama data, and WebUI data have been preserved.
exit /b 0
