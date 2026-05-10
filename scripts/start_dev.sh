#!/bin/bash
# Quick start for development (not production)
# Starts all services locally without systemd

set -euo pipefail

echo "Starting JARVIS dev environment..."

# Load .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo "Warning: .env not found, using .env.example defaults"
  export $(grep -v '^#' .env.example | xargs 2>/dev/null || true)
fi

export JARVIS_HOME="${JARVIS_HOME:-$HOME/jarvis}"
export PYTHONPATH="$(pwd)/src"

# Start Ollama if installed
if command -v ollama &>/dev/null; then
  echo "Starting Ollama..."
  ollama serve &>/dev/null &
  sleep 2
fi

# Start Docker sandbox
docker-compose up -d jarvis-sandbox 2>/dev/null || echo "Docker not available, sandbox disabled"

# Start API server
echo "Starting JARVIS API on port ${JARVIS_PORT:-8080}..."
python -m uvicorn jarvis.api.main:app \
  --host "${JARVIS_HOST:-0.0.0.0}" \
  --port "${JARVIS_PORT:-8080}" \
  --reload \
  --log-level info &

API_PID=$!

# Start Telegram bot if configured
if [ -n "${TELEGRAM_BOT_TOKEN:-}" ]; then
  echo "Starting Telegram bot..."
  python -c "import asyncio; from jarvis.api.telegram_bot import start_telegram_bot; asyncio.run(start_telegram_bot())" &
  TG_PID=$!
fi

echo ""
echo "JARVIS running:"
echo "  API:       http://localhost:${JARVIS_PORT:-8080}"
echo "  Dashboard: http://localhost:${JARVIS_PORT:-8080}/dashboard"
echo "  Docs:      http://localhost:${JARVIS_PORT:-8080}/docs"
echo ""
echo "Press Ctrl+C to stop"

# Wait for Ctrl+C
trap 'echo "Stopping..."; kill $API_PID ${TG_PID:-} 2>/dev/null; exit 0' INT TERM
wait $API_PID
