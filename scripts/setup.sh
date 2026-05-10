#!/bin/bash
# JARVIS v6.0 — Complete setup script
# Tested on Ubuntu 22.04 LTS (Hetzner CX32 or Oracle Cloud Free Tier)
#
# Usage:
#   chmod +x scripts/setup.sh
#   sudo ./scripts/setup.sh [--with-ollama] [--with-lab]

set -euo pipefail

JARVIS_USER="${JARVIS_USER:-jarvis}"
JARVIS_HOME="${JARVIS_HOME:-/home/$JARVIS_USER}"
PYTHON_VER="${PYTHON_VER:-3.11}"
WITH_OLLAMA=false
WITH_LAB=false

for arg in "$@"; do
  case $arg in
    --with-ollama) WITH_OLLAMA=true ;;
    --with-lab) WITH_LAB=true ;;
  esac
done

info() { echo -e "\033[0;36m[SETUP]\033[0m $*"; }
ok()   { echo -e "\033[0;32m[OK]\033[0m $*"; }
warn() { echo -e "\033[0;33m[WARN]\033[0m $*"; }
err()  { echo -e "\033[0;31m[ERROR]\033[0m $*"; exit 1; }

# ─── System requirements ──────────────────────────────────────────────────────
info "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
  python${PYTHON_VER} python${PYTHON_VER}-venv python${PYTHON_VER}-dev \
  python3-pip git curl wget sqlite3 \
  build-essential libssl-dev libffi-dev \
  ffmpeg libsndfile1 portaudio19-dev \
  docker.io docker-compose-v2 \
  caddy logrotate \
  nmap tcpdump net-tools  # lab tools (safe to install, gated by JARVIS_LAB_MODE)

# ─── Create jarvis user ───────────────────────────────────────────────────────
if ! id "$JARVIS_USER" &>/dev/null; then
  info "Creating user $JARVIS_USER..."
  useradd -m -s /bin/bash "$JARVIS_USER"
fi
usermod -aG docker "$JARVIS_USER"

# ─── Create directory structure ───────────────────────────────────────────────
info "Creating directory structure..."
for dir in "$JARVIS_HOME/jarvis/data" "$JARVIS_HOME/jarvis/logs" \
           "$JARVIS_HOME/jarvis/backups" "$JARVIS_HOME/jarvis/workspace" \
           "$JARVIS_HOME/jarvis/lab" "$JARVIS_HOME/jarvis/vault" \
           "$JARVIS_HOME/jarvis/skills" "/etc/jarvis/secrets"; do
  mkdir -p "$dir"
done
chown -R "$JARVIS_USER:$JARVIS_USER" "$JARVIS_HOME/jarvis"
chmod 700 "/etc/jarvis/secrets"
chmod 700 "$JARVIS_HOME/jarvis/vault"

# ─── Python virtual environment ───────────────────────────────────────────────
info "Setting up Python virtual environment..."
sudo -u "$JARVIS_USER" python${PYTHON_VER} -m venv "$JARVIS_HOME/venv"
VENV="$JARVIS_HOME/venv/bin"

# Install core dependencies
info "Installing Python packages..."
"$VENV/pip" install --upgrade pip setuptools wheel
"$VENV/pip" install -r "$(dirname "$0")/../requirements.txt"

# ─── spaCy Greek model ────────────────────────────────────────────────────────
info "Downloading spaCy Greek model..."
"$VENV/python" -m spacy download el_core_news_sm || warn "Greek spaCy model failed — sentence splitting will use fallback"

# ─── Install Ollama (free local LLM) ─────────────────────────────────────────
if [ "$WITH_OLLAMA" = "true" ]; then
  info "Installing Ollama..."
  curl -fsSL https://ollama.ai/install.sh | sh
  
  info "Starting Ollama and downloading models..."
  systemctl start ollama || ollama serve &
  sleep 5
  
  ollama pull llama3.2:3b   || warn "llama3.2:3b download failed"
  ollama pull mistral:7b     || warn "mistral:7b download failed"
  ollama pull nomic-embed-text || warn "nomic-embed-text download failed"
  
  ok "Ollama installed with models"
fi

# ─── Docker sandbox setup ────────────────────────────────────────────────────
info "Setting up Docker sandbox..."
cd "$(dirname "$0")/.."
docker-compose up -d jarvis-sandbox || warn "Docker sandbox setup failed"

if [ "$WITH_LAB" = "true" ]; then
  docker-compose --profile lab up -d jarvis-lab || warn "Lab container setup failed"
  ok "Lab container started"
fi

# ─── systemd service ─────────────────────────────────────────────────────────
info "Installing systemd service..."
cp "$(dirname "$0")/../config/jarvis.service" /etc/systemd/system/
systemctl daemon-reload

# ─── Session secret ───────────────────────────────────────────────────────────
if [ ! -f "/etc/jarvis/session_secret" ]; then
  python${PYTHON_VER} -c "import os; open('/etc/jarvis/session_secret','wb').write(os.urandom(32))"
  chmod 600 /etc/jarvis/session_secret
  ok "Session secret generated"
fi

# ─── logrotate ───────────────────────────────────────────────────────────────
cp "$(dirname "$0")/../config/logrotate.conf" /etc/logrotate.d/jarvis

# ─── Copy source to jarvis home ──────────────────────────────────────────────
info "Installing JARVIS source..."
cp -r "$(dirname "$0")/.." "$JARVIS_HOME/jarvis-src"
chown -R "$JARVIS_USER:$JARVIS_USER" "$JARVIS_HOME/jarvis-src"

# ─── Copy .env ───────────────────────────────────────────────────────────────
if [ ! -f "$JARVIS_HOME/.env" ]; then
  cp "$JARVIS_HOME/jarvis-src/.env.example" "$JARVIS_HOME/.env"
  chmod 600 "$JARVIS_HOME/.env"
  warn "Created $JARVIS_HOME/.env — EDIT IT before starting!"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
ok "JARVIS v6.0 setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit $JARVIS_HOME/.env (at minimum: TELEGRAM_BOT_TOKEN, TELEGRAM_USER_ID)"
echo "  2. Edit CLAUDE.md to personalize your assistant"
echo "  3. sudo systemctl start jarvis && sudo systemctl enable jarvis"
echo "  4. Access dashboard: http://localhost:8080 (or https://your-domain)"
echo ""
echo "Free stack active by default:"
echo "  LLM:  Ollama (local) — set ANTHROPIC_API_KEY for Claude"
echo "  STT:  Whisper (local) — set DEEPGRAM_API_KEY for faster"
echo "  TTS:  edge-tts (free online) — set ELEVENLABS_API_KEY for higher quality"
echo "  Cost: €0/month (only VPS hosting cost)"
