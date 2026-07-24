#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cleanup() {
    echo ""
    echo "Shutting down..."
    kill 0 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT

echo "=== Starting Language Learner Dev Servers ==="

# espeak-ng library for phonemizer (macOS Homebrew)
if [ -f "/opt/homebrew/lib/libespeak-ng.dylib" ]; then
    export PHONEMIZER_ESPEAK_LIBRARY="/opt/homebrew/lib/libespeak-ng.dylib"
fi

# Backend
echo "Starting backend on http://localhost:8000 ..."
cd "$ROOT_DIR/backend"
uv run uvicorn app.main:app --reload --port 8000 &

# Frontend
echo "Starting frontend on http://localhost:5173 ..."
cd "$ROOT_DIR/frontend"
npm run dev &

wait
