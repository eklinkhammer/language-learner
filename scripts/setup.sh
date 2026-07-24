#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Language Learner Setup ==="

# Backend
echo ""
echo "--- Installing Python dependencies (uv) ---"
cd "$ROOT_DIR/backend"
uv sync --all-extras
echo "Backend dependencies installed."

# Frontend
echo ""
echo "--- Installing Node dependencies ---"
cd "$ROOT_DIR/frontend"
npm install
echo "Frontend dependencies installed."

echo ""
echo "=== Setup complete ==="
echo "Run scripts/dev.sh to start the development servers."
