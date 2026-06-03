#!/usr/bin/env bash
# One-time setup: venv, dashboard deps, export pipeline JSONs, optional spaCy model
set -e
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3.12}"
if ! command -v "$PYTHON" &>/dev/null; then
  PYTHON=python3
fi

echo "==> Creating virtual environment (.venv) with $PYTHON"
"$PYTHON" -m venv .venv
source .venv/bin/activate

echo "==> Installing dashboard dependencies"
pip install --upgrade pip
pip install -r requirements-dashboard.txt

echo "==> Exporting data/outputs and FAISS index (use --fast to skip BERTopic)"
python scripts/export_dashboard_inputs.py "$@"

echo ""
echo "Setup complete. Run the dashboard:"
echo "  source .venv/bin/activate"
echo "  ./run_server.sh"
echo "  Open http://localhost:8000"
