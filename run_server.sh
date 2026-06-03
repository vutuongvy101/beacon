#!/usr/bin/env bash
# Start the dashboard API (serves frontend at http://localhost:8000 when frontend/ exists)
set -e
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
PYTHON="${PWD}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Create the venv first: python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dashboard.txt"
  exit 1
fi
exec "$PYTHON" -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
