#!/usr/bin/env bash
# Bootstrap deps then run the Python scorer.
# stdout/stderr are streamed to participants so keep messages clear.
set -euo pipefail

echo "[scoring] installing dependencies..."
pip install --quiet --no-warn-script-location -r requirements.txt

echo "[scoring] launching scorer..."
exec python3 score.py
