#!/bin/bash
# Start OpenClaw Eyes (vision pipeline only)
#
# Usage: ./scripts/start_eyes.sh [--fps 1.0] [--analyzer cascade]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start OpenClaw Eyes
echo "Starting OpenClaw Eyes..."
python open_eyes/main.py "$@"
