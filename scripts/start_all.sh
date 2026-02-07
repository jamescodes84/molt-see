#!/bin/bash
# Start both Molt-Speak and Molt-See together
#
# Usage: ./scripts/start_all.sh [--molt-speak-dir /path/to/molt-speak/runtime]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Auto-detect Molt-Speak runtime directory
MOLT_SPEAK_DIR=""
POSSIBLE_PATHS=(
    "$PROJECT_DIR/../Molt-Speak/molt-speak/runtime"
    "$PROJECT_DIR/../molt-speak/runtime"
    "$HOME/molt-speak/runtime"
)

for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        MOLT_SPEAK_DIR="$path"
        break
    fi
done

if [ -n "$MOLT_SPEAK_DIR" ]; then
    echo "Molt-Speak runtime found: $MOLT_SPEAK_DIR"
    export MOLT_SPEAK_RUNTIME_DIR="$MOLT_SPEAK_DIR"
else
    echo "Molt-Speak runtime not found. Running standalone."
fi

# Start the coordinator (which starts OpenClaw Eyes)
echo "Starting Molt-See Vision System..."
python main.py "$@"
