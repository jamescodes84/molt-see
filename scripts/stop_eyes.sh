#!/bin/bash
# Stop OpenClaw Eyes and Vision Coordinator

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RUNTIME_DIR="$PROJECT_DIR/runtime"

echo "Stopping Molt-See..."

# Stop Eyes process
if [ -f "$RUNTIME_DIR/eyes.pid" ]; then
    PID=$(cat "$RUNTIME_DIR/eyes.pid")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "OpenClaw Eyes stopped (PID: $PID)"
    fi
    rm -f "$RUNTIME_DIR/eyes.pid"
fi

# Stop Coordinator process
if [ -f "$RUNTIME_DIR/vision_coordinator.pid" ]; then
    PID=$(cat "$RUNTIME_DIR/vision_coordinator.pid")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "Vision Coordinator stopped (PID: $PID)"
    fi
    rm -f "$RUNTIME_DIR/vision_coordinator.pid"
fi

# Clean up signal files
rm -f "$RUNTIME_DIR/eyes_status.txt"
rm -f "$RUNTIME_DIR/eyes_pause.signal"
rm -f "$RUNTIME_DIR/agent_instructions.active"

echo "Molt-See stopped."
