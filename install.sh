#!/bin/bash
# Molt-See Installation Script
#
# Sets up virtual environment and installs all dependencies.
# Usage:
#   ./install.sh          Install everything (recommended)
#   ./install.sh --minimal  Core only, no YOLO/viewer/menu bar

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MINIMAL=false
if [ "$1" = "--minimal" ]; then
    MINIMAL=true
fi

PIP="$SCRIPT_DIR/venv/bin/pip"
PYTHON="$SCRIPT_DIR/venv/bin/python3"

echo "=== Molt-See Installation ==="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Upgrade pip
"$PIP" install --upgrade pip

# Install core dependencies
echo ""
echo "Installing core dependencies..."
"$PIP" install -r "$SCRIPT_DIR/requirements.txt"

if [ "$MINIMAL" = false ]; then
    # Object detection (YOLO)
    echo ""
    echo "Installing YOLO object detection..."
    "$PIP" install "ultralytics>=8.0.0" "torch>=2.0.0"

    # Camera viewer
    echo ""
    echo "Installing camera viewer (PyQt6)..."
    "$PIP" install "PyQt6>=6.5.0"

    # macOS menu bar
    if [ "$(uname)" = "Darwin" ]; then
        echo ""
        echo "Installing macOS menu bar support..."
        "$PIP" install "rumps>=0.4.0" "pyobjc-framework-Cocoa>=9.0"
    fi
else
    echo ""
    echo "Minimal install — skipping YOLO, viewer, and menu bar."
    echo "You can install them later:"
    echo "  $PIP install ultralytics torch    # YOLO detection"
    echo "  $PIP install PyQt6               # Camera viewer"
    echo "  $PIP install rumps pyobjc-framework-Cocoa  # macOS menu bar"
fi

# Create runtime directory
mkdir -p "$SCRIPT_DIR/runtime" "$SCRIPT_DIR/logs" "$SCRIPT_DIR/resources"

# Copy .env.example if .env doesn't exist
if [ ! -f "$SCRIPT_DIR/.env" ] && [ -f "$SCRIPT_DIR/.env.example" ]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo ""
    echo "Created .env from .env.example — edit with your settings"
fi

# Make scripts executable
chmod +x "$SCRIPT_DIR/scripts/"*.sh

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit $SCRIPT_DIR/.env with your settings (especially ANTHROPIC_API_KEY for VLM)"
echo "  2. Grant camera permissions when prompted on first run"
echo "  3. Start with: $PYTHON $SCRIPT_DIR/open_eyes/main.py"
echo "  4. Or with Molt-Speak: $PYTHON $SCRIPT_DIR/main.py --molt-speak-dir /path/to/molt-speak/runtime"
echo "  5. Launch viewer: $PYTHON $SCRIPT_DIR/open_eyes/scripts/viewer.py"
echo ""
