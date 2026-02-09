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

# Install CLI (moltsee command)
echo ""
echo "Installing moltsee CLI..."
"$PIP" install -e "$SCRIPT_DIR"

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

# Ensure moltsee is on PATH
MOLTSEE_BIN="$SCRIPT_DIR/venv/bin/moltsee"
if [ -x "$MOLTSEE_BIN" ]; then
    BIN_DIR="$SCRIPT_DIR/venv/bin"
else
    # Find where pip installed the script (user install)
    BIN_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('scripts', 'posix_user'))" 2>/dev/null || true)
fi

if [ -n "$BIN_DIR" ] && ! echo "$PATH" | tr ':' '\n' | grep -qx "$BIN_DIR"; then
    SHELL_RC=""
    if [ -n "$ZSH_VERSION" ] || [ "$(basename "$SHELL")" = "zsh" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ] || [ "$(basename "$SHELL")" = "bash" ]; then
        SHELL_RC="$HOME/.bashrc"
    fi

    if [ -n "$SHELL_RC" ]; then
        EXPORT_LINE="export PATH=\"\$PATH:$BIN_DIR\""
        if ! grep -qF "$BIN_DIR" "$SHELL_RC" 2>/dev/null; then
            echo "" >> "$SHELL_RC"
            echo "# Molt-See CLI" >> "$SHELL_RC"
            echo "$EXPORT_LINE" >> "$SHELL_RC"
            echo ""
            echo "Added $BIN_DIR to PATH in $SHELL_RC"
            echo "Run: source $SHELL_RC   (or open a new terminal)"
        fi
        export PATH="$PATH:$BIN_DIR"
    fi
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit $SCRIPT_DIR/.env with your settings (especially ANTHROPIC_API_KEY for VLM)"
echo "  2. Grant camera permissions when prompted on first run"
echo "  3. Run everything:  moltsee run"
echo "  4. Stop everything: moltsee stop"
echo "  5. Check status:    moltsee status"
echo ""
