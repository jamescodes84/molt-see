#!/usr/bin/env python3
"""
Molt-See CLI — unified commands for the vision system.

Usage:
    moltsee run                          Start pipeline + viewer together
    moltsee run --analyzer cascade -- --mode expanded
    moltsee start [--analyzer cascade]   Start pipeline only
    moltsee viewer [--mode expanded]     Start viewer only
    moltsee stop                         Stop all processes
    moltsee status                       Show current status
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNTIME_DIR = PROJECT_ROOT / "runtime"

PID_FILES = [
    ("eyes.pid", "OpenClaw Eyes"),
    ("vision_coordinator.pid", "Vision Coordinator"),
]


def cmd_start(remaining: list[str]) -> None:
    """Start the OpenClaw Eyes vision pipeline."""
    sys.argv = ["moltsee-start"] + remaining
    from open_eyes.main import main as start_main

    start_main()


def cmd_viewer(remaining: list[str]) -> None:
    """Start the PyQt6 camera viewer with detection overlays."""
    sys.argv = ["moltsee-viewer"] + remaining
    from open_eyes.scripts.viewer import main as viewer_main

    viewer_main()


def cmd_run(remaining: list[str]) -> None:
    """Start the pipeline in the background and viewer in the foreground."""
    # Split remaining args: everything before '--' goes to pipeline,
    # everything after goes to viewer. If no '--', all go to pipeline
    # and viewer uses defaults.
    pipeline_args: list[str] = []
    viewer_args: list[str] = []
    target = pipeline_args
    for arg in remaining:
        if arg == "--":
            target = viewer_args
            continue
        target.append(arg)

    # Find the Python interpreter (prefer venv)
    venv_python = PROJECT_ROOT / "venv" / "bin" / "python3"
    python = str(venv_python) if venv_python.exists() else sys.executable

    # Start pipeline as background process
    pipeline_cmd = [python, str(PROJECT_ROOT / "open_eyes" / "main.py")] + pipeline_args
    print(f"Starting pipeline: {' '.join(pipeline_cmd)}")
    pipeline_proc = subprocess.Popen(
        pipeline_cmd,
        stdin=subprocess.DEVNULL,
        cwd=str(PROJECT_ROOT),
    )
    print(f"Pipeline running (PID: {pipeline_proc.pid})")

    # Launch viewer in foreground — blocks until viewer is closed
    print("Starting viewer...")
    sys.argv = ["moltsee-viewer"] + viewer_args
    try:
        from open_eyes.scripts.viewer import main as viewer_main

        viewer_main()
    except KeyboardInterrupt:
        pass
    finally:
        # Viewer closed — stop the pipeline
        print("\nViewer closed, stopping pipeline...")
        pipeline_proc.terminate()
        try:
            pipeline_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pipeline_proc.kill()
        # Clean up PID files
        for name, _ in PID_FILES:
            (RUNTIME_DIR / name).unlink(missing_ok=True)
        print("Molt-See stopped.")


def cmd_stop(_remaining: list[str]) -> None:
    """Stop all Molt-See processes."""
    stopped = False

    for pid_name, label in PID_FILES:
        pid_file = RUNTIME_DIR / pid_name
        if not pid_file.exists():
            continue

        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            print(f"{label} stopped (PID: {pid})")
            stopped = True
        except (ProcessLookupError, ValueError):
            pass
        finally:
            pid_file.unlink(missing_ok=True)

    # Clean up signal files
    for name in [
        "eyes_status.txt",
        "eyes_pause.signal",
    ]:
        signal_file = RUNTIME_DIR / name
        signal_file.unlink(missing_ok=True)

    if stopped:
        print("Molt-See stopped.")
    else:
        print("No running Molt-See processes found.")


def cmd_status(_remaining: list[str]) -> None:
    """Show current pipeline status."""
    # Check PID files
    for pid_name, label in PID_FILES:
        pid_file = RUNTIME_DIR / pid_name
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)  # check if process exists
                print(f"{label}: running (PID: {pid})")
            except (ProcessLookupError, ValueError):
                print(f"{label}: stale PID file (not running)")
        else:
            print(f"{label}: not running")

    # Show latest status
    status_file = RUNTIME_DIR / "eyes_status.txt"
    if status_file.exists():
        content = status_file.read_text().strip()
        if content:
            last_line = content.splitlines()[-1]
            print(f"\nLast status: {last_line}")

    # Show visual context summary
    context_file = RUNTIME_DIR / "visual_context.txt"
    if context_file.exists():
        content = context_file.read_text().strip()
        if content:
            print(f"\n--- Visual Context ---\n{content}")


COMMANDS = {
    "run": ("Start pipeline + viewer together", cmd_run),
    "start": ("Start the vision pipeline only", cmd_start),
    "viewer": ("Start camera viewer only", cmd_viewer),
    "stop": ("Stop all Molt-See processes", cmd_stop),
    "status": ("Show pipeline status", cmd_status),
}


def _print_help() -> None:
    """Print top-level help."""
    print("usage: moltsee <command> [options]\n")
    print("Molt-See — vision perception for AI agents\n")
    print("Commands:")
    for name, (desc, _) in COMMANDS.items():
        print(f"  {name:<10} {desc}")
    print("\nRun 'moltsee <command> --help' for command-specific options.")


def main() -> None:
    """Molt-See CLI entry point."""
    # Manual argv parsing so --help forwards to subcommands correctly
    argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        sys.exit(0)

    command = argv[0]
    remaining = argv[1:]

    if command not in COMMANDS:
        print(f"moltsee: unknown command '{command}'\n")
        _print_help()
        sys.exit(1)

    _, handler = COMMANDS[command]
    handler(remaining)


if __name__ == "__main__":
    main()
