#!/usr/bin/env python3
"""
OpenClaw Eyes - Vision perception for AI agents.

Entry point for the vision pipeline. Captures webcam frames,
detects scene changes, analyzes scenes, and reports contextually
relevant observations.
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logging_utils import configure_logging
from open_eyes.src.core.vision_pipeline import VisionPipeline

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenClaw Eyes - Vision perception for AI agents"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=float(os.getenv("CAPTURE_FPS", "1.0")),
        help="Capture frames per second (default: 1.0)",
    )
    parser.add_argument(
        "--analyzer",
        choices=["yolo", "vlm", "cascade", "florence", "composite"],
        default=os.getenv("ANALYZER_MODE", "cascade"),
        help="Scene analyzer mode (default: cascade)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("RELEVANCE_THRESHOLD", "0.6")),
        help="Relevance threshold 0-1 (default: 0.6)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=int(os.getenv("CAMERA_DEVICE_INDEX", "0")),
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--molt-speak-dir",
        type=str,
        default=os.getenv("MOLT_SPEAK_RUNTIME_DIR"),
        help="Path to Molt-Speak runtime directory",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable terminal visualization",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for OpenClaw Eyes."""
    args = parse_args()

    # Configure logging
    configure_logging(level=args.log_level)

    logger.info("Starting OpenClaw Eyes...")
    logger.info(
        f"Config: fps={args.fps}, analyzer={args.analyzer}, "
        f"threshold={args.threshold}, camera={args.camera}"
    )

    # Resolve Molt-Speak runtime directory
    molt_speak_dir = Path(args.molt_speak_dir) if args.molt_speak_dir else None

    # Create and start pipeline
    pipeline = VisionPipeline(
        capture_fps=args.fps,
        analyzer_mode=args.analyzer,
        relevance_threshold=args.threshold,
        camera_device=args.camera,
        molt_speak_runtime=molt_speak_dir,
        enable_display=not args.no_display,
    )

    # Write PID file
    pid_file = Path(__file__).parent.parent / "runtime" / "eyes.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))

    # Signal handlers for graceful shutdown
    def shutdown(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        pipeline.stop()
        if pid_file.exists():
            pid_file.unlink()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start pipeline
    if not pipeline.start():
        logger.error("Failed to start vision pipeline")
        sys.exit(1)

    logger.info("OpenClaw Eyes running. Press 'q' + Enter or Ctrl+C to stop.")

    # Keep main thread alive, accept 'q' to quit gracefully
    import select
    import time

    try:
        while True:
            # Check for stdin input (non-blocking with 1s timeout)
            if select.select([sys.stdin], [], [], 1.0)[0]:
                line = sys.stdin.readline().strip().lower()
                if line in ("q", "quit", "exit"):
                    break
    except (EOFError, KeyboardInterrupt):
        pass

    shutdown(signal.SIGINT, None)


if __name__ == "__main__":
    main()
