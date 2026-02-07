#!/usr/bin/env python3
"""
Molt-See Camera Viewer - standalone PyQt6 camera viewer.

Displays live camera feed with detection overlays from the
OpenClaw Eyes vision pipeline.

Usage:
    python open_eyes/scripts/viewer.py
    python open_eyes/scripts/viewer.py --fps 30 --mode expanded
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Molt-See Camera Viewer - live camera with detection overlays"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=int(os.getenv("VIEWER_FPS", "15")),
        help="Viewer camera FPS (default: 15)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=int(
            os.getenv(
                "VIEWER_CAMERA_DEVICE",
                os.getenv("CAMERA_DEVICE_INDEX", "0"),
            )
        ),
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--mode",
        choices=["compact", "expanded"],
        default=os.getenv("VIEWER_DEFAULT_MODE", "compact"),
        help="Initial viewer mode (default: compact)",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--no-on-top",
        action="store_true",
        help="Do not keep viewer window on top",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the camera viewer."""
    args = parse_args()

    configure_logging(level=args.log_level)

    logger.info("Starting Molt-See Camera Viewer...")
    logger.info(f"Config: fps={args.fps}, camera={args.camera}, mode={args.mode}")

    # Apply CLI args to env so viewer_settings picks them up
    os.environ["VIEWER_FPS"] = str(args.fps)
    os.environ["VIEWER_CAMERA_DEVICE"] = str(args.camera)
    os.environ["VIEWER_DEFAULT_MODE"] = args.mode

    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        logger.error(
            "PyQt6 is required for the camera viewer. "
            "Install with: pip install PyQt6"
        )
        sys.exit(1)

    from open_eyes.src.ui.main_window import ViewerWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Molt-See Viewer")

    window = ViewerWindow()

    if args.no_on_top:
        window.setWindowFlags(
            window.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint
        )

    if not window.start():
        logger.error("Failed to start camera viewer")
        sys.exit(1)

    window.show()
    logger.info("Camera viewer running. Press Q to quit, Tab to toggle mode.")

    # Allow Ctrl+C in terminal to quit
    signal.signal(signal.SIGINT, lambda *_: app.quit())

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
