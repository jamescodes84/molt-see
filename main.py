#!/usr/bin/env python3
"""
Molt-See - Vision perception system for AI agents.

Main entry point that starts the VisionCoordinator, which manages
the OpenClaw Eyes pipeline and agent integration.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from src.utils.logging_utils import configure_logging
from src.core.coordinator import VisionCoordinator

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Molt-See - Vision perception for AI agents"
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
    return parser.parse_args()


def main() -> None:
    """Main entry point for Molt-See."""
    args = parse_args()

    # Configure logging
    configure_logging(level=args.log_level)

    logger.info("Starting Molt-See Vision System...")

    # Resolve Molt-Speak runtime directory
    molt_speak_dir = Path(args.molt_speak_dir) if args.molt_speak_dir else None

    # Create and start coordinator
    coordinator = VisionCoordinator(
        molt_speak_runtime_dir=molt_speak_dir,
    )

    coordinator.start()


if __name__ == "__main__":
    main()
