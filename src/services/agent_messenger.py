"""
Agent messenger service.

Sends vision instructions to AI agents by writing signal
files that agents can read.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from ..config import settings

logger = logging.getLogger(__name__)


class VisionMessenger:
    """
    Sends vision signals to the AI agent.

    The agent instructions live in AGENT_INSTRUCTIONS.txt at the project root.
    This class handles shutdown signaling and cleanup.
    """

    def __init__(
        self,
        shutdown_signal_file: Optional[Path] = None,
    ):
        """
        Initialize vision messenger.

        Args:
            shutdown_signal_file: Path for shutdown signal.
        """
        self.shutdown_file = shutdown_signal_file or settings.AGENT_SHUTDOWN_SIGNAL_FILE
        self.instructions_file = settings.PROJECT_DIR / "AGENT_INSTRUCTIONS.txt"

    def get_instructions_path(self) -> Path:
        """Return the path to the agent instructions file."""
        return self.instructions_file

    def send_shutdown(self, message: str = "Vision system shutting down") -> None:
        """Signal the agent that vision is shutting down."""
        try:
            self.shutdown_file.parent.mkdir(parents=True, exist_ok=True)
            self.shutdown_file.write_text(
                f"{time.time()}|{message}"
            )
            logger.info("Shutdown signal sent to agent")
        except Exception as e:
            logger.error(f"Failed to write shutdown signal: {e}")

    def cleanup(self) -> None:
        """Remove signal files."""
        try:
            if self.shutdown_file.exists():
                self.shutdown_file.unlink()
        except Exception as e:
            logger.debug(f"Failed to cleanup {self.shutdown_file}: {e}")
