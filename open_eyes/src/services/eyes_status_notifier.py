"""
Eyes status notifier.

Writes status to runtime/eyes_status.txt so other components
can monitor the vision pipeline state.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from ..config import settings

logger = logging.getLogger(__name__)


class EyesStatusNotifier:
    """
    Notifies other systems of the eyes pipeline status.

    Writes to runtime/eyes_status.txt with format:
    timestamp|status|details

    Status values:
    - IDLE: Not actively processing
    - CAPTURING: Camera is capturing frames
    - ANALYZING: Running scene analysis
    - OBSERVING: Observation in progress
    - PAUSED: Paused by coordinator or user
    - ERROR: System error
    """

    def __init__(self, status_file_path: Optional[Path] = None):
        """
        Initialize the eyes status notifier.

        Args:
            status_file_path: Path to eyes_status.txt.
        """
        self.status_file = status_file_path or settings.EYES_STATUS_FILE
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self._last_status = None

        # Create status file immediately
        self.notify_idle()

    def notify_idle(self) -> None:
        """Notify that eyes pipeline is idle."""
        self._write_status("IDLE", "waiting for scene changes")

    def notify_capturing(self, frame_count: int = 0) -> None:
        """Notify that camera is capturing."""
        self._write_status("CAPTURING", f"frame={frame_count}")

    def notify_analyzing(self, method: str = "") -> None:
        """Notify that scene analysis is running."""
        self._write_status("ANALYZING", f"method={method}")

    def notify_observing(self, description: str = "") -> None:
        """Notify that an observation is being made."""
        self._write_status("OBSERVING", description[:100])

    def notify_paused(self) -> None:
        """Notify that eyes pipeline is paused."""
        self._write_status("PAUSED", "paused by coordinator")

    def notify_error(self, message: str = "") -> None:
        """Notify of an error."""
        self._write_status("ERROR", message[:200])

    def _write_status(self, status: str, details: str) -> None:
        """Write status to file."""
        # Only write if status changed (reduce I/O)
        if status == self._last_status and status not in ("CAPTURING", "OBSERVING"):
            return

        try:
            timestamp = time.time()
            content = f"{timestamp}|{status}|{details}"
            self.status_file.write_text(content)
            self._last_status = status
        except Exception as e:
            logger.debug(f"Failed to write eyes status: {e}")

    def cleanup(self) -> None:
        """Clean up status file."""
        try:
            if self.status_file.exists():
                self.status_file.unlink()
        except Exception as e:
            logger.debug(f"Failed to cleanup eyes status: {e}")
