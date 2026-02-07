"""
Molt-Speak status monitor.

Reads Molt-Speak's runtime status files to understand the current
state of the voice system (who is speaking, conversation activity).
"""

import logging
import time
from pathlib import Path
from typing import Optional

from ..config import settings

logger = logging.getLogger(__name__)


class SpeakStatusMonitor:
    """
    Monitors Molt-Speak's status files.

    Reads mouth_status.txt and ears_status.txt from Molt-Speak's
    runtime directory to coordinate with the voice system.
    """

    def __init__(
        self,
        molt_speak_runtime_dir: Optional[Path] = settings.MOLT_SPEAK_RUNTIME_DIR,
        poll_interval: float = settings.SPEAK_STATUS_POLL_INTERVAL,
    ):
        """
        Initialize speak status monitor.

        Args:
            molt_speak_runtime_dir: Path to Molt-Speak's runtime directory.
            poll_interval: How often to check status files (seconds).
        """
        self.molt_speak_dir = molt_speak_runtime_dir
        self.poll_interval = poll_interval

        self._mouth_status_file: Optional[Path] = None
        self._ears_status_file: Optional[Path] = None

        if self.molt_speak_dir:
            self._mouth_status_file = self.molt_speak_dir / "mouth_status.txt"
            self._ears_status_file = self.molt_speak_dir / "ears_status.txt"

    @property
    def is_connected(self) -> bool:
        """Check if Molt-Speak runtime directory exists."""
        return self.molt_speak_dir is not None and self.molt_speak_dir.exists()

    def is_agent_speaking(self) -> bool:
        """Check if the agent is currently speaking via Molt-Speak."""
        status = self._read_status(self._mouth_status_file)
        return status == "SPEAKING"

    def is_user_speaking(self) -> bool:
        """Check if the user is currently speaking."""
        status = self._read_status(self._ears_status_file)
        return status == "SPEECH_DETECTED"

    def is_conversation_idle(self) -> bool:
        """Check if both mouth and ears are idle."""
        return not self.is_agent_speaking() and not self.is_user_speaking()

    def _read_status(self, status_file: Optional[Path]) -> Optional[str]:
        """Read status from a Molt-Speak status file."""
        if not status_file or not status_file.exists():
            return None

        try:
            content = status_file.read_text().strip()
            if "|" in content:
                parts = content.split("|")
                if len(parts) >= 2:
                    return parts[1]
            return content
        except Exception:
            return None
