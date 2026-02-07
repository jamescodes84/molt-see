"""
Pipeline status and observation history reader.

Polls runtime/eyes_status.txt and runtime/visual_observations.txt
to provide pipeline state and observation history to the viewer.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PipelineStatusInfo:
    """Parsed pipeline status from eyes_status.txt."""

    timestamp: float = 0.0
    status: str = "UNKNOWN"
    details: str = ""
    is_connected: bool = False

    @property
    def age_seconds(self) -> float:
        if self.timestamp <= 0:
            return float("inf")
        return time.time() - self.timestamp


@dataclass
class ObservationEntry:
    """A single parsed observation from visual_observations.txt."""

    timestamp_iso: str = ""
    observation_type: str = ""
    description: str = ""


class ObservationReader:
    """
    Reads pipeline status and observation history from runtime files.

    Parses the existing IPC formats:
    - eyes_status.txt: "timestamp|STATUS|details"
    - visual_observations.txt: "ISO_TIMESTAMP|TYPE|description"
    """

    def __init__(
        self,
        status_file: Path,
        observations_file: Path,
        max_history: int = 50,
        pipeline_timeout: float = 10.0,
    ):
        self._status_file = status_file
        self._observations_file = observations_file
        self._pipeline_timeout = pipeline_timeout
        self._max_history = max_history

        self._last_status_mtime: float = 0.0
        self._last_obs_size: int = 0

        self._status = PipelineStatusInfo()
        self._observations: deque[ObservationEntry] = deque(maxlen=max_history)

    def read_status(self) -> PipelineStatusInfo:
        """Read and parse eyes_status.txt if changed."""
        try:
            if not self._status_file.exists():
                self._status = PipelineStatusInfo()
                return self._status

            mtime = self._status_file.stat().st_mtime
            if mtime == self._last_status_mtime:
                self._status.is_connected = (
                    self._status.age_seconds < self._pipeline_timeout
                )
                return self._status

            self._last_status_mtime = mtime
            content = self._status_file.read_text().strip()
            if not content:
                return self._status

            parts = content.split("|", 2)
            if len(parts) >= 2:
                self._status = PipelineStatusInfo(
                    timestamp=float(parts[0]),
                    status=parts[1],
                    details=parts[2] if len(parts) > 2 else "",
                    is_connected=True,
                )
        except Exception as e:
            logger.debug(f"Failed to read pipeline status: {e}")

        return self._status

    def read_observations(self) -> list[ObservationEntry]:
        """Read new observations appended since last check."""
        try:
            if not self._observations_file.exists():
                return list(self._observations)

            current_size = self._observations_file.stat().st_size
            if current_size == self._last_obs_size:
                return list(self._observations)

            with open(self._observations_file, "r") as f:
                if self._last_obs_size > 0 and current_size > self._last_obs_size:
                    f.seek(self._last_obs_size)
                new_lines = f.readlines()

            self._last_obs_size = current_size

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|", 2)
                if len(parts) >= 3:
                    self._observations.append(
                        ObservationEntry(
                            timestamp_iso=parts[0],
                            observation_type=parts[1],
                            description=parts[2],
                        )
                    )
        except Exception as e:
            logger.debug(f"Failed to read observations: {e}")

        return list(self._observations)
