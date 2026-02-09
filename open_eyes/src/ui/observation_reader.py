"""
Pipeline status and observation history reader.

Polls runtime/eyes_status.txt and runtime/visual_observations.txt
to provide pipeline state and observation history to the viewer.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
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


@dataclass
class VisualContextSnapshot:
    """Parsed tiered visual context from visual_context.txt."""

    scene: str = ""
    objects: str = ""
    activity: str = ""
    expression: str = ""
    recent_events: list[str] = field(default_factory=list)
    last_updated: str = ""


class VisualContextReader:
    """
    Reads and parses visual_context.txt written by the vision pipeline.

    The file has a tiered format with SCENE, ACTIVITY, EXPRESSION, and
    RECENT EVENTS sections. This reader parses each section and provides
    a structured snapshot for the viewer to display.
    """

    _HEADERS = {"SCENE:", "ACTIVITY:", "EXPRESSION:", "RECENT EVENTS:"}
    # Sections that persist across blank lines (content may span multiple paragraphs)
    _PERSISTENT_SECTIONS = {"SCENE:", "RECENT EVENTS:"}

    def __init__(self, context_file: Path):
        self._file = context_file
        self._last_mtime: float = 0.0
        self._latest = VisualContextSnapshot()

    def read(self) -> VisualContextSnapshot:
        """Read visual context if the file has changed."""
        try:
            if not self._file.exists():
                return self._latest

            mtime = self._file.stat().st_mtime
            if mtime == self._last_mtime:
                return self._latest

            self._last_mtime = mtime
            content = self._file.read_text()
            self._latest = self._parse(content)
        except Exception as e:
            logger.debug(f"Failed to read visual context: {e}")

        return self._latest

    def _parse(self, content: str) -> VisualContextSnapshot:
        """Parse visual_context.txt into a VisualContextSnapshot."""
        snap = VisualContextSnapshot()

        # Group lines by section header
        sections: dict[str, list[str]] = {}
        current = ""
        for line in content.splitlines():
            stripped = line.strip()
            if stripped in self._HEADERS:
                current = stripped
                sections.setdefault(current, [])
                continue
            if stripped.startswith("[Last updated:"):
                snap.last_updated = stripped.strip("[]")
                continue
            if not stripped:
                if current not in self._PERSISTENT_SECTIONS:
                    current = ""
                continue
            if current:
                sections.setdefault(current, []).append(stripped)

        # Extract each tier from its collected lines
        self._extract_scene(snap, sections.get("SCENE:", []))
        self._extract_text(snap, "activity", sections.get("ACTIVITY:", []))
        self._extract_text(snap, "expression", sections.get("EXPRESSION:", []))
        snap.recent_events = [
            ln[2:] for ln in sections.get("RECENT EVENTS:", []) if ln.startswith("- ")
        ]
        return snap

    @staticmethod
    def _extract_scene(snap: VisualContextSnapshot, lines: list[str]) -> None:
        """Extract scene description and objects from SCENE section lines."""
        desc_parts = []
        for line in lines:
            if line.startswith("Objects:"):
                snap.objects = line[len("Objects:"):].strip()
            else:
                desc_parts.append(line)
        snap.scene = " ".join(desc_parts)

    @staticmethod
    def _extract_text(snap: VisualContextSnapshot, attr: str, lines: list[str]) -> None:
        """Set a text attribute from section lines (first line only for expression)."""
        if lines:
            setattr(snap, attr, " ".join(lines))
