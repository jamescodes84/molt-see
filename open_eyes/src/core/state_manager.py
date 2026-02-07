"""
Vision pipeline state manager.

Thread-safe state tracking for the vision pipeline.
"""

import logging
import threading
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    """Vision pipeline status values."""

    IDLE = "IDLE"
    CAPTURING = "CAPTURING"
    ANALYZING = "ANALYZING"
    OBSERVING = "OBSERVING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


class VisionState:
    """Current state of the vision pipeline."""

    def __init__(self):
        self.status: PipelineStatus = PipelineStatus.IDLE
        self.current_description: str = ""
        self.queue_size: int = 0
        self.frame_count: int = 0
        self.observations_made: int = 0
        self.observations_reported: int = 0
        self.last_change_magnitude: float = 0.0
        self.error_message: Optional[str] = None


class StateManager:
    """Thread-safe state manager for the vision pipeline."""

    def __init__(self):
        self._state = VisionState()
        self._lock = threading.Lock()

    @property
    def state(self) -> VisionState:
        """Get current state (read-only snapshot)."""
        with self._lock:
            return self._state

    def set_idle(self) -> None:
        """Set pipeline to idle."""
        with self._lock:
            self._state.status = PipelineStatus.IDLE
            self._state.current_description = ""

    def set_capturing(self, frame_count: int = 0) -> None:
        """Set pipeline to capturing frames."""
        with self._lock:
            self._state.status = PipelineStatus.CAPTURING
            self._state.frame_count = frame_count

    def set_analyzing(self, description: str = "") -> None:
        """Set pipeline to analyzing a frame."""
        with self._lock:
            self._state.status = PipelineStatus.ANALYZING
            self._state.current_description = description

    def set_observing(self, description: str) -> None:
        """Set pipeline to reporting an observation."""
        with self._lock:
            self._state.status = PipelineStatus.OBSERVING
            self._state.current_description = description
            self._state.observations_made += 1

    def set_paused(self) -> None:
        """Set pipeline to paused."""
        with self._lock:
            self._state.status = PipelineStatus.PAUSED

    def set_error(self, message: str) -> None:
        """Set pipeline to error state."""
        with self._lock:
            self._state.status = PipelineStatus.ERROR
            self._state.error_message = message
            logger.error(f"Pipeline error: {message}")

    def set_stopped(self) -> None:
        """Set pipeline to stopped."""
        with self._lock:
            self._state.status = PipelineStatus.STOPPED

    def increment_reported(self) -> None:
        """Increment the count of reported observations."""
        with self._lock:
            self._state.observations_reported += 1

    def update_change_magnitude(self, magnitude: float) -> None:
        """Update last change detection magnitude."""
        with self._lock:
            self._state.last_change_magnitude = magnitude

    def update_queue_size(self, size: int) -> None:
        """Update the queue size."""
        with self._lock:
            self._state.queue_size = size
