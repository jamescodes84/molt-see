"""
Vision memory service.

Tracks visual observations over time for deduplication and continuity.
Maintains a sliding window of recent observations and tracks what has
been reported to the agent.
"""

import json
import logging
import time
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from ..config import settings
from ..models.vision_models import SceneDescription

logger = logging.getLogger(__name__)


class VisionMemory:
    """
    Tracks visual observations for deduplication and scene continuity.

    Maintains a sliding window of recent observations, tracks what has
    been reported vs. merely observed, and provides context for
    relevance scoring.
    """

    def __init__(
        self,
        max_observations: int = settings.MAX_OBSERVATIONS,
        max_reported: int = settings.MAX_REPORTED,
        similarity_threshold: float = settings.MEMORY_SIMILARITY_THRESHOLD,
        state_file: Optional[Path] = None,
    ):
        """
        Initialize vision memory.

        Args:
            max_observations: Max observations in sliding window.
            max_reported: Max reported observations to track.
            similarity_threshold: Text similarity threshold for dedup (0-1).
            state_file: Path for state persistence.
        """
        self.max_observations = max_observations
        self.max_reported = max_reported
        self.similarity_threshold = similarity_threshold
        self.state_file = state_file or settings.VISION_MEMORY_FILE

        self._observations: deque[dict] = deque(maxlen=max_observations)
        self._reported: deque[dict] = deque(maxlen=max_reported)

    def record_observation(self, observation: SceneDescription) -> None:
        """
        Record an observation (seen but not necessarily reported).

        Args:
            observation: Scene description to record.
        """
        self._observations.append({
            "timestamp": observation.timestamp,
            "description": observation.description,
            "labels": observation.object_labels,
            "method": observation.analysis_method,
        })

    def record_reported(self, observation: SceneDescription) -> None:
        """
        Record that an observation was reported to the agent.

        Args:
            observation: Scene description that was reported.
        """
        self._reported.append({
            "timestamp": observation.timestamp,
            "description": observation.description,
            "labels": observation.object_labels,
            "reported_at": time.time(),
        })

    def is_duplicate(self, observation: SceneDescription) -> bool:
        """
        Check if this observation is substantially similar to recent ones.

        Uses text similarity between descriptions and label overlap.

        Args:
            observation: Scene description to check.

        Returns:
            True if the observation is too similar to a recent one.
        """
        for recent in reversed(self._observations):
            similarity = SequenceMatcher(
                None, observation.description, recent["description"]
            ).ratio()
            if similarity >= self.similarity_threshold:
                return True

            # Also check label overlap
            if observation.object_labels and recent["labels"]:
                label_set = set(observation.object_labels)
                recent_set = set(recent["labels"])
                if label_set and recent_set:
                    overlap = len(label_set & recent_set) / len(
                        label_set | recent_set
                    )
                    if overlap >= self.similarity_threshold:
                        # Labels are very similar — check if description also close
                        if similarity >= 0.7:
                            return True

        return False

    def was_recently_reported(self, observation: SceneDescription) -> bool:
        """
        Check if a similar observation was recently reported.

        Args:
            observation: Scene description to check.

        Returns:
            True if a similar observation was reported recently.
        """
        for reported in reversed(self._reported):
            similarity = SequenceMatcher(
                None, observation.description, reported["description"]
            ).ratio()
            if similarity >= self.similarity_threshold:
                return True
        return False

    def get_recent_observations(self, count: int = 10) -> list[dict]:
        """
        Get the N most recent observations.

        Args:
            count: Number of observations to return.

        Returns:
            List of observation dicts.
        """
        items = list(self._observations)
        return items[-count:] if len(items) > count else items

    def get_recent_reported(self, count: int = 5) -> list[dict]:
        """
        Get the N most recently reported observations.

        Args:
            count: Number of reported observations to return.

        Returns:
            List of reported observation dicts.
        """
        items = list(self._reported)
        return items[-count:] if len(items) > count else items

    def get_scene_summary(self) -> str:
        """
        Generate a brief summary of the current scene state.

        Useful for injecting into agent context.

        Returns:
            A concise summary string.
        """
        recent = self.get_recent_observations(5)
        if not recent:
            return "No visual observations yet."

        # Collect unique labels from recent observations
        all_labels = set()
        for obs in recent:
            all_labels.update(obs.get("labels", []))

        latest = recent[-1]
        parts = [f"Latest: {latest['description']}"]
        if all_labels:
            parts.append(f"Visible: {', '.join(sorted(all_labels))}")

        return " | ".join(parts)

    def save_state(self) -> None:
        """Persist memory to disk."""
        try:
            state = {
                "observations": list(self._observations),
                "reported": list(self._reported),
            }
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save vision memory: {e}")

    def load_state(self) -> None:
        """Load memory from disk."""
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text())
                self._observations = deque(
                    state.get("observations", []),
                    maxlen=self.max_observations,
                )
                self._reported = deque(
                    state.get("reported", []),
                    maxlen=self.max_reported,
                )
                logger.info(
                    f"Loaded vision memory: {len(self._observations)} observations, "
                    f"{len(self._reported)} reported"
                )
        except Exception as e:
            logger.warning(f"Failed to load vision memory: {e}")

    def clear(self) -> None:
        """Clear all memory."""
        self._observations.clear()
        self._reported.clear()
