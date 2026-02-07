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

    # Max items to scan for deduplication (avoids O(n) over full history)
    _DEDUP_SCAN_DEPTH = 15

    def _is_match(self, desc: str, current_labels: set, recent: dict) -> bool:
        """Check if a single recent observation matches the candidate."""
        recent_labels = set(recent["labels"]) if recent["labels"] else set()
        # Fast path: label overlap pre-filter (set ops are cheap)
        if current_labels and recent_labels:
            overlap = len(current_labels & recent_labels) / len(
                current_labels | recent_labels
            )
            if overlap >= self.similarity_threshold:
                similarity = SequenceMatcher(
                    None, desc, recent["description"]
                ).ratio()
                return similarity >= 0.7
            return False

        # Fall back to text similarity for label-less observations
        similarity = SequenceMatcher(
            None, desc, recent["description"]
        ).ratio()
        return similarity >= self.similarity_threshold

    def is_duplicate(self, observation: SceneDescription) -> bool:
        """
        Check if this observation is substantially similar to recent ones.

        Uses label overlap as a fast pre-filter, then text similarity
        only when labels suggest a match. Scans only the most recent
        observations to bound latency.

        Args:
            observation: Scene description to check.

        Returns:
            True if the observation is too similar to a recent one.
        """
        current_labels = set(observation.object_labels)
        desc = observation.description
        items = list(self._observations)[-self._DEDUP_SCAN_DEPTH:]

        return any(self._is_match(desc, current_labels, r) for r in reversed(items))

    def was_recently_reported(
        self,
        observation: SceneDescription,
        window_seconds: float = settings.RECENTLY_REPORTED_WINDOW,
    ) -> bool:
        """
        Check if a similar observation was reported within a time window.

        Only considers observations reported in the last *window_seconds*.
        This prevents stale reports from permanently blocking new ones
        when the scene hasn't changed much.

        Args:
            observation: Scene description to check.
            window_seconds: Only consider reports within this many seconds.

        Returns:
            True if a similar observation was reported recently.
        """
        desc = observation.description
        cutoff = time.time() - window_seconds
        items = [
            r for r in list(self._reported)[-self._DEDUP_SCAN_DEPTH:]
            if r.get("reported_at", 0) >= cutoff
        ]

        return any(
            SequenceMatcher(None, desc, r["description"]).ratio()
            >= self.similarity_threshold
            for r in reversed(items)
        )

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

    def clear_session(self) -> None:
        """Clear short-term dedup state for a fresh session.

        Keeps observations (scene context) but clears the reported
        tracker so the first observation of a new session always
        passes the novelty check.
        """
        self._reported.clear()

    def clear(self) -> None:
        """Clear all memory."""
        self._observations.clear()
        self._reported.clear()
