"""Unit tests for vision memory."""

import time

import pytest

from open_eyes.src.models.vision_models import SceneDescription
from open_eyes.src.services.vision_memory import VisionMemory


class TestVisionMemory:
    """Tests for VisionMemory."""

    def _make_observation(
        self, description: str, labels: list[str] = None
    ) -> SceneDescription:
        """Helper to create a SceneDescription."""
        return SceneDescription(
            timestamp=time.time(),
            description=description,
            object_labels=labels or [],
            analysis_method="test",
        )

    def test_record_observation(self):
        """Recording an observation should be retrievable."""
        memory = VisionMemory()
        obs = self._make_observation("A person at a desk")
        memory.record_observation(obs)
        recent = memory.get_recent_observations()
        assert len(recent) == 1
        assert recent[0]["description"] == "A person at a desk"

    def test_record_reported(self):
        """Recording a reported observation should track it."""
        memory = VisionMemory()
        obs = self._make_observation("Person detected")
        memory.record_reported(obs)
        reported = memory.get_recent_reported()
        assert len(reported) == 1

    def test_duplicate_detection(self):
        """Similar observations should be detected as duplicates."""
        memory = VisionMemory(similarity_threshold=0.85)
        obs1 = self._make_observation(
            "A person sitting at a desk with a laptop",
            labels=["person", "desk", "laptop"],
        )
        memory.record_observation(obs1)

        obs2 = self._make_observation(
            "A person sitting at the desk with a laptop",
            labels=["person", "desk", "laptop"],
        )
        assert memory.is_duplicate(obs2) is True

    def test_different_observation_not_duplicate(self):
        """Substantially different observations should not be duplicates."""
        memory = VisionMemory(similarity_threshold=0.85)
        obs1 = self._make_observation("A cat sleeping on the couch")
        memory.record_observation(obs1)

        obs2 = self._make_observation("A person walking through the door")
        assert memory.is_duplicate(obs2) is False

    def test_sliding_window_eviction(self):
        """Old observations should be evicted from window."""
        memory = VisionMemory(max_observations=3)
        for i in range(5):
            obs = self._make_observation(f"Observation number {i}")
            memory.record_observation(obs)

        recent = memory.get_recent_observations(count=10)
        assert len(recent) == 3

    def test_was_recently_reported(self):
        """Should detect if similar observation was recently reported."""
        memory = VisionMemory(similarity_threshold=0.85)
        obs = self._make_observation("A person sitting at a desk")
        memory.record_reported(obs)

        similar = self._make_observation("A person sitting at the desk")
        assert memory.was_recently_reported(similar) is True

    def test_was_not_recently_reported(self):
        """Should not flag unrelated observations as reported."""
        memory = VisionMemory()
        obs = self._make_observation("A person sitting at a desk")
        memory.record_reported(obs)

        different = self._make_observation("A cat on the windowsill")
        assert memory.was_recently_reported(different) is False

    def test_scene_summary(self):
        """Should generate a summary of recent observations."""
        memory = VisionMemory()
        obs = self._make_observation(
            "A person at a desk with coffee",
            labels=["person", "desk", "cup"],
        )
        memory.record_observation(obs)

        summary = memory.get_scene_summary()
        assert "person" in summary.lower() or "desk" in summary.lower()

    def test_scene_summary_empty(self):
        """Empty memory should return appropriate message."""
        memory = VisionMemory()
        summary = memory.get_scene_summary()
        assert "no visual" in summary.lower()

    def test_state_persistence(self, tmp_path):
        """Memory should persist to and load from disk."""
        state_file = tmp_path / "memory.json"
        memory = VisionMemory(state_file=state_file)
        obs = self._make_observation(
            "Test observation", labels=["test"]
        )
        memory.record_observation(obs)
        memory.save_state()

        memory2 = VisionMemory(state_file=state_file)
        memory2.load_state()
        assert len(memory2.get_recent_observations()) == 1

    def test_clear(self):
        """Clear should remove all observations."""
        memory = VisionMemory()
        memory.record_observation(
            self._make_observation("Test")
        )
        memory.record_reported(
            self._make_observation("Test")
        )
        memory.clear()
        assert len(memory.get_recent_observations()) == 0
        assert len(memory.get_recent_reported()) == 0

    def test_get_recent_limits(self):
        """get_recent should respect count limit."""
        memory = VisionMemory()
        for i in range(10):
            memory.record_observation(
                self._make_observation(f"Observation {i}")
            )

        recent = memory.get_recent_observations(count=3)
        assert len(recent) == 3
