"""Unit tests for visual context file output."""

import time
from unittest.mock import patch

import pytest

from open_eyes.src.core.vision_pipeline import VisionPipeline
from open_eyes.src.models.vision_models import ObservationType, SceneDescription


class TestVisualContext:
    """Tests for the visual context file writing."""

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

    def _make_pipeline_stub(self, context_file):
        """Create a minimal object with the attributes _write_visual_context needs."""
        stub = object.__new__(VisionPipeline)
        stub._recent_changes = []
        stub._max_recent_changes = 5
        return stub

    def test_context_file_created(self, tmp_path):
        """Writing visual context should create the file."""
        context_file = tmp_path / "visual_context.txt"
        stub = self._make_pipeline_stub(context_file)
        obs = self._make_observation(
            "A person sitting at a desk",
            labels=["person", "desk"],
        )

        with patch("open_eyes.src.core.vision_pipeline.settings") as mock_settings:
            mock_settings.VISUAL_CONTEXT_FILE = context_file
            stub._write_visual_context(
                obs, ObservationType.SCENE_CHANGE, "2026-02-08T14:23:15+00:00"
            )

        assert context_file.exists()

    def test_context_file_contains_current_scene(self, tmp_path):
        """Context file should include the current scene description."""
        context_file = tmp_path / "visual_context.txt"
        stub = self._make_pipeline_stub(context_file)
        obs = self._make_observation(
            "A man in a hoodie at a desk with a guitar on the wall",
            labels=["person", "desk", "guitar"],
        )

        with patch("open_eyes.src.core.vision_pipeline.settings") as mock_settings:
            mock_settings.VISUAL_CONTEXT_FILE = context_file
            stub._write_visual_context(
                obs, ObservationType.SCENE_CHANGE, "2026-02-08T14:23:15+00:00"
            )

        content = context_file.read_text()
        assert "CURRENT SCENE:" in content
        assert "A man in a hoodie at a desk with a guitar on the wall" in content

    def test_context_file_contains_objects(self, tmp_path):
        """Context file should list detected objects."""
        context_file = tmp_path / "visual_context.txt"
        stub = self._make_pipeline_stub(context_file)
        obs = self._make_observation(
            "A person at a desk",
            labels=["person", "desk", "laptop"],
        )

        with patch("open_eyes.src.core.vision_pipeline.settings") as mock_settings:
            mock_settings.VISUAL_CONTEXT_FILE = context_file
            stub._write_visual_context(
                obs, ObservationType.SCENE_CHANGE, "2026-02-08T14:23:15+00:00"
            )

        content = context_file.read_text()
        assert "OBJECTS:" in content
        assert "desk" in content
        assert "laptop" in content
        assert "person" in content

    def test_context_file_contains_timestamp(self, tmp_path):
        """Context file should include the last-updated timestamp."""
        context_file = tmp_path / "visual_context.txt"
        stub = self._make_pipeline_stub(context_file)
        obs = self._make_observation("A room")

        with patch("open_eyes.src.core.vision_pipeline.settings") as mock_settings:
            mock_settings.VISUAL_CONTEXT_FILE = context_file
            stub._write_visual_context(
                obs, ObservationType.SCENE_CHANGE, "2026-02-08T14:23:15+00:00"
            )

        content = context_file.read_text()
        assert "2026-02-08T14:23:15+00:00" in content

    def test_context_file_recent_changes_accumulate(self, tmp_path):
        """Multiple observations should accumulate in RECENT CHANGES."""
        context_file = tmp_path / "visual_context.txt"
        stub = self._make_pipeline_stub(context_file)

        with patch("open_eyes.src.core.vision_pipeline.settings") as mock_settings:
            mock_settings.VISUAL_CONTEXT_FILE = context_file

            obs1 = self._make_observation("Person entered the room", labels=["person"])
            stub._write_visual_context(
                obs1, ObservationType.PERSON_DETECTED, "2026-02-08T14:20:00+00:00"
            )

            obs2 = self._make_observation("Person picked up coffee mug", labels=["person", "cup"])
            stub._write_visual_context(
                obs2, ObservationType.OBJECT_CHANGE, "2026-02-08T14:23:15+00:00"
            )

        content = context_file.read_text()
        assert "RECENT CHANGES:" in content
        assert "Person entered the room" in content
        assert "Person picked up coffee mug" in content

    def test_context_file_bounded_recent_changes(self, tmp_path):
        """Recent changes should be bounded to max_recent_changes."""
        context_file = tmp_path / "visual_context.txt"
        stub = self._make_pipeline_stub(context_file)
        stub._max_recent_changes = 3

        with patch("open_eyes.src.core.vision_pipeline.settings") as mock_settings:
            mock_settings.VISUAL_CONTEXT_FILE = context_file

            for i in range(5):
                obs = self._make_observation(
                    f"Observation number {i}", labels=["person"]
                )
                stub._write_visual_context(
                    obs, ObservationType.SCENE_CHANGE, f"2026-02-08T14:2{i}:00+00:00"
                )

        content = context_file.read_text()
        # First two should have been evicted
        assert "Observation number 0" not in content
        assert "Observation number 1" not in content
        # Last three should remain
        assert "Observation number 2" in content
        assert "Observation number 3" in content
        assert "Observation number 4" in content

    def test_context_file_no_objects_section_when_empty(self, tmp_path):
        """Context file should omit OBJECTS section when no labels present."""
        context_file = tmp_path / "visual_context.txt"
        stub = self._make_pipeline_stub(context_file)
        obs = self._make_observation("An empty room", labels=[])

        with patch("open_eyes.src.core.vision_pipeline.settings") as mock_settings:
            mock_settings.VISUAL_CONTEXT_FILE = context_file
            stub._write_visual_context(
                obs, ObservationType.SCENE_CHANGE, "2026-02-08T14:23:15+00:00"
            )

        content = context_file.read_text()
        assert "OBJECTS:" not in content

    def test_context_file_atomic_write(self, tmp_path):
        """Context file should be written atomically (no .tmp file left behind)."""
        context_file = tmp_path / "visual_context.txt"
        tmp_file = context_file.with_suffix(".tmp")
        stub = self._make_pipeline_stub(context_file)
        obs = self._make_observation("A room", labels=["desk"])

        with patch("open_eyes.src.core.vision_pipeline.settings") as mock_settings:
            mock_settings.VISUAL_CONTEXT_FILE = context_file
            stub._write_visual_context(
                obs, ObservationType.SCENE_CHANGE, "2026-02-08T14:23:15+00:00"
            )

        assert context_file.exists()
        assert not tmp_file.exists()
