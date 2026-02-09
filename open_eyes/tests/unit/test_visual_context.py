"""Unit tests for visual context file output."""

import threading
import time
from unittest.mock import patch

import pytest

from open_eyes.src.core.vision_pipeline import VisionPipeline
from open_eyes.src.models.vision_models import (
    ObservationTier,
    ObservationType,
    SceneDescription,
)


class TestVisualContext:
    """Tests for the tiered visual context file writing."""

    def _make_observation(
        self,
        description: str,
        labels: list[str] = None,
        analysis_method: str = "test",
        tier: ObservationTier = None,
    ) -> SceneDescription:
        """Helper to create a SceneDescription."""
        return SceneDescription(
            timestamp=time.time(),
            description=description,
            object_labels=labels or [],
            analysis_method=analysis_method,
            tier=tier,
        )

    def _make_pipeline_stub(self, context_file):
        """Create a minimal object with the attributes the tiered output needs."""
        stub = object.__new__(VisionPipeline)
        # Per-tier state
        stub._scene_description = ""
        stub._scene_objects = []
        stub._current_activity = ""
        stub._recent_events = []
        stub._max_recent_events = 5
        # Face state
        stub._face_state = None
        stub._face_lock = threading.Lock()
        stub._last_face_expression = "none"
        # Pose state
        stub._pose_state = None
        stub._pose_lock = threading.Lock()
        stub._last_gesture = "none"
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
            stub._update_tier_state(
                obs, ObservationTier.ACTIVITY, "2026-02-08T14:23:15+00:00"
            )
            stub._rebuild_visual_context()

        assert context_file.exists()

    def test_context_file_scene_section(self, tmp_path):
        """SCENE tier should populate the SCENE section."""
        context_file = tmp_path / "visual_context.txt"
        stub = self._make_pipeline_stub(context_file)
        obs = self._make_observation(
            "A man in a hoodie at a desk with a guitar on the wall",
            labels=["person", "desk", "guitar"],
        )

        with patch("open_eyes.src.core.vision_pipeline.settings") as mock_settings:
            mock_settings.VISUAL_CONTEXT_FILE = context_file
            stub._update_tier_state(
                obs, ObservationTier.SCENE, "2026-02-08T14:23:15+00:00"
            )
            stub._rebuild_visual_context()

        content = context_file.read_text()
        assert "SCENE:" in content
        assert "A man in a hoodie at a desk with a guitar on the wall" in content

    def test_context_file_contains_objects(self, tmp_path):
        """SCENE tier should list detected objects."""
        context_file = tmp_path / "visual_context.txt"
        stub = self._make_pipeline_stub(context_file)
        obs = self._make_observation(
            "A person at a desk",
            labels=["person", "desk", "laptop"],
        )

        with patch("open_eyes.src.core.vision_pipeline.settings") as mock_settings:
            mock_settings.VISUAL_CONTEXT_FILE = context_file
            stub._update_tier_state(
                obs, ObservationTier.SCENE, "2026-02-08T14:23:15+00:00"
            )
            stub._rebuild_visual_context()

        content = context_file.read_text()
        assert "Objects:" in content
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
            stub._update_tier_state(
                obs, ObservationTier.ACTIVITY, "2026-02-08T14:23:15+00:00"
            )
            stub._rebuild_visual_context()

        content = context_file.read_text()
        assert "[Last updated:" in content

    def test_context_file_activity_section(self, tmp_path):
        """ACTIVITY tier should populate the ACTIVITY section."""
        context_file = tmp_path / "visual_context.txt"
        stub = self._make_pipeline_stub(context_file)
        obs = self._make_observation(
            "Person sitting at desk typing on laptop",
            labels=["person", "laptop"],
        )

        with patch("open_eyes.src.core.vision_pipeline.settings") as mock_settings:
            mock_settings.VISUAL_CONTEXT_FILE = context_file
            stub._update_tier_state(
                obs, ObservationTier.ACTIVITY, "2026-02-08T14:23:15+00:00"
            )
            stub._rebuild_visual_context()

        content = context_file.read_text()
        assert "ACTIVITY:" in content
        assert "Person sitting at desk typing on laptop" in content

    def test_context_file_recent_events_accumulate(self, tmp_path):
        """Multiple EVENT observations should accumulate in RECENT EVENTS."""
        context_file = tmp_path / "visual_context.txt"
        stub = self._make_pipeline_stub(context_file)

        with patch("open_eyes.src.core.vision_pipeline.settings") as mock_settings:
            mock_settings.VISUAL_CONTEXT_FILE = context_file

            obs1 = self._make_observation("Person entered the room", labels=["person"])
            stub._update_tier_state(
                obs1, ObservationTier.EVENT, "2026-02-08T14:20:00+00:00"
            )

            obs2 = self._make_observation("Person picked up coffee mug", labels=["person", "cup"])
            stub._update_tier_state(
                obs2, ObservationTier.EVENT, "2026-02-08T14:23:15+00:00"
            )
            stub._rebuild_visual_context()

        content = context_file.read_text()
        assert "RECENT EVENTS:" in content
        assert "Person entered the room" in content
        assert "Person picked up coffee mug" in content

    def test_context_file_bounded_recent_events(self, tmp_path):
        """Recent events should be bounded to max_recent_events."""
        context_file = tmp_path / "visual_context.txt"
        stub = self._make_pipeline_stub(context_file)
        stub._max_recent_events = 3

        with patch("open_eyes.src.core.vision_pipeline.settings") as mock_settings:
            mock_settings.VISUAL_CONTEXT_FILE = context_file

            for i in range(5):
                obs = self._make_observation(
                    f"Observation number {i}", labels=["person"]
                )
                stub._update_tier_state(
                    obs, ObservationTier.EVENT, f"2026-02-08T14:2{i}:00+00:00"
                )
            stub._rebuild_visual_context()

        content = context_file.read_text()
        # First two should have been evicted
        assert "Observation number 0" not in content
        assert "Observation number 1" not in content
        # Last three should remain
        assert "Observation number 2" in content
        assert "Observation number 3" in content
        assert "Observation number 4" in content

    def test_context_file_no_objects_when_activity(self, tmp_path):
        """ACTIVITY tier should not add Objects line to SCENE section."""
        context_file = tmp_path / "visual_context.txt"
        stub = self._make_pipeline_stub(context_file)
        obs = self._make_observation("Typing on keyboard", labels=["keyboard"])

        with patch("open_eyes.src.core.vision_pipeline.settings") as mock_settings:
            mock_settings.VISUAL_CONTEXT_FILE = context_file
            stub._update_tier_state(
                obs, ObservationTier.ACTIVITY, "2026-02-08T14:23:15+00:00"
            )
            stub._rebuild_visual_context()

        content = context_file.read_text()
        # No SCENE section means no Objects line
        assert "Objects:" not in content

    def test_context_file_atomic_write(self, tmp_path):
        """Context file should be written atomically (no .tmp file left behind)."""
        context_file = tmp_path / "visual_context.txt"
        tmp_file = context_file.with_suffix(".tmp")
        stub = self._make_pipeline_stub(context_file)
        obs = self._make_observation("A room", labels=["desk"])

        with patch("open_eyes.src.core.vision_pipeline.settings") as mock_settings:
            mock_settings.VISUAL_CONTEXT_FILE = context_file
            stub._update_tier_state(
                obs, ObservationTier.SCENE, "2026-02-08T14:23:15+00:00"
            )
            stub._rebuild_visual_context()

        assert context_file.exists()
        assert not tmp_file.exists()

    def test_context_file_all_tiers(self, tmp_path):
        """All tiers should appear in the output when populated."""
        context_file = tmp_path / "visual_context.txt"
        stub = self._make_pipeline_stub(context_file)

        with patch("open_eyes.src.core.vision_pipeline.settings") as mock_settings:
            mock_settings.VISUAL_CONTEXT_FILE = context_file

            # Scene tier
            stub._update_tier_state(
                self._make_observation(
                    "A living room with a desk and guitar",
                    labels=["desk", "guitar", "lamp"],
                ),
                ObservationTier.SCENE,
                "2026-02-08T14:20:00+00:00",
            )
            # Activity tier
            stub._update_tier_state(
                self._make_observation(
                    "Person sitting at desk typing",
                    labels=["person", "laptop"],
                ),
                ObservationTier.ACTIVITY,
                "2026-02-08T14:21:00+00:00",
            )
            # Event tier
            stub._update_tier_state(
                self._make_observation(
                    "Person picked up coffee mug",
                    labels=["person", "cup"],
                ),
                ObservationTier.EVENT,
                "2026-02-08T14:22:00+00:00",
            )

            stub._rebuild_visual_context()

        content = context_file.read_text()
        assert "SCENE:" in content
        assert "A living room with a desk and guitar" in content
        assert "Objects:" in content
        assert "ACTIVITY:" in content
        assert "Person sitting at desk typing" in content
        assert "RECENT EVENTS:" in content
        assert "Person picked up coffee mug" in content
