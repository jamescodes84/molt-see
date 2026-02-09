"""Unit tests for multi-tier observation architecture."""

import time

import pytest

from open_eyes.src.core.vision_pipeline import VisionPipeline
from open_eyes.src.models.vision_models import (
    FaceState,
    ObservationTier,
    SceneDescription,
)
from open_eyes.src.services.relevance_engine import RelevanceEngine


# ============================================================================
# Tier Classification
# ============================================================================


class TestTierClassification:
    """Tests for VisionPipeline._classify_tier."""

    def _make_observation(
        self,
        description: str,
        labels: list[str] = None,
        analysis_method: str = "cascade_local",
    ) -> SceneDescription:
        return SceneDescription(
            timestamp=time.time(),
            description=description,
            object_labels=labels or [],
            analysis_method=analysis_method,
        )

    def test_event_tier_for_person_entered(self):
        obs = self._make_observation("Someone entered the room", labels=["person"])
        assert VisionPipeline._classify_tier(obs) == ObservationTier.EVENT

    def test_event_tier_for_person_left(self):
        obs = self._make_observation("Person left the frame", labels=["person"])
        assert VisionPipeline._classify_tier(obs) == ObservationTier.EVENT

    def test_event_tier_for_picked_up(self):
        obs = self._make_observation("Person picked up a mug", labels=["person", "cup"])
        assert VisionPipeline._classify_tier(obs) == ObservationTier.EVENT

    def test_event_tier_for_safety(self):
        obs = self._make_observation("Detected: knife", labels=["knife"])
        assert VisionPipeline._classify_tier(obs) == ObservationTier.EVENT

    def test_scene_tier_for_cascade_detail(self):
        obs = self._make_observation(
            "A cozy room with a desk and guitar",
            labels=["desk", "guitar"],
            analysis_method="cascade_detail",
        )
        assert VisionPipeline._classify_tier(obs) == ObservationTier.SCENE

    def test_scene_tier_for_moondream(self):
        obs = self._make_observation(
            "A person at a desk",
            analysis_method="moondream",
        )
        assert VisionPipeline._classify_tier(obs) == ObservationTier.SCENE

    def test_scene_tier_for_vlm(self):
        obs = self._make_observation(
            "A room with warm lighting",
            analysis_method="vlm",
        )
        assert VisionPipeline._classify_tier(obs) == ObservationTier.SCENE

    def test_scene_tier_for_room_keyword(self):
        obs = self._make_observation("The room is dimly lit")
        assert VisionPipeline._classify_tier(obs) == ObservationTier.SCENE

    def test_activity_tier_default(self):
        obs = self._make_observation(
            "Detected: person, laptop",
            labels=["person", "laptop"],
        )
        assert VisionPipeline._classify_tier(obs) == ObservationTier.ACTIVITY

    def test_activity_tier_for_simple_detection(self):
        obs = self._make_observation("Detected: cup", labels=["cup"])
        assert VisionPipeline._classify_tier(obs) == ObservationTier.ACTIVITY


# ============================================================================
# Per-Tier Cooldowns
# ============================================================================


class TestPerTierCooldowns:
    """Tests for per-tier cooldown independence in RelevanceEngine."""

    def _make_observation(self, description: str = "Test") -> SceneDescription:
        return SceneDescription(
            timestamp=time.time(),
            description=description,
            object_labels=["person"],
            analysis_method="test",
        )

    def test_tier_cooldowns_are_independent(self):
        """Reporting an EVENT should not block a SCENE report."""
        engine = RelevanceEngine(
            relevance_threshold=0.0,  # always relevant
            tier_cooldowns={"EVENT": 5.0, "SCENE": 180.0, "ACTIVITY": 15.0},
        )

        # Report an event
        engine.mark_reported(tier="EVENT")

        # SCENE should still be eligible (different tier)
        assert engine._cooldown_elapsed(tier="SCENE") is True
        # EVENT should be in cooldown
        assert engine._cooldown_elapsed(tier="EVENT") is False

    def test_global_cooldown_used_without_tier(self):
        """Without tier, falls back to global cooldown."""
        engine = RelevanceEngine(cooldown_seconds=10.0)
        engine.mark_reported()

        assert engine._cooldown_elapsed() is False

    def test_tier_cooldown_respects_time(self):
        """Per-tier cooldown should respect elapsed time."""
        engine = RelevanceEngine(
            tier_cooldowns={"EVENT": 0.0},  # zero cooldown
        )
        engine.mark_reported(tier="EVENT")

        assert engine._cooldown_elapsed(tier="EVENT") is True

    def test_mark_reported_updates_both_global_and_tier(self):
        """mark_reported with tier should update both global and tier timestamps."""
        engine = RelevanceEngine(cooldown_seconds=10.0)
        engine.mark_reported(tier="ACTIVITY")

        # Global cooldown should also be set
        assert engine._cooldown_elapsed() is False
        # Tier cooldown should be set
        assert engine._tier_last_report.get("ACTIVITY", 0.0) > 0.0

    def test_unknown_tier_falls_back_to_global(self):
        """An unrecognized tier should fall back to global cooldown."""
        engine = RelevanceEngine(cooldown_seconds=0.0)
        engine.mark_reported()

        assert engine._cooldown_elapsed(tier="UNKNOWN_TIER") is True


# ============================================================================
# FaceState Model
# ============================================================================


class TestFaceStateModel:
    """Tests for the FaceState data model."""

    def test_face_state_creation(self):
        state = FaceState(
            timestamp=time.time(),
            num_faces=1,
            primary_expression="smiling",
            expressions=["smiling"],
        )
        assert state.primary_expression == "smiling"
        assert state.num_faces == 1

    def test_face_state_defaults(self):
        state = FaceState(timestamp=time.time())
        assert state.num_faces == 0
        assert state.primary_expression == "none"
        assert state.expressions == []
        assert state.blendshapes is None

    def test_face_state_with_blendshapes(self):
        bs = {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.7}
        state = FaceState(
            timestamp=time.time(),
            num_faces=1,
            primary_expression="smiling",
            expressions=["smiling"],
            blendshapes=bs,
        )
        assert state.blendshapes["mouthSmileLeft"] == 0.8


# ============================================================================
# ObservationTier on SceneDescription
# ============================================================================


class TestSceneDescriptionTier:
    """Tests for the tier field on SceneDescription."""

    def test_tier_defaults_to_none(self):
        obs = SceneDescription(
            timestamp=time.time(),
            description="test",
            analysis_method="test",
        )
        assert obs.tier is None

    def test_tier_can_be_set(self):
        obs = SceneDescription(
            timestamp=time.time(),
            description="test",
            analysis_method="test",
            tier=ObservationTier.SCENE,
        )
        assert obs.tier == ObservationTier.SCENE

    def test_tier_can_be_mutated(self):
        obs = SceneDescription(
            timestamp=time.time(),
            description="test",
            analysis_method="test",
        )
        obs.tier = ObservationTier.EVENT
        assert obs.tier == ObservationTier.EVENT


# ============================================================================
# Build Face State
# ============================================================================


class TestBuildFaceState:
    """Tests for VisionPipeline._build_face_state."""

    def test_build_from_empty_result(self):
        result = SceneDescription(
            timestamp=time.time(),
            description="No faces detected",
            detected_objects=[],
            object_labels=[],
            analysis_method="mediapipe_face",
        )
        state = VisionPipeline._build_face_state(result)
        assert state.num_faces == 0
        assert state.primary_expression == "none"
        assert state.blendshapes is None
