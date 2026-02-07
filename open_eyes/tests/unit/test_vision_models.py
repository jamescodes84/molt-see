"""Unit tests for vision data models."""

import time

import pytest
from pydantic import ValidationError

from open_eyes.src.models.vision_models import (
    CapturedFrame,
    DetectedObject,
    ObservationType,
    RelevanceScore,
    SceneDescription,
    VisualObservation,
)


class TestDetectedObject:
    """Tests for DetectedObject model."""

    def test_valid_object(self):
        """Valid object should be created successfully."""
        obj = DetectedObject(
            label="person",
            confidence=0.95,
            bbox=(10, 20, 100, 200),
            area_fraction=0.15,
        )
        assert obj.label == "person"
        assert obj.confidence == 0.95

    def test_confidence_out_of_range(self):
        """Confidence above 1.0 should fail validation."""
        with pytest.raises(ValidationError):
            DetectedObject(
                label="person",
                confidence=1.5,
                bbox=(10, 20, 100, 200),
            )

    def test_negative_confidence(self):
        """Negative confidence should fail validation."""
        with pytest.raises(ValidationError):
            DetectedObject(
                label="person",
                confidence=-0.1,
                bbox=(10, 20, 100, 200),
            )

    def test_default_area_fraction(self):
        """Area fraction should default to 0.0."""
        obj = DetectedObject(
            label="cup",
            confidence=0.8,
            bbox=(0, 0, 50, 50),
        )
        assert obj.area_fraction == 0.0


class TestCapturedFrame:
    """Tests for CapturedFrame model."""

    def test_valid_frame(self):
        """Valid frame metadata should be created."""
        frame = CapturedFrame(
            frame_number=1,
            timestamp=time.time(),
            resolution=(640, 480),
            change_magnitude=0.15,
        )
        assert frame.frame_number == 1
        assert frame.resolution == (640, 480)

    def test_change_magnitude_bounds(self):
        """Change magnitude outside 0-1 should fail."""
        with pytest.raises(ValidationError):
            CapturedFrame(
                frame_number=1,
                timestamp=time.time(),
                resolution=(640, 480),
                change_magnitude=1.5,
            )

    def test_default_change_magnitude(self):
        """Change magnitude should default to 0.0."""
        frame = CapturedFrame(
            frame_number=0,
            timestamp=time.time(),
            resolution=(640, 480),
        )
        assert frame.change_magnitude == 0.0


class TestSceneDescription:
    """Tests for SceneDescription model."""

    def test_valid_scene(self):
        """Valid scene description should be created."""
        scene = SceneDescription(
            timestamp=time.time(),
            description="A person at a desk",
            object_labels=["person", "desk"],
            analysis_method="yolo",
        )
        assert "person" in scene.object_labels
        assert scene.analysis_method == "yolo"

    def test_empty_defaults(self):
        """Default lists should be empty."""
        scene = SceneDescription(
            timestamp=time.time(),
            description="Empty scene",
            analysis_method="test",
        )
        assert scene.detected_objects == []
        assert scene.object_labels == []
        assert scene.processing_time_ms == 0.0

    def test_with_detected_objects(self):
        """Scene with detected objects should work."""
        obj = DetectedObject(
            label="person", confidence=0.9, bbox=(10, 20, 100, 200)
        )
        scene = SceneDescription(
            timestamp=time.time(),
            description="Person detected",
            detected_objects=[obj],
            object_labels=["person"],
            analysis_method="yolo",
        )
        assert len(scene.detected_objects) == 1
        assert scene.detected_objects[0].label == "person"


class TestRelevanceScore:
    """Tests for RelevanceScore model."""

    def test_valid_score(self):
        """Valid relevance score should be created."""
        score = RelevanceScore(
            overall_score=0.75,
            novelty=0.9,
            context_match=0.6,
            intrinsic_interest=0.7,
            timing=0.8,
            should_report=True,
            reason="Novel observation matching conversation context",
        )
        assert score.overall_score == 0.75
        assert score.should_report is True

    def test_overall_score_bounds(self):
        """Overall score outside 0-1 should fail."""
        with pytest.raises(ValidationError):
            RelevanceScore(overall_score=1.5)

    def test_defaults(self):
        """Default values should be sensible."""
        score = RelevanceScore(overall_score=0.5)
        assert score.novelty == 0.0
        assert score.should_report is False
        assert score.reason == ""


class TestObservationType:
    """Tests for ObservationType enum."""

    def test_all_types_exist(self):
        """All expected observation types should exist."""
        expected = [
            "SCENE_CHANGE",
            "PERSON_DETECTED",
            "PERSON_LEFT",
            "OBJECT_CHANGE",
            "CONTEXT_RELEVANT",
            "ENVIRONMENT_CHANGE",
            "SAFETY_ALERT",
        ]
        for name in expected:
            assert hasattr(ObservationType, name)

    def test_string_value(self):
        """Enum values should be strings matching their names."""
        assert ObservationType.PERSON_DETECTED.value == "PERSON_DETECTED"


class TestVisualObservation:
    """Tests for VisualObservation model."""

    def test_valid_observation(self):
        """Valid observation should be created."""
        obs = VisualObservation(
            timestamp="2026-02-06T14:23:15",
            observation_type=ObservationType.PERSON_DETECTED,
            description="Someone walked into the room",
            relevance_score=0.85,
        )
        assert obs.observation_type == ObservationType.PERSON_DETECTED
        assert obs.was_reported is False

    def test_relevance_score_bounds(self):
        """Relevance score outside 0-1 should fail."""
        with pytest.raises(ValidationError):
            VisualObservation(
                timestamp="2026-02-06T14:23:15",
                observation_type=ObservationType.SCENE_CHANGE,
                description="Test",
                relevance_score=1.5,
            )

    def test_optional_scene_context(self):
        """Scene context should be optional."""
        obs = VisualObservation(
            timestamp="2026-02-06T14:23:15",
            observation_type=ObservationType.SCENE_CHANGE,
            description="Scene changed",
            relevance_score=0.6,
            scene_context="Person at desk with laptop",
        )
        assert obs.scene_context == "Person at desk with laptop"
