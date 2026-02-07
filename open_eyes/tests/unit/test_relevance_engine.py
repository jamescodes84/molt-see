"""Unit tests for relevance engine."""

import time

import pytest

from open_eyes.src.models.vision_models import SceneDescription
from open_eyes.src.services.relevance_engine import RelevanceEngine
from open_eyes.src.services.vision_memory import VisionMemory


class TestRelevanceEngine:
    """Tests for RelevanceEngine."""

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

    def test_novel_observation_scores_high(self):
        """First-time observation should score high on novelty."""
        memory = VisionMemory()
        engine = RelevanceEngine(vision_memory=memory)
        obs = self._make_observation(
            "A person sat down at the desk",
            labels=["person", "desk"],
        )
        score = engine.evaluate(obs)
        assert score.novelty >= 0.7

    def test_duplicate_observation_scores_low(self):
        """Repeated observation should score low on novelty."""
        memory = VisionMemory()
        obs = self._make_observation(
            "A person sitting at a desk",
            labels=["person", "desk"],
        )
        memory.record_observation(obs)
        memory.record_reported(obs)

        engine = RelevanceEngine(vision_memory=memory)
        similar = self._make_observation(
            "A person sitting at the desk",
            labels=["person", "desk"],
        )
        score = engine.evaluate(similar)
        assert score.novelty < 0.3

    def test_context_relevant_observation(self):
        """Observation matching conversation should score high on context."""
        engine = RelevanceEngine()
        obs = self._make_observation(
            "A coffee mug on the desk", labels=["cup"]
        )
        score = engine.evaluate(
            obs, conversation_context="Let's take a coffee break soon"
        )
        assert score.context_match >= 0.4

    def test_no_context_moderate_score(self):
        """No conversation context should give moderate context score."""
        engine = RelevanceEngine()
        obs = self._make_observation("A cat on the desk", labels=["cat"])
        score = engine.evaluate(obs, conversation_context="")
        assert 0.4 <= score.context_match <= 0.6

    def test_person_high_intrinsic_interest(self):
        """Person detection should have high intrinsic interest."""
        engine = RelevanceEngine()
        obs = self._make_observation(
            "A person appeared", labels=["person"]
        )
        score = engine.evaluate(obs)
        assert score.intrinsic_interest >= 0.9

    def test_chair_low_intrinsic_interest(self):
        """Static furniture should have low intrinsic interest."""
        engine = RelevanceEngine()
        obs = self._make_observation("A chair", labels=["chair"])
        score = engine.evaluate(obs)
        assert score.intrinsic_interest <= 0.4

    def test_cooldown_prevents_rapid_observations(self):
        """Observations within cooldown should not be reported."""
        engine = RelevanceEngine(
            cooldown_seconds=30.0, relevance_threshold=0.0
        )
        obs = self._make_observation("Person waving", labels=["person"])

        # First evaluation — cooldown not started
        score1 = engine.evaluate(obs)
        assert score1.should_report is True

        # Mark as reported
        engine.mark_reported()

        # Immediate second evaluation — within cooldown
        score2 = engine.evaluate(obs)
        assert score2.should_report is False

    def test_cooldown_elapsed(self):
        """After cooldown passes, reporting should be allowed."""
        engine = RelevanceEngine(
            cooldown_seconds=0.1, relevance_threshold=0.0
        )
        engine.mark_reported()
        time.sleep(0.15)

        obs = self._make_observation("Person waving", labels=["person"])
        score = engine.evaluate(obs)
        assert score.should_report is True

    def test_high_priority_override(self):
        """High-priority events should bypass normal scoring."""
        engine = RelevanceEngine(relevance_threshold=0.99)
        obs = self._make_observation(
            "Someone walked in through the door",
            labels=["person"],
        )
        score = engine.evaluate(obs)
        assert score.should_report is True
        assert "high-priority" in score.reason

    def test_overall_score_is_weighted(self):
        """Overall score should be weighted combination of factors."""
        engine = RelevanceEngine()
        obs = self._make_observation("Test observation", labels=[])
        score = engine.evaluate(obs)

        # Verify overall is in valid range
        assert 0.0 <= score.overall_score <= 1.0

    def test_reason_string(self):
        """Score should include a human-readable reason."""
        memory = VisionMemory()
        engine = RelevanceEngine(vision_memory=memory)
        obs = self._make_observation("A person appeared", labels=["person"])
        score = engine.evaluate(obs)
        assert len(score.reason) > 0


class TestRelevanceEngineNoMemory:
    """Tests for RelevanceEngine without VisionMemory."""

    def test_works_without_memory(self):
        """Engine should work without VisionMemory."""
        engine = RelevanceEngine(vision_memory=None)
        obs = SceneDescription(
            timestamp=time.time(),
            description="Test scene",
            object_labels=["person"],
            analysis_method="test",
        )
        score = engine.evaluate(obs)
        assert 0.0 <= score.overall_score <= 1.0
