"""
Relevance engine.

The core intelligence layer that determines if a visual observation
is worth surfacing to the agent given conversation context, novelty,
intrinsic interest, and timing.

The goal: "like how a human might casually reference something
they've just noticed."
"""

import logging
import time
from typing import Optional

from ..config import settings
from ..models.vision_models import RelevanceScore, SceneDescription
from .context_reader import ConversationContextReader
from .vision_memory import VisionMemory

logger = logging.getLogger(__name__)


class RelevanceEngine:
    """
    Context-aware relevance scoring for visual observations.

    Evaluates whether a visual observation is worth reporting given:
    1. Current conversation topic/context
    2. What has already been reported (no repeats)
    3. How "interesting" the observation is intrinsically
    4. Whether now is an appropriate time to interject
    """

    # Scoring weights
    WEIGHTS = {
        "novelty": 0.30,
        "context_match": 0.25,
        "intrinsic_interest": 0.25,
        "timing": 0.20,
    }

    # Object classes ranked by inherent interest
    HIGH_INTEREST = {"person", "face", "cat", "dog", "bird", "knife", "scissors"}
    MEDIUM_INTEREST = {
        "cell phone", "laptop", "book", "cup", "bottle",
        "backpack", "handbag", "umbrella",
    }
    LOW_INTEREST = {
        "chair", "couch", "bed", "dining table", "desk",
        "tv", "keyboard", "mouse", "monitor",
    }

    def __init__(
        self,
        relevance_threshold: float = settings.RELEVANCE_THRESHOLD,
        cooldown_seconds: float = settings.OBSERVATION_COOLDOWN,
        context_reader: Optional[ConversationContextReader] = None,
        vision_memory: Optional[VisionMemory] = None,
    ):
        """
        Initialize relevance engine.

        Args:
            relevance_threshold: Minimum score to report (0-1).
            cooldown_seconds: Minimum seconds between reports.
            context_reader: For reading conversation context.
            vision_memory: For checking novelty/duplicates.
        """
        self.relevance_threshold = relevance_threshold
        self.cooldown_seconds = cooldown_seconds
        self.context_reader = context_reader
        self.vision_memory = vision_memory

        self._last_report_time: float = 0.0

    def evaluate(
        self,
        observation: SceneDescription,
        conversation_context: Optional[str] = None,
    ) -> RelevanceScore:
        """
        Score an observation's relevance.

        Args:
            observation: Scene description to evaluate.
            conversation_context: Optional override for conversation context.

        Returns:
            RelevanceScore with overall score and component breakdown.
        """
        # Get conversation context if not provided
        if conversation_context is None and self.context_reader:
            conversation_context = self.context_reader.get_recent_context()

        # Score each factor
        novelty = self._score_novelty(observation)
        context_match = self._score_context_match(
            observation, conversation_context or ""
        )
        intrinsic = self._score_intrinsic_interest(observation)
        timing = self._score_timing()

        # Weighted combination
        overall = (
            self.WEIGHTS["novelty"] * novelty
            + self.WEIGHTS["context_match"] * context_match
            + self.WEIGHTS["intrinsic_interest"] * intrinsic
            + self.WEIGHTS["timing"] * timing
        )

        # Check for high-priority overrides
        is_override = self._is_high_priority(observation)
        should_report = (
            (overall >= self.relevance_threshold) or is_override
        ) and self._cooldown_elapsed()

        # Build reason
        reason = self._build_reason(
            novelty, context_match, intrinsic, timing, is_override
        )

        return RelevanceScore(
            overall_score=round(overall, 3),
            novelty=round(novelty, 3),
            context_match=round(context_match, 3),
            intrinsic_interest=round(intrinsic, 3),
            timing=round(timing, 3),
            should_report=should_report,
            reason=reason,
        )

    def mark_reported(self) -> None:
        """Record that a report was just made (for cooldown tracking)."""
        self._last_report_time = time.time()

    def _score_novelty(self, observation: SceneDescription) -> float:
        """
        How different is this from recent observations?

        High novelty = first time seeing this. Low = seen it before.
        """
        if self.vision_memory is None:
            return 0.8  # Default to fairly novel if no memory

        # Check if this was recently reported
        if self.vision_memory.was_recently_reported(observation):
            return 0.1

        # Check if it's a duplicate of any recent observation
        if self.vision_memory.is_duplicate(observation):
            return 0.2

        # Check label overlap with recent observations
        recent = self.vision_memory.get_recent_observations(10)
        if not recent:
            return 1.0  # First observation ever

        recent_labels = set()
        for obs in recent:
            recent_labels.update(obs.get("labels", []))

        current_labels = set(observation.object_labels)
        if current_labels and recent_labels:
            new_labels = current_labels - recent_labels
            if new_labels:
                return 0.9  # Something new appeared
            overlap = len(current_labels & recent_labels) / len(
                current_labels | recent_labels
            )
            return max(0.3, 1.0 - overlap)

        return 0.7  # Default moderate novelty

    def _score_context_match(
        self, observation: SceneDescription, context: str
    ) -> float:
        """
        Does the observation relate to the current conversation?

        Uses keyword overlap between observation and conversation.
        """
        if not context:
            return 0.5  # No Molt-Speak connected = neutral default

        # Tokenize both
        obs_words = set(observation.description.lower().split())
        ctx_words = set(context.lower().split())

        # Also include object labels
        label_words = set()
        for label in observation.object_labels:
            label_words.update(label.lower().split())
        obs_words.update(label_words)

        # Filter stopwords (minimal set for speed)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "to", "of", "in",
            "for", "on", "with", "at", "by", "and", "or", "but", "it",
            "this", "that", "no", "not", "i", "you", "we", "they",
            "detected:", "user", "agent",
        }
        obs_words -= stopwords
        ctx_words -= stopwords

        if not obs_words or not ctx_words:
            return 0.2

        # Calculate overlap
        overlap = obs_words & ctx_words
        if overlap:
            score = min(len(overlap) / min(len(obs_words), 5), 1.0)
            return max(0.4, score)

        return 0.2  # No keyword overlap

    def _score_intrinsic_interest(
        self, observation: SceneDescription
    ) -> float:
        """
        How inherently noteworthy is this observation?

        People appearing/leaving > animate objects > object changes > static.
        """
        labels = set(observation.object_labels)
        # Normalize compound labels (e.g., "face: smiling" -> also add "face")
        base_labels = set()
        for label in labels:
            base_labels.add(label)
            if ":" in label:
                base_labels.add(label.split(":")[0].strip())
        labels = base_labels

        # Check for high-interest items
        high_matches = labels & self.HIGH_INTEREST
        if high_matches:
            if "person" in high_matches:
                return 0.95  # People are almost always interesting
            return 0.85  # Animals, safety items

        # Check for medium-interest items
        medium_matches = labels & self.MEDIUM_INTEREST
        if medium_matches:
            return 0.6

        # Check for low-interest items
        low_matches = labels & self.LOW_INTEREST
        if low_matches:
            return 0.3

        # Check description for interesting keywords
        desc_lower = observation.description.lower()
        interesting_keywords = [
            "someone", "walked", "entered", "left", "picked up",
            "put down", "moved", "changed", "appeared", "disappeared",
        ]
        for keyword in interesting_keywords:
            if keyword in desc_lower:
                return 0.8

        return 0.4  # Default moderate interest

    def _score_timing(self) -> float:
        """
        Is now a good time to mention something?

        High score during conversation pauses, low during active speech.
        """
        if self.context_reader is None:
            return 0.6  # Default moderate timing

        if self.context_reader.is_good_time_to_interject():
            return 0.9  # Great timing — nobody speaking

        if self.context_reader.is_agent_speaking():
            return 0.2  # Bad timing — agent is speaking

        if self.context_reader.is_user_speaking():
            return 0.3  # Bad timing — user is speaking

        return 0.6  # Default

    def _is_high_priority(self, observation: SceneDescription) -> bool:
        """Check for high-priority events that bypass normal scoring."""
        desc_lower = observation.description.lower()

        # New person appearing is always worth mentioning
        priority_triggers = [
            "someone walked in",
            "person entered",
            "someone just",
            "new person",
        ]
        for trigger in priority_triggers:
            if trigger in desc_lower:
                return True

        # Safety alerts bypass scoring
        if any(label in {"knife", "scissors"} for label in observation.object_labels):
            return True

        return False

    def _cooldown_elapsed(self) -> bool:
        """Check if enough time has passed since last report."""
        return (time.time() - self._last_report_time) >= self.cooldown_seconds

    def _build_reason(
        self,
        novelty: float,
        context_match: float,
        intrinsic: float,
        timing: float,
        is_override: bool,
    ) -> str:
        """Build a human-readable reason for the score."""
        parts = []
        if is_override:
            parts.append("high-priority override")
        if novelty >= 0.8:
            parts.append("novel observation")
        elif novelty <= 0.3:
            parts.append("similar to recent")
        if context_match >= 0.6:
            parts.append("matches conversation")
        if intrinsic >= 0.8:
            parts.append("inherently interesting")
        if timing >= 0.8:
            parts.append("good timing")
        elif timing <= 0.3:
            parts.append("bad timing (active speech)")

        return "; ".join(parts) if parts else "moderate relevance"
