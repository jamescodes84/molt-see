"""
Conversation context reader.

Reads Molt-Speak runtime files to understand conversation context
for informing relevance scoring. Monitors transcriptions.txt and
speech_output.txt for recent conversation history.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from ..config import settings

logger = logging.getLogger(__name__)


class ConversationContextReader:
    """
    Reads Molt-Speak runtime files to understand conversation context.

    Monitors transcriptions.txt and speech_output.txt for recent
    conversation history to inform relevance scoring.
    """

    def __init__(
        self,
        molt_speak_runtime_dir: Optional[Path] = settings.MOLT_SPEAK_RUNTIME_DIR,
        context_window_lines: int = settings.CONTEXT_WINDOW_LINES,
    ):
        """
        Initialize context reader.

        Args:
            molt_speak_runtime_dir: Path to Molt-Speak's runtime directory.
            context_window_lines: Number of recent lines to track.
        """
        self.molt_speak_dir = molt_speak_runtime_dir
        self.context_window_lines = context_window_lines

        # Molt-Speak file paths
        self._transcriptions_file: Optional[Path] = None
        self._speech_output_file: Optional[Path] = None
        self._mouth_status_file: Optional[Path] = None
        self._ears_status_file: Optional[Path] = None

        # File read cache: {path_str: (mtime, content)}
        self._read_cache: dict[str, tuple[float, str]] = {}

        if self.molt_speak_dir:
            self._transcriptions_file = self.molt_speak_dir / "transcriptions.txt"
            self._speech_output_file = self.molt_speak_dir / "speech_output.txt"
            self._mouth_status_file = self.molt_speak_dir / "mouth_status.txt"
            self._ears_status_file = self.molt_speak_dir / "ears_status.txt"

    @property
    def is_connected(self) -> bool:
        """Check if Molt-Speak runtime directory exists."""
        return self.molt_speak_dir is not None and self.molt_speak_dir.exists()

    def get_recent_context(self, max_lines: int = 10) -> str:
        """
        Get recent conversation as a single string.

        Combines recent transcriptions (user speech) and speech output
        (agent responses) into a single context string.

        Args:
            max_lines: Maximum number of lines to include.

        Returns:
            Recent conversation text, or empty string if unavailable.
        """
        if not self.is_connected:
            return ""

        lines = []

        # Read user speech (transcriptions)
        user_lines = self._read_tail(
            self._transcriptions_file, max_lines // 2
        )
        for line in user_lines:
            lines.append(f"[user] {line}")

        # Read agent speech
        agent_lines = self._read_tail(
            self._speech_output_file, max_lines // 2
        )
        for line in agent_lines:
            lines.append(f"[agent] {line}")

        return "\n".join(lines[-max_lines:])

    def get_conversation_topics(self) -> list[str]:
        """
        Extract likely topics from recent conversation.

        Uses simple keyword extraction — pulls out nouns and significant
        words from recent conversation lines.

        Returns:
            List of topic keywords.
        """
        context = self.get_recent_context(max_lines=10)
        if not context:
            return []

        # Simple keyword extraction: split words, filter short/common
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "each", "every", "both", "few", "more", "most", "other",
            "some", "such", "no", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "because", "but", "and", "or",
            "if", "while", "about", "up", "it", "its", "this", "that",
            "these", "those", "i", "you", "he", "she", "we", "they",
            "me", "him", "her", "us", "them", "my", "your", "his",
            "our", "their", "what", "which", "who", "whom",
            "user", "agent",
        }

        words = context.lower().split()
        topics = []
        for word in words:
            # Strip punctuation
            cleaned = "".join(c for c in word if c.isalnum())
            if cleaned and len(cleaned) > 2 and cleaned not in stopwords:
                if cleaned not in topics:
                    topics.append(cleaned)

        return topics[:20]  # Cap at 20 topics

    def is_conversation_active(self) -> bool:
        """
        Check if there is an active conversation (recent speech).

        Returns:
            True if there has been recent speech activity.
        """
        if not self.is_connected:
            return False

        # Check if transcriptions file has been modified recently (30s)
        if self._transcriptions_file and self._transcriptions_file.exists():
            mtime = self._transcriptions_file.stat().st_mtime
            if time.time() - mtime < 30.0:
                return True

        return False

    def is_agent_speaking(self) -> bool:
        """
        Check if the agent is currently speaking.

        Returns:
            True if mouth_status.txt indicates SPEAKING.
        """
        return self._read_status(self._mouth_status_file) == "SPEAKING"

    def is_user_speaking(self) -> bool:
        """
        Check if the user is currently speaking.

        Returns:
            True if ears_status.txt indicates SPEECH_DETECTED.
        """
        return self._read_status(self._ears_status_file) == "SPEECH_DETECTED"

    def is_good_time_to_interject(self) -> bool:
        """
        Check if the conversation is in a pause (good time to mention something).

        Returns:
            True if neither the agent nor user is actively speaking.
        """
        if not self.is_connected:
            return True  # If not connected, always allow observations

        return not self.is_agent_speaking() and not self.is_user_speaking()

    def _read_cached(self, file_path: Path) -> str:
        """Read file contents, returning cached value if mtime unchanged."""
        key = str(file_path)
        try:
            mtime = file_path.stat().st_mtime
            cached = self._read_cache.get(key)
            if cached and cached[0] == mtime:
                return cached[1]
            content = file_path.read_text()
            self._read_cache[key] = (mtime, content)
            return content
        except Exception:
            return ""

    def _read_status(self, status_file: Optional[Path]) -> Optional[str]:
        """Read status from a Molt-Speak status file."""
        if not status_file or not status_file.exists():
            return None

        try:
            content = self._read_cached(status_file).strip()
            if "|" in content:
                parts = content.split("|")
                if len(parts) >= 2:
                    return parts[1]
            return content
        except Exception:
            return None

    def _read_tail(
        self, file_path: Optional[Path], max_lines: int
    ) -> list[str]:
        """Read the last N lines from a file."""
        if not file_path or not file_path.exists():
            return []

        try:
            content = self._read_cached(file_path)
            lines = [
                line.strip()
                for line in content.splitlines()
                if line.strip()
            ]
            return lines[-max_lines:]
        except Exception:
            return []
