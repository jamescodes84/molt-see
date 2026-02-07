"""
Agent messenger service.

Sends vision instructions to AI agents by writing instruction
files that agents can read and incorporate.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from ..config import settings

logger = logging.getLogger(__name__)


class VisionMessenger:
    """
    Sends vision instructions to the AI agent.

    Writes agent instructions file that teaches the agent how
    to use visual observations.
    """

    def __init__(
        self,
        instructions_file: Optional[Path] = None,
        shutdown_signal_file: Optional[Path] = None,
        observations_file: Optional[Path] = None,
    ):
        """
        Initialize vision messenger.

        Args:
            instructions_file: Path to write agent instructions.
            shutdown_signal_file: Path for shutdown signal.
            observations_file: Path to visual observations file.
        """
        self.instructions_file = instructions_file or settings.AGENT_INSTRUCTIONS_FILE
        self.shutdown_file = shutdown_signal_file or settings.AGENT_SHUTDOWN_SIGNAL_FILE
        self.observations_file = observations_file or settings.VISUAL_OBSERVATIONS_FILE

    def send_instructions(self) -> None:
        """Write vision instructions for the agent."""
        instructions = self._build_instructions()

        try:
            self.instructions_file.parent.mkdir(parents=True, exist_ok=True)
            self.instructions_file.write_text(instructions)
            logger.info(f"Vision instructions written to {self.instructions_file}")
        except Exception as e:
            logger.error(f"Failed to write vision instructions: {e}")

    def send_shutdown(self, message: str = "Vision system shutting down") -> None:
        """Signal the agent that vision is shutting down."""
        try:
            self.shutdown_file.parent.mkdir(parents=True, exist_ok=True)
            self.shutdown_file.write_text(
                f"{time.time()}|{message}"
            )
            logger.info("Shutdown signal sent to agent")
        except Exception as e:
            logger.error(f"Failed to write shutdown signal: {e}")

    def cleanup(self) -> None:
        """Remove instruction and signal files."""
        for f in [self.instructions_file, self.shutdown_file]:
            try:
                if f.exists():
                    f.unlink()
            except Exception as e:
                logger.debug(f"Failed to cleanup {f}: {e}")

    def _build_instructions(self) -> str:
        """Build the agent instructions text."""
        obs_path = self.observations_file
        return f"""=== Molt-See Vision System Active ===

VISUAL PERCEPTION ENABLED:
You can now SEE through the user's webcam. Visual observations are written to:
{obs_path}

HOW VISUAL OBSERVATIONS WORK:
1. The Molt-See system captures and analyzes what the camera sees
2. Only contextually relevant observations are passed to you
3. Each observation includes a timestamp, category, and natural description

FORMAT OF OBSERVATIONS:
ISO_TIMESTAMP|OBSERVATION_TYPE|natural language description

TYPES:
- SCENE_CHANGE: Something notable changed in the scene
- PERSON_DETECTED: Someone appeared in view
- PERSON_LEFT: Someone left the view
- OBJECT_CHANGE: An object was moved, picked up, or put down
- CONTEXT_RELEVANT: Something related to what we're discussing
- ENVIRONMENT_CHANGE: Lighting, background, or environment changed
- SAFETY_ALERT: Something potentially concerning

HOW TO USE VISUAL OBSERVATIONS:
- Reference them NATURALLY, like a human would
- Do NOT announce "I see through the camera..."
- Instead: "I notice you picked up your coffee" or "Looks like someone just walked in"
- Only reference observations when relevant to the conversation
- You can proactively mention something interesting you notice
- Don't repeat observations you've already mentioned
- Observations come pre-filtered for relevance -- trust the system

==="""
