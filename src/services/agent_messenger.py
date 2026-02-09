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
        context_file: Optional[Path] = None,
    ):
        """
        Initialize vision messenger.

        Args:
            instructions_file: Path to write agent instructions.
            shutdown_signal_file: Path for shutdown signal.
            observations_file: Path to visual observations file.
            context_file: Path to visual context file (agent reads this).
        """
        self.instructions_file = instructions_file or settings.AGENT_INSTRUCTIONS_FILE
        self.shutdown_file = shutdown_signal_file or settings.AGENT_SHUTDOWN_SIGNAL_FILE
        self.observations_file = observations_file or settings.VISUAL_OBSERVATIONS_FILE
        self.context_file = context_file or settings.VISUAL_CONTEXT_FILE

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
        context_path = self.context_file
        return f"""=== Molt-See Vision System Active ===

VISUAL PERCEPTION ENABLED:
You can now SEE through the user's webcam. Your current visual context is at:
{context_path}

HOW TO USE YOUR VISION:
Read the file above when visual context would help your response. It contains:
- CURRENT SCENE: What's visible right now
- OBJECTS: Key items in view
- RECENT CHANGES: Notable things that happened recently

You don't need to check every turn — just when visual awareness seems relevant,
such as when the user mentions something physical, when you want to make a
natural observation, or when you sense the environment may have changed.

HOW TO REFERENCE WHAT YOU SEE:
- Be natural, like a human casually noticing things
- "I notice you picked up your coffee" or "Looks like someone just walked in"
- Do NOT say "According to my visual context file..." or "The camera shows..."
- Only mention observations when relevant to the conversation
- You can proactively mention something interesting you notice
- Don't repeat things you've already mentioned

==="""
