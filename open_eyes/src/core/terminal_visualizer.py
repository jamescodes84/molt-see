"""
Terminal visualizer for OpenClaw Eyes.

Displays live pipeline status in the terminal.
"""

import logging
import os
import time
from typing import Optional

from .state_manager import StateManager, PipelineStatus

logger = logging.getLogger(__name__)


class TerminalVisualizer:
    """
    Displays live vision pipeline status in the terminal.

    Shows current pipeline state, frame count, observations made,
    and scene change magnitude.
    """

    STATUS_ICONS = {
        PipelineStatus.IDLE: "  ",
        PipelineStatus.CAPTURING: "  ",
        PipelineStatus.ANALYZING: "  ",
        PipelineStatus.OBSERVING: "  ",
        PipelineStatus.PAUSED: "  ",
        PipelineStatus.ERROR: "  ",
        PipelineStatus.STOPPED: "  ",
    }

    def __init__(self, state_manager: StateManager):
        """
        Initialize terminal visualizer.

        Args:
            state_manager: Pipeline state manager to read from.
        """
        self.state_manager = state_manager
        self._last_update = 0.0

    def update(self) -> None:
        """Update the terminal display."""
        state = self.state_manager.state
        icon = self.STATUS_ICONS.get(state.status, "")

        # Build status line
        parts = [
            f"{icon} {state.status.value}",
            f"Frames: {state.frame_count}",
            f"Reported: {state.observations_reported}",
            f"Change: {state.last_change_magnitude:.2f}",
        ]

        if state.current_description:
            # Truncate long descriptions
            desc = state.current_description[:60]
            if len(state.current_description) > 60:
                desc += "..."
            parts.append(f"| {desc}")

        if state.error_message:
            parts.append(f"ERR: {state.error_message[:40]}")

        line = " | ".join(parts)

        # Clear line and print
        terminal_width = os.get_terminal_size().columns if hasattr(os, "get_terminal_size") else 80
        print(f"\r{line:<{terminal_width}}", end="", flush=True)
