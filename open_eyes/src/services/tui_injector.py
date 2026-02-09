"""
Agent TUI text injector.

Sends text directly to the agent's Terminal window via AppleScript,
mirroring Molt-Speak's _send_to_openclaw_tui() mechanism.
"""

import logging
import subprocess

from ..config import settings

logger = logging.getLogger(__name__)


class TuiInjector:
    """
    Injects text into the agent's Terminal window via AppleScript.

    Uses macOS `do script` to send text to a Terminal tab matching
    a window name pattern (default: "openclaw"), without clipboard
    or window activation.
    """

    def __init__(
        self,
        window_pattern: str = settings.TARGET_WINDOW_PATTERN,
        enabled: bool = settings.ENABLE_TUI_INJECTION,
    ):
        self._pattern = window_pattern
        self._enabled = enabled

    def inject(self, text: str) -> bool:
        """
        Inject text into the agent's Terminal window.

        Returns True if injection succeeded.
        """
        if not self._enabled:
            return False

        escaped = text.replace("\\", "\\\\").replace('"', '\\"')

        applescript = f'''
tell application "Terminal"
    set foundWindow to missing value

    if "{self._pattern}" is not "" then
        repeat with w in windows
            if name of w contains "{self._pattern}" then
                set foundWindow to w
                exit repeat
            end if
        end repeat
    end if

    if foundWindow is missing value and (count of windows) > 0 then
        set foundWindow to front window
    end if

    if foundWindow is not missing value then
        do script "{escaped}" in selected tab of foundWindow
        return "ok"
    else
        error "No Terminal windows available"
    end if
end tell
'''

        try:
            result = subprocess.run(
                ["osascript", "-e", applescript],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return True
            logger.debug(f"AppleScript failed: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            logger.debug("AppleScript timed out")
        except Exception as e:
            logger.debug(f"TUI injection error: {e}")

        return False

    def send_online(self) -> None:
        """Announce vision system online."""
        ctx_path = settings.VISUAL_CONTEXT_FILE
        self.inject(
            f"[MOLT SEE: ONLINE] - Visual context at {ctx_path}"
        )

    def send_offline(self) -> None:
        """Announce vision system offline."""
        self.inject("[MOLT SEE: OFFLINE]")

    def send_observation(self, description: str) -> None:
        """Send a visual observation to the agent."""
        self.inject(f"[VISUAL] {description}")
