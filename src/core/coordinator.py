"""
Vision coordinator.

Manages the lifecycle of the OpenClaw Eyes vision subsystem,
sends agent instructions, and coordinates with Molt-Speak.
"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from ..config import settings
from ..services.agent_messenger import VisionMessenger
from ..services.speak_status_monitor import SpeakStatusMonitor

logger = logging.getLogger(__name__)


class VisionCoordinator:
    """
    Coordinates the Molt-See vision system.

    Manages starting/stopping the OpenClaw Eyes pipeline,
    sends vision instructions to the agent, and monitors
    Molt-Speak status for integration.
    """

    def __init__(
        self,
        molt_speak_runtime_dir: Optional[Path] = settings.MOLT_SPEAK_RUNTIME_DIR,
    ):
        """
        Initialize vision coordinator.

        Args:
            molt_speak_runtime_dir: Path to Molt-Speak runtime directory.
        """
        self.molt_speak_dir = molt_speak_runtime_dir
        self._speak_monitor = SpeakStatusMonitor(molt_speak_runtime_dir)
        self._messenger = VisionMessenger()
        self._eyes_process: Optional[subprocess.Popen] = None
        self._running = False

    def start(self) -> None:
        """Start the vision coordinator and OpenClaw Eyes."""
        logger.info("Starting Vision Coordinator...")
        self._running = True

        # Write PID
        pid_file = settings.COORDINATOR_PID_FILE
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()))

        # Send vision instructions to agent
        self._messenger.send_instructions()

        # Start OpenClaw Eyes as subprocess
        self._start_eyes()

        # Log integration status
        if self._speak_monitor.is_connected:
            logger.info(
                f"Molt-Speak integration active: {self.molt_speak_dir}"
            )
        else:
            logger.info("Running standalone (no Molt-Speak integration)")

        # Monitor loop
        try:
            self._monitor_loop()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the coordinator and OpenClaw Eyes."""
        logger.info("Stopping Vision Coordinator...")
        self._running = False

        # Stop eyes process
        self._stop_eyes()

        # Send shutdown signal to agent
        self._messenger.send_shutdown()

        # Cleanup
        self._messenger.cleanup()

        # Remove PID file
        pid_file = settings.COORDINATOR_PID_FILE
        if pid_file.exists():
            try:
                pid_file.unlink()
            except Exception:
                pass

        logger.info("Vision Coordinator stopped")

    def _start_eyes(self) -> None:
        """Start OpenClaw Eyes as a subprocess."""
        eyes_main = Path(__file__).parent.parent.parent / "open_eyes" / "main.py"

        if not eyes_main.exists():
            logger.error(f"OpenClaw Eyes entry point not found: {eyes_main}")
            return

        cmd = [
            sys.executable,
            str(eyes_main),
            "--log-level", settings.LOG_LEVEL,
        ]

        if self.molt_speak_dir:
            cmd.extend(["--molt-speak-dir", str(self.molt_speak_dir)])

        try:
            self._eyes_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info(
                f"OpenClaw Eyes started (PID: {self._eyes_process.pid})"
            )
        except Exception as e:
            logger.error(f"Failed to start OpenClaw Eyes: {e}")

    def _stop_eyes(self) -> None:
        """Stop the OpenClaw Eyes subprocess."""
        if self._eyes_process and self._eyes_process.poll() is None:
            try:
                self._eyes_process.send_signal(signal.SIGTERM)
                self._eyes_process.wait(timeout=5.0)
                logger.info("OpenClaw Eyes stopped")
            except subprocess.TimeoutExpired:
                self._eyes_process.kill()
                logger.warning("OpenClaw Eyes force-killed")
            except Exception as e:
                logger.error(f"Error stopping OpenClaw Eyes: {e}")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            # Check if Eyes is still running
            if self._eyes_process and self._eyes_process.poll() is not None:
                logger.warning(
                    f"OpenClaw Eyes exited with code "
                    f"{self._eyes_process.returncode}"
                )
                # Restart after brief delay
                time.sleep(2.0)
                if self._running:
                    logger.info("Restarting OpenClaw Eyes...")
                    self._start_eyes()

            time.sleep(1.0)
