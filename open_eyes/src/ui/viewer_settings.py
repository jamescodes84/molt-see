"""
Configuration settings for the camera viewer.

Loads viewer-specific configuration from environment variables.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Re-import shared paths and defaults from main settings
from ..config import settings

RUNTIME_DIR = settings.RUNTIME_DIR

# ============================================================================
# Viewer Frame Polling
# ============================================================================
VIEWER_FPS: int = int(os.getenv("VIEWER_FPS", "15"))

# ============================================================================
# Polling Intervals (milliseconds)
# ============================================================================
DETECTION_POLL_MS: int = int(os.getenv("DETECTION_POLL_MS", "500"))
STATUS_POLL_MS: int = int(os.getenv("STATUS_POLL_MS", "1000"))
OBSERVATION_POLL_MS: int = int(os.getenv("OBSERVATION_POLL_MS", "2000"))

# ============================================================================
# Display Configuration
# ============================================================================
DETECTION_STALE_SECONDS: float = float(os.getenv("DETECTION_STALE_SECONDS", "5.0"))
MAX_OBSERVATION_HISTORY: int = int(os.getenv("MAX_OBSERVATION_HISTORY", "50"))
DEFAULT_MODE: str = os.getenv("VIEWER_DEFAULT_MODE", "compact")
WINDOW_OPACITY: float = float(os.getenv("VIEWER_WINDOW_OPACITY", "1.0"))

# ============================================================================
# File Paths (derived from shared settings)
# ============================================================================
LATEST_DETECTIONS_FILE = settings.LATEST_DETECTIONS_FILE
LATEST_FRAME_FILE = settings.LATEST_FRAME_FILE
EYES_STATUS_FILE = settings.EYES_STATUS_FILE
VISUAL_OBSERVATIONS_FILE = settings.VISUAL_OBSERVATIONS_FILE
