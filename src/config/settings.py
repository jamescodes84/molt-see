"""
Configuration settings for Molt-See Vision System.

Loads configuration from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# Logging Configuration
# ============================================================================
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Optional[str] = os.getenv("LOG_FILE")

# ============================================================================
# Project Directories
# ============================================================================
PROJECT_DIR: Path = Path(__file__).parent.parent.parent  # molt-see root
RUNTIME_DIR: Path = PROJECT_DIR / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# Configuration file
CONFIG_FILE: Path = RUNTIME_DIR / "molt_see_config.json"

# ============================================================================
# Signal & Status Files
# ============================================================================
EYES_STATUS_FILE: Path = RUNTIME_DIR / "eyes_status.txt"
VISUAL_OBSERVATIONS_FILE: Path = RUNTIME_DIR / "visual_observations.txt"
VISUAL_CONTEXT_FILE: Path = RUNTIME_DIR / "visual_context.txt"
VISION_MEMORY_FILE: Path = RUNTIME_DIR / "vision_memory.json"
EYES_PID_FILE: Path = RUNTIME_DIR / "eyes.pid"
COORDINATOR_PID_FILE: Path = RUNTIME_DIR / "vision_coordinator.pid"
EYES_PAUSE_SIGNAL_FILE: Path = RUNTIME_DIR / "eyes_pause.signal"
AGENT_INSTRUCTIONS_FILE: Path = RUNTIME_DIR / "agent_instructions.active"
AGENT_SHUTDOWN_SIGNAL_FILE: Path = RUNTIME_DIR / "agent_shutdown.signal"

# ============================================================================
# Molt-Speak Integration
# ============================================================================
MOLT_SPEAK_RUNTIME_DIR: Optional[Path] = (
    Path(os.getenv("MOLT_SPEAK_RUNTIME_DIR"))
    if os.getenv("MOLT_SPEAK_RUNTIME_DIR")
    else None
)
ENABLE_SPEAK_INTEGRATION: bool = os.getenv(
    "ENABLE_SPEAK_INTEGRATION", "true"
).lower() == "true"

# Optionally place visual_observations.txt in Molt-Speak's runtime dir
VISUAL_OBSERVATIONS_DIR: Optional[Path] = (
    Path(os.getenv("VISUAL_OBSERVATIONS_DIR"))
    if os.getenv("VISUAL_OBSERVATIONS_DIR")
    else None
)

# ============================================================================
# Monitoring Configuration
# ============================================================================
SPEAK_STATUS_POLL_INTERVAL: float = float(
    os.getenv("SPEAK_STATUS_POLL_INTERVAL", "0.1")
)

# ============================================================================
# Integration Mode
# ============================================================================
ENABLE_INTEGRATION: bool = os.getenv(
    "ENABLE_INTEGRATION", "true"
).lower() == "true"

# ============================================================================
# Performance Tuning
# ============================================================================
MAX_QUEUE_SIZE: int = int(os.getenv("MAX_QUEUE_SIZE", "10"))
PROCESSING_TIMEOUT: float = float(os.getenv("PROCESSING_TIMEOUT", "30.0"))
