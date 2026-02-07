"""
Configuration settings for OpenClaw Eyes vision subsystem.

Loads configuration from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# Project Directories
# ============================================================================
PROJECT_DIR: Path = Path(__file__).parent.parent.parent.parent  # molt-see root
RUNTIME_DIR: Path = PROJECT_DIR / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Camera Configuration
# ============================================================================
CAMERA_DEVICE_INDEX: int = int(os.getenv("CAMERA_DEVICE_INDEX", "0"))
CAPTURE_FPS: float = float(os.getenv("CAPTURE_FPS", "1.0"))
CAPTURE_RESOLUTION_W: int = int(os.getenv("CAPTURE_RESOLUTION_W", "640"))
CAPTURE_RESOLUTION_H: int = int(os.getenv("CAPTURE_RESOLUTION_H", "480"))
WARMUP_FRAMES: int = int(os.getenv("WARMUP_FRAMES", "5"))

# ============================================================================
# Change Detection
# ============================================================================
CHANGE_SIMILARITY_THRESHOLD: float = float(
    os.getenv("CHANGE_SIMILARITY_THRESHOLD", "0.92")
)
CHANGE_MIN_AREA: float = float(os.getenv("CHANGE_MIN_AREA", "0.05"))
CHANGE_METHOD: str = os.getenv("CHANGE_METHOD", "ssim")

# ============================================================================
# Scene Analysis
# ============================================================================
ANALYZER_MODE: str = os.getenv("ANALYZER_MODE", "cascade")
YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8n")
YOLO_CONFIDENCE: float = float(os.getenv("YOLO_CONFIDENCE", "0.5"))
VLM_PROVIDER: str = os.getenv("VLM_PROVIDER", "anthropic")
VLM_MODEL: str = os.getenv("VLM_MODEL", "claude-sonnet-4-5-20250929")
VLM_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY") or os.getenv("VLM_API_KEY")
VLM_MAX_TOKENS: int = int(os.getenv("VLM_MAX_TOKENS", "150"))
FLORENCE_MODEL: str = os.getenv("FLORENCE_MODEL", "microsoft/Florence-2-large")

# Moondream (local VLM, free)
MOONDREAM_MODEL: str = os.getenv("MOONDREAM_MODEL", "vikhyatk/moondream2")

# YOLO-World (open-vocabulary detection)
YOLO_WORLD_MODEL: str = os.getenv("YOLO_WORLD_MODEL", "yolov8s-world")
YOLO_WORLD_CONFIDENCE: float = float(os.getenv("YOLO_WORLD_CONFIDENCE", "0.3"))
YOLO_WORLD_CLASSES: list[str] = [
    c.strip()
    for c in os.getenv(
        "YOLO_WORLD_CLASSES",
        "person,guitar,piano,microphone,headphones,camera,monitor,keyboard,"
        "desk,chair,book,cup,mug,bottle,phone,laptop,cat,dog,backpack,"
        "plant,picture frame,clock,lamp,speaker,cable,whiteboard",
    ).split(",")
    if c.strip()
]

# MediaPipe Face Detection
MEDIAPIPE_FACE_CONFIDENCE: float = float(
    os.getenv("MEDIAPIPE_FACE_CONFIDENCE", "0.5")
)
MEDIAPIPE_MAX_FACES: int = int(os.getenv("MEDIAPIPE_MAX_FACES", "3"))

# ============================================================================
# Relevance Engine
# ============================================================================
RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.6"))
OBSERVATION_COOLDOWN: float = float(os.getenv("OBSERVATION_COOLDOWN", "30.0"))
CONTEXT_WINDOW_LINES: int = int(os.getenv("CONTEXT_WINDOW_LINES", "20"))

# ============================================================================
# Vision Memory
# ============================================================================
MAX_OBSERVATIONS: int = int(os.getenv("MAX_OBSERVATIONS", "100"))
MAX_REPORTED: int = int(os.getenv("MAX_REPORTED", "50"))
MEMORY_SIMILARITY_THRESHOLD: float = float(
    os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.85")
)

# ============================================================================
# Molt-Speak Integration
# ============================================================================
MOLT_SPEAK_RUNTIME_DIR: Optional[Path] = (
    Path(os.getenv("MOLT_SPEAK_RUNTIME_DIR"))
    if os.getenv("MOLT_SPEAK_RUNTIME_DIR")
    else None
)

# ============================================================================
# Signal & Status Files
# ============================================================================
EYES_STATUS_FILE: Path = RUNTIME_DIR / "eyes_status.txt"
VISUAL_OBSERVATIONS_FILE: Path = RUNTIME_DIR / "visual_observations.txt"
VISION_MEMORY_FILE: Path = RUNTIME_DIR / "vision_memory.json"
LATEST_DETECTIONS_FILE: Path = RUNTIME_DIR / "latest_detections.json"
STRUCTURED_OBSERVATIONS_FILE: Path = RUNTIME_DIR / "structured_observations.jsonl"
LATEST_FRAME_FILE: Path = RUNTIME_DIR / "latest_frame.jpg"
MAX_STRUCTURED_FILE_SIZE: int = int(
    os.getenv("MAX_STRUCTURED_FILE_SIZE", str(10 * 1024 * 1024))  # 10MB default
)

# ============================================================================
# Performance Tuning
# ============================================================================
MAX_QUEUE_SIZE: int = int(os.getenv("MAX_QUEUE_SIZE", "10"))
PROCESSING_TIMEOUT: float = float(os.getenv("PROCESSING_TIMEOUT", "30.0"))

# ============================================================================
# Feature Flags
# ============================================================================
ENABLE_TERMINAL_DISPLAY: bool = os.getenv(
    "ENABLE_TERMINAL_DISPLAY", "true"
).lower() == "true"
DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
