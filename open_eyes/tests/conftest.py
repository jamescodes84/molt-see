"""Shared fixtures for OpenClaw Eyes tests."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from open_eyes.src.models.vision_models import (
    DetectedObject,
    SceneDescription,
)


@pytest.fixture
def sample_frame():
    """Create a sample image frame for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def black_frame():
    """Create a solid black frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_frame():
    """Create a solid white frame."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 255


@pytest.fixture
def runtime_dir(tmp_path):
    """Create a temporary runtime directory."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    return runtime


@pytest.fixture
def mock_detected_object():
    """Create a sample DetectedObject."""
    return DetectedObject(
        label="person",
        confidence=0.92,
        bbox=(50, 100, 200, 400),
        area_fraction=0.15,
    )


@pytest.fixture
def mock_scene_description():
    """Create a sample SceneDescription."""
    return SceneDescription(
        timestamp=time.time(),
        description="A person sitting at a desk with a laptop and coffee mug",
        detected_objects=[],
        object_labels=["person", "desk", "laptop", "cup"],
        analysis_method="test",
        processing_time_ms=15.0,
        frame_number=1,
    )
