"""Unit tests for camera capture."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from open_eyes.src.services.camera_capture import CameraCapture


class TestCameraCapture:
    """Tests for CameraCapture."""

    def test_init_defaults(self):
        """Default initialization should set sensible values."""
        capture = CameraCapture()
        assert capture.device_index == 0
        assert 0.1 <= capture.target_fps <= 30.0
        assert capture.frame_count == 0

    def test_fps_clamped(self):
        """FPS should be clamped to valid range."""
        capture = CameraCapture(target_fps=100.0)
        assert capture.target_fps == 30.0

        capture = CameraCapture(target_fps=0.01)
        assert capture.target_fps == 0.1

    def test_get_frame_none_before_start(self):
        """get_frame should return None before camera is started."""
        capture = CameraCapture()
        assert capture.get_frame() is None

    def test_is_available_before_start(self):
        """Camera should not be available before start."""
        capture = CameraCapture()
        assert capture.is_available() is False

    @patch("open_eyes.src.services.camera_capture.cv2.VideoCapture")
    def test_start_failure(self, mock_cv2_capture):
        """Start should return False if camera can't be opened."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2_capture.return_value = mock_cap

        capture = CameraCapture(warmup_frames=0)
        result = capture.start()
        assert result is False

    @patch("open_eyes.src.services.camera_capture.cv2.VideoCapture")
    def test_start_success(self, mock_cv2_capture):
        """Start should return True and begin capture thread."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (
            True,
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
        mock_cap.get.return_value = 640.0
        mock_cv2_capture.return_value = mock_cap

        capture = CameraCapture(warmup_frames=0, target_fps=1.0)
        result = capture.start()
        assert result is True
        assert capture.is_available() is True

        capture.stop()

    @patch("open_eyes.src.services.camera_capture.cv2.VideoCapture")
    def test_list_devices(self, mock_cv2_capture):
        """list_devices should probe camera indices."""
        # First device available, second not
        mock_cap_open = MagicMock()
        mock_cap_open.isOpened.return_value = True
        mock_cap_open.get.return_value = 640.0

        mock_cap_closed = MagicMock()
        mock_cap_closed.isOpened.return_value = False

        mock_cv2_capture.side_effect = [
            mock_cap_open,
            mock_cap_closed,
            mock_cap_closed,
        ]

        devices = CameraCapture.list_devices(max_devices=3)
        assert len(devices) == 1
        assert devices[0]["index"] == 0
        assert devices[0]["available"] is True
