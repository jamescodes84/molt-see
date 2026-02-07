"""
Camera capture service.

Manages webcam access and frame capture using OpenCV.
Supports configurable FPS, resolution, and device selection.
"""

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Manages webcam access and frame capture.

    Uses OpenCV VideoCapture for reliable macOS camera access.
    Frames are captured in a background thread at the configured FPS.
    """

    def __init__(
        self,
        device_index: int = settings.CAMERA_DEVICE_INDEX,
        target_fps: float = settings.CAPTURE_FPS,
        resolution: tuple[int, int] = (
            settings.CAPTURE_RESOLUTION_W,
            settings.CAPTURE_RESOLUTION_H,
        ),
        warmup_frames: int = settings.WARMUP_FRAMES,
    ):
        """
        Initialize camera capture.

        Args:
            device_index: Camera device index (0 = default webcam).
            target_fps: Target frames per second.
            resolution: Capture resolution (width, height).
            warmup_frames: Frames to discard for auto-exposure adjustment.
        """
        self.device_index = device_index
        self.target_fps = max(0.1, min(target_fps, 30.0))
        self.resolution = resolution
        self.warmup_frames = warmup_frames

        self._capture: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_count = 0

    def start(self) -> bool:
        """
        Open camera and begin capture thread.

        Returns:
            True if camera opened successfully.
        """
        self._capture = cv2.VideoCapture(self.device_index)

        if not self._capture.isOpened():
            logger.error(
                f"Failed to open camera device {self.device_index}"
            )
            return False

        # Set resolution
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Discard warmup frames (auto-exposure adjustment)
        for _ in range(self.warmup_frames):
            self._capture.read()

        self._running = True
        self._frame_count = 0
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="camera-capture"
        )
        self._capture_thread.start()

        actual_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            f"Camera started: device={self.device_index}, "
            f"resolution={actual_w}x{actual_h}, fps={self.target_fps}"
        )
        return True

    def stop(self) -> None:
        """Release camera resources."""
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self._capture:
            self._capture.release()
            self._capture = None
        logger.info("Camera stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest captured frame (non-blocking).

        Returns:
            BGR numpy array of the latest frame, or None if unavailable.
        """
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return None

    @property
    def frame_count(self) -> int:
        """Number of frames captured."""
        return self._frame_count

    def is_available(self) -> bool:
        """Check if camera is accessible."""
        return self._capture is not None and self._capture.isOpened()

    def _capture_loop(self) -> None:
        """Background thread: capture frames at target FPS."""
        frame_interval = 1.0 / self.target_fps

        while self._running and self._capture and self._capture.isOpened():
            start_time = time.time()

            ret, frame = self._capture.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                self._frame_count += 1
            else:
                logger.warning("Failed to read frame from camera")

            # Sleep to maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    @staticmethod
    def list_devices(max_devices: int = 5) -> list[dict]:
        """
        List available camera devices.

        Args:
            max_devices: Maximum number of devices to probe.

        Returns:
            List of dicts with device info.
        """
        devices = []
        for i in range(max_devices):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                devices.append({
                    "index": i,
                    "resolution": f"{w}x{h}",
                    "available": True,
                })
                cap.release()
        return devices
