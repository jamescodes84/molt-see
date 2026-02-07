"""
Scene change detection service.

Detects meaningful scene changes to avoid re-processing static frames.
Uses structural similarity (SSIM) to filter noise while catching real changes.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from ..config import settings

logger = logging.getLogger(__name__)


class ChangeDetector:
    """
    Detects meaningful scene changes between frames.

    Compares incoming frames to a reference frame using SSIM or
    pixel-difference thresholds. Prevents flooding the analysis
    pipeline with identical or near-identical frames.
    """

    def __init__(
        self,
        similarity_threshold: float = settings.CHANGE_SIMILARITY_THRESHOLD,
        min_change_area: float = settings.CHANGE_MIN_AREA,
        method: str = settings.CHANGE_METHOD,
    ):
        """
        Initialize change detector.

        Args:
            similarity_threshold: SSIM threshold below which a change
                is considered significant (0-1, higher = less sensitive).
            min_change_area: Minimum fraction of frame area that must
                change to trigger detection (0-1).
            method: Detection method ("ssim", "mse", "pixel_diff").
        """
        self.similarity_threshold = similarity_threshold
        self.min_change_area = min_change_area
        self.method = method
        self._reference_frame: Optional[np.ndarray] = None

    def has_scene_changed(self, current_frame: np.ndarray) -> bool:
        """
        Compare current frame to reference.

        Args:
            current_frame: BGR numpy array of the current frame.

        Returns:
            True if the scene has changed enough to warrant analysis.
        """
        if self._reference_frame is None:
            # First frame always counts as a change
            return True

        magnitude = self.get_change_magnitude(current_frame)
        threshold = 1.0 - self.similarity_threshold  # Convert to change scale
        return bool(magnitude >= threshold)

    def update_reference(self, frame: np.ndarray) -> None:
        """
        Update the reference frame.

        Should be called after a changed frame is successfully processed.

        Args:
            frame: BGR numpy array to use as the new reference.
        """
        self._reference_frame = self._to_gray(frame)

    def get_change_magnitude(self, current_frame: np.ndarray) -> float:
        """
        Get the magnitude of change between current and reference frame.

        Args:
            current_frame: BGR numpy array.

        Returns:
            Change magnitude from 0.0 (identical) to 1.0 (completely different).
        """
        if self._reference_frame is None:
            return 1.0

        current_gray = self._to_gray(current_frame)

        # Ensure frames are the same size
        if current_gray.shape != self._reference_frame.shape:
            current_gray = cv2.resize(
                current_gray, self._reference_frame.shape[::-1]
            )

        if self.method == "ssim":
            return self._ssim_change(current_gray)
        elif self.method == "mse":
            return self._mse_change(current_gray)
        else:
            return self._pixel_diff_change(current_gray)

    def get_change_regions(
        self, current_frame: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """
        Get bounding boxes of regions that changed.

        Args:
            current_frame: BGR numpy array.

        Returns:
            List of (x, y, w, h) bounding boxes for changed regions.
        """
        if self._reference_frame is None:
            return []

        current_gray = self._to_gray(current_frame)
        if current_gray.shape != self._reference_frame.shape:
            current_gray = cv2.resize(
                current_gray, self._reference_frame.shape[::-1]
            )

        # Compute absolute difference
        diff = cv2.absdiff(current_gray, self._reference_frame)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Find contours of changed regions
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frame_area = current_gray.shape[0] * current_gray.shape[1]
        min_area = frame_area * self.min_change_area
        regions = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append((x, y, w, h))

        return regions

    def _ssim_change(self, current_gray: np.ndarray) -> float:
        """Compute change magnitude using SSIM."""
        score = ssim(self._reference_frame, current_gray)
        return 1.0 - score  # Convert similarity to change magnitude

    def _mse_change(self, current_gray: np.ndarray) -> float:
        """Compute change magnitude using Mean Squared Error."""
        mse = np.mean((self._reference_frame.astype(float) - current_gray.astype(float)) ** 2)
        # Normalize to 0-1 range (max MSE for uint8 is 255^2 = 65025)
        return min(mse / 65025.0, 1.0)

    def _pixel_diff_change(self, current_gray: np.ndarray) -> float:
        """Compute change magnitude using pixel difference ratio."""
        diff = cv2.absdiff(current_gray, self._reference_frame)
        changed_pixels = np.count_nonzero(diff > 30)
        total_pixels = current_gray.shape[0] * current_gray.shape[1]
        return changed_pixels / total_pixels

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to grayscale."""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
