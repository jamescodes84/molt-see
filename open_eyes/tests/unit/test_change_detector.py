"""Unit tests for change detection."""

import numpy as np
import pytest

from open_eyes.src.services.change_detector import ChangeDetector


class TestChangeDetector:
    """Tests for ChangeDetector."""

    def test_first_frame_always_changed(self):
        """First frame with no reference should always detect change."""
        detector = ChangeDetector(similarity_threshold=0.92)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert detector.has_scene_changed(frame) is True

    def test_identical_frames_no_change(self, black_frame):
        """Identical frames should not trigger change detection."""
        detector = ChangeDetector(similarity_threshold=0.92)
        detector.update_reference(black_frame)
        assert detector.has_scene_changed(black_frame) is False

    def test_significant_change_detected(self, black_frame, white_frame):
        """Large scene change should be detected."""
        detector = ChangeDetector(similarity_threshold=0.92)
        detector.update_reference(black_frame)
        assert detector.has_scene_changed(white_frame) is True

    def test_minor_noise_ignored(self, black_frame):
        """Small pixel noise should not trigger change."""
        detector = ChangeDetector(similarity_threshold=0.92)
        detector.update_reference(black_frame)

        # Add small noise to <1% of pixels
        noisy = black_frame.copy()
        rng = np.random.default_rng(42)
        for _ in range(50):
            y, x = rng.integers(0, 480), rng.integers(0, 640)
            noisy[y, x] = [128, 128, 128]

        assert detector.has_scene_changed(noisy) is False

    def test_change_magnitude_identical(self, black_frame):
        """Identical frames should have magnitude 0."""
        detector = ChangeDetector()
        detector.update_reference(black_frame)
        magnitude = detector.get_change_magnitude(black_frame)
        assert magnitude < 0.01

    def test_change_magnitude_max(self, black_frame, white_frame):
        """Black to white should have high magnitude."""
        detector = ChangeDetector()
        detector.update_reference(black_frame)
        magnitude = detector.get_change_magnitude(white_frame)
        assert magnitude > 0.5

    def test_change_magnitude_no_reference(self, black_frame):
        """No reference frame should return magnitude 1.0."""
        detector = ChangeDetector()
        assert detector.get_change_magnitude(black_frame) == 1.0

    def test_update_reference(self, black_frame, white_frame):
        """Updating reference should affect subsequent comparisons."""
        detector = ChangeDetector(similarity_threshold=0.92)
        detector.update_reference(black_frame)
        assert detector.has_scene_changed(white_frame) is True

        # Now update reference to white
        detector.update_reference(white_frame)
        assert detector.has_scene_changed(white_frame) is False

    def test_mse_method(self, black_frame, white_frame):
        """MSE method should detect changes."""
        detector = ChangeDetector(
            similarity_threshold=0.92, method="mse"
        )
        detector.update_reference(black_frame)
        assert detector.has_scene_changed(white_frame) is True

    def test_pixel_diff_method(self, black_frame, white_frame):
        """Pixel diff method should detect changes."""
        detector = ChangeDetector(
            similarity_threshold=0.92, method="pixel_diff"
        )
        detector.update_reference(black_frame)
        assert detector.has_scene_changed(white_frame) is True

    def test_change_regions_empty_for_identical(self, black_frame):
        """No change regions for identical frames."""
        detector = ChangeDetector()
        detector.update_reference(black_frame)
        regions = detector.get_change_regions(black_frame)
        assert len(regions) == 0

    def test_change_regions_for_different(self, black_frame):
        """Changed region should be detected."""
        detector = ChangeDetector(min_change_area=0.01)
        detector.update_reference(black_frame)

        # Draw a white rectangle on the frame
        changed = black_frame.copy()
        changed[100:300, 200:400] = 255

        regions = detector.get_change_regions(changed)
        assert len(regions) > 0

    def test_grayscale_input(self):
        """Should handle grayscale frames correctly."""
        detector = ChangeDetector()
        gray_frame = np.zeros((480, 640), dtype=np.uint8)
        detector.update_reference(gray_frame)
        assert detector.has_scene_changed(gray_frame) is False
