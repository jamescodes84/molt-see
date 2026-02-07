"""
Image processing utilities.

Helper functions for frame encoding, resizing, and format conversion.
"""

import base64
import io
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def resize_frame(
    frame: np.ndarray,
    max_width: int = 640,
    max_height: int = 480,
) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.

    Args:
        frame: BGR numpy array.
        max_width: Maximum width.
        max_height: Maximum height.

    Returns:
        Resized frame.
    """
    h, w = frame.shape[:2]
    if w <= max_width and h <= max_height:
        return frame

    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def frame_to_jpeg_bytes(
    frame: np.ndarray, quality: int = 85
) -> Optional[bytes]:
    """
    Encode frame as JPEG bytes.

    Args:
        frame: BGR numpy array.
        quality: JPEG quality (0-100).

    Returns:
        JPEG-encoded bytes, or None on failure.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if success:
        return buffer.tobytes()
    return None


def frame_to_base64(frame: np.ndarray, quality: int = 85) -> Optional[str]:
    """
    Encode frame as base64 JPEG string.

    Used for sending frames to vision LLM APIs.

    Args:
        frame: BGR numpy array.
        quality: JPEG quality (0-100).

    Returns:
        Base64-encoded JPEG string, or None on failure.
    """
    jpeg_bytes = frame_to_jpeg_bytes(frame, quality)
    if jpeg_bytes:
        return base64.b64encode(jpeg_bytes).decode("utf-8")
    return None


def frame_to_pil(frame: np.ndarray) -> Image.Image:
    """
    Convert BGR numpy frame to PIL Image.

    Args:
        frame: BGR numpy array.

    Returns:
        PIL Image in RGB mode.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_frame(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to BGR numpy frame.

    Args:
        image: PIL Image.

    Returns:
        BGR numpy array.
    """
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
