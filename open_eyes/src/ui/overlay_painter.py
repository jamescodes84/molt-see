"""
Overlay painter for detection bounding boxes and labels.

Draws bounding boxes, confidence labels, and freshness indicators
on the camera feed using QPainter.
"""

import logging

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen

from .detection_reader import DetectionSnapshot

logger = logging.getLogger(__name__)


# Color palette for different object classes
LABEL_COLORS: dict[str, QColor] = {
    "person": QColor(0, 255, 128),
    "cat": QColor(255, 165, 0),
    "dog": QColor(255, 165, 0),
    "bird": QColor(255, 165, 0),
    "knife": QColor(255, 0, 0),
    "scissors": QColor(255, 0, 0),
    "cell phone": QColor(100, 200, 255),
    "laptop": QColor(100, 200, 255),
    "book": QColor(100, 200, 255),
    "cup": QColor(180, 140, 255),
    "bottle": QColor(180, 140, 255),
    "face": QColor(255, 105, 180),
    "guitar": QColor(200, 160, 80),
    "microphone": QColor(200, 160, 80),
}

DEFAULT_COLOR = QColor(255, 255, 100)


class OverlayPainter:
    """
    Paints detection overlays on a QPainter surface.

    Scales bounding box coordinates from the pipeline's frame
    resolution to the widget's display size.
    """

    def __init__(self):
        self._label_font = QFont("Menlo", 11, QFont.Weight.Bold)
        self._info_font = QFont("Menlo", 9)
        self._box_width: int = 2
        self._label_padding: int = 4

    def paint_detections(
        self,
        painter: QPainter,
        snapshot: DetectionSnapshot,
        display_width: int,
        display_height: int,
        x_offset: int = 0,
        y_offset: int = 0,
    ) -> None:
        """
        Paint all detection overlays scaled to the display size.

        Args:
            painter: Active QPainter on the widget surface.
            snapshot: Current detection data from pipeline.
            display_width: Scaled image width in pixels.
            display_height: Scaled image height in pixels.
            x_offset: Horizontal offset for centered image.
            y_offset: Vertical offset for centered image.
        """
        if not snapshot.detected_objects:
            return

        src_w, src_h = snapshot.frame_resolution
        if src_w <= 0 or src_h <= 0:
            return

        scale_x = display_width / src_w
        scale_y = display_height / src_h

        for obj in snapshot.detected_objects:
            x1, y1, x2, y2 = obj.bbox
            sx1 = int(x1 * scale_x) + x_offset
            sy1 = int(y1 * scale_y) + y_offset
            sx2 = int(x2 * scale_x) + x_offset
            sy2 = int(y2 * scale_y) + y_offset

            color = LABEL_COLORS.get(obj.label, None)
            if color is None and obj.label.startswith("face"):
                color = LABEL_COLORS.get("face", DEFAULT_COLOR)
            if color is None:
                color = DEFAULT_COLOR

            # Bounding box
            pen = QPen(color, self._box_width)
            painter.setPen(pen)
            painter.drawRect(sx1, sy1, sx2 - sx1, sy2 - sy1)

            # Label background + text
            label_text = f"{obj.label} {obj.confidence:.0%}"
            painter.setFont(self._label_font)
            fm = QFontMetrics(self._label_font)
            text_rect = fm.boundingRect(label_text)
            pad = self._label_padding

            bg_x = sx1
            bg_y = sy1 - text_rect.height() - pad * 2
            bg_w = text_rect.width() + pad * 2
            bg_h = text_rect.height() + pad * 2

            # Push label below box if it would go above the widget
            if bg_y < y_offset:
                bg_y = sy2

            painter.fillRect(
                QRectF(bg_x, bg_y, bg_w, bg_h), QColor(0, 0, 0, 180)
            )
            painter.setPen(color)
            painter.drawText(bg_x + pad, bg_y + bg_h - pad, label_text)

    def paint_freshness_indicator(
        self,
        painter: QPainter,
        snapshot: DetectionSnapshot,
        x: int,
        y: int,
    ) -> None:
        """
        Paint a freshness dot showing detection data age.

        Green = fresh (<2s), Yellow = aging (2-5s), Red = stale (>5s),
        Gray = no data.
        """
        age = snapshot.age_seconds
        if snapshot.timestamp <= 0:
            color = QColor(128, 128, 128)
        elif age < 2.0:
            color = QColor(0, 255, 128)
        elif age < 5.0:
            color = QColor(255, 200, 0)
        else:
            color = QColor(255, 60, 60)

        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(x, y, 10, 10)

    def paint_compact_info(
        self,
        painter: QPainter,
        snapshot: DetectionSnapshot,
        x: int,
        y: int,
        width: int,
    ) -> None:
        """
        Paint compact info bar at the bottom of the frame.

        Shows: object count, analysis method, processing time.
        """
        if snapshot.timestamp <= 0:
            return

        painter.setFont(self._info_font)

        count = snapshot.object_count
        method = snapshot.analysis_method
        ms = snapshot.processing_time_ms
        info_text = f"{count} objects | {method} | {ms:.0f}ms"

        bar_height = 24
        painter.fillRect(x, y - bar_height, width, bar_height, QColor(0, 0, 0, 160))
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(x + 8, y - 7, info_text)
