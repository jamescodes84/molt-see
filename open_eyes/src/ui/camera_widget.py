"""
Camera display widget with detection overlay.

Reads frames written by the vision pipeline to disk and composites
detection bounding boxes on top. Does NOT open its own camera —
the pipeline is the sole camera owner.
"""

import logging
from typing import Optional

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QWidget

from .detection_reader import DetectionReader, DetectionSnapshot
from .overlay_painter import OverlayPainter
from . import viewer_settings as vs

logger = logging.getLogger(__name__)


class CameraWidget(QWidget):
    """
    Live camera feed widget with detection overlay.

    Reads frames from the pipeline's latest_frame.jpg file.
    Reads detection data from the pipeline via DetectionReader.
    Paints frames + overlays in paintEvent.
    """

    detection_updated = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._detection_reader = DetectionReader(
            detections_file=vs.LATEST_DETECTIONS_FILE,
            stale_threshold=vs.DETECTION_STALE_SECONDS,
        )

        self._overlay_painter = OverlayPainter()

        self._current_pixmap: Optional[QPixmap] = None
        self._current_detections = DetectionSnapshot()
        self._last_frame_mtime: float = 0.0

        self._frame_timer = QTimer(self)
        self._frame_timer.timeout.connect(self._on_frame_tick)

        self._detection_timer = QTimer(self)
        self._detection_timer.timeout.connect(self._on_detection_tick)

        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: black;")

    def start(self) -> bool:
        """Start frame polling timers."""
        frame_interval_ms = max(16, int(1000 / vs.VIEWER_FPS))
        self._frame_timer.start(frame_interval_ms)
        self._detection_timer.start(vs.DETECTION_POLL_MS)

        logger.info(
            f"Camera widget started: poll_fps={vs.VIEWER_FPS}, "
            f"detection_poll={vs.DETECTION_POLL_MS}ms"
        )
        return True

    def stop(self) -> None:
        """Stop polling timers."""
        self._frame_timer.stop()
        self._detection_timer.stop()
        logger.info("Camera widget stopped")

    def _on_frame_tick(self) -> None:
        """Read latest frame from pipeline's frame file and trigger repaint."""
        pixmap = self._read_frame_file()
        if pixmap is not None:
            self._current_pixmap = pixmap
            self.update()

    def _read_frame_file(self) -> Optional[QPixmap]:
        """Read latest_frame.jpg if it has changed since last read."""
        try:
            frame_path = vs.LATEST_FRAME_FILE
            if not frame_path.exists():
                return None

            mtime = frame_path.stat().st_mtime
            if mtime == self._last_frame_mtime:
                return None

            self._last_frame_mtime = mtime
            pixmap = QPixmap(str(frame_path))
            if pixmap.isNull():
                return None
            return pixmap

        except OSError as e:
            logger.debug(f"Failed to read frame file: {e}")
            return None

    def _on_detection_tick(self) -> None:
        """Poll for new detection data from pipeline."""
        snapshot = self._detection_reader.read()
        self._current_detections = snapshot
        self.detection_updated.emit(snapshot)

    def paintEvent(self, event) -> None:
        """Paint the camera frame with detection overlays."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        if self._current_pixmap is not None:
            scaled = self._current_pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x_off = (w - scaled.width()) // 2
            y_off = (h - scaled.height()) // 2
            painter.drawPixmap(x_off, y_off, scaled)

            # Detection bounding boxes
            self._overlay_painter.paint_detections(
                painter, self._current_detections,
                scaled.width(), scaled.height(),
                x_off, y_off,
            )

            # Freshness indicator (top-right of image area)
            self._overlay_painter.paint_freshness_indicator(
                painter, self._current_detections,
                x_off + scaled.width() - 18, y_off + 8,
            )

            # Compact info bar (bottom of image area)
            self._overlay_painter.paint_compact_info(
                painter, self._current_detections,
                x_off, y_off + scaled.height(),
                scaled.width(),
            )
        else:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "Waiting for pipeline...",
            )

        painter.end()
