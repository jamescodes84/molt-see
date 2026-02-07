"""
Detection data reader.

Polls runtime/latest_detections.json written by the vision pipeline
and provides structured detection data to the viewer.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DetectionOverlay:
    """Parsed detection data for overlay rendering."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    area_fraction: float = 0.0


@dataclass
class DetectionSnapshot:
    """Complete detection state from one pipeline analysis cycle."""

    timestamp: float = 0.0
    frame_number: int = 0
    analysis_method: str = ""
    description: str = ""
    processing_time_ms: float = 0.0
    detected_objects: list[DetectionOverlay] = field(default_factory=list)
    frame_resolution: tuple[int, int] = (640, 480)
    is_stale: bool = True

    @property
    def age_seconds(self) -> float:
        """Seconds since this detection was produced."""
        if self.timestamp <= 0:
            return float("inf")
        return time.time() - self.timestamp

    @property
    def object_count(self) -> int:
        return len(self.detected_objects)


class DetectionReader:
    """
    Reads and parses latest_detections.json from the vision pipeline.

    Designed to be called on a QTimer at regular intervals.
    Uses file mtime to skip re-reading unchanged files.
    """

    def __init__(
        self,
        detections_file: Path,
        stale_threshold: float = 5.0,
    ):
        self._file = detections_file
        self._stale_threshold = stale_threshold
        self._last_mtime: float = 0.0
        self._latest = DetectionSnapshot()

    def read(self) -> DetectionSnapshot:
        """
        Read latest detections if the file has changed.

        Returns:
            DetectionSnapshot with current detection data.
        """
        try:
            if not self._file.exists():
                self._latest.is_stale = True
                return self._latest

            mtime = self._file.stat().st_mtime
            if mtime == self._last_mtime:
                self._latest.is_stale = (
                    self._latest.age_seconds > self._stale_threshold
                )
                return self._latest

            self._last_mtime = mtime
            data = json.loads(self._file.read_text())
            self._latest = self._parse(data)
            return self._latest

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.debug(f"Failed to read detections: {e}")
            self._latest.is_stale = True
            return self._latest

    def _parse(self, data: dict) -> DetectionSnapshot:
        """Parse raw JSON dict into DetectionSnapshot."""
        objects = [
            DetectionOverlay(
                label=obj["label"],
                confidence=obj["confidence"],
                bbox=tuple(obj["bbox"]),
                area_fraction=obj.get("area_fraction", 0.0),
            )
            for obj in data.get("detected_objects", [])
        ]

        resolution = data.get("frame_resolution", [640, 480])
        timestamp = data.get("timestamp", 0.0)

        return DetectionSnapshot(
            timestamp=timestamp,
            frame_number=data.get("frame_number", 0),
            analysis_method=data.get("analysis_method", ""),
            description=data.get("description", ""),
            processing_time_ms=data.get("processing_time_ms", 0.0),
            detected_objects=objects,
            frame_resolution=(resolution[0], resolution[1]),
            is_stale=(time.time() - timestamp) > self._stale_threshold,
        )
