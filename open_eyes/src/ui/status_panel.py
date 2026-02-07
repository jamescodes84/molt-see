"""
Status panel widget for expanded viewer mode.

Displays pipeline status, current detections detail, and
observation history in a side panel.
"""

import logging
from typing import Optional

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .detection_reader import DetectionSnapshot
from .observation_reader import ObservationReader
from .styles import LIST_STYLE, SECTION_HEADER_STYLE, STATUS_BOX_STYLE
from . import viewer_settings as vs

logger = logging.getLogger(__name__)

# Status label color map
_STATUS_COLORS = {
    "IDLE": "#6ec6ff",
    "CAPTURING": "#69f0ae",
    "ANALYZING": "#ffd740",
    "OBSERVING": "#ff8a65",
    "PAUSED": "#b0bec5",
    "ERROR": "#ef5350",
    "STOPPED": "#9e9e9e",
    "UNKNOWN": "#616161",
}


class StatusPanel(QWidget):
    """
    Side panel displaying pipeline status and observation history.

    Sections:
    1. Pipeline status (color-coded)
    2. Current detection details (object list with confidence)
    3. Observation history (scrollable list)
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._observation_reader = ObservationReader(
            status_file=vs.EYES_STATUS_FILE,
            observations_file=vs.VISUAL_OBSERVATIONS_FILE,
            max_history=vs.MAX_OBSERVATION_HISTORY,
        )

        self._last_obs_timestamp: str = ""
        self._setup_ui()
        self._setup_timers()

    def _setup_ui(self) -> None:
        """Build the panel layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.setMinimumWidth(280)
        self.setMaximumWidth(500)
        self.setStyleSheet("background-color: #1a1a2e; color: #e0e0e0;")

        # --- Section 1: Pipeline Status ---
        status_header = QLabel("Pipeline Status")
        status_header.setStyleSheet(SECTION_HEADER_STYLE)
        layout.addWidget(status_header)

        self._status_label = QLabel("UNKNOWN")
        self._status_label.setFont(QFont("Menlo", 14, QFont.Weight.Bold))
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet(
            f"color: #616161; {STATUS_BOX_STYLE}"
        )
        layout.addWidget(self._status_label)

        self._status_details = QLabel("")
        self._status_details.setFont(QFont("Menlo", 9))
        self._status_details.setStyleSheet("color: #888;")
        layout.addWidget(self._status_details)

        layout.addWidget(self._make_divider())

        # --- Section 2: Detected Objects ---
        det_header = QLabel("Detected Objects")
        det_header.setStyleSheet(SECTION_HEADER_STYLE)
        layout.addWidget(det_header)

        self._detection_list = QListWidget()
        self._detection_list.setMaximumHeight(150)
        self._detection_list.setStyleSheet(LIST_STYLE)
        layout.addWidget(self._detection_list)

        self._analysis_info = QLabel("")
        self._analysis_info.setFont(QFont("Menlo", 9))
        self._analysis_info.setStyleSheet("color: #888;")
        layout.addWidget(self._analysis_info)

        layout.addWidget(self._make_divider())

        # --- Section 3: Observation History ---
        obs_header = QLabel("Observation History")
        obs_header.setStyleSheet(SECTION_HEADER_STYLE)
        layout.addWidget(obs_header)

        self._observation_list = QListWidget()
        self._observation_list.setWordWrap(True)
        self._observation_list.setStyleSheet(LIST_STYLE)
        layout.addWidget(self._observation_list, stretch=1)

    def _setup_timers(self) -> None:
        """Set up polling timers for status and observations."""
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._poll_status)
        self._status_timer.start(vs.STATUS_POLL_MS)

        self._obs_timer = QTimer(self)
        self._obs_timer.timeout.connect(self._poll_observations)
        self._obs_timer.start(vs.OBSERVATION_POLL_MS)

    def update_detections(self, snapshot: DetectionSnapshot) -> None:
        """
        Update the detection detail section.

        Called when CameraWidget emits detection_updated signal.
        """
        self._detection_list.clear()
        for obj in snapshot.detected_objects:
            text = f"{obj.label}  {obj.confidence:.0%}  ({obj.area_fraction:.1%} area)"
            self._detection_list.addItem(QListWidgetItem(text))

        if not snapshot.detected_objects:
            self._detection_list.addItem(QListWidgetItem("No objects detected"))

        self._analysis_info.setText(
            f"Method: {snapshot.analysis_method} | "
            f"Time: {snapshot.processing_time_ms:.0f}ms | "
            f"Age: {snapshot.age_seconds:.1f}s"
        )

    def _poll_status(self) -> None:
        """Poll pipeline status file."""
        status = self._observation_reader.read_status()

        self._status_label.setText(status.status)
        color = _STATUS_COLORS.get(status.status, "#e0e0e0")
        self._status_label.setStyleSheet(
            f"color: {color}; {STATUS_BOX_STYLE}"
        )

        if status.is_connected:
            self._status_details.setText(
                f"{status.details} | {status.age_seconds:.0f}s ago"
            )
        else:
            self._status_details.setText("Pipeline not connected")

    def _poll_observations(self) -> None:
        """Poll observation history file."""
        observations = self._observation_reader.read_observations()
        if not observations:
            if self._last_obs_timestamp:
                self._last_obs_timestamp = ""
                self._observation_list.clear()
            return

        newest_ts = observations[-1].timestamp_iso
        if newest_ts == self._last_obs_timestamp:
            return

        self._last_obs_timestamp = newest_ts
        self._observation_list.clear()
        for obs in reversed(observations):  # newest first
            time_str = obs.timestamp_iso[11:19] if len(obs.timestamp_iso) >= 19 else ""
            text = f"[{time_str}] {obs.description}"
            item = QListWidgetItem(text)
            item.setToolTip(obs.timestamp_iso)
            self._observation_list.addItem(item)

    @staticmethod
    def _make_divider() -> QFrame:
        """Create a horizontal divider line."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #2a2a4a;")
        return line
