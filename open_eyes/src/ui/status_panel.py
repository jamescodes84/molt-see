"""
Status panel widget for expanded viewer mode.

Displays pipeline status, current detections detail, and
tiered visual context (Scene, Activity, Expression, Recent Events).
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
from .observation_reader import ObservationReader, VisualContextReader
from .styles import (
    LIST_STYLE,
    SECTION_HEADER_STYLE,
    STATUS_BOX_STYLE,
    TIER_ACTIVITY_STYLE,
    TIER_CONTENT_STYLE,
    TIER_EVENTS_STYLE,
    TIER_EXPRESSION_STYLE,
    TIER_OBJECTS_STYLE,
    TIER_SCENE_STYLE,
)
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
    Side panel displaying pipeline status and tiered visual context.

    Sections:
    1. Pipeline status (color-coded)
    2. Current detection details (object list with confidence)
    3. Scene tier (rich environment description)
    4. Activity tier (what's happening)
    5. Expression tier (facial expression)
    6. Recent Events tier (scrollable list)
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._observation_reader = ObservationReader(
            status_file=vs.EYES_STATUS_FILE,
            observations_file=vs.VISUAL_OBSERVATIONS_FILE,
            max_history=vs.MAX_OBSERVATION_HISTORY,
        )
        self._context_reader = VisualContextReader(
            context_file=vs.VISUAL_CONTEXT_FILE,
        )

        self._setup_ui()
        self._setup_timers()

    def _setup_ui(self) -> None:
        """Build the panel layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

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
        self._detection_list.setMaximumHeight(120)
        self._detection_list.setStyleSheet(LIST_STYLE)
        layout.addWidget(self._detection_list)

        self._analysis_info = QLabel("")
        self._analysis_info.setFont(QFont("Menlo", 9))
        self._analysis_info.setStyleSheet("color: #888;")
        layout.addWidget(self._analysis_info)

        layout.addWidget(self._make_divider())

        # --- Section 3: Scene Tier ---
        self._scene_header = QLabel("Scene")
        self._scene_header.setStyleSheet(TIER_SCENE_STYLE)
        layout.addWidget(self._scene_header)

        self._scene_label = QLabel("")
        self._scene_label.setWordWrap(True)
        self._scene_label.setStyleSheet(TIER_CONTENT_STYLE)
        layout.addWidget(self._scene_label)

        self._scene_objects_label = QLabel("")
        self._scene_objects_label.setWordWrap(True)
        self._scene_objects_label.setStyleSheet(TIER_OBJECTS_STYLE)
        layout.addWidget(self._scene_objects_label)

        # --- Section 4: Activity Tier ---
        self._activity_header = QLabel("Activity")
        self._activity_header.setStyleSheet(TIER_ACTIVITY_STYLE)
        layout.addWidget(self._activity_header)

        self._activity_label = QLabel("")
        self._activity_label.setWordWrap(True)
        self._activity_label.setStyleSheet(TIER_CONTENT_STYLE)
        layout.addWidget(self._activity_label)

        # --- Section 5: Expression Tier ---
        self._expression_header = QLabel("Expression")
        self._expression_header.setStyleSheet(TIER_EXPRESSION_STYLE)
        layout.addWidget(self._expression_header)

        self._expression_label = QLabel("")
        self._expression_label.setFont(QFont("Menlo", 13))
        self._expression_label.setStyleSheet("color: #ff80ab; padding: 2px 0;")
        layout.addWidget(self._expression_label)

        # --- Section 6: Recent Events Tier ---
        self._events_header = QLabel("Recent Events")
        self._events_header.setStyleSheet(TIER_EVENTS_STYLE)
        layout.addWidget(self._events_header)

        self._events_list = QListWidget()
        self._events_list.setStyleSheet(LIST_STYLE)
        layout.addWidget(self._events_list, stretch=1)

        # Initially hide tier sections until data arrives
        self._set_tiers_visible(False)

    def _set_tiers_visible(self, visible: bool) -> None:
        """Show or hide all tier sections at once."""
        for widget in (
            self._scene_header, self._scene_label, self._scene_objects_label,
            self._activity_header, self._activity_label,
            self._expression_header, self._expression_label,
            self._events_header, self._events_list,
        ):
            widget.setVisible(visible)

    def _setup_timers(self) -> None:
        """Set up polling timers for status and visual context."""
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._poll_status)
        self._status_timer.start(vs.STATUS_POLL_MS)

        self._context_timer = QTimer(self)
        self._context_timer.timeout.connect(self._poll_visual_context)
        self._context_timer.start(vs.CONTEXT_POLL_MS)

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

    def _poll_visual_context(self) -> None:
        """Poll visual_context.txt and update tier sections."""
        ctx = self._context_reader.read()

        has_scene = bool(ctx.scene)
        self._scene_header.setVisible(has_scene)
        self._scene_label.setVisible(has_scene)
        self._scene_objects_label.setVisible(has_scene and bool(ctx.objects))
        if has_scene:
            self._scene_label.setText(ctx.scene)
            if ctx.objects:
                self._scene_objects_label.setText(f"Objects: {ctx.objects}")

        has_activity = bool(ctx.activity)
        self._activity_header.setVisible(has_activity)
        self._activity_label.setVisible(has_activity)
        if has_activity:
            self._activity_label.setText(ctx.activity)

        has_expression = bool(ctx.expression)
        self._expression_header.setVisible(has_expression)
        self._expression_label.setVisible(has_expression)
        if has_expression:
            self._expression_label.setText(ctx.expression)

        has_events = bool(ctx.recent_events)
        self._events_header.setVisible(has_events)
        self._events_list.setVisible(has_events)
        if has_events:
            self._events_list.clear()
            for event in reversed(ctx.recent_events):
                self._events_list.addItem(QListWidgetItem(event))

    @staticmethod
    def _make_divider() -> QFrame:
        """Create a horizontal divider line."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #2a2a4a;")
        return line
