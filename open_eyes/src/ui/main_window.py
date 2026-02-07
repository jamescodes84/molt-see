"""
Main viewer window.

QMainWindow that composes the camera widget and status panel,
handles mode switching between compact and expanded layouts,
and manages keyboard shortcuts.
"""

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QToolBar,
    QWidget,
)

from .camera_widget import CameraWidget
from .status_panel import StatusPanel
from .styles import MAIN_WINDOW_STYLE, TOOLBAR_STYLE
from . import viewer_settings as vs

logger = logging.getLogger(__name__)


class ViewerWindow(QMainWindow):
    """
    Main camera viewer window.

    Supports two modes:
    - Compact: Camera feed with overlays only
    - Expanded: Camera feed + side panel with status and history

    Keyboard shortcuts:
    - Tab: Toggle compact/expanded mode
    - Esc: Minimize window
    - Q: Quit viewer
    - F: Toggle fullscreen
    """

    COMPACT_SIZE = (660, 500)
    EXPANDED_SIZE = (1100, 560)

    def __init__(self):
        super().__init__()

        self._mode = vs.DEFAULT_MODE

        self._setup_window()
        self._setup_widgets()
        self._setup_toolbar()
        self._setup_shortcuts()
        self._apply_mode()

    def _setup_window(self) -> None:
        """Configure window properties."""
        self.setWindowTitle("Molt-See Camera Viewer")
        self.setStyleSheet(MAIN_WINDOW_STYLE)
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowOpacity(vs.WINDOW_OPACITY)

    def _setup_widgets(self) -> None:
        """Create and lay out the camera widget and status panel."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._camera_widget = CameraWidget()
        layout.addWidget(self._camera_widget, stretch=1)

        self._status_panel = StatusPanel()
        layout.addWidget(self._status_panel)

        # Wire detection updates from camera to status panel
        self._camera_widget.detection_updated.connect(
            self._status_panel.update_detections
        )

    def _setup_toolbar(self) -> None:
        """Create a minimal toolbar with mode toggle."""
        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        toolbar.setStyleSheet(TOOLBAR_STYLE)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        self._mode_action = QAction("Expand", self)
        self._mode_action.setCheckable(True)
        self._mode_action.setChecked(self._mode == "expanded")
        self._mode_action.triggered.connect(self._toggle_mode)
        self._mode_action.setShortcut(QKeySequence(Qt.Key.Key_Tab))
        toolbar.addAction(self._mode_action)

        toolbar.addSeparator()

        fs_action = QAction("Fullscreen", self)
        fs_action.setShortcut(QKeySequence(Qt.Key.Key_F))
        fs_action.triggered.connect(self._toggle_fullscreen)
        toolbar.addAction(fs_action)

    def _setup_shortcuts(self) -> None:
        """Set up additional keyboard shortcuts."""
        esc_action = QAction(self)
        esc_action.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        esc_action.triggered.connect(self.showMinimized)
        self.addAction(esc_action)

        quit_action = QAction(self)
        quit_action.setShortcut(QKeySequence("Q"))
        quit_action.triggered.connect(self.close)
        self.addAction(quit_action)

    def _toggle_mode(self) -> None:
        """Switch between compact and expanded modes."""
        self._mode = "expanded" if self._mode == "compact" else "compact"
        self._apply_mode()

    def _apply_mode(self) -> None:
        """Apply the current mode to the layout."""
        is_expanded = self._mode == "expanded"
        self._status_panel.setVisible(is_expanded)
        self._mode_action.setChecked(is_expanded)
        self._mode_action.setText("Compact" if is_expanded else "Expand")

        if is_expanded:
            self.resize(*self.EXPANDED_SIZE)
        else:
            self.resize(*self.COMPACT_SIZE)

        logger.debug(f"Viewer mode: {self._mode}")

    def _toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def start(self) -> bool:
        """Start the camera and all polling."""
        return self._camera_widget.start()

    def closeEvent(self, event) -> None:
        """Clean up on window close."""
        logger.info("Viewer window closing")
        self._camera_widget.stop()
        event.accept()
