"""
QSS stylesheet constants for the viewer.

Centralizes all styling to keep widget code clean.
"""

MAIN_WINDOW_STYLE = """
QMainWindow {
    background-color: #0f0f1a;
}
"""

TOOLBAR_STYLE = """
QToolBar {
    background-color: #1a1a2e;
    border-bottom: 1px solid #2a2a4a;
    spacing: 4px;
    padding: 2px;
}
QToolButton {
    color: #e0e0e0;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 4px 8px;
    font-size: 11px;
}
QToolButton:hover {
    background-color: #2a2a4a;
    border-color: #3a3a5a;
}
QToolButton:checked {
    background-color: #3a3a5a;
    border-color: #5a5a7a;
}
"""

STATUS_PANEL_STYLE = """
QWidget#StatusPanel {
    background-color: #1a1a2e;
    color: #e0e0e0;
}
"""

SECTION_HEADER_STYLE = "color: #7eb8da; font-size: 12px; font-weight: bold;"

STATUS_BOX_STYLE = (
    "background-color: #16213e; border-radius: 4px; padding: 8px;"
)

LIST_STYLE = """
QListWidget {
    background-color: #16213e;
    border: none;
    color: #e0e0e0;
    font-size: 11px;
}
QListWidget::item {
    padding: 3px;
    border-bottom: 1px solid #2a2a4a;
}
QListWidget::item:selected {
    background-color: #2a2a4a;
}
"""
