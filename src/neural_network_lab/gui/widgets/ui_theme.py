"""Shared visual language for the Neural Network Lab interface.

This module is the single source of truth for presentation constants:
spacing, margins, colours, and the application-wide Qt style sheet.
Pages compose these tokens instead of repeating literal sizes or inline
``setStyleSheet`` calls, which is what keeps every section of the
application looking and behaving like one coherent product.

Nothing here renders a page, owns application state, or contains any
neural-network mathematics: :func:`apply_theme` only installs style
rules on the shared :class:`QApplication`.
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

# ---------------------------------------------------------------------------
# Spacing and sizing tokens (pixels)
# ---------------------------------------------------------------------------

#: Tight spacing inside a control group (labels, fields, small rows).
FIELD_SPACING = 8

#: Default spacing between stacked elements inside one section.
CONTENT_SPACING = 12

#: Extra spacing between two sections of a page.
SECTION_SPACING = 14

#: Margin around the content of a page, a section card, or the window.
CONTENT_MARGIN = 16

#: Margin inside a section card around its own contents.
CARD_MARGIN = 12

#: Minimum width of the navigation panel, and its maximum before it is
#: allowed to steal space from the content area.
NAVIGATION_MIN_WIDTH = 190
NAVIGATION_MAX_WIDTH = 340

#: Smallest practical desktop window; below this the UI is unusable.
MIN_WINDOW_WIDTH = 1000
MIN_WINDOW_HEIGHT = 640

#: Default window size on a normal desktop screen.
DEFAULT_WINDOW_WIDTH = 1320
DEFAULT_WINDOW_HEIGHT = 840

#: Widest usable content column; wider windows centre the page instead
#: of stretching a form across the whole screen.
MAX_CONTENT_WIDTH = 1240

#: Smallest size a chart keeps, so its axes box can never collapse to
#: zero width or height (Matplotlib needs a positive box for aspect
#: calculations and layout).
CHART_MIN_WIDTH = 320
CHART_MIN_HEIGHT = 240

# ---------------------------------------------------------------------------
# Colour tokens
# ---------------------------------------------------------------------------

#: Application background behind the pages.
CANVAS = "#eef1f5"

#: Card and input background.
SURFACE = "#ffffff"

#: Alternating table rows and inset strips.
SURFACE_ALT = "#f6f8fa"

#: Default border colour for cards, inputs, and tables.
BORDER = "#d3dae2"

#: Border colour for inputs that need more emphasis.
BORDER_STRONG = "#b9c4d0"

#: Default text colour.
TEXT = "#22303c"

#: Secondary text colour (descriptions, hints, empty states).
TEXT_MUTED = "#5d6b7a"

#: Primary action colour (buttons, selection, progress).
PRIMARY_COLOR = "#2f6f9f"
PRIMARY_HOVER = "#3a82b6"
PRIMARY_PRESSED = "#275f8a"
PRIMARY_DISABLED = "#a9bccd"
ON_PRIMARY = "#ffffff"

#: Validation problem colour.
ERROR = "#b3261e"
ERROR_BG = "#fdecea"
ERROR_BORDER = "#f2c4bf"

#: Caution colour used for persistent notices.
WARNING = "#8a5a00"
WARNING_BG = "#fff6e5"
WARNING_BORDER = "#eedcb4"

#: Informational strip background.
INFO_BG = "#eef3f8"

#: Highlight behind text the user selected for copying.
SELECTION_BG = "#cfe3f3"

#: Matplotlib figure background, kept equal to :data:`SURFACE`.
CHART_BG = "#ffffff"

# ---------------------------------------------------------------------------
# Style sheet
# ---------------------------------------------------------------------------


def _selector(class_name: str, ids: tuple[str, ...], state: str = "") -> str:
    """Return a comma-separated selector list for the given widget ids.

    The interaction state is appended to every id, so one shared hover,
    pressed, or disabled rule covers each button it applies to.
    """
    return ", ".join(f"{class_name}#{name}{state}" for name in ids)


#: Buttons rendered as the single main action of their section.
_PRIMARY_IDS = (
    "primaryButton",
    "trainScratchButton",
    "predictButton",
    "evaluateButton",
    "applyButton",
    "browseButton",
    "compareButton",
)

#: Buttons that replace or discard state and need a warning style.
_DANGER_IDS = ("dangerButton", "cancelTrainingButton")

#: Declarations shared by every primary action button.
_PRIMARY_DECL = (
    f"background: {PRIMARY_COLOR}; color: {ON_PRIMARY}; "
    f"border: 1px solid {PRIMARY_PRESSED}; font-weight: 600;"
)

#: Declarations shared by every danger (state-replacing) action button.
_DANGER_DECL = (
    f"color: {ERROR}; border-color: {ERROR_BORDER}; background: {SURFACE};"
)

#: ``(selector, declarations)`` pairs forming the application style sheet.
#: Font *sizes* and weights are only set on leaf labels and titles: a font
#: declared on a container widget would be inherited by every child.
_RULES: tuple[tuple[str, str], ...] = (
    ("QWidget", f"color: {TEXT};"),
    ("QMainWindow", f"background: {CANVAS};"),
    ("QWidget#centralArea", f"background: {CANVAS};"),
    ("QWidget#pageContent", "background: transparent;"),
    # Headings and explanatory text.
    (
        "QLabel#pageTitle",
        f"font-size: 15pt; font-weight: 600; color: {TEXT};",
    ),
    (
        "QLabel#appTitle",
        f"font-size: 11pt; font-weight: 600; color: {TEXT};",
    ),
    ("QLabel#appSubtitle", f"font-size: 8pt; color: {TEXT_MUTED};"),
    ("QLabel#hintText", f"color: {TEXT_MUTED};"),
    ("QLabel#infoText", f"color: {TEXT};"),
    # Highlight for text the user selected so copying is visible.
    ("QLabel::selected", f"background: {SELECTION_BG}; color: {TEXT};"),
    (
        "QLabel#statusText",
        f"color: {TEXT}; background: {INFO_BG}; "
        f"border: 1px solid {BORDER}; border-radius: 4px; padding: 6px;",
    ),
    (
        "QLabel#errorText",
        f"color: {ERROR}; background: {ERROR_BG}; "
        f"border: 1px solid {ERROR_BORDER}; border-radius: 4px; padding: 6px;",
    ),
    (
        "QLabel#warningText",
        f"color: {WARNING}; background: {WARNING_BG}; "
        f"border: 1px solid {WARNING_BORDER}; border-radius: 4px; padding: 6px;",
    ),
    # Cards: every group box in the application is a titled panel.
    (
        "QGroupBox",
        f"background: {SURFACE}; border: 1px solid {BORDER}; "
        "border-radius: 6px; margin-top: 16px; padding: 10px 10px 8px 10px;",
    ),
    (
        "QGroupBox::title",
        f"subcontrol-origin: margin; subcontrol-position: top left; "
        f"left: 12px; padding: 0 4px; color: {TEXT}; font-weight: 600;",
    ),
    # Navigation.
    (
        "QListWidget#navigationList",
        f"background: {SURFACE}; border: 1px solid {BORDER}; "
        "border-radius: 6px; padding: 6px; outline: none;",
    ),
    (
        "QListWidget#navigationList::item",
        f"padding: 8px 10px; border-radius: 4px; color: {TEXT};",
    ),
    (
        "QListWidget#navigationList::item:hover:!selected",
        f"background: {INFO_BG};",
    ),
    (
        "QListWidget#navigationList::item:selected",
        f"background: {PRIMARY_COLOR}; color: {ON_PRIMARY}; font-weight: 600;",
    ),
    # Buttons: one primary action per section, everything else secondary.
    (
        "QPushButton",
        f"background: {SURFACE}; border: 1px solid {BORDER_STRONG}; "
        "border-radius: 4px; padding: 5px 14px; min-height: 20px;",
    ),
    ("QPushButton:hover", f"background: {INFO_BG}; border-color: {PRIMARY_COLOR};"),
    ("QPushButton:pressed", f"background: {SURFACE_ALT};"),
    ("QPushButton:disabled", "color: #9aa7b4; background: #f1f3f5;"),
    (_selector("QPushButton", _PRIMARY_IDS), _PRIMARY_DECL),
    (
        _selector("QPushButton", _PRIMARY_IDS, ":hover"),
        f"background: {PRIMARY_HOVER};",
    ),
    (
        _selector("QPushButton", _PRIMARY_IDS, ":pressed"),
        f"background: {PRIMARY_PRESSED};",
    ),
    (
        _selector("QPushButton", _PRIMARY_IDS, ":disabled"),
        f"background: {PRIMARY_DISABLED}; color: #f0f4f7; "
        f"border-color: {PRIMARY_DISABLED};",
    ),
    (_selector("QPushButton", _DANGER_IDS), _DANGER_DECL),
    (
        _selector("QPushButton", _DANGER_IDS, ":hover"),
        f"background: {ERROR_BG}; border-color: {ERROR};",
    ),
    (
        _selector("QPushButton", _DANGER_IDS, ":disabled"),
        "color: #9aa7b4; background: #f1f3f5;",
    ),
    # Inputs.
    (
        "QLineEdit, QSpinBox, QComboBox, QPlainTextEdit",
        f"background: {SURFACE}; border: 1px solid {BORDER_STRONG}; "
        "border-radius: 4px; padding: 4px;",
    ),
    (
        "QLineEdit:focus, QSpinBox:focus, QComboBox:focus, "
        "QPlainTextEdit:focus",
        f"border: 1px solid {PRIMARY_COLOR};",
    ),
    ("QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled", "color: #9aa7b4;"),
    # Tables.
    (
        "QTableWidget",
        f"background: {SURFACE}; alternate-background-color: {SURFACE_ALT}; "
        f"gridline-color: {BORDER}; border: 1px solid {BORDER}; "
        "border-radius: 4px;",
    ),
    (
        "QTableWidget::item:selected",
        f"background: {PRIMARY_HOVER}; color: {ON_PRIMARY};",
    ),
    (
        "QHeaderView::section",
        f"background: {INFO_BG}; color: {TEXT}; font-weight: 600; "
        f"border: none; border-right: 1px solid {BORDER}; "
        f"border-bottom: 1px solid {BORDER}; padding: 5px;",
    ),
    # Tabs and splitters.
    ("QTabWidget::pane", f"border: 1px solid {BORDER}; border-radius: 4px; "
     f"background: {SURFACE}; top: -1px;"),
    (
        "QTabBar::tab",
        f"background: {INFO_BG}; color: {TEXT_MUTED}; "
        f"border: 1px solid {BORDER}; border-bottom: none; "
        "border-top-left-radius: 4px; border-top-right-radius: 4px; "
        "padding: 6px 14px; margin-right: 2px;",
    ),
    (
        "QTabBar::tab:selected",
        f"background: {SURFACE}; color: {TEXT}; font-weight: 600;",
    ),
    ("QTabBar::tab:hover:!selected", f"background: {SURFACE_ALT};"),
    ("QSplitter::handle", f"background: {BORDER};"),
    ("QSplitter::handle:hover", f"background: {PRIMARY_COLOR};"),
    ("QScrollArea", "border: none; background: transparent;"),
    # Progress feedback.
    (
        "QProgressBar",
        f"background: {SURFACE}; border: 1px solid {BORDER_STRONG}; "
        "border-radius: 4px; text-align: center; min-height: 16px;",
    ),
    ("QProgressBar::chunk", f"background: {PRIMARY_COLOR}; border-radius: 3px;"),
    ("QToolTip", f"background: {SURFACE}; color: {TEXT}; "
     f"border: 1px solid {BORDER_STRONG}; padding: 4px;"),
)


def application_stylesheet() -> str:
    """Return the application-wide Qt style sheet as one string."""
    return "\n".join(
        f"{selector} {{\n    {declarations}\n}}" for selector, declarations in _RULES
    )


def apply_theme(app: QApplication) -> QApplication:
    """Install the shared style sheet on ``app`` (idempotent).

    Args:
        app: The shared application instance.

    Returns:
        The same application, so callers can chain the call.
    """
    app.setStyleSheet(application_stylesheet())
    return app