"""Reusable page scaffolding for the Neural Network Lab interface.

Every page in the application is built from the same small set of
presentation components so that spacing, headings, card styling, and
feedback messages stay consistent:

- :class:`ScrollablePage` — a titled page whose content scrolls instead
  of being clipped on a small window.
- :class:`PageHeader` — the page title and its one-line explanation.
- :class:`SectionCard` — a titled panel with an optional description and
  a body layout to group related controls and results.
- :class:`FeedbackArea` — a status line above an error line, both hidden
  while they carry no message.
- :class:`MessageLabel` — a word-wrapped label that collapses when empty.
- :func:`make_button` — a consistently sized primary/secondary/danger
  action button.

The module is presentation only: it holds no application state and no
neural-network mathematics.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame
from PySide6.QtWidgets import QGroupBox
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QScrollArea
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget

from neural_network_lab.gui.widgets.ui_theme import CARD_MARGIN
from neural_network_lab.gui.widgets.ui_theme import CONTENT_MARGIN
from neural_network_lab.gui.widgets.ui_theme import CONTENT_SPACING
from neural_network_lab.gui.widgets.ui_theme import FIELD_SPACING
from neural_network_lab.gui.widgets.ui_theme import MAX_CONTENT_WIDTH

#: Object names used by the style sheet for each message kind.
MESSAGE_OBJECT_NAMES: dict[str, str] = {
    "info": "infoText",
    "hint": "hintText",
    "status": "statusText",
    "warning": "warningText",
    "error": "errorText",
}

#: Object names used by the style sheet for each button kind.
BUTTON_OBJECT_NAMES: dict[str, str] = {
    "primary": "primaryButton",
    "secondary": "secondaryButton",
    "danger": "dangerButton",
}

#: Interaction flags that make display-only text selectable and copyable
#: with the mouse and the keyboard, without turning the label editable.
SELECTABLE_TEXT_FLAGS = (
    Qt.TextInteractionFlag.TextSelectableByMouse
    | Qt.TextInteractionFlag.TextSelectableByKeyboard
)


def make_selectable(widget: QWidget):
    """Let the user select and copy the text of a display-only widget.

    Read-only text stays read-only: this only adds text interaction, so
    messages, metrics, and summaries can be copied. Editable inputs are
    never passed here.
    """
    widget.setTextInteractionFlags(SELECTABLE_TEXT_FLAGS)
    widget.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    return widget


class MessageLabel(QLabel):
    """A word-wrapped label that hides itself while it has no message.

    Collapsing when empty keeps pages free of empty coloured strips, so
    the surrounding layout closes up around the content that is actually
    present. Its text is selectable and copyable, so a long error or
    result line can be taken away in full.
    """

    def __init__(self, kind: str = "info", parent: QWidget | None = None) -> None:
        """Create an empty label styled for ``kind``.

        Args:
            kind: One of :data:`MESSAGE_OBJECT_NAMES`; unknown kinds
                fall back to the neutral information style.
            parent: Optional parent widget.
        """
        super().__init__("", parent)
        object_name = MESSAGE_OBJECT_NAMES.get(
            kind, MESSAGE_OBJECT_NAMES["info"]
        )
        self.setObjectName(object_name)
        self.setWordWrap(True)
        # ``MinimumExpanding`` keeps a wrapped label from demanding the
        # full text width, which would otherwise fight splitters.
        self.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum
        )
        make_selectable(self)
        self.setVisible(False)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        """Show the label only when ``text`` carries a message."""
        super().setText(text)
        self.setVisible(bool(text and text.strip()))


class FeedbackArea(QWidget):
    """A status line above an error line, both optional.

    Views keep their existing ``_status_label`` / ``_error_label``
    references pointing at these two labels, so feedback presentation is
    shared while every view stays in control of its own messages.
    """

    def __init__(
        self, parent: QWidget | None = None, *, status_text: str = ""
    ) -> None:
        """Create the area, optionally showing an initial status message."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(FIELD_SPACING)
        self._status = MessageLabel("status", self)
        self._error = MessageLabel("error", self)
        if status_text:
            self._status.setText(status_text)
        layout.addWidget(self._status)
        layout.addWidget(self._error)

    def status_label(self) -> MessageLabel:
        """Return the status line (operation results and progress)."""
        return self._status

    def error_label(self) -> MessageLabel:
        """Return the error line (validation and operation failures)."""
        return self._error

    def show_status(self, message: str) -> None:
        """Replace the status message."""
        self._status.setText(message)

    def clear_status(self) -> None:
        """Clear the status message."""
        self._status.setText("")

    def show_error(self, message: str) -> None:
        """Replace the error message."""
        self._error.setText(message)

    def clear_error(self) -> None:
        """Clear the error message."""
        self._error.setText("")


class PageHeader(QWidget):
    """The page title plus a one-line explanation of the page.

    Both parts are selectable so a user can copy a section's name or
    its explanation.
    """

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        parent: QWidget | None = None,
    ) -> None:
        """Create the heading block for one page."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._title = QLabel(title, self)
        self._title.setObjectName("pageTitle")
        self._title.setWordWrap(True)
        make_selectable(self._title)
        layout.addWidget(self._title)
        self._subtitle = MessageLabel("hint", self)
        self._subtitle.setText(subtitle)
        layout.addWidget(self._subtitle)

    def title_label(self) -> QLabel:
        """Return the page title label."""
        return self._title

    def subtitle_label(self) -> MessageLabel:
        """Return the page subtitle label."""
        return self._subtitle


class SectionCard(QGroupBox):
    """A titled panel that groups related controls or results."""

    def __init__(
        self,
        title: str = "",
        *,
        description: str = "",
        parent: QWidget | None = None,
    ) -> None:
        """Create the card with its title, description, and body layout."""
        super().__init__(title, parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            CARD_MARGIN, CARD_MARGIN, CARD_MARGIN, CARD_MARGIN
        )
        outer.setSpacing(CONTENT_SPACING)
        self._description = MessageLabel("hint", self)
        self._description.setText(description)
        outer.addWidget(self._description)
        self._body = QVBoxLayout()
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(FIELD_SPACING)
        outer.addLayout(self._body)

    def body(self) -> QVBoxLayout:
        """Return the layout callers add their own content to."""
        return self._body

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        """Add ``widget`` to the card body and return it."""
        self._body.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout) -> None:
        """Add a prepared layout to the card body."""
        self._body.addLayout(layout)

    def set_description(self, text: str) -> None:
        """Replace the card's explanatory line."""
        self._description.setText(text)


class ScrollablePage(QWidget):
    """A titled page whose content scrolls when the window is small.

    The page keeps a page-width limit and centres the column, so a wide
    monitor does not stretch a short form across the whole screen, while
    a narrow window scrolls instead of clipping controls or charts.
    """

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        parent: QWidget | None = None,
    ) -> None:
        """Create the scrollable page shell with its header."""
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._content = QWidget()
        self._content.setObjectName("pageContent")
        self._content.setMaximumWidth(MAX_CONTENT_WIDTH)
        self._body = QVBoxLayout(self._content)
        self._body.setContentsMargins(
            CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN, CONTENT_MARGIN
        )
        self._body.setSpacing(CONTENT_SPACING)
        self._header = PageHeader(title, subtitle, self._content)
        self._body.addWidget(self._header)
        self._finished = False
        frame = QWidget()
        frame_layout = QHBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        frame_layout.addStretch(1)
        frame_layout.addWidget(self._content, 10)
        frame_layout.addStretch(1)
        self._scroll.setWidget(frame)
        outer.addWidget(self._scroll)

    def header(self) -> PageHeader:
        """Return the page header block."""
        return self._header

    def body(self) -> QVBoxLayout:
        """Return the layout that page content is added to."""
        return self._body

    def content(self) -> QWidget:
        """Return the page content widget (the scroll area's column)."""
        return self._content

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        """Add ``widget`` to the page body and return it."""
        self._body.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout, stretch: int = 0) -> None:
        """Add a prepared layout to the page body.

        Args:
            layout: The layout to add.
            stretch: Stretch factor, mirroring :meth:`add`, so a page can
                let a layout-based row take the leftover window height.
        """
        self._body.addLayout(layout, stretch)

    def finish(self) -> None:
        """Append a trailing stretch for pages that stay top-aligned.

        Pages whose main content should grow with the window (a table,
        a chart pane, a splitter) give that widget a stretch factor
        instead and do not call this, because inside a scroll area the
        extra height is handed to stretch items, not to the last
        widget.
        """
        if not self._finished:
            self._body.addStretch(1)
            self._finished = True


def install_page(
    host: QWidget, title: str, subtitle: str = ""
) -> ScrollablePage:
    """Create a :class:`ScrollablePage` and fill ``host`` with it.

    Views use this instead of setting their own layout so the page always
    fills the window area it is placed in.

    Args:
        host: The view widget that becomes the page container.
        title: Page title.
        subtitle: One-line explanation under the title.

    Returns:
        The created page.
    """
    outer = QVBoxLayout(host)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    page = ScrollablePage(title, subtitle, host)
    outer.addWidget(page)
    return page


def make_button(
    text: str,
    *,
    kind: str = "secondary",
    tooltip: str = "",
    parent: QWidget | None = None,
) -> QPushButton:
    """Create a consistently sized action button.

    Args:
        text: Button label.
        kind: ``"primary"`` for the single main action of a section,
            ``"danger"`` for state-replacing or destructive actions, and
            ``"secondary"`` for everything else.
        tooltip: Optional explanatory tooltip.
        parent: Optional parent widget.

    Returns:
        The configured button.
    """
    button = QPushButton(text, parent)
    object_name = BUTTON_OBJECT_NAMES.get(
        kind, BUTTON_OBJECT_NAMES["secondary"]
    )
    button.setObjectName(object_name)
    if tooltip:
        button.setToolTip(tooltip)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


def button_row(*widgets: QWidget, trailing_stretch: bool = True) -> QHBoxLayout:
    """Return a horizontal layout holding ``widgets`` side by side."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(FIELD_SPACING)
    for widget in widgets:
        row.addWidget(widget)
    if trailing_stretch:
        row.addStretch(1)
    return row


def hint_text(text: str, parent: QWidget | None = None) -> MessageLabel:
    """Return a muted explanatory label (hidden when ``text`` is empty)."""
    label = MessageLabel("hint", parent)
    label.setText(text)
    return label


def selectable_label(
    text: str, parent: QWidget | None = None
) -> QLabel:
    """Return a word-wrapped, selectable label for display-only text.

    Used for informational labels that are not part of a form row, such
    as the dataset-driven input size on the network configuration page.
    """
    label = QLabel(text, parent)
    label.setWordWrap(True)
    return make_selectable(label)