"""Dashboard: a read-only overview of the current application state.

Every value shown here comes from the existing pages and services
through the injected providers (the dataset editor's applied dataset,
the training view's model, run state, and last-run summary, and the
experiment view's loaded records). The dashboard computes nothing,
stores nothing, and never invents metrics: when a piece of state does
not exist yet it says so and names the page that owns it.

The page is organised as a grid of small status cards plus one
"next step" card whose buttons navigate to the sections that resolve
the current state, so the workflow is discoverable without hunting
through the navigation list.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QAbstractItemView
from PySide6.QtWidgets import QGridLayout
from PySide6.QtWidgets import QTableWidget
from PySide6.QtWidgets import QTableWidgetItem
from PySide6.QtWidgets import QWidget

from neural_network_lab.data.dataset import Dataset
from neural_network_lab.data.dataset import TARGET_KIND_LABELS
from neural_network_lab.gui.widgets.page_scaffold import FeedbackArea
from neural_network_lab.gui.widgets.page_scaffold import MessageLabel
from neural_network_lab.gui.widgets.page_scaffold import SectionCard
from neural_network_lab.gui.widgets.page_scaffold import button_row
from neural_network_lab.gui.widgets.page_scaffold import install_page
from neural_network_lab.gui.widgets.page_scaffold import make_button
from neural_network_lab.gui.widgets.ui_theme import CONTENT_SPACING

#: Text shown wherever a piece of state does not exist yet.
UNKNOWN_TEXT = "—"

#: Text used for a metric the run did not record (never shown as zero).
NOT_RECORDED_TEXT = "— (not recorded)"

#: Precision used for every floating-point value in the results table.
METRIC_PRECISION = 6

#: Sections the dashboard can send the user to, in workflow order.
NAVIGATION_TARGETS: tuple[tuple[str, str], ...] = (
    ("datasets", "Dataset Management"),
    ("network", "Network Configuration"),
    ("training", "Training"),
    ("testing", "Testing"),
    ("visualization", "Visualization"),
    ("experiments", "Experiment History"),
)


def format_number(value: object) -> str:
    """Format a count, marking absent values with a dash."""
    if value is None:
        return UNKNOWN_TEXT
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return repr(value)


def format_layers(sizes: object) -> str:
    """Format layer sizes as ``[n0, n1, ...]`` or a dash."""
    if not sizes:
        return UNKNOWN_TEXT
    try:
        return "[" + ", ".join(str(int(size)) for size in sizes) + "]"  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return repr(sizes)


def format_metric(value: object) -> str:
    """Format a measured number with one consistent, short precision."""
    if value is None:
        return NOT_RECORDED_TEXT
    try:
        return f"{float(value):.{METRIC_PRECISION}g}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def format_count(value: object) -> str:
    """Format an integer count, marking absent values clearly."""
    if value is None:
        return NOT_RECORDED_TEXT
    try:
        return str(int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def format_hidden_layers(sizes: object) -> str:
    """Describe hidden layers, or state that there are none."""
    try:
        hidden = tuple(int(size) for size in sizes)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return NOT_RECORDED_TEXT
    if not hidden:
        return "none (inputs map straight to outputs)"
    plural = "" if len(hidden) == 1 else " layers"
    return f"{len(hidden)} hidden{plural}: [{', '.join(str(size) for size in hidden)}]"


class DashboardView(QWidget):
    """Summarize the model, dataset, training, and experiment state."""

    def __init__(self, parent=None) -> None:
        """Create the status cards, next-step actions, and refresh."""
        super().__init__(parent)
        self._dataset_provider: Callable[[], Dataset | None] = lambda: None
        self._network_provider: Callable[[], object | None] = lambda: None
        self._layer_sizes_provider: Callable[[], object | None] = lambda: None
        self._training_active_provider: Callable[[], bool] = lambda: False
        self._last_run_provider: Callable[[], object] = lambda: None
        self._experiment_state_provider: (
            Callable[[], tuple[int, int, str | None]]
        ) = lambda: (0, 0, None)
        self._navigator: Callable[[str], bool] = lambda key: False

        self._page = install_page(
            self,
            "Dashboard",
            "Live overview of the dataset, network, training run, and "
            "saved experiments. Every value is read from the section "
            "that owns it; nothing here is estimated.",
        )

        cards = QGridLayout()
        cards.setContentsMargins(0, 0, 0, 0)
        cards.setSpacing(CONTENT_SPACING)
        self._card_position = 0
        self._cards_layout = cards
        self._dataset_label = self._status_card(
            "Dataset", "The dataset other sections read. No dataset loaded."
        )
        self._model_label = self._status_card(
            "Network", "No model yet — configure one in Network Configuration."
        )
        self._match_label = self._status_card(
            "Dataset and network compatibility",
            "Unknown — a model and a labeled dataset are both needed.",
        )
        self._training_label = self._status_card(
            "Training", "Idle — no training run in progress."
        )
        self._page.add_layout(cards)

        run_card = SectionCard(
            "Last training run",
            description=(
                "Every value is one the run itself recorded. Training, "
                "validation, and held-out test results are listed "
                "separately, and anything the run did not measure is "
                "marked as not recorded."
            ),
            parent=self._page,
        )
        self._run_table = QTableWidget(run_card)
        self._run_table.setObjectName("dashboardRunTable")
        self._run_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._run_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._run_table.setAlternatingRowColors(True)
        self._run_table.setWordWrap(False)
        self._run_table.setMinimumHeight(220)
        self._run_table.setColumnCount(3)
        self._run_table.setHorizontalHeaderLabels(["Section", "Field", "Value"])
        self._run_table.verticalHeader().setVisible(False)
        self._run_table.horizontalHeader().setStretchLastSection(True)
        run_card.add(self._run_table, 1)
        self._page.add(run_card)

        self._experiments_label = self._status_card(
            "Experiments", "Experiment History not opened yet.", standalone=True
        )

        next_card = SectionCard(
            "Next step",
            description=(
                "The recommended action for the current state, with "
                "buttons that open the section that performs it."
            ),
            parent=self._page,
        )
        self._next_label = MessageLabel("info", next_card)
        self._next_label.setText(UNKNOWN_TEXT)
        next_card.add(self._next_label)
        self._go_button = make_button(
            "Go to section",
            kind="primary",
            parent=next_card,
        )
        self._go_button.clicked.connect(self._on_go_clicked)
        next_card.add_layout(button_row(self._go_button))
        self._target_buttons: dict[str, QWidget] = {}
        target_row = button_row()
        for key, title in NAVIGATION_TARGETS:
            button = make_button(
                title,
                kind="secondary",
                tooltip=f"Open {title}.",
                parent=next_card,
            )
            button.clicked.connect(
                lambda _checked=False, target=key: self._navigate(target)
            )
            self._target_buttons[key] = button
            target_row.addWidget(button)
        next_card.add_layout(target_row)
        self._recommended_key: str | None = None
        self._page.add(next_card)

        self._refresh_button = make_button(
            "Refresh overview",
            kind="secondary",
            tooltip="Re-read the current state from each section.",
            parent=self._page,
        )
        self._refresh_button.clicked.connect(self.refresh)
        self._feedback = FeedbackArea(parent=self._page)
        self._error_label = self._feedback.error_label()
        self._page.add_layout(button_row(self._refresh_button))
        self._page.add(self._feedback)
        self._page.finish()

    def _status_card(
        self, title: str, initial: str, *, standalone: bool = False
    ) -> MessageLabel:
        """Create a status card for the overview grid (or full width).

        Args:
            title: Card title.
            initial: Value shown before anything is known.
            standalone: Place the card on its own row instead of in the
                two-column overview grid.

        Returns:
            The card's value label.
        """
        card = SectionCard(title, parent=self._page)
        value = MessageLabel("info", card)
        value.setText(initial)
        card.add(value)
        if standalone:
            self._page.add(card)
        else:
            index = self._card_position
            self._cards_layout.addWidget(card, index // 2, index % 2)
            self._card_position = index + 1
        return value

    def set_dataset_provider(
        self, provider: Callable[[], Dataset | None]
    ) -> None:
        """Set the callable supplying the dataset editor's dataset."""
        self._dataset_provider = provider

    def set_network_provider(
        self, provider: Callable[[], object | None]
    ) -> None:
        """Set the callable supplying the training view's model."""
        self._network_provider = provider

    def set_layer_sizes_provider(
        self, provider: Callable[[], object | None]
    ) -> None:
        """Set the callable supplying the model's layer sizes."""
        self._layer_sizes_provider = provider

    def set_training_active_provider(
        self, provider: Callable[[], bool]
    ) -> None:
        """Set the callable reporting whether training is running."""
        self._training_active_provider = provider

    def set_last_run_provider(
        self, provider: Callable[[], object]
    ) -> None:
        """Set the callable supplying the last finished run's data.

        The provider may return the run's structured field mapping (see
        :meth:`TrainingView.last_run_details`) or, for compatibility, its
        one-line summary text; both are rendered, the structured form as
        a table.
        """
        self._last_run_provider = provider

    def set_experiment_state_provider(
        self, provider: Callable[[], tuple[int, int, str | None]]
    ) -> None:
        """Set the callable supplying the experiment-history state.

        The returned tuple is ``(loaded records, skipped files,
        directory)``; an empty directory means nothing was loaded.
        """
        self._experiment_state_provider = provider

    def set_navigator(self, navigator: Callable[[str], bool]) -> None:
        """Set the callable opening another section by navigation key."""
        self._navigator = navigator

    def show_error(self, message: str) -> None:
        """Present an error inline."""
        self._error_label.setText(message)

    def clear_error(self) -> None:
        """Clear any inline error message."""
        self._error_label.setText("")

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Re-read every state source whenever the page becomes visible."""
        super().showEvent(event)
        self.refresh()

    def _safe_call(self, provider, fallback):
        """Call a provider, reporting failures instead of raising."""
        try:
            return provider()
        except Exception as exc:
            self.show_error(f"Could not read the current state: {exc}")
            return fallback

    def refresh(self) -> None:
        """Re-read every state source and redraw the cards."""
        self.clear_error()
        dataset = self._safe_call(self._dataset_provider, None)
        if not isinstance(dataset, Dataset):
            dataset = None
        network = self._safe_call(self._network_provider, None)
        layer_sizes = self._safe_call(self._layer_sizes_provider, None)
        training_active = bool(
            self._safe_call(self._training_active_provider, False)
        )
        last_run = self._safe_call(self._last_run_provider, None)
        records, skipped, directory = self._safe_call(
            self._experiment_state_provider, (0, 0, None)
        )
        self._refresh_dataset(dataset)
        self._refresh_model(network, layer_sizes)
        self._refresh_match(network, dataset)
        self._refresh_training(training_active)
        self._refresh_last_run(last_run)
        self._refresh_experiments(records, skipped, directory)
        key, message = self._next_step(
            dataset=dataset,
            has_model=network is not None,
            compatible=self._is_compatible(network, dataset),
            training_active=training_active,
        )
        self._refresh_next_step(key, message)

    def _refresh_dataset(self, dataset: Dataset | None) -> None:
        """Describe the dataset editor's applied dataset."""
        if dataset is None:
            self._dataset_label.setText(
                "No dataset loaded — validate one in Dataset Management."
            )
            return
        target_text = "labeled" if dataset.has_targets else "unlabeled"
        task_text = (
            ""
            if not dataset.has_targets
            else f" · {TARGET_KIND_LABELS.get(dataset.target_kind, '')}"
        )
        self._dataset_label.setText(
            f"{dataset.n_samples} samples · {dataset.n_features} features "
            f"· {target_text}{task_text}"
        )

    def _refresh_model(self, network: object | None, layer_sizes) -> None:
        """Describe the training view's current model."""
        if network is None:
            self._model_label.setText(
                "No model yet — configure one in Network Configuration, "
                "then train it."
            )
            return
        shown = layer_sizes if layer_sizes else UNKNOWN_TEXT
        self._model_label.setText(f"active · layers {format_layers(shown)}")

    @staticmethod
    def _is_compatible(network: object | None, dataset: Dataset | None) -> bool | None:
        """Return whether model and dataset match, or None if unknown.

        Uses the dataset's own compatibility check; nothing is trained
        or modified here.
        """
        if network is None or dataset is None:
            return None
        layer_sizes = getattr(network, "layer_sizes", None)
        if not layer_sizes:
            return None
        sizes = tuple(int(size) for size in layer_sizes)
        checker = getattr(dataset, "is_compatible_with", None)
        if not callable(checker):
            return None
        try:
            return bool(checker(sizes[0], sizes[-1] if dataset.has_targets else None))
        except (TypeError, ValueError):
            return None

    def _refresh_match(
        self, network: object | None, dataset: Dataset | None
    ) -> None:
        """Report whether the model and the dataset can be used together."""
        compatible = self._is_compatible(network, dataset)
        unlabeled = dataset is not None and not dataset.has_targets
        if compatible is None:
            self._match_label.setText(
                "Unknown — a model and a labeled dataset are both needed."
            )
        elif compatible:
            if unlabeled:
                self._match_label.setText(
                    "Input sizes match (the dataset has no targets, so the "
                    "output size is not checked)."
                )
            else:
                self._match_label.setText(
                    "Compatible: the model's input and output sizes match "
                    "the dataset."
                )
        else:
            self._match_label.setText(
                "Incompatible: the dataset's features or target columns "
                "do not match the model; train again or load a matching "
                "model."
            )

    def _refresh_training(self, training_active: bool) -> None:
        """Report the training view's run state."""
        if training_active:
            self._training_label.setText(
                "A training run is in progress — progress is reported on "
                "the Training page."
            )
        else:
            self._training_label.setText("Idle — no training run in progress.")

    def _refresh_last_run(self, last_run: object) -> None:
        """Render the last finished run's recorded fields as a table."""
        if isinstance(last_run, str):
            # A plain summary (legacy provider): keep it on one row.
            rows = [("Run", "Summary", last_run or NOT_RECORDED_TEXT)]
        elif isinstance(last_run, dict):
            rows = self._run_rows(last_run)
        else:
            rows = [("Run", "Status", "No training run finished yet.")]
        table = self._run_table
        table.setRowCount(len(rows))
        for row_index, (section, field, value) in enumerate(rows):
            for column, text in enumerate((section, field, value)):
                table.setItem(row_index, column, QTableWidgetItem(str(text)))
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)

    @staticmethod
    def _run_rows(details: dict) -> list[tuple[str, str, str]]:
        """Turn a run's recorded fields into ``(section, field, value)`` rows.

        A value the run did not record — or that a legacy record did not
        carry — is rendered as "not recorded" instead of being guessed.
        """
        mode = str(details.get("mode") or "")
        mode_text = {
            "from_scratch": "from scratch (previous parameters replaced)",
            "continued": "continued (parameters preserved as starting point)",
        }.get(mode, mode or NOT_RECORDED_TEXT)
        experiment_id = details.get("experiment_id")
        record_path = details.get("record_path")
        model_path = details.get("model_path")
        if record_path:
            saved = f"experiment record: {record_path}"
        else:
            saved = "experiment record not saved"
        if model_path:
            saved += f" · model state: {model_path}"
        test_rows = [
            ("Test", "Held-out samples", format_count(details.get("test_samples"))),
            ("Test", "Test MSE", format_metric(details.get("test_mse"))),
            ("Test", "Test MAE", format_metric(details.get("test_mae"))),
            ("Test", "Test RMSE", format_metric(details.get("test_rmse"))),
        ]
        if details.get("test_accuracy") is None:
            test_rows.append(
                (
                    "Test",
                    "Test accuracy",
                    "not applicable (continuous targets)",
                )
            )
        else:
            test_rows.append(
                ("Test", "Test accuracy", format_metric(details["test_accuracy"]))
            )
        return [
            (
                "Experiment",
                "Experiment id",
                str(experiment_id) if experiment_id else NOT_RECORDED_TEXT,
            ),
            (
                "Experiment",
                "Recorded at",
                str(details.get("created_at") or NOT_RECORDED_TEXT),
            ),
            ("Experiment", "Training mode", mode_text),
            (
                "Experiment",
                "Parameters preserved",
                "yes" if details.get("preserved_parameters") else "no",
            ),
            ("Dataset", "Training samples",
             format_count(details.get("train_samples"))),
            ("Dataset", "Input features",
             format_count(details.get("train_features"))),
            ("Dataset", "Target columns",
             format_count(details.get("train_outputs"))),
            (
                "Dataset",
                "Declared task",
                str(details.get("task_label") or NOT_RECORDED_TEXT),
            ),
            ("Model", "Architecture", format_layers(details.get("layer_sizes"))),
            ("Model", "Inputs", format_count(details.get("n_inputs"))),
            ("Model", "Hidden layers",
             format_hidden_layers(details.get("hidden_layers"))),
            ("Model", "Outputs", format_count(details.get("n_outputs"))),
            ("Training", "Learning rate",
             format_metric(details.get("learning_rate"))),
            ("Training", "Epochs", format_count(details.get("epochs"))),
            ("Training", "Final epoch training MSE",
             format_metric(details.get("final_train_mse"))),
            ("Validation", "Validation rows",
             format_count(details.get("validation_samples"))),
            ("Validation", "Final epoch validation MSE",
             format_metric(details.get("final_validation_mse"))),
            ("Validation", "Best epoch",
             format_count(details.get("best_epoch"))),
            ("Validation", "Best validation MSE",
             format_metric(details.get("best_validation_mse"))),
            *test_rows,
            ("Saved", "Artifacts", saved),
        ]

    def _refresh_experiments(
        self, records: int, skipped: int, directory: str | None
    ) -> None:
        """Report what the Experiment History page has loaded."""
        if not directory:
            self._experiments_label.setText(
                "Experiment History not opened yet — open it to load saved "
                "records."
            )
            return
        text = f"{format_number(records)} record(s) loaded from {directory}"
        if skipped:
            text += f" · {format_number(skipped)} file(s) skipped"
        self._experiments_label.setText(text)

    @staticmethod
    def _next_step(
        *,
        dataset: Dataset | None,
        has_model: bool,
        compatible: bool | None,
        training_active: bool,
    ) -> tuple[str, str]:
        """Return ``(navigation key, advice)`` for the current state."""
        if training_active:
            return (
                "training",
                "Training is running — watch the Training page for "
                "per-epoch progress.",
            )
        if dataset is None:
            return (
                "datasets",
                "Select and validate a dataset in Dataset Management "
                "(built-in datasets are available there).",
            )
        if not has_model:
            return (
                "training",
                "Train a model in Training: it builds the architecture "
                "from the dataset and your settings.",
            )
        if compatible is False:
            return (
                "training",
                "Retrain from the Training page so the model matches the "
                "current dataset, or load a matching model there.",
            )
        return (
            "testing",
            "Model and dataset are ready: evaluate them in Testing, or "
            "see the decision boundary and architecture in "
            "Visualization.",
        )

    def _refresh_next_step(self, key: str, message: str) -> None:
        """Show the advice and point the action buttons at ``key``."""
        self._next_label.setText(message)
        self._recommended_key = key
        titles = dict(NAVIGATION_TARGETS)
        title = titles.get(key, "section")
        self._go_button.setText(f"Go to {title}")
        self._go_button.setToolTip(f"Open {title}.")
        for target, button in self._target_buttons.items():
            button.setToolTip(
                "Recommended next step."
                if target == key
                else f"Open {titles[target]}."
            )

    def _on_go_clicked(self) -> None:
        """Navigate to the section recommended for the current state."""
        if self._recommended_key is not None:
            self._navigate(self._recommended_key)

    def _navigate(self, key: str) -> None:
        """Ask the window to open ``key``, reporting a refused request."""
        if not self._navigator(key):
            self.show_error(
                f"{key} could not be opened. Resolve any pending dialog "
                "or unsaved-changes prompt first."
            )