"""Matplotlib chart widgets for recorded training and evaluation results.

Display-only: every plotted value comes from recorded
``TrainingResult`` loss histories, ``EvaluationResult``
predictions/targets supplied by the owning views, or the owning view's
prepared grid of model predictions. No loss, gradient, accuracy, or
threshold mathematics beyond reusing the documented classification
threshold for labeling lives here.

Four widgets share this module and its single matplotlib dependency:
``LossHistoryChart`` (training/validation loss), ``PredictionChart``
(per-sample predictions), ``DecisionBoundaryChart`` (two-feature class
regions plus samples), and ``ArchitectureChart`` (the current model's
layers, neurons, and connections).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.patches import Patch
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QToolBar

from neural_network_lab.application.evaluation_service import (
    CLASSIFICATION_THRESHOLD,
)
from neural_network_lab.gui.widgets.ui_theme import CHART_BG
from neural_network_lab.gui.widgets.ui_theme import CHART_MIN_HEIGHT
from neural_network_lab.gui.widgets.ui_theme import CHART_MIN_WIDTH
from neural_network_lab.gui.widgets.ui_theme import TEXT
from neural_network_lab.gui.widgets.ui_theme import TEXT_MUTED

#: Largest layer drawn neuron-by-neuron before the column is summarized.
MAX_DRAWABLE_NEURONS = 12

#: Default chart figure size (inches). Slightly larger than before so
#: tick labels, legends, and inline annotations have room to fit.
FIGURE_SIZE = (5.6, 3.4)

#: Padding (inches of figure space) reserved around the axes content.
FIGURE_PADDING = 0.6

#: Navigation actions kept on the chart toolbar: looking around the
#: figure and nothing else. In particular the "Configure plot options"
#: editor is left out — it rebuilds the figure from its own widgets
#: (clearing the recorded data), offers scale choices that do not apply
#: to these linear plots, and is the source of the box/aspect and
#: log-limit failures reported during testing.
NAVIGATION_ACTIONS = ("home", "back", "forward", "pan", "zoom")


def create_navigation_toolbar(canvas: FigureCanvas, parent=None) -> QToolBar:
    """Return a zoom/pan toolbar for ``canvas``.

    The toolbar is matplotlib's standard Qt navigation toolbar, which
    keeps the chart consistent with the other matplotlib widgets in the
    application. Only the navigation actions are shown: pan, zoom,
    back, forward, and home. Figure editors and file-writing actions are
    hidden, so the toolbar can never discard recorded results or apply
    a scale the plot does not use.

    Args:
        canvas: The chart canvas the toolbar drives.
        parent: Optional parent widget, usually the panel holding the
            canvas.

    Returns:
        The configured toolbar.
    """
    toolbar = NavigationToolbar2QT(canvas, parent)
    for action in toolbar.actions():
        label = f"{action.text()} {action.toolTip()}".strip().lower()
        if not any(word in label for word in NAVIGATION_ACTIONS):
            action.setVisible(False)
    toolbar.setToolTip(
        "Zoom, pan, and step through the chart's recent views."
    )
    return toolbar


def _is_number(value: object) -> bool:
    """Return True for real numeric scalars, excluding booleans."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def validate_loss_history(history: object) -> list[float]:
    """Validate a recorded per-epoch loss history.

    Args:
        history: Non-empty sequence of finite loss values, one per epoch.

    Returns:
        The values as plain Python floats.

    Raises:
        TypeError: For non-sequence input or non-numeric entries.
        ValueError: For empty histories or non-finite values.
    """
    if isinstance(history, (str, bytes)) or not isinstance(history, (list, tuple)):
        try:
            items = list(history)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "loss history must be a sequence of numbers, "
                f"got {type(history).__name__}"
            ) from exc
    else:
        items = list(history)
    if len(items) == 0:
        raise ValueError("loss history is empty; nothing to plot")
    values: list[float] = []
    for index, entry in enumerate(items):
        if not _is_number(entry):
            raise TypeError(
                f"loss history entry {index} is not a number: {entry!r}"
            )
        value = float(entry)
        if not math.isfinite(value):
            raise ValueError(
                f"loss history entry {index} is not finite: {entry!r}"
            )
        values.append(value)
    return values


def _as_single_output_series(values: object, name: str) -> list[float]:
    """Validate one per-sample series as a flat finite float list.

    Accepts flat sequences or ``(n, 1)`` arrays; anything else is
    rejected so multi-output or ragged data can never be misplotted.
    """
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence, got text")
    try:
        items = list(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a numeric sequence, got {type(values).__name__}"
        ) from exc
    try:
        array = np.asarray(items, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only numbers: {exc}") from exc
    if array.ndim == 2:
        if array.shape[1] != 1:
            raise ValueError(
                f"{name} has {array.shape[1]} outputs per sample; "
                "charts support single-output models only"
            )
        array = array[:, 0]
    elif array.ndim != 1:
        raise ValueError(f"{name} must be one value per sample, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} is empty; nothing to plot")
    bad = np.where(~np.isfinite(array))[0]
    if len(bad) > 0:
        raise ValueError(f"{name} entry {int(bad[0])} is not finite")
    return [float(value) for value in array]


def validate_prediction_inputs(
    probabilities: object, targets: object = None
) -> tuple[list[float], list[float] | None]:
    """Validate per-sample prediction (and optional target) series.

    Returns:
        ``(probabilities, targets)`` as plain float lists, with targets
        ``None`` when not supplied.

    Raises:
        TypeError: For non-sequence or non-numeric input.
        ValueError: For empty, ragged, multi-output, mismatched-length,
            or non-finite input.
    """
    probs = _as_single_output_series(probabilities, "probabilities")
    if targets is None:
        return probs, None
    actual = _as_single_output_series(targets, "targets")
    if len(actual) != len(probs):
        raise ValueError(
            f"targets length {len(actual)} does not match "
            f"probabilities length {len(probs)}"
        )
    return probs, actual


def _base_figure() -> tuple[Figure, object]:
    """Create a figure with one axes for chart widgets."""
    figure = Figure(figsize=FIGURE_SIZE, facecolor=CHART_BG)
    return figure, figure.add_subplot(111)


def _style_axes(axes) -> None:
    """Apply the shared chart typography to one axes object."""
    axes.tick_params(labelsize=8, colors=TEXT_MUTED)
    for label in (axes.title, axes.xaxis.label, axes.yaxis.label):
        label.set_fontsize(9)
    axes.xaxis.label.set_color(TEXT_MUTED)
    axes.yaxis.label.set_color(TEXT_MUTED)
    axes.title.set_color(TEXT)


def _init_canvas(canvas: FigureCanvas, figure: Figure) -> None:
    """Give a chart canvas the shared size policy and figure colours.

    The minimum size keeps the axes box strictly positive even in a
    narrow splitter, which Matplotlib needs for aspect calculations and
    for the layout pass.
    """
    canvas.setSizePolicy(
        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
    )
    canvas.setMinimumWidth(CHART_MIN_WIDTH)
    canvas.setMinimumHeight(CHART_MIN_HEIGHT)
    figure.patch.set_facecolor(CHART_BG)


def _add_legend(axes, *, handles=None, loc=None) -> bool:
    """Add a legend only when labeled artists exist.

    Matplotlib warns ("No artists with labels found to put in legend.")
    and draws nothing when the legend has no handles, so the legend is
    skipped in that case instead. Legends with real content are
    unchanged.

    Returns:
        True when a legend was added.
    """
    if handles is None:
        handles = [
            artist
            for artist in list(axes.get_lines()) + list(axes.collections)
            if artist.get_label() and not artist.get_label().startswith("_")
        ]
    if not handles:
        return False
    if loc is None:
        axes.legend(handles=handles, fontsize=8)
    else:
        axes.legend(handles=handles, loc=loc, fontsize=8)
    return True


def _apply_tight_layout(figure: Figure) -> None:
    """Fit titles, tick labels, and legends inside the figure.

    ``tight_layout`` grows the margins to fit the axes decorations. When
    the canvas is resized very small the required margins cannot fit,
    and Matplotlib then keeps the previous layout — the chart is still
    drawn correctly, so that one fallback notice is not repeated on
    every redraw of the same figure.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Tight layout not applied\..*",
            category=UserWarning,
        )
        figure.tight_layout(pad=FIGURE_PADDING)


def _ordered_limits(low: float, high: float, fallback: tuple[float, float]):
    """Return limits that are strictly increasing and finite.

    Guards the diagram against an inverted or collapsed axis, which a
    fixed aspect or a log scale would otherwise turn into an invalid
    view. The fallback pair is used only if the requested pair is not
    usable.
    """
    try:
        values = (float(low), float(high))
    except (TypeError, ValueError):
        return fallback
    if not all(math.isfinite(value) for value in values):
        return fallback
    if values[0] < values[1]:
        return values
    if values[1] < values[0]:
        return (values[1], values[0])
    return fallback


def _show_empty_message(figure: Figure, axes, title: str, text: str) -> None:
    """Replace a chart's content with a centered explanatory message."""
    axes.clear()
    # An empty state carries no data to keep a fixed aspect, and an equal
    # aspect here would fight the layout pass for the message text.
    axes.set_aspect("auto")
    axes.set_title(title)
    axes.text(
        0.5, 0.5, text, ha="center", va="center", wrap=True,
        transform=axes.transAxes, fontsize=9, color=TEXT_MUTED,
    )
    axes.set_xticks([])
    axes.set_yticks([])
    _style_axes(axes)
    _apply_tight_layout(figure)


def _as_float_matrix(values: object, name: str) -> np.ndarray:
    """Convert input to a finite 2D float array or raise."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric matrix, got text")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only numbers: {exc}") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} is empty; nothing to plot")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values (no NaN or inf)")
    return array


def _as_float_vector(values: object, name: str) -> list[float]:
    """Convert input to a non-empty finite 1D float list or raise."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence, got text")
    try:
        items = list(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a sequence of numbers, got {type(values).__name__}"
        ) from exc
    try:
        array = np.asarray(items, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only numbers: {exc}") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one value per entry, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} is empty; nothing to plot")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values (no NaN or inf)")
    return [float(value) for value in array]


def validate_decision_boundary_inputs(
    features: object, targets: object
) -> tuple[list[float], list[float], list[float]]:
    """Validate two-feature samples with binary single-output targets.

    Args:
        features: ``(number_of_samples, 2)`` numeric input matrix.
        targets: ``(number_of_samples, 1)`` or ``(number_of_samples,)``
            binary target values (0.0 or 1.0).

    Returns:
        ``(x_values, y_values, classes)`` as plain float lists, one entry
        per sample, in the samples' original order.

    Raises:
        TypeError: For non-numeric input.
        ValueError: For anything other than exactly two feature columns,
            a single target column, non-binary targets, a row-count
            mismatch, empty input, or non-finite values.
    """
    matrix = _as_float_matrix(features, "features")
    if matrix.shape[1] != 2:
        raise ValueError(
            f"decision boundary needs exactly 2 input features, got {matrix.shape[1]}"
        )
    try:
        target_array = np.asarray(targets, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"targets must contain only numbers: {exc}") from exc
    if target_array.ndim == 1:
        target_array = target_array.reshape(-1, 1)
    actual = _as_float_matrix(target_array, "targets")
    if actual.shape[1] != 1:
        raise ValueError(
            f"decision boundary needs a single binary target column, "
            f"got {actual.shape[1]}"
        )
    if actual.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"targets length {actual.shape[0]} does not match "
            f"features length {matrix.shape[0]}"
        )
    classes: list[float] = []
    for value in actual[:, 0]:
        if value not in (0.0, 1.0):
            raise ValueError(
                "decision boundary needs binary targets (0.0 or 1.0), "
                f"found {value!r}"
            )
        classes.append(float(value))
    return (
        [float(value) for value in matrix[:, 0]],
        [float(value) for value in matrix[:, 1]],
        classes,
    )


def validate_layer_sizes(layer_sizes: object) -> tuple[int, ...]:
    """Validate an architecture as at least two positive layer sizes.

    Returns:
        The layer sizes as a tuple of plain ints.

    Raises:
        TypeError: For non-numeric or non-integer entries.
        ValueError: For fewer than two layers or non-positive sizes.
    """
    if isinstance(layer_sizes, (str, bytes)):
        raise TypeError("layer sizes must be a sequence of positive integers")
    try:
        items = list(layer_sizes)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            "layer sizes must be a sequence of positive integers"
        ) from exc
    if len(items) < 2:
        raise ValueError("layer sizes must contain at least input and output layers")
    sizes: list[int] = []
    for index, entry in enumerate(items):
        if not _is_number(entry) or float(entry) != int(float(entry)):
            raise TypeError(f"layer size {index} is not an integer: {entry!r}")
        size = int(entry)
        if size < 1:
            raise ValueError(f"layer size {index} must be positive, got {size}")
        sizes.append(size)
    return tuple(sizes)


class LossHistoryChart(FigureCanvas):
    """Line chart of recorded per-epoch training loss (MSE)."""

    def __init__(self, parent=None) -> None:
        """Create the chart showing its empty-state message."""
        figure, axes = _base_figure()
        self._figure = figure
        self._axes = axes
        super().__init__(self._figure)
        _init_canvas(self, self._figure)
        self.setObjectName("lossHistoryChart")
        self.show_message(
            "No training history yet. Train the model to record loss values."
        )

    def show_message(self, text: str) -> None:
        """Replace the chart with a centered informational message."""
        _show_empty_message(
            self._figure, self._axes, "Training loss (MSE) by epoch", text
        )
        self.draw()

    def plot_history(
        self,
        history: object,
        validation_history: object = None,
        best_epoch: int | None = None,
    ) -> bool:
        """Plot recorded losses at epochs 1..N, replacing prior content.

        When a validation history is supplied, both series share the
        axes and the best validation epoch is marked. ``best_epoch``
        comes from the training run when available; otherwise the
        earliest epoch with the lowest recorded validation MSE is used,
        which is a labeling choice, not a new measurement.

        Returns True when drawn, False with a message when the history
        is missing or invalid. Mismatched series lengths are refused so
        the two curves can never be shown as if they shared epochs.
        """
        try:
            values = validate_loss_history(history)
        except (TypeError, ValueError) as exc:
            self.show_message(f"Cannot plot training loss: {exc}")
            return False
        validation_values: list[float] | None = None
        if validation_history is not None:
            try:
                validation_values = validate_loss_history(validation_history)
            except (TypeError, ValueError) as exc:
                self.show_message(f"Cannot plot validation loss: {exc}")
                return False
            if len(validation_values) != len(values):
                self.show_message(
                    "Cannot plot validation loss: "
                    f"{len(validation_values)} validation value(s) do not "
                    f"match {len(values)} training epoch(s)."
                )
                return False

        epochs = list(range(1, len(values) + 1))
        axes = self._axes
        axes.clear()
        axes.plot(epochs, values, marker="o", label="training MSE")
        best_position = self._best_epoch_position(
            best_epoch, validation_values, len(values)
        )
        if validation_values is not None:
            axes.plot(
                epochs, validation_values, marker="s", label="validation MSE"
            )
            if best_position is not None:
                axes.plot(
                    [best_position],
                    [validation_values[best_position - 1]],
                    marker="*",
                    markersize=14,
                    linestyle="",
                    label=f"best validation epoch ({best_position})",
                )
            axes.set_title("Training and validation loss (MSE) by epoch")
            axes.set_ylabel("loss (MSE)")
        else:
            axes.set_title("Training loss (MSE) by epoch")
            axes.set_ylabel("training loss (MSE)")
        axes.set_xlabel("epoch")
        _add_legend(axes)
        axes.grid(True, alpha=0.25, linewidth=0.6)
        _style_axes(axes)
        _apply_tight_layout(self._figure)
        self.draw()
        return True

    @staticmethod
    def _best_epoch_position(
        best_epoch: object, validation_values: list[float] | None, epochs: int
    ) -> int | None:
        """Return a plottable 1-based best-epoch position, or None.

        An explicitly reported epoch is used when it lies inside the
        recorded range; otherwise the earliest epoch with the lowest
        recorded validation MSE is used.
        """
        if validation_values is None:
            return None
        if isinstance(best_epoch, bool):
            return None
        if isinstance(best_epoch, (int, np.integer)) and 1 <= int(best_epoch) <= epochs:
            return int(best_epoch)
        lowest = min(validation_values)
        return validation_values.index(lowest) + 1


class PredictionChart(FigureCanvas):
    """Per-sample predicted probabilities, classes, and actual targets."""

    def __init__(self, parent=None) -> None:
        """Create the chart showing its empty-state message."""
        figure, axes = _base_figure()
        self._figure = figure
        self._axes = axes
        super().__init__(self._figure)
        _init_canvas(self, self._figure)
        self.setObjectName("predictionChart")
        self.show_message(
            "No evaluation yet. Evaluate a labeled dataset to compare "
            "predictions against targets."
        )

    def show_message(self, text: str) -> None:
        """Replace the chart with a centered informational message."""
        _show_empty_message(
            self._figure, self._axes, "Predicted vs actual targets", text
        )
        self.draw()

    def plot_predictions(
        self,
        probabilities: object,
        targets: object = None,
        *,
        threshold: float = CLASSIFICATION_THRESHOLD,
        classify: bool = True,
    ) -> bool:
        """Plot per-sample probabilities, classes, and optional targets.

        Args:
            probabilities: Per-sample outputs of shape
                ``(number_of_samples, 1)`` or a flat sequence.
            targets: Optional actual values in the same shape.
            threshold: Decision threshold for the class series.
            classify: Draw the thresholded class series. It is ``False``
                for continuous regression, where thresholding would
                invent a class label that the data does not have.

        Returns True when drawn, False with a message when the inputs
        are missing, mismatched, or otherwise invalid.
        """
        try:
            probs, actual = validate_prediction_inputs(probabilities, targets)
            if not _is_number(threshold) or not math.isfinite(float(threshold)):
                raise ValueError(f"threshold is not finite: {threshold!r}")
            limit = float(threshold)
        except (TypeError, ValueError) as exc:
            self.show_message(f"Cannot plot predictions: {exc}")
            return False
        classes = [1.0 if value >= limit else 0.0 for value in probs]
        samples = list(range(1, len(probs) + 1))
        axes = self._axes
        axes.clear()
        axes.plot(samples, probs, marker="o", label="predicted probability")
        if classify:
            axes.plot(
                samples,
                classes,
                marker="s",
                linestyle="--",
                label=f"predicted class (threshold {limit})",
            )
        if actual is not None:
            axes.plot(samples, actual, marker="^", linestyle="",
                      label="actual target")
            if classify:
                axes.set_title("Predicted vs actual targets (labeled evaluation)")
            else:
                axes.set_title("Predicted vs actual values (regression)")
        elif classify:
            axes.set_title("Predicted probabilities and classes (no targets)")
        else:
            axes.set_title("Predicted values (no targets)")
        axes.set_xlabel("sample index")
        axes.set_ylabel("output value")
        _add_legend(axes)
        axes.grid(True, alpha=0.25, linewidth=0.6)
        _style_axes(axes)
        _apply_tight_layout(self._figure)
        self.draw()
        return True


class DecisionBoundaryChart(FigureCanvas):
    """Two-feature samples over the model's predicted class regions.

    The owning view supplies the sampled prediction grid produced by the
    existing prediction API; this widget only validates and draws it.
    """

    def __init__(self, parent=None) -> None:
        """Create the chart showing its empty-state message."""
        figure, axes = _base_figure()
        self._figure = figure
        self._axes = axes
        super().__init__(self._figure)
        _init_canvas(self, self._figure)
        self.setObjectName("decisionBoundaryChart")
        self.show_message(
            "No two-feature dataset and model available yet. Train or "
            "configure a two-input binary classifier."
        )

    def show_message(self, text: str) -> None:
        """Replace the chart with a centered informational message."""
        _show_empty_message(
            self._figure,
            self._axes,
            "Decision boundary (two input features)",
            text,
        )
        self.draw()

    def plot_boundary(
        self,
        features: object,
        targets: object,
        region_x: object,
        region_y: object,
        region_prediction: object,
        *,
        threshold: float = CLASSIFICATION_THRESHOLD,
    ) -> bool:
        """Draw the predicted regions, the boundary, and the samples.

        Args:
            features: ``(n, 2)`` sample inputs, in dataset order.
            targets: ``(n, 1)`` or ``(n,)`` binary sample targets.
            region_x: Grid coordinates along the first feature.
            region_y: Grid coordinates along the second feature.
            region_prediction: ``(len(region_y), len(region_x))`` model
                outputs for the grid, matching ``region_x``/``region_y``.
            threshold: Classification threshold used for the boundary
                line label; default :data:`CLASSIFICATION_THRESHOLD`.

        Returns True when drawn, False with a message when any input is
        missing, mismatched, or invalid — never a partial picture.
        """
        try:
            x_values, y_values, classes = validate_decision_boundary_inputs(
                features, targets
            )
            grid_x = _as_float_vector(region_x, "region x grid")
            grid_y = _as_float_vector(region_y, "region y grid")
            region = _as_float_matrix(region_prediction, "region prediction")
            if region.shape != (len(grid_y), len(grid_x)):
                raise ValueError(
                    f"region prediction shape {region.shape} does not match "
                    f"the grid {len(grid_y)}x{len(grid_x)}"
                )
            if not _is_number(threshold) or not math.isfinite(float(threshold)):
                raise ValueError(f"threshold is not finite: {threshold!r}")
            limit = float(threshold)
        except (TypeError, ValueError) as exc:
            self.show_message(f"Cannot plot the decision boundary: {exc}")
            return False
        axes = self._axes
        axes.clear()
        axes.contourf(
            grid_x,
            grid_y,
            region,
            levels=[0.0, limit, 1.0],
            colors=["#dbe9f6", "#fbd9d3"],
            alpha=0.75,
        )
        boundary = axes.contour(
            grid_x, grid_y, region, levels=[limit], colors="black", linewidths=2
        )
        # ``allsegs`` is empty when no level was drawn, and its entries
        # are NumPy vertex arrays that cannot be tested for truth.
        segments = boundary.allsegs[0] if len(boundary.allsegs) else []
        if any(len(segment) > 0 for segment in segments):
            boundary.clabel(fmt={limit: f"prediction = {limit}"}, inline=True)
        handles = [
            Patch(
                facecolor="#dbe9f6", edgecolor="none",
                label="predicted class 0 region",
            ),
            Patch(
                facecolor="#fbd9d3", edgecolor="none",
                label="predicted class 1 region",
            ),
            Line2D(
                [], [], color="black", linewidth=2,
                label=f"decision boundary (prediction = {limit})",
            ),
        ]
        for class_value, marker, label in (
            (0.0, "o", "class 0 samples"),
            (1.0, "^", "class 1 samples"),
        ):
            sample_x = [
                x for x, value in zip(x_values, classes) if value == class_value
            ]
            sample_y = [
                y for y, value in zip(y_values, classes) if value == class_value
            ]
            if not sample_x:
                continue
            artist = axes.scatter(
                sample_x,
                sample_y,
                marker=marker,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
            artist.set_label(label)
            handles.append(artist)
        axes.set_xlabel("feature 1")
        axes.set_ylabel("feature 2")
        axes.set_title("Decision boundary and dataset samples")
        _add_legend(axes, handles=handles, loc="upper right")
        _style_axes(axes)
        _apply_tight_layout(self._figure)
        self.draw()
        return True


class RegressionSurfaceChart(FigureCanvas):
    """Predicted value surface for one output over two input features.

    The counterpart of :class:`DecisionBoundaryChart` for continuous
    targets: where that widget draws class regions, this one draws the
    predicted value itself with a labelled colour scale, plus the
    dataset's samples coloured by their actual value. The owning view
    supplies the sampled prediction grid produced by the existing
    forward-pass API; this widget only validates and draws it.
    """

    #: Number of filled bands used for the predicted value.
    SURFACE_LEVELS = 14

    def __init__(self, parent=None) -> None:
        """Create the chart showing its empty-state message."""
        figure, axes = _base_figure()
        self._figure = figure
        self._axes = axes
        super().__init__(self._figure)
        _init_canvas(self, self._figure)
        self.setObjectName("regressionSurfaceChart")
        self.show_message(
            "No prediction surface yet. Train or configure a model with a "
            "continuous target and validate a dataset for it."
        )

    def show_message(self, text: str) -> None:
        """Replace the chart with a centered informational message."""
        _show_empty_message(
            self._figure,
            self._axes,
            "Predicted value surface",
            text,
        )
        self.draw()

    def plot_surface(
        self,
        features: object,
        values: object,
        region_x: object,
        region_y: object,
        region_prediction: object,
        *,
        output_index: int = 0,
        x_label: str = "feature 1",
        y_label: str = "feature 2",
    ) -> bool:
        """Draw the predicted values and the samples.

        Args:
            features: ``(n, 2)`` sample inputs projected onto the two
                selected features, in dataset order.
            values: ``(n,)`` actual values for the selected output.
            region_x: Grid coordinates along the first selected feature.
            region_y: Grid coordinates along the second selected feature.
            region_prediction: ``(len(region_y), len(region_x))`` model
                outputs for the grid.
            output_index: Zero-based output drawn, used in the labels.
            x_label: Label for the horizontal axis.
            y_label: Label for the vertical axis.

        Returns True when drawn, False with a message when any input is
        missing, mismatched, or invalid — never a partial picture.
        """
        try:
            matrix = _as_float_matrix(features, "features")
            if matrix.shape[1] != 2:
                raise ValueError(
                    f"the surface needs exactly two feature columns, got "
                    f"{matrix.shape[1]}"
                )
            actual = _as_float_vector(values, "values")
            if len(actual) != matrix.shape[0]:
                raise ValueError(
                    f"values length {len(actual)} does not match "
                    f"{matrix.shape[0]} samples"
                )
            grid_x = _as_float_vector(region_x, "region x grid")
            grid_y = _as_float_vector(region_y, "region y grid")
            region = _as_float_matrix(region_prediction, "region prediction")
            if region.shape != (len(grid_y), len(grid_x)):
                raise ValueError(
                    f"region prediction shape {region.shape} does not match "
                    f"the grid {len(grid_y)}x{len(grid_x)}"
                )
        except (TypeError, ValueError) as exc:
            self.show_message(f"Cannot plot the predicted surface: {exc}")
            return False
        axes = self._axes
        axes.clear()
        filled = axes.contourf(
            grid_x,
            grid_y,
            region,
            levels=self.SURFACE_LEVELS,
            cmap="viridis",
        )
        predicted_bar = self._figure.colorbar(
            filled, ax=axes, fraction=0.05, pad=0.03
        )
        predicted_bar.set_label(
            f"predicted value (output {output_index})", fontsize=8
        )
        predicted_bar.ax.tick_params(labelsize=7)
        samples = axes.scatter(
            [float(value) for value in matrix[:, 0]],
            [float(value) for value in matrix[:, 1]],
            c=actual,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
            label="samples (colour = actual value)",
        )
        actual_bar = self._figure.colorbar(
            samples, ax=axes, fraction=0.05, pad=0.12
        )
        actual_bar.set_label(
            f"actual value (output {output_index})", fontsize=8
        )
        actual_bar.ax.tick_params(labelsize=7)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.set_title(
            f"Predicted values for output {output_index} over "
            f"{x_label} and {y_label}"
        )
        _add_legend(axes)
        _style_axes(axes)
        _apply_tight_layout(self._figure)
        self.draw()
        return True


class ArchitectureChart(FigureCanvas):
    """Neurons and connections of the current network's layers."""

    def __init__(self, parent=None) -> None:
        """Create the diagram showing its empty-state message."""
        figure, axes = _base_figure()
        self._figure = figure
        self._axes = axes
        super().__init__(self._figure)
        _init_canvas(self, self._figure)
        self.setObjectName("architectureChart")
        self.show_message(
            "No model available yet. Train or configure one to see its "
            "layers and connections."
        )

    def show_message(self, text: str) -> None:
        """Replace the diagram with a centered informational message."""
        _show_empty_message(
            self._figure, self._axes, "Network architecture", text
        )
        self.draw()

    @staticmethod
    def _layer_roles(count: int) -> list[str]:
        """Name each column: input, output, or hidden layer."""
        if count <= 1:
            return ["layer"] * count
        roles = ["hidden layer"] * count
        roles[0] = "input layer"
        roles[-1] = "output layer"
        return roles

    def plot_architecture(self, layer_sizes: object) -> bool:
        """Draw the layers' neurons and their connections.

        Every neuron drawn in a layer is connected to every neuron drawn
        in the next layer, so the diagram shows the real adjacency of the
        reported sizes. Columns with more than :data:`MAX_DRAWABLE_NEURONS`
        neurons are summarized as that many dots plus a caption stating the
        true count and that the drawn lines stand for the full layer, so a
        large architecture stays readable without changing or
        approximating the model anywhere else.

        Returns True when drawn, False with a message when the layer
        sizes are missing or invalid.
        """
        try:
            sizes = validate_layer_sizes(layer_sizes)
        except (TypeError, ValueError) as exc:
            self.show_message(f"Cannot show the architecture: {exc}")
            return False
        drawn = [min(size, MAX_DRAWABLE_NEURONS) for size in sizes]
        tallest = max(drawn)
        step = 1.0 / max(1, tallest - 1)
        columns = [
            [
                index * step - (count - 1) * step / 2.0
                for index in range(count)
            ]
            for count in drawn
        ]
        axes = self._axes
        axes.clear()
        # Connections are derived from the actual adjacent layers: every
        # neuron drawn in layer L links to every neuron drawn in layer
        # L + 1, between those two layers' own column positions.
        for index in range(len(columns) - 1):
            for source in columns[index]:
                for target in columns[index + 1]:
                    axes.plot(
                        [index, index + 1],
                        [source, target],
                        color="#9aa7b4",
                        linewidth=0.6,
                        zorder=1,
                    )
        for index, positions in enumerate(columns):
            for position in positions:
                axes.add_patch(
                    Circle(
                        (index, position),
                        0.075,
                        facecolor="#2f6f9f",
                        edgecolor="black",
                        linewidth=0.6,
                        zorder=2,
                    )
                )
        roles = self._layer_roles(len(sizes))
        summarized = False
        for index, (positions, size) in enumerate(zip(columns, sizes)):
            caption = f"{roles[index]}\n{size} neuron" + ("" if size == 1 else "s")
            if size != len(positions):
                caption += f"\n({len(positions)} shown)"
                summarized = True
            axes.text(index, -0.62, caption, ha="center", va="top", fontsize=8)
        if not summarized:
            # Neuron numbering stays readable for small architectures;
            # a summarized column would be too crowded to label.
            for index, positions in enumerate(columns):
                for neuron, position in enumerate(positions, start=1):
                    axes.text(
                        index,
                        position,
                        str(neuron),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="#ffffff",
                        zorder=3,
                    )
        bottom = -1.05
        if summarized:
            # A capped column shows a sample of that layer, so state that
            # the drawn lines stand in for the connections of the full
            # layer instead of implying that few connections exist.
            axes.text(
                (len(sizes) - 1) / 2.0,
                -1.05,
                "a layer is larger than the drawing limit: lines connect "
                "the displayed neurons of adjacent layers",
                ha="center",
                va="top",
                fontsize=7,
                color=TEXT_MUTED,
            )
            bottom = -1.45
        # The diagram owns its limits (captions included) and hides its
        # axes, so fixed margins are laid out instead of a tight layout:
        # there is nothing left to measure and the view can never end up
        # with a collapsed, non-positive axes box.
        axes.set_xlim(*_ordered_limits(-0.5, len(sizes) - 0.5, (-0.5, 1.5)))
        axes.set_ylim(*_ordered_limits(bottom, 0.75, (-1.45, 0.75)))
        axes.set_aspect("equal")
        axes.axis("off")
        axes.set_title(f"Network architecture: [{', '.join(str(s) for s in sizes)}]")
        self._figure.subplots_adjust(
            left=0.04, right=0.96, top=0.95, bottom=0.05
        )
        self.draw()
        return True
