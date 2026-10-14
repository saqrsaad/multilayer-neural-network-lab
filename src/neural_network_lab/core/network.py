"""Feedforward multilayer perceptron with forward propagation.

Conventions follow ``docs/PROJECT_SPEC.md`` and ``docs/ARCHITECTURE.md``:
NumPy mathematics only, independent of the GUI. Parameter shapes:

- layer sizes ``[n0, n1, ..., nL]``
- weight matrix for layer ``l``: ``(n_l, n_(l+1))``
- bias array for layer ``l``: ``(1, n_(l+1))``
- batch input ``X``: ``(number_of_samples, n0)``
- final output: ``(number_of_samples, nL)``

Sigmoid activation is applied at every layer in this initial
implementation. Full-batch gradient descent training is provided via
:mod:`neural_network_lab.core.backpropagation` and
:mod:`neural_network_lab.core.optimizers`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neural_network_lab.core.activations import sigmoid
from neural_network_lab.core.backpropagation import compute_gradients
from neural_network_lab.core.backpropagation import validate_targets
from neural_network_lab.core.losses import mean_squared_error
from neural_network_lab.core.optimizers import sgd_update
from neural_network_lab.core.optimizers import validate_learning_rate


def _validate_layer_sizes(layer_sizes: Sequence[int]) -> tuple[int, ...]:
    """Validate layer sizes and return them as a tuple of ints."""
    if isinstance(layer_sizes, (str, bytes)):
        raise TypeError("layer_sizes must be a sequence of positive integers")
    try:
        sizes = tuple(layer_sizes)
    except TypeError as exc:
        raise TypeError(
            "layer_sizes must be a sequence of positive integers"
        ) from exc
    if len(sizes) < 2:
        raise ValueError("layer_sizes must contain at least input and output sizes")
    validated: list[int] = []
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
            raise TypeError("each layer size must be a positive integer")
        if int(size) < 1:
            raise ValueError("each layer size must be a positive integer")
        validated.append(int(size))
    return tuple(validated)


def _random_parameters(
    layer_sizes: tuple[int, ...], rng: np.random.Generator
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Draw fresh Xavier-uniform weights and zero biases."""
    weights: list[NDArray[np.float64]] = []
    biases: list[NDArray[np.float64]] = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
        weights.append(
            rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(float)
        )
        biases.append(np.zeros((1, fan_out), dtype=float))
    return weights, biases


def _validate_epochs(epochs: int) -> int:
    """Validate an epoch count and return it as an int."""
    if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)):
        raise TypeError("epochs must be a positive integer")
    if int(epochs) < 1:
        raise ValueError("epochs must be a positive integer")
    return int(epochs)


@dataclass(frozen=True)
class TrainingResult:
    """Structured outcome of a full-batch training run.

    ``loss_history`` holds one actual measured post-update MSE value per
    requested epoch, in order. ``mode`` is ``"continued"`` when training
    started from the current parameters or ``"from_scratch"`` when the
    parameters were explicitly reinitialized first.
    """

    epochs_completed: int
    loss_history: tuple[float, ...]
    learning_rate: float
    mode: str

    @property
    def final_loss(self) -> float:
        """Measured MSE after the final epoch."""
        return self.loss_history[-1]


def _validate_parameters(
    layer_sizes: tuple[int, ...],
    weights: Sequence[ArrayLike] | None,
    biases: Sequence[ArrayLike] | None,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Validate explicit weights/biases against layer sizes."""
    n_layers = len(layer_sizes) - 1
    if (weights is None) != (biases is None):
        raise ValueError("weights and biases must be provided together or not at all")
    if weights is None or biases is None:
        raise ValueError("weights and biases must be provided together or not at all")
    weight_list = list(weights)
    bias_list = list(biases)
    if len(weight_list) != n_layers or len(bias_list) != n_layers:
        raise ValueError(
            f"expected {n_layers} weight matrices and bias arrays, "
            f"got {len(weight_list)} weights and {len(bias_list)} biases"
        )
    clean_weights: list[NDArray[np.float64]] = []
    clean_biases: list[NDArray[np.float64]] = []
    for index, (w, b) in enumerate(zip(weight_list, bias_list)):
        try:
            w_arr = np.array(w, dtype=float, copy=True)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"weights[{index}] must be numeric: {exc}") from exc
        try:
            b_arr = np.array(b, dtype=float, copy=True)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"biases[{index}] must be numeric: {exc}") from exc
        expected_w = (layer_sizes[index], layer_sizes[index + 1])
        expected_b = (1, layer_sizes[index + 1])
        if w_arr.shape != expected_w:
            raise ValueError(
                f"weights[{index}] shape {w_arr.shape} != expected {expected_w}"
            )
        if b_arr.shape != expected_b:
            raise ValueError(
                f"biases[{index}] shape {b_arr.shape} != expected {expected_b}"
            )
        if not np.all(np.isfinite(w_arr)):
            raise ValueError(f"weights[{index}] must contain only finite values")
        if not np.all(np.isfinite(b_arr)):
            raise ValueError(f"biases[{index}] must contain only finite values")
        clean_weights.append(w_arr)
        clean_biases.append(b_arr)
    return clean_weights, clean_biases


class NeuralNetwork:
    """Configurable feedforward MLP using sigmoid activations.

    Layer/parameter ordering: index ``l`` refers to the transformation
    from layer ``l`` to layer ``l + 1``. ``weights[l]`` has shape
    ``(n_l, n_(l+1))`` and ``biases[l]`` has shape ``(1, n_(l+1))``.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        weights: Sequence[ArrayLike] | None = None,
        biases: Sequence[ArrayLike] | None = None,
        seed: int | None = None,
    ) -> None:
        """Create a network from layer sizes.

        Args:
            layer_sizes: Sequence like ``[n0, n1, ..., nL]`` with at
                least two positive integers.
            weights: Optional explicit weight matrices matching the
                documented shapes. Must be given together with ``biases``.
            biases: Optional explicit bias arrays of shape
                ``(1, n_(l+1))``. Must be given together with ``weights``.
            seed: Optional seed for reproducible random initialization
                when ``weights``/``biases`` are omitted.

        Raises:
            TypeError: For non-integer sizes or non-numeric parameters.
            ValueError: For invalid sizes, shapes, or non-finite values.
        """
        sizes = _validate_layer_sizes(layer_sizes)
        self._layer_sizes = sizes
        if weights is not None or biases is not None:
            self._weights, self._biases = _validate_parameters(
                sizes, weights, biases
            )
        else:
            rng = np.random.default_rng(seed)
            self._weights, self._biases = _random_parameters(sizes, rng)

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        """Layer sizes ``(n0, n1, ..., nL)``."""
        return self._layer_sizes

    @property
    def n_layers(self) -> int:
        """Number of weight layers (``len(layer_sizes) - 1``)."""
        return len(self._layer_sizes) - 1

    @property
    def weights(self) -> list[NDArray[np.float64]]:
        """Copies of the weight matrices in layer order."""
        return [w.copy() for w in self._weights]

    @property
    def biases(self) -> list[NDArray[np.float64]]:
        """Copies of the bias arrays in layer order."""
        return [b.copy() for b in self._biases]

    def set_parameters(
        self,
        weights: Sequence[ArrayLike],
        biases: Sequence[ArrayLike],
    ) -> None:
        """Replace all weights and biases with validated values.

        Every array is validated against the current architecture
        before anything is stored, so a rejected edit leaves the
        existing parameters completely unchanged. Shapes must match
        the current layers exactly; reshaping is never performed.

        Args:
            weights: One ``(n_l, n_(l+1))`` matrix per layer.
            biases: One ``(1, n_(l+1))`` array per layer.

        Raises:
            TypeError: For non-numeric values.
            ValueError: For wrong counts, shapes, or non-finite values.
        """
        new_weights, new_biases = _validate_parameters(
            self._layer_sizes, weights, biases
        )
        self._weights, self._biases = new_weights, new_biases

    def to_state_dict(self) -> dict:
        """Export architecture and parameters as a JSON-compatible dict.

        Returns:
            ``{"layer_sizes": [...], "weights": [...], "biases": [...]}``
            with plain Python floats. The storage layer wraps this
            payload with its file envelope when saving.
        """
        return {
            "layer_sizes": list(self._layer_sizes),
            "weights": [w.tolist() for w in self._weights],
            "biases": [b.tolist() for b in self._biases],
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> NeuralNetwork:
        """Rebuild a network from a state dict, validating it first.

        Args:
            state: Dictionary as produced by :meth:`to_state_dict`.

        Returns:
            A new network with the stored architecture and parameters.

        Raises:
            TypeError: If ``state`` is not a dict of the expected shape.
            ValueError: If architecture or parameter data is missing,
                misshapen, or non-finite. The existing constructor
                performs the final shape/value validation.
        """
        if not isinstance(state, dict):
            raise TypeError(
                f"model state must be a dict, got {type(state).__name__}"
            )
        for key in ("layer_sizes", "weights", "biases"):
            if key not in state:
                raise ValueError(f"model state is missing {key!r}")
        layer_sizes = state["layer_sizes"]
        weights = state["weights"]
        biases = state["biases"]
        if (
            not isinstance(layer_sizes, (list, tuple))
            or len(layer_sizes) < 2
            or any(
                isinstance(size, bool)
                or not isinstance(size, (int, np.integer))
                or int(size) < 1
                for size in layer_sizes
            )
        ):
            raise ValueError(
                "model state needs 'layer_sizes' with at least "
                "two positive integers"
            )
        if not isinstance(weights, list) or not isinstance(biases, list):
            raise ValueError("model state 'weights' and 'biases' must be lists")
        if len(weights) != len(layer_sizes) - 1 or len(biases) != len(layer_sizes) - 1:
            raise ValueError(
                "model state parameter count does not match 'layer_sizes'"
            )
        return cls(
            layer_sizes=list(layer_sizes), weights=weights, biases=biases
        )

    def _validate_input(self, X: ArrayLike) -> NDArray[np.float64]:
        """Validate a batch input and return it as a float array."""
        try:
            arr = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"X must be numeric array-like: {exc}") from exc
        if arr.ndim != 2:
            raise ValueError(
                f"X must be 2D with shape (number_of_samples, {self._layer_sizes[0]}), "
                f"got shape {arr.shape}"
            )
        if arr.shape[0] < 1:
            raise ValueError("X must contain at least one sample")
        if arr.shape[1] != self._layer_sizes[0]:
            raise ValueError(
                f"X second dimension {arr.shape[1]} != network input size "
                f"{self._layer_sizes[0]}; refusing to reshape incompatible input"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("X must contain only finite values (no NaN or inf)")
        return arr

    def forward_with_cache(
        self, X: ArrayLike
    ) -> tuple[
        NDArray[np.float64],
        list[NDArray[np.float64]],
        list[NDArray[np.float64]],
    ]:
        """Run forward propagation and keep intermediates for backpropagation.

        For each layer ``l``: ``Z[l] = A[l] @ weights[l] + biases[l]`` and
        ``A[l+1] = sigmoid(Z[l])``, with ``A[0] = X``.

        Args:
            X: Batch input of shape ``(number_of_samples, n0)``.

        Returns:
            Tuple ``(output, activations, pre_activations)`` where
            ``activations`` is ``[A0, ..., AL]`` (length ``L + 1``) and
            ``pre_activations`` is ``[Z0, ..., Z(L-1)]`` (length ``L``).
            The final output ``AL`` has shape ``(number_of_samples, nL)``.
            Parameters are not modified.
        """
        current = self._validate_input(X)
        activations: list[NDArray[np.float64]] = [current]
        pre_activations: list[NDArray[np.float64]] = []
        for weight, bias in zip(self._weights, self._biases):
            linear = current @ weight + bias
            pre_activations.append(linear)
            current = sigmoid(linear)
            activations.append(current)
        return current, activations, pre_activations

    def forward(self, X: ArrayLike) -> NDArray[np.float64]:
        """Compute network predictions for a batch input.

        Args:
            X: Batch input of shape ``(number_of_samples, n0)``.

        Returns:
            Predictions of shape ``(number_of_samples, nL)``.
        """
        output, _, _ = self.forward_with_cache(X)
        return output

    # Alias: prediction is the forward pass and never updates parameters.
    predict = forward

    def apply_gradients(
        self,
        weight_grads: Sequence[ArrayLike],
        bias_grads: Sequence[ArrayLike],
        learning_rate: float,
    ) -> None:
        """Update parameters in place with precomputed gradients.

        Applies ``W_l = W_l - learning_rate * dL/dW_l`` (and the same
        rule for biases). Gradients must already have been computed
        from a single consistent pre-update parameter state.

        Args:
            weight_grads: One gradient array per layer with shape
                ``(n_l, n_(l+1))``.
            bias_grads: One gradient array per layer with shape
                ``(1, n_(l+1))``.
            learning_rate: Positive, finite step size.

        Raises:
            TypeError: For non-numeric gradients or learning rate.
            ValueError: For count/shape mismatches, non-finite
                gradients, invalid learning rates, or updates that
                would leave parameters non-finite.
        """
        rate = validate_learning_rate(learning_rate)
        new_weights = sgd_update(self._weights, list(weight_grads), rate)
        new_biases = sgd_update(self._biases, list(bias_grads), rate)
        for index, (w, b) in enumerate(zip(new_weights, new_biases)):
            if not np.all(np.isfinite(w)) or not np.all(np.isfinite(b)):
                raise ValueError(
                    f"update for layer {index} produced non-finite parameters; "
                    "try a smaller learning rate"
                )
        self._weights = new_weights
        self._biases = new_biases

    def train_step(self, X: ArrayLike, y: ArrayLike, learning_rate: float) -> float:
        """Run one full-batch gradient descent epoch.

        Gradients for all layers are computed from the current
        (pre-update) parameters first; only then are the parameters
        updated. Returns the actual measured post-update MSE.

        Args:
            X: Batch input of shape ``(number_of_samples, n0)``.
            y: Targets of shape ``(number_of_samples, nL)``.
            learning_rate: Positive, finite step size.

        Returns:
            Post-update Mean Squared Error on ``(X, y)``.
        """
        weight_grads, bias_grads = compute_gradients(self, X, y)
        self.apply_gradients(weight_grads, bias_grads, learning_rate)
        return mean_squared_error(self.forward(X), y)

    def _train_loop(
        self, X: ArrayLike, y: ArrayLike, epochs: int, learning_rate: float
    ) -> list[float]:
        """Run ``epochs`` full-batch steps, recording measured MSE each time."""
        history: list[float] = []
        for _ in range(epochs):
            history.append(self.train_step(X, y, learning_rate))
        return history

    def train(
        self, X: ArrayLike, y: ArrayLike, *, epochs: int, learning_rate: float
    ) -> TrainingResult:
        """Continue training from the current parameters (continued training).

        The existing weights and biases are preserved as the starting
        state; they are only changed by gradient descent updates. Never
        reinitializes or resets parameters.

        Args:
            X: Training inputs of shape ``(number_of_samples, n0)``.
            y: Training targets of shape ``(number_of_samples, nL)``.
            epochs: Positive number of full-batch epochs.
            learning_rate: Positive, finite step size.

        Returns:
            :class:`TrainingResult` with ``mode="continued"`` and one
            measured post-update MSE per epoch.
        """
        epochs_count = _validate_epochs(epochs)
        rate = validate_learning_rate(learning_rate)
        history = self._train_loop(X, y, epochs_count, rate)
        return TrainingResult(
            epochs_completed=epochs_count,
            loss_history=tuple(history),
            learning_rate=rate,
            mode="continued",
        )

    def train_from_scratch(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        epochs: int,
        learning_rate: float,
        seed: int | None = None,
    ) -> TrainingResult:
        """Reinitialize parameters explicitly, then train (train from scratch).

        All inputs and settings are validated *before* the current
        parameters are replaced, so invalid requests leave the network
        untouched. Replacement is explicit and complete: fresh
        Xavier-uniform weights and zero biases are drawn, then
        full-batch training runs for ``epochs`` epochs.

        Args:
            X: Training inputs of shape ``(number_of_samples, n0)``.
            y: Training targets of shape ``(number_of_samples, nL)``.
            epochs: Positive number of full-batch epochs.
            learning_rate: Positive, finite step size.
            seed: Optional seed for the fresh parameter state.

        Returns:
            :class:`TrainingResult` with ``mode="from_scratch"`` and one
            measured post-update MSE per epoch.
        """
        epochs_count = _validate_epochs(epochs)
        rate = validate_learning_rate(learning_rate)
        inputs = self._validate_input(X)
        validate_targets(y, inputs.shape[0], self._layer_sizes[-1])
        self._weights, self._biases = _random_parameters(
            self._layer_sizes, np.random.default_rng(seed)
        )
        history = self._train_loop(inputs, y, epochs_count, rate)
        return TrainingResult(
            epochs_completed=epochs_count,
            loss_history=tuple(history),
            learning_rate=rate,
            mode="from_scratch",
        )
