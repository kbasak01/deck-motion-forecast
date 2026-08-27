"""The shared forecasting interface.

Shape contract, enforced by convention and by ``tests/test_onnx_parity.py``:

- Input ``x``: ``(B, L, C_in)`` -- batch, lookback samples, input channels.
- Point output: ``(B, H, C_out)`` -- batch, horizon samples, target channels.
- Quantile output: ``(B, H, C_out, Q)`` -- as above plus quantile levels, ascending.

Inputs reaching ``forward`` are de-meaned and scaled (dimensionless). Outputs are likewise
dimensionless and are mapped back to corpus units -- degrees for roll and pitch, metres for
heave -- by :func:`dmf.data.normalize.invert_norm` before any metric is computed.
"""

from typing import ClassVar, Literal, Protocol, runtime_checkable

from torch import Tensor, nn

__all__ = ["BaseForecaster", "FitKind", "ForecastModel", "QuantileForecastModel"]

#: How a model acquires its parameters. ``none`` needs no training data at all,
#: ``closed_form`` is solved exactly from accumulated moments, ``sgd`` goes through
#: :func:`dmf.train.loop.fit`. The experiment driver dispatches on this rather than on
#: the class, so a new model joins a run by declaring its fit kind.
FitKind = Literal["none", "closed_form", "sgd"]


@runtime_checkable
class ForecastModel(Protocol):
    """Protocol for a direct multi-horizon point forecaster."""

    def forward(self, x: Tensor) -> Tensor:
        """Forecast all horizon steps in one pass.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, de-meaned and scaled
                (dimensionless).

        Returns:
            Point forecasts, shape ``(B, H, C_out)``, dimensionless.
        """
        ...


@runtime_checkable
class QuantileForecastModel(Protocol):
    """Protocol for a direct multi-horizon quantile forecaster."""

    def forward(self, x: Tensor) -> Tensor:
        """Forecast a quantile fan for all horizon steps in one pass.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, de-meaned and scaled
                (dimensionless).

        Returns:
            Quantile forecasts, shape ``(B, H, C_out, Q)``, dimensionless, with the
            quantile axis sorted ascending. Sorting is applied post-hoc rather than
            assumed: an unconstrained quantile head produces crossing quantiles, which
            makes interval width meaningless.
        """
        ...


class BaseForecaster(nn.Module):
    """Common base for the learned forecasters.

    Holds the shape bookkeeping every model needs and nothing else; the architectures
    themselves live in their own modules.

    Attributes:
        FIT_KIND: How this model acquires its parameters. Read by
            :func:`dmf.train.experiment.run_experiment` to decide whether the model needs
            no fitting at all, a closed-form solve from accumulated moments, or a run
            through the SGD loop.
    """

    FIT_KIND: ClassVar[FitKind] = "sgd"

    def __init__(
        self,
        lookback: int,
        max_horizon: int,
        n_input_channels: int,
        n_target_channels: int,
        n_quantiles: int = 0,
    ) -> None:
        """Record the shape contract for this instance.

        Args:
            lookback: Input window length ``L``, samples.
            max_horizon: Forecast length ``H``, samples.
            n_input_channels: Input channel count ``C_in``.
            n_target_channels: Target channel count ``C_out``.
            n_quantiles: Quantile count ``Q``, or 0 for a point model.

        Raises:
            ValueError: If any shape count is not positive, if ``n_quantiles`` is negative,
                or if ``n_target_channels`` exceeds ``n_input_channels``.
        """
        super().__init__()
        for name, value in (
            ("lookback", lookback),
            ("max_horizon", max_horizon),
            ("n_input_channels", n_input_channels),
            ("n_target_channels", n_target_channels),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if n_quantiles < 0:
            raise ValueError(f"n_quantiles must be non-negative, got {n_quantiles}")
        if n_target_channels > n_input_channels:
            raise ValueError(
                f"n_target_channels ({n_target_channels}) exceeds n_input_channels "
                f"({n_input_channels}); the targets are a prefix of the inputs (P2-D4)"
            )
        self.lookback = lookback
        self.max_horizon = max_horizon
        self.n_input_channels = n_input_channels
        self.n_target_channels = n_target_channels
        self.n_quantiles = n_quantiles

    def forward(self, x: Tensor) -> Tensor:
        """Forecast all horizon steps in one pass.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Forecasts, shape ``(B, H, C_out)`` or ``(B, H, C_out, Q)``, dimensionless.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    def output_shape(self, batch_size: int) -> tuple[int, ...]:
        """Return the expected output shape for a given batch size.

        Args:
            batch_size: Batch size ``B``.

        Returns:
            ``(B, H, C_out)`` for a point model, ``(B, H, C_out, Q)`` for a quantile
            model. Used by the ONNX export and parity checks to assert the graph's shape
            without running the model.
        """
        if self.n_quantiles > 0:
            return (batch_size, self.max_horizon, self.n_target_channels, self.n_quantiles)
        return (batch_size, self.max_horizon, self.n_target_channels)

    @property
    def n_fitted_parameters(self) -> int:
        """Number of values fitted from data, for the parameter-count column.

        Not the same as ``sum(p.numel() for p in self.parameters())``. Closed-form models
        hold their coefficients in **buffers**, not ``nn.Parameter``, because nothing
        differentiates through them -- so the naive count reports 0 for AR(40), which
        actually carries 216 900 fitted coefficients at the P3 geometry. Printing 0 beside
        DLinear's count would make the budget-matching column wrong in exactly the direction
        that flatters the baseline, so models with buffer-held coefficients override this.

        Returns:
            Fitted value count. The default counts trainable parameters only.
        """
        return sum(int(p.numel()) for p in self.parameters())
