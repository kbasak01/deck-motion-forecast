"""The shared forecasting interface.

Shape contract, enforced by convention and by ``tests/test_onnx_parity.py``:

- Input ``x``: ``(B, L, C_in)`` -- batch, lookback samples, input channels.
- Point output: ``(B, H, C_out)`` -- batch, horizon samples, target channels.
- Quantile output: ``(B, H, C_out, Q)`` -- as above plus quantile levels, ascending.

Inputs reaching ``forward`` are de-meaned and scaled (dimensionless). Outputs are likewise
dimensionless and are mapped back to corpus units -- degrees for roll and pitch, metres for
heave -- by :func:`dmf.data.normalize.invert_norm` before any metric is computed.
"""

from typing import Protocol, runtime_checkable

from torch import Tensor, nn

__all__ = ["BaseForecaster", "ForecastModel", "QuantileForecastModel"]


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
    """

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
        """
        raise NotImplementedError

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
        raise NotImplementedError
