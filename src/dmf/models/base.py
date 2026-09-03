"""The shared forecasting interface.

Shape contract, enforced by convention and by ``tests/test_onnx_parity.py``:

- Input ``x``: ``(B, L, C_in)`` -- batch, lookback samples, input channels.
- Point output: ``(B, H, C_out)`` -- batch, horizon samples, target channels.
- Quantile output: ``(B, H, C_out, Q)`` -- as above plus quantile levels, ascending.
- Gaussian output: ``(B, H, C_out, 2)`` -- as above plus ``(mean, log_var)``, which is
  **not** a quantile axis and is never sorted (``docs/protocol.md`` P5-D1).

Which of the three a model emits is set by its ``head`` kind, and the trailing width ``K``
comes from :func:`dmf.models.heads.n_output_params` -- one definition, imported by every
model rather than restated in each.

Inputs reaching ``forward`` are de-meaned and scaled (dimensionless). Outputs are likewise
dimensionless and are mapped back to corpus units -- degrees for roll and pitch, metres for
heave -- by :func:`dmf.data.normalize.invert_norm` before any metric is computed.
"""

from typing import ClassVar, Literal, Protocol, runtime_checkable

from torch import Tensor, nn

from dmf.models.heads import HeadKind, n_output_params, quantile_fan

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
        SUPPORTED_HEADS: The head kinds this class can emit. Checked in ``__init__``, so a
            config asking for a head a model does not implement fails at construction.
    """

    FIT_KIND: ClassVar[FitKind] = "sgd"

    #: Head kinds this class can actually emit. Point-only by default, because most models
    #: in this project are: persistence, the window mean, damped persistence, AR and the
    #: closed-form DLinear have no distributional parameters to project. A subclass that
    #: implements the reshape branches widens it.
    #:
    #: It exists because the accepted-argument check in
    #: :func:`dmf.train.registry.build_model` cannot see this on its own. A model that does
    #: not define its own ``__init__`` inherits this one, ``head`` is therefore an accepted
    #: keyword, and ``head: gaussian`` on ``persistence`` would build a model that records
    #: itself as Gaussian and emits a rank-3 point forecast -- a config-vs-behaviour
    #: mismatch that shows up nowhere until a probabilistic metric reads an axis that is
    #: not there.
    SUPPORTED_HEADS: ClassVar[tuple[HeadKind, ...]] = ("point",)

    def __init__(
        self,
        lookback: int,
        max_horizon: int,
        n_input_channels: int,
        n_target_channels: int,
        n_quantiles: int = 0,
        head: HeadKind | None = None,
    ) -> None:
        """Record the shape contract for this instance.

        Args:
            lookback: Input window length ``L``, samples.
            max_horizon: Forecast length ``H``, samples.
            n_input_channels: Input channel count ``C_in``.
            n_target_channels: Target channel count ``C_out``.
            n_quantiles: Quantile count ``Q``, or 0 for a point model.
            head: Output head kind. Defaults to ``"quantile"`` when ``n_quantiles > 0`` and
                ``"point"`` otherwise, so every construction site that predates Phase 5
                keeps its exact behaviour and ``"gaussian"`` must be asked for by name --
                which is the point. A Gaussian head is *not* a two-wide quantile fan, and
                the two are kept apart here rather than distinguished later by arity
                (``docs/protocol.md`` P5-D1).

        Raises:
            ValueError: If any shape count is not positive, if ``n_quantiles`` is negative,
                if ``head`` is not in this class's :attr:`SUPPORTED_HEADS`, if ``head`` and
                ``n_quantiles`` disagree -- ``n_quantiles > 0`` if and only
                if the head is ``"quantile"`` -- or if ``n_target_channels`` exceeds
                ``n_input_channels``.
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
        resolved: HeadKind = head if head is not None else ("quantile" if n_quantiles else "point")
        if resolved not in self.SUPPORTED_HEADS:
            raise ValueError(
                f"{type(self).__name__} cannot carry a {resolved!r} head; it supports "
                f"{list(self.SUPPORTED_HEADS)}. A model that accepts the keyword and "
                f"ignores it would report a distribution it does not emit."
            )
        if (resolved == "quantile") != (n_quantiles > 0):
            raise ValueError(
                f"head={resolved!r} and n_quantiles={n_quantiles} disagree: n_quantiles is "
                f"positive if and only if the head is 'quantile'. A gaussian head is two "
                f"parameters, not a two-level fan, and conflating them puts (mean, log_var) "
                f"on the axis sort_quantiles sorts (docs/protocol.md P5-D1)."
            )
        self.lookback = lookback
        self.max_horizon = max_horizon
        self.n_input_channels = n_input_channels
        self.n_target_channels = n_target_channels
        self.n_quantiles = n_quantiles
        #: Named ``head_kind`` and not ``head`` because three models already carry an
        #: ``nn.Linear`` named ``self.head`` -- their final projection -- whose assignment
        #: would move this string out of ``__dict__`` into ``_modules`` and leave the kind
        #: unreadable. Renaming their layer instead would change the ``head.weight`` key of
        #: every checkpoint already written under ``artifacts/``.
        self.head_kind: HeadKind = resolved
        #: Fan levels, empty unless the head is ``quantile``. Recovered from the width,
        #: because a model is constructed with ``n_quantiles`` and never with the levels.
        self.quantile_levels: tuple[float, ...] = (
            quantile_fan(n_quantiles) if resolved == "quantile" else ()
        )
        #: Trailing output width ``K``: 1, ``Q`` or 2. The models' head-width arithmetic
        #: reads this rather than recomputing ``max(n_quantiles, 1)``.
        self.n_output_params = n_output_params(resolved, self.quantile_levels)

    def forward(self, x: Tensor) -> Tensor:
        """Forecast all horizon steps in one pass.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Forecasts, shape ``(B, H, C_out)``, ``(B, H, C_out, Q)`` or
            ``(B, H, C_out, 2)`` according to the head kind, dimensionless.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    def output_shape(self, batch_size: int) -> tuple[int, ...]:
        """Return the expected output shape for a given batch size.

        Args:
            batch_size: Batch size ``B``.

        Returns:
            ``(B, H, C_out)`` for a point model, ``(B, H, C_out, Q)`` for a quantile model,
            ``(B, H, C_out, 2)`` for a Gaussian one. Used by the ONNX export and parity
            checks to assert the graph's shape without running the model.
        """
        if self.head_kind == "point":
            return (batch_size, self.max_horizon, self.n_target_channels)
        return (batch_size, self.max_horizon, self.n_target_channels, self.n_output_params)

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
