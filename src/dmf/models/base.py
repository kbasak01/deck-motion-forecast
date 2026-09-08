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

``forward`` is a **template method** and architectures override :meth:`BaseForecaster._predict`
instead. It exists to hold one seam: the optional reversible instance normalisation of
``DataConfig.revin``, applied on the way in and inverted on the way out, so that a model
acquires it by existing rather than by remembering to call it. RevIN composes *with* the
pipeline's normalisation rather than replacing it -- see :meth:`BaseForecaster.enable_revin`.
"""

from typing import ClassVar, Literal, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from dmf.data.normalize import RevIN
from dmf.models.heads import HeadKind, n_output_params, quantile_fan

__all__ = [
    "REVIN_EPS",
    "BaseForecaster",
    "FitKind",
    "ForecastModel",
    "QuantileForecastModel",
    "revin_applies",
]

#: Numerical floor added to RevIN's per-window standard deviation, dimensionless. Named
#: here rather than left to the layer's default so that the one value the project fits with
#: sits in the same file as the wiring that applies it.
REVIN_EPS: float = 1e-5

#: How a model acquires its parameters. ``none`` needs no training data at all,
#: ``closed_form`` is solved exactly from accumulated moments, ``sgd`` goes through
#: :func:`dmf.train.loop.fit`. The experiment driver dispatches on this rather than on
#: the class, so a new model joins a run by declaring its fit kind.
FitKind = Literal["none", "closed_form", "sgd"]


def revin_applies(model_cls: "type[BaseForecaster]") -> bool:
    """Report whether RevIN may be attached to a model class.

    One tested predicate rather than a condition restated at each call site, because the
    two exclusions below are both silent failures if they are got wrong.

    Args:
        model_cls: The model class.

    Returns:
        True for ``FIT_KIND == "sgd"`` only. ``closed_form`` models are solved from moments
        accumulated over the dataset's own windows, so a RevIN forward pass would evaluate
        their coefficients in a space they were never solved in; ``none`` models
        (``persistence``, ``window_mean``) are mathematically unchanged by RevIN but not
        *bitwise* unchanged, and ``dmf.eval.controls`` asserts persistence is bitwise the
        inline expression every skill denominator is defined by (P2-D9).

        The consequence is worth stating where it will be read: **on a ``revin: true`` arm
        only the SGD rows differ from the reference arm.** The ``persistence``,
        ``window_mean``, ``ar*`` and ``dlinear_ols`` rows are the reference arm's rows, and
        a table that lists them under the RevIN arm must say so.
    """
    return model_cls.FIT_KIND == "sgd"


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
        #: The reversible instance-normalisation layer, or None. Off unless
        #: :meth:`enable_revin` is called, which :func:`dmf.train.registry.build_model` does
        #: when ``DataConfig.revin`` is set -- so every model built before Phase 6 keeps its
        #: exact forward pass and its exact state-dict keys.
        self.revin: RevIN | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Forecast all horizon steps in one pass, through the RevIN seam.

        **Subclasses override :meth:`_predict`, not this method.** This is the one place the
        optional reversible instance normalisation is applied, so a model acquires it by
        existing rather than by remembering to call it. ``tests/test_models.py`` asserts
        that no registered model defines its own ``forward``: one that did would be silently
        excluded from the RevIN ablation while still carrying its config's ``revin: true``.

        Written as a template method rather than as a ``register_forward_pre_hook`` pair
        because :mod:`dmf.eval.runner`, :mod:`dmf.eval.prob_runner` and
        :mod:`dmf.eval.controls` all call ``model.forward(x)`` directly rather than
        ``model(x)``. Hooks fire on ``__call__`` only, so a hook-based wiring would apply
        RevIN during training and skip it during scoring -- the worst available failure
        mode, since it is invisible in both the loss curve and the metric table.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless -- already de-meaned per
                window and divided by the training-split channel scale by
                :class:`dmf.data.dataset.DeckMotionDataset`.

        Returns:
            Forecasts, shape ``(B, H, C_out)``, ``(B, H, C_out, Q)`` or
            ``(B, H, C_out, 2)`` according to the head kind, dimensionless and in the same
            normalised space as ``x``, whether or not RevIN is enabled.
        """
        if self.revin is None:
            return self._predict(x)
        return self._revin_inverse(self._predict(self.revin(x)))

    def _predict(self, x: Tensor) -> Tensor:
        """Forecast all horizon steps in one pass.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless. When RevIN is enabled
                this is the RevIN-normalised window, not the dataset's.

        Returns:
            Forecasts, shape ``(B, H, C_out)``, ``(B, H, C_out, Q)`` or
            ``(B, H, C_out, 2)`` according to the head kind, dimensionless.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError

    def enable_revin(self, *, eps: float = REVIN_EPS, affine: bool = True) -> None:
        """Attach a :class:`dmf.data.normalize.RevIN` layer to this model's forward pass.

        Called by :func:`dmf.train.registry.build_model` when ``DataConfig.revin`` is set,
        after construction rather than through every model's ``__init__`` -- a constructor
        keyword would have to be accepted by all eight model classes and would be rejected
        by ``build_model``'s accepted-argument check for any that had not been edited.

        **What RevIN composes with here.** The window reaching :meth:`forward` has already
        been de-meaned per window and divided by the training-split channel scale, and the
        output is expected back in that same space, so RevIN is applied *inside* the model
        rather than replacing the dataset's normalisation:

        - the mean subtraction is a numerical no-op (the window mean is already zero), so
          nothing is de-meaned twice;
        - what RevIN adds is the per-window, per-channel **standard deviation**, which the
          pipeline's single train-split constant per channel does not remove -- so the
          ablation measures instance-level amplitude normalisation and nothing else;
        - :meth:`_revin_inverse` puts the forecast back in the dataset's space before it
          leaves :meth:`forward`, so the loss, the early-stopping criterion,
          :func:`dmf.data.normalize.invert_norm` and every metric downstream are the same
          quantities as in every other arm. That is what makes the RevIN arm comparable to
          the reference arm at all.

        ``affine=True`` adds ``2 * C_in`` parameters (12 at the shipped ``C_in = 6``), which
        :attr:`n_fitted_parameters` counts, so the RevIN row's parameter column is honest.

        Args:
            eps: Numerical floor added to the per-window standard deviation, dimensionless.
            affine: Whether RevIN learns a per-channel affine transform after normalising.

        Raises:
            ValueError: If RevIN is already enabled, or if this model is not fitted by SGD.
                RevIN is refused on ``closed_form`` models because their coefficients are
                solved from moments accumulated by
                :func:`dmf.train.closed_form.accumulate_training_moments` over the
                *dataset's* windows -- a RevIN forward pass would apply them in a space they
                were never solved in, and the failure would be silent. It is refused on
                ``none`` models because RevIN is the identity on ``persistence`` and
                ``window_mean`` but not *bitwise* the identity, and
                ``dmf.eval.controls._pipeline_persistence_sse`` asserts bitwise equality
                against the inline persistence expression every skill denominator is
                defined by (``docs/protocol.md`` P2-D9).
        """
        if self.revin is not None:
            raise ValueError(f"{type(self).__name__} already has RevIN enabled")
        if not revin_applies(type(self)):
            raise ValueError(
                f"{type(self).__name__} declares FIT_KIND={self.FIT_KIND!r}; RevIN is "
                f"applied to SGD-fitted models only. A closed-form model's coefficients are "
                f"solved from moments accumulated in the dataset's normalised space, and "
                f"persistence must stay bitwise equal to the inline expression the Gate 2 "
                f"pipeline control validated (docs/protocol.md P2-D9)."
            )
        self.revin = RevIN(self.n_input_channels, eps=eps, affine=affine)

    def _revin_inverse(self, y: Tensor) -> Tensor:
        """Undo the RevIN normalisation of the current forward pass on a model output.

        :meth:`dmf.data.normalize.RevIN.inverse` handles ``(B, H, C_in)`` only, while this
        project's outputs are ``(B, H, C_out[, K])`` with ``C_out <= C_in``, so the
        inversion is done here against the layer's cached statistics -- the same tensors,
        read rather than recomputed, so the forward and inverse transforms cannot drift
        apart. ``tests/test_models.py`` asserts this agrees with ``RevIN.inverse`` bitwise
        on the one geometry that layer accepts.

        The target channels are the **leading** ``C_out`` input channels (the P2-D4 prefix
        rule, enforced in :meth:`__init__`), so the statistics are sliced, never gathered.

        Args:
            y: Model output in RevIN space, shape ``(B, H, C_out)``, ``(B, H, C_out, Q)`` or
                ``(B, H, C_out, 2)``.

        Returns:
            The same shape, in the dataset's normalised space.

        Raises:
            RuntimeError: If called before the RevIN layer has seen a window.
        """
        revin = self.revin
        if revin is None:  # pragma: no cover - unreachable from forward
            raise RuntimeError("_revin_inverse called with RevIN disabled")
        # Private on the layer because nothing outside a forward pass has any business
        # reading them; this *is* inside the forward pass they belong to.
        mean, std = revin._mean, revin._std
        if mean is None or std is None:
            raise RuntimeError("_revin_inverse called before the RevIN forward pass")
        n_out = self.n_target_channels
        mean, std = mean[..., :n_out], std[..., :n_out]
        weight: Tensor | float = revin.weight[:n_out] if revin.affine else 1.0
        bias: Tensor | float = revin.bias[:n_out] if revin.affine else 0.0
        if self.head_kind == "point":
            # No dtype cast: `RevIN.inverse` does not cast either, so under autocast both
            # promote a bf16 output to the float32 the statistics carry, identically.
            return (y - bias) / weight * std + mean
        if self.head_kind == "quantile":
            fan_weight = weight[:, None] if isinstance(weight, Tensor) else weight
            fan_bias = bias[:, None] if isinstance(bias, Tensor) else bias
            return (y - fan_bias) / fan_weight * std[..., None] + mean[..., None]
        # Gaussian: a location shift does not change a spread, so the mean gets the full
        # affine map and the log-variance gets only twice the log of its scale factor --
        # the same split `dmf.models.heads.PredictiveDistribution.affine` makes.
        location = (y[..., 0] - bias) / weight * std + mean
        log_scale = torch.log(std) - (
            torch.log(torch.abs(weight)) if isinstance(weight, Tensor) else 0.0
        )
        return torch.stack((location, y[..., 1] + 2.0 * log_scale), dim=-1)

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
