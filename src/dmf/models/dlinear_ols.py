"""DLinear solved exactly, as the optimisation-gap reference for the SGD ``dlinear`` row.

``DLinear(individual=False)`` is an affine map of the decomposed window, so under the MSE
objective its optimum is the ordinary-least-squares solution and can be written down. This
file is that solution; :mod:`dmf.models.dlinear` stays exactly as it was and keeps its own
row in the table.

**Why both rows ship.** Every ``ar*`` row in ``results/baselines.csv`` sits at its exact
ridge-regularised global optimum, while the ``dlinear`` row sits wherever 60 epochs of Adam
landed. Reporting them side by side therefore conflates an optimisation gap of unknown size
with an architecture gap. With ``dlinear_ols`` present the difference between the two
DLinear rows *measures* the optimisation shortfall instead of hiding it inside the
architecture comparison, and the SGD row survives as the end-to-end evidence that the
shared training loop works on the corpus -- which is what Phase 4 inherits.

Measured on ``id``: 0.431545 validation MSE here against 0.432852 +- 0.000007 over the SGD
row's three seeds, evaluated by the same :func:`dmf.train.loop.validate` call on the same
windows. SGD is 0.30 % above the solve in loss, which is a median +0.0052 skill on
``id``/test and up to +0.0758 -- enough to move the median AR(20)-over-DLinear advantage
from +0.0194 to +0.0056.

**The design is exactly rank-deficient, by construction.** ``series_decompose`` writes
``trend = A x`` and ``remainder = (I - A) x``, so the ``2L`` columns of
``[trend_lags, remainder_lags]`` span an at-most-``L``-dimensional space: the Gram matrix is
singular however many windows are accumulated. That is a property of the parameterisation,
not of the corpus (measured ``cond(R) = 6.0e19`` on ``id``/train), so the solve keeps the
same centred-and-whitened ridge treatment as the AR fit
(:func:`dmf.train.closed_form.solve_ar_coefficients`).

**The ridge is not cosmetic here.** Measured on ``id`` from one moments pass: at
``ridge = 1e-6`` the fit reaches training MSE 0.3761 and validation MSE 0.4315; at
``ridge = 0`` it reaches training MSE 0.3182 and validation MSE **2539.8**. The
unregularised minimiser of the training loss is real and reachable -- the ``lstsq`` fallback
keeps directions sitting at the ``rcond`` cutoff and amplifies them by about 1e10 -- and it
is useless. What this class reports is therefore the exact optimum of the *regularised*
objective, solved under the same penalty every ``ar*`` row is solved under, not the
unattainable-in-practice unregularised one.

Units: inputs, outputs and coefficients are dimensionless; ``kernel_size`` is in samples.
"""

from typing import ClassVar, cast

import numpy as np
import torch
from torch import nn

from dmf.models.base import FitKind
from dmf.models.dlinear import DLinear
from dmf.models.heads import HeadKind
from dmf.train.registry import register_model
from dmf.typedefs import FloatArray

__all__ = ["DLinearOLS"]


@register_model("dlinear_ols")
class DLinearOLS(DLinear):
    """Channel-independent DLinear whose coefficients are solved, not trained.

    Identical forward pass to :class:`dmf.models.dlinear.DLinear` -- it is inherited
    unchanged, so the two rows of the results table are the same function class and differ
    only in how the coefficients were obtained. ``individual`` is forced False: the
    closed-form derivation assumes one shared ``(L -> H)`` map per component, applied to
    every target channel, which is what makes the design a single stacked regression over
    ``N * C_out`` rows.

    The coefficients stay ``nn.Parameter`` rather than moving to buffers, unlike
    :class:`dmf.models.ar.ARForecaster`: the tensors are the *same* tensors the SGD twin
    trains, so the parameter count, the ONNX graph and the state dict all stay comparable
    between the two rows. ``FIT_KIND = "closed_form"`` is what keeps an optimiser away from
    them.
    """

    # The value is inside the ``FitKind`` union ``BaseForecaster`` declares, but mypy
    # re-infers the class variable's type from DLinear's own bare assignment
    # (``FIT_KIND = "sgd"``) and so rejects any override in a grandchild class.
    # Silenced here rather than fixed by annotating DLinear, which CLAUDE.md forbids
    # editing to accommodate a new model.
    FIT_KIND: ClassVar[FitKind] = "closed_form"  # type: ignore[assignment]

    #: Point only, narrowed from the parent's three: this row is a least-squares solve, and
    #: neither pinball loss nor a Gaussian NLL has normal equations to accumulate. Stated on
    #: the class so that a config asking this model for a head fails at construction rather
    #: than inheriting a capability the fitting path does not have.
    SUPPORTED_HEADS: ClassVar[tuple[HeadKind, ...]] = ("point",)

    def __init__(
        self,
        lookback: int,
        max_horizon: int,
        n_input_channels: int,
        n_target_channels: int,
        kernel_size: int = 25,
        ridge: float = 0.0,
    ) -> None:
        """Configure the model.

        Args:
            lookback: Input window length ``L``, samples.
            max_horizon: Forecast length ``H``, samples.
            n_input_channels: Input channel count ``C_in``.
            n_target_channels: Target channel count ``C_out``.
            kernel_size: Trend-extraction window, samples. Must match the SGD twin's, or
                the two rows are not the same function class.
            ridge: Tikhonov strength on the whitened correlation matrix, dimensionless.
                Recorded on the instance so the fitted model carries the regularisation it
                was solved under.

        Raises:
            ValueError: If ``kernel_size`` is outside ``[1, lookback]`` or ``ridge`` is
                negative.
        """
        super().__init__(
            lookback,
            max_horizon,
            n_input_channels,
            n_target_channels,
            kernel_size=kernel_size,
            individual=False,
            # Hardcoded, and matched by `SUPPORTED_HEADS` above: a closed-form solve has no
            # probabilistic head.
            n_quantiles=0,
        )
        if ridge < 0.0:
            raise ValueError(f"ridge must be non-negative, got {ridge}")
        self.ridge = ridge

    def set_coefficients(self, weight: FloatArray, bias: FloatArray) -> None:
        """Install a solved coefficient matrix.

        The trend and remainder biases are only ever used summed, so the identifiable
        intercept is a single vector: it is written to the trend map and the remainder map
        is left at zero. Any other split would give the same forecasts and a different
        state dict.

        Args:
            weight: Coefficients, shape ``(2 * L, H)``, dimensionless. Rows ``[:L]`` are
                the trend lags in window order (oldest first, matching the input layout),
                rows ``[L:]`` the remainder lags.
            bias: Combined intercept, shape ``(H,)``, dimensionless.

        Raises:
            ValueError: If either array has the wrong shape.
        """
        w = np.asarray(weight, dtype=np.float64)
        b = np.asarray(bias, dtype=np.float64)
        expected_w = (2 * self.lookback, self.max_horizon)
        if w.shape != expected_w:
            raise ValueError(f"weight has shape {w.shape}, expected {expected_w}")
        if b.shape != (self.max_horizon,):
            raise ValueError(f"bias has shape {b.shape}, expected {(self.max_horizon,)}")
        trend = cast(nn.Linear, self.trend[0])
        remainder = cast(nn.Linear, self.remainder[0])
        with torch.no_grad():
            # nn.Linear applies x @ W.T, so the (in, out) solve transposes on the way in.
            trend.weight.copy_(torch.from_numpy(w[: self.lookback].T.copy()).float())
            remainder.weight.copy_(torch.from_numpy(w[self.lookback :].T.copy()).float())
            trend.bias.copy_(torch.from_numpy(b.copy()).float())
            remainder.bias.zero_()
