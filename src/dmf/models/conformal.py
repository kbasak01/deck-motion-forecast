"""Split-conformal calibration of an already-fitted interval head.

This is the wrapper ``dmf.models.heads.IntervalPredictor`` was declared for and that
``docs/IMPLEMENTATION_PLAN.md`` §Phase 5 asked to be possible "without touching the models".
Until now the seam had one implementer and it was a test double
(``tests/test_models.py::test_the_conformal_seam_moves_coverage_without_importing_a_model``);
this module is the first real one, and it holds a model by **composition** and imports no
architecture, so the property that test asserts is a property of the shipped code too.

**What it does.** Given a fitted head and a calibration split disjoint from both train and
test, it scales the head's predictive distribution about its own point forecast by one
dimensionless factor per ``(horizon, target channel)``:

    quantile head:  ``c_k(x) = m(x) + gamma * (f_k(x) - m(x))``   for all Q levels
    gaussian head:  ``(mean, log_var + 2*log(gamma))``

Those are the same operation: a Gaussian's quantiles are ``mean + z_k * sigma``, so scaling
deviations from the mean by ``gamma`` is exactly ``sigma -> gamma * sigma``. One code path,
two head kinds, and the Gaussian rows stay Gaussian rather than being silently materialised
into a fan.

**Why this scaling is multiplicative.** ``docs/protocol.md`` P10-D1 pre-registered this as the
*primary* arm and committed to a per-level **additive** secondary arm beside it; that secondary arm
was never built, which P10-D4 records as a broken commitment rather than a decision. The reasoning
below is why the primary arm is multiplicative, not an argument that the additive one was rejected
on evidence. The scored quantity is
conditional sharpness: at ``id``/pitch/10 s the committed ``width_ratio_mean`` runs from
0.737 (``dlinear_quantile``) to 0.284 (``lstm_quantile``), a 2.6x spread, and P6-D6 built the
floor contrast specifically to measure it. An additive offset widens the sharpest window and
the widest window by the same absolute amount and compresses that structure; a multiplicative
one moves only the scale, so ``width_ratio_calibrated / width_ratio_uncalibrated == gamma``
per cell and the sharpness ordering across heads is untouched.

**Three invariants that are structural here rather than checked downstream.**

1. **The point forecast is bitwise unchanged.** ``m + gamma*(m - m) == m`` for the quantile
   branch and the Gaussian mean is not touched, so no RMSE, MAE, skill, nrmse or phase-lag
   number in this project can move when this wrapper is applied. Gate 5 requires the
   out-of-distribution degradation be *reported, not fixed* (P5-D2); this is the strongest
   available form of that promise, and ``tests/test_conformal.py`` asserts it with
   ``torch.equal`` rather than a tolerance.
2. **Crossing is impossible.** ``gamma > 0`` times a non-decreasing fan is non-decreasing, so
   the raw -- pre-sort -- output is already ordered, the same property P5-D16 records for
   :class:`dmf.models.residual_interval.EmpiricalResidualInterval`.
3. **An uncalibrated instance raises.** ``gamma = 1`` is a perfectly well-formed
   passthrough, which would be scored as a head that needed no calibration rather than as a
   bug -- the same reasoning behind ``EmpiricalResidualInterval``'s ``is_fitted`` flag.

**One measurement this wrapper destroys, and it must be published as such.**
``dmf.eval.prob_runner`` reads ``crossing_terms`` off the raw tensor a model returns, before
:class:`dmf.models.heads.PredictiveDistribution` sorts it. This wrapper has to sort the base
fan before scaling -- otherwise ``gamma`` multiplies deviations from something that is not
the median -- so a calibrated row's ``crossing_rate`` is exactly 0.0 **by construction**.
That is the removal of a measurement, not the removal of crossing. The uncalibrated row
keeps the real number: 0.0 median over all uncalibrated rows, 1.5% over the quantile heads
alone, 6.3% over ``dlinear_quantile``'s own rows, and a worst cell of 96.6%
(``dlinear_quantile``/``unseen_seastate``/heave at 1 s). The two columns must be read together.

**Not registered, deliberately.** ``dmf.train.registry.build_model``
constructs a model from ``(cfg, geometry, n_in, n_out)`` and ``cfg.params`` alone; there is no
way to hand it a *fitted base model* at a particular regime and seed. A registry entry that
``build_model`` cannot build would be worse than no entry, so this class is constructed by
:mod:`dmf.eval.conformal_runner` and by nothing else.

Units: ``gamma`` is dimensionless and calibration happens in the model's normalised space,
like every other model's output, reaching corpus units only through
:meth:`dmf.models.heads.PredictiveDistribution.affine`.
"""

from typing import ClassVar

import numpy as np
import torch
from torch import Tensor

from dmf.models.base import BaseForecaster, FitKind
from dmf.models.heads import HeadKind, PredictiveDistribution
from dmf.typedefs import FloatArray

__all__ = ["ConformalInterval"]

#: Largest ``|log(gamma)|`` accepted on installation. ``gamma`` is a ratio of interval
#: half-widths; a factor of 100 in either direction means the base head's fan is not merely
#: mis-scaled but meaningless at that cell, and silently installing it would publish a
#: calibrated row that is a rescaling of noise. Raise instead, and let the caller decide.
MAX_ABS_LOG_GAMMA: float = float(np.log(100.0))


class ConformalInterval(BaseForecaster):
    """A fitted interval head plus one conformal scale factor per ``(horizon, channel)``.

    The base model is held by **composition, not inheritance**, for the same reason
    :class:`dmf.models.residual_interval.EmpiricalResidualInterval` gives: it is not an LSTM
    or a TCN, it *has* one, and the fit dispatch in :mod:`dmf.train.experiment` keys on
    ``issubclass``. Holding it as a submodule also means ``.to(device)``, ``.eval()`` and
    ``state_dict()`` reach the base without this class forwarding anything by hand.

    ``FIT_KIND`` is ``"closed_form"``: nothing here trains. The scale factor is one order
    statistic of held-out conformity scores. The *base*, however, is whatever it was --
    usually an SGD head carrying a seed -- so a wrapped row is **not** deterministic and is
    reported as ``mean +/- std`` over the base's three seeds like any other learned row.
    """

    #: Solved from held-out scores, never trained.
    FIT_KIND: ClassVar[FitKind] = "closed_form"

    #: Both interval heads are supported, and each keeps its own kind: this wrapper is
    #: head-preserving, so a Gaussian base stays Gaussian and is scored on exactly the
    #: object the committed Gaussian rows are scored on.
    SUPPORTED_HEADS: ClassVar[tuple[HeadKind, ...]] = ("quantile", "gaussian")

    #: Declared at class level because ``nn.Module.__getattr__`` types a registered buffer as
    #: ``Tensor | Module``; without these two lines every use below needs a cast.
    gamma: Tensor
    is_calibrated: Tensor

    def __init__(
        self,
        base: BaseForecaster,
    ) -> None:
        """Wrap an already-fitted interval head.

        The geometry is taken from the base rather than passed in, so a wrapper cannot be
        built at a geometry its base does not have -- the failure ``load_state_dict(strict=True)``
        exists to catch one layer down.

        Args:
            base: A fitted forecaster with a non-point head. Its parameters are not
                modified, read, or re-initialised.

        Raises:
            ValueError: If ``base`` carries a point head, which has no interval to calibrate.
        """
        if base.head_kind == "point":
            raise ValueError(
                "ConformalInterval needs an interval to calibrate and the base model has "
                f"head_kind={base.head_kind!r}. A point forecaster carries no predictive "
                f"distribution, so there is nothing for a conformity score to be about."
            )
        super().__init__(
            base.lookback,
            base.max_horizon,
            base.n_input_channels,
            base.n_target_channels,
            len(base.quantile_levels),
            base.head_kind,
        )
        #: The calibrated head. A submodule, so device moves and ``state_dict`` reach it.
        self.base = base
        #: Conformal scale, ``(H, C_out)``, dimensionless. A **buffer**: nothing
        #: differentiates through it and it must travel with the state dict, exactly as the
        #: AR coefficients and the empirical residual fan do.
        self.register_buffer(
            "gamma",
            torch.ones(base.max_horizon, base.n_target_channels, dtype=torch.float32),
        )
        #: Calibrated flag, as a buffer so it survives ``load_state_dict``. ``gamma = 1`` is
        #: a well-formed passthrough and would score as a head that needed no calibration.
        self.register_buffer("is_calibrated", torch.zeros((), dtype=torch.bool))

    def set_gamma(self, gamma: FloatArray) -> None:
        """Install the fitted conformal scale factors.

        Args:
            gamma: Conformal scale, shape ``(H, C_out)``, dimensionless and strictly
                positive. One factor per ``(horizon, target channel)``; see P10-D1 for why
                this granularity rather than one pooled factor.

        Raises:
            ValueError: If the shape is wrong, if any value is not finite, if any value is
                not strictly positive, or if any value is further than
                :data:`MAX_ABS_LOG_GAMMA` from 1 in log space. Non-positivity is the check
                that makes "this row cannot cross" a property of the object: a negative
                factor would reverse the fan and a zero one would collapse it to the point
                forecast, which scores as perfect sharpness and no coverage.
        """
        array = np.asarray(gamma, dtype=np.float64)
        expected = (self.max_horizon, self.n_target_channels)
        if array.shape != expected:
            raise ValueError(f"gamma has shape {array.shape}, expected {expected}")
        if not np.all(np.isfinite(array)):
            raise ValueError("gamma contains non-finite values")
        if np.any(array <= 0.0):
            worst = float(np.min(array))
            raise ValueError(
                f"gamma must be strictly positive; the smallest value is {worst:.3e}. A "
                f"non-positive scale would reverse or collapse the fan, and a collapsed fan "
                f"scores as perfect sharpness with no coverage rather than as a bug."
            )
        excess = float(np.max(np.abs(np.log(array))))
        if excess > MAX_ABS_LOG_GAMMA:
            raise ValueError(
                f"gamma reaches a factor of {float(np.exp(excess)):.1f} from 1, beyond the "
                f"{float(np.exp(MAX_ABS_LOG_GAMMA)):.0f}x bound. At that magnitude the base "
                f"head's interval is not mis-scaled but uninformative at the cell, and "
                f"rescaling it would publish a calibrated row that is a rescaling of noise."
            )
        with torch.no_grad():
            self.gamma.copy_(torch.from_numpy(array).float())
            self.is_calibrated.fill_(True)

    def _predict(self, x: Tensor) -> Tensor:
        """Scale the base head's predictive distribution about its own point forecast.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            The calibrated head parameters, dimensionless: ``(B, H, C_out, Q)`` for a
            quantile base, non-decreasing along ``Q`` **before** any post-hoc sort, and
            ``(B, H, C_out, 2)`` carrying ``(mean, log_var)`` for a Gaussian base.

        Raises:
            RuntimeError: If ``set_gamma`` has not been called.
        """
        if not bool(self.is_calibrated):
            raise RuntimeError(
                "ConformalInterval has no conformal scale; call "
                "dmf.eval.conformal.calibrate first. An uncalibrated instance would emit "
                "gamma = 1, which is a well-formed passthrough and would be scored as a "
                "head that needed no calibration rather than as a bug."
            )
        raw: Tensor = self.base(x)
        scale = self.gamma.to(dtype=raw.dtype)[None, :, :, None]
        if self.head_kind == "gaussian":
            # (mean, log_var): scaling deviations from the mean by gamma is sigma -> gamma *
            # sigma, i.e. log_var -> log_var + 2 log gamma. The mean is untouched, which is
            # what makes the point forecast bitwise identical to the base's.
            mean = raw[..., :1]
            log_var = raw[..., 1:] + 2.0 * torch.log(scale)
            return torch.cat((mean, log_var), dim=-1)
        # Quantile: sort first, so that the 0.5 member really is the median and gamma
        # multiplies deviations from it. This is what makes the row's crossing_rate
        # structurally 0.0 -- see the module docstring.
        fan = PredictiveDistribution(raw, "quantile", self.base.quantile_levels).raw
        median = fan[..., self.median_index : self.median_index + 1]
        return median + scale * (fan - median)

    @property
    def median_index(self) -> int:
        """Index of the 0.5 level in the base's fan.

        Returns:
            The position of the median in ``quantile_levels``, so the scaling is about the
            point forecast P5-D5 reads and not about an arbitrary fan member.

        Raises:
            ValueError: If the base is a quantile head whose fan carries no 0.5 level.
        """
        levels = self.base.quantile_levels
        for index, level in enumerate(levels):
            if abs(level - 0.5) <= 1e-9:
                return index
        raise ValueError(
            f"the base fan {levels} carries no 0.5 level, so it has no median to scale "
            f"about. dmf.models.heads.QUANTILE_FAN_9 contains 0.5 exactly for this reason."
        )

    def predict(self, x: Tensor) -> PredictiveDistribution:
        """Return the calibrated predictive distribution.

        This is the :class:`dmf.models.heads.IntervalPredictor` seam, implemented for the
        first time by something that is not a test double.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            The calibrated distribution, in ``x``'s units.
        """
        return PredictiveDistribution(self(x), self.head_kind, self.base.quantile_levels)

    @property
    def n_fitted_parameters(self) -> int:
        """Parameters of the base plus the installed scale factors.

        Returns:
            The base's own count plus ``H * C_out``. Counted in the direction that does not
            flatter this arm: the calibration is cheap, and saying so is the point.
        """
        return int(self.base.n_fitted_parameters + self.gamma.numel())
