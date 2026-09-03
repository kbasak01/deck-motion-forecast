"""Head kinds, the predictive-distribution value object, and the calibration seam.

This module holds no ``nn.Module``. The four ``PointHead``/``QuantileHead``/``GaussianHead``
stubs that shipped from Phase 0 declared ``forward(z: Tensor[B, d_in])`` -- a projection of
a flat encoder representation -- and DLinear, which Phase 5 is required to carry both heads
on, has no such ``z``: its head is a pair of ``Linear(lookback, .)`` maps applied per
channel to the trend and remainder of the decomposed *input series*. Adopting that contract
would have left one of the two models the phase must serve outside the abstraction, so the
three modules are removed and the head becomes a **kind**, resolved inside each model's own
projection arithmetic through :func:`n_output_params`. ``docs/protocol.md`` P5-D1 records
the deviation.

What lives here instead:

- the head-kind vocabulary and the single definition of the fan width ``K``;
- :class:`PredictiveDistribution`, a frozen value object over a raw model output that
  answers "point forecast", "quantile at level", "interval at alpha" identically in
  normalised and in corpus units;
- :func:`sort_quantiles`, unchanged from Phase 0;
- :class:`IntervalPredictor`, the protocol a split-conformal ``ConformalWrapper``
  (Project 6) implements so that calibration can move interval endpoints **without
  importing or subclassing any model**.

**The trap this design closes.** The four SGD models already emit ``(B, H, C, K)`` for
``K = max(n_quantiles, 1)``, so a Gaussian head could be had for free by building with
``n_quantiles = 2`` and reading channel 0 as the mean and channel 1 as the log-variance.
That puts ``(mean, log_var)`` on exactly the axis :func:`sort_quantiles` is defined to sort
ascending, and sorting them is silent, shape-preserving and catastrophic -- it swaps the two
on every element where ``log_var < mean``, producing intervals that are wrong without being
malformed. ``head`` is therefore a first-class kind and the ``gaussian`` branch never
reaches :func:`sort_quantiles`.

Units: everything here is unit-agnostic. A distribution constructed on a model's raw output
is dimensionless; :meth:`PredictiveDistribution.affine` is the one place it becomes corpus
units -- degrees for roll and pitch, degrees per second for their rates, metres for heave,
metres per second for heave rate.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

if TYPE_CHECKING:  # pragma: no cover - import cycle: base.py imports this module
    from dmf.models.base import BaseForecaster

__all__ = [
    "LOG_VAR_MAX",
    "LOG_VAR_MIN",
    "QUANTILE_FAN_9",
    "HeadKind",
    "IntervalPredictor",
    "PredictiveDistribution",
    "clamp_log_variance",
    "n_output_params",
    "point_view",
    "quantile_fan",
    "sort_quantiles",
]

#: The three output heads. ``point`` emits ``(B, H, C)``, ``quantile`` emits
#: ``(B, H, C, Q)`` with the quantile axis ascending, ``gaussian`` emits ``(B, H, C, 2)``
#: carrying ``(mean, log_var)`` on the trailing axis -- which is **not** a quantile axis and
#: is never sorted.
HeadKind = Literal["point", "quantile", "gaussian"]

#: The project's quantile fan: 0.05 to 0.95 in steps of 0.1125. The levels are chosen so
#: that nothing has to be interpolated -- 0.5 is present exactly, so the point forecast of a
#: quantile model is *read off* the fan (``docs/protocol.md`` P5-D5), and 0.05/0.95 are
#: present exactly, so PICP@90 and the 90 percent interval width are read off it too.
QUANTILE_FAN_9: tuple[float, ...] = (
    0.05,
    0.1625,
    0.275,
    0.3875,
    0.5,
    0.6125,
    0.725,
    0.8375,
    0.95,
)

#: Clamp bounds on a Gaussian head's log-variance, dimensionless (the head is fitted in
#: normalised space, where the target has unit scale by construction).
#:
#: The NLL carries ``exp(-log_var) * (target - mean)^2``, so an unclamped head can drive
#: ``log_var`` down without limit on any element it fits well and multiply that element's
#: squared error by ``exp(-log_var)`` -- which is how a Gaussian NLL diverges in practice,
#: and it does so as a loss that keeps improving rather than as a NaN. The bounds put
#: ``sigma`` in ``[exp(-7), exp(7)] = [9.1e-4, 1.1e3]`` against a unit-scale target, i.e.
#: three orders of magnitude of headroom either side of the signal, and cap the NLL's
#: precision factor at ``exp(14) = 1.2e6``. Both ends stay far inside bf16's exponent range
#: (bf16 has fp32's exponent field, so the binding constraint is the precision factor, not
#: the representable range), so the clamp is what keeps the ``amp_dtype: bf16`` path from
#: producing an Inf that only shows up as a NaN gradient several steps later.
LOG_VAR_MIN: float = -14.0
LOG_VAR_MAX: float = 14.0

#: Absolute tolerance for matching a requested quantile level against the fan a model was
#: built with. Levels are exact doubles on both sides -- ``interval(0.1)`` forms
#: ``0.1/2 == 0.05`` and ``1 - 0.05 == 0.95`` exactly -- so this only absorbs a config file
#: that writes ``0.1625`` with a trailing digit, never a genuinely different level.
_LEVEL_TOL: float = 1e-9


def quantile_fan(n_quantiles: int) -> tuple[float, ...]:
    """Return the project's quantile fan of a given width.

    Models are constructed with a fan *width* (``n_quantiles``), not with the levels
    themselves -- :func:`dmf.train.registry.build_model` passes
    ``n_quantiles=len(cfg.quantiles)`` -- so the levels have to be recoverable from the
    width alone for :class:`PredictiveDistribution` to name them. There is exactly one fan
    in this project and it is :data:`QUANTILE_FAN_9`; other widths are defined by the same
    rule (evenly spaced from 0.05 to 0.95) so that a synthetic-geometry test can build a
    narrower fan without inventing a second convention.

    Args:
        n_quantiles: Fan width ``Q``, at least 2.

    Returns:
        ``Q`` levels in (0, 1), strictly ascending. Exactly :data:`QUANTILE_FAN_9` -- the
        same object, so the levels are bitwise the shipped ones -- when ``Q`` is 9.

    Raises:
        ValueError: If ``n_quantiles`` is less than 2. A one-level "fan" has no interval
            and would make ``interval(alpha)`` meaningless rather than merely narrow.
    """
    if n_quantiles < 2:
        raise ValueError(f"a quantile fan needs at least 2 levels, got {n_quantiles}")
    if n_quantiles == len(QUANTILE_FAN_9):
        return QUANTILE_FAN_9
    lo, hi = QUANTILE_FAN_9[0], QUANTILE_FAN_9[-1]
    step = (hi - lo) / (n_quantiles - 1)
    return tuple(lo + i * step for i in range(n_quantiles - 1)) + (hi,)


def clamp_log_variance(fanned: Tensor) -> Tensor:
    """Clamp the log-variance component of a Gaussian head's output.

    Defined here, beside :data:`LOG_VAR_MIN` and :data:`LOG_VAR_MAX`, so that all four
    architectures clamp at the same bounds by construction rather than by four copies of a
    literal agreeing.

    Args:
        fanned: Gaussian head output, shape ``(B, H, C, 2)``, carrying ``(mean, log_var)``
            on the trailing axis, dimensionless.

    Returns:
        The same tensor with ``log_var`` clamped to ``[LOG_VAR_MIN, LOG_VAR_MAX]``, shape
        ``(B, H, C, 2)``. The mean is untouched.

    Raises:
        ValueError: If the trailing axis is not of width 2.
    """
    if fanned.ndim != 4 or int(fanned.shape[-1]) != 2:
        raise ValueError(f"expected a (B, H, C, 2) gaussian output, got {tuple(fanned.shape)}")
    return torch.stack((fanned[..., 0], fanned[..., 1].clamp(LOG_VAR_MIN, LOG_VAR_MAX)), dim=-1)


def n_output_params(head: HeadKind, quantiles: tuple[float, ...] = ()) -> int:
    """Return the trailing fan width ``K`` a head projects to, per (horizon, channel).

    **The single definition of ``K`` in the project.** Every model's head-width arithmetic,
    :meth:`dmf.models.base.BaseForecaster.output_shape`, and the reshape branch of every
    ``forward`` read it from here, so a fourth head kind is added in one place rather than
    in five that have to agree.

    Args:
        head: The head kind.
        quantiles: Quantile levels, required and used only for the ``quantile`` kind.

    Returns:
        1 for ``point`` (the horizon axis carries no fan), ``len(quantiles)`` for
        ``quantile``, 2 for ``gaussian`` (mean and log-variance).

    Raises:
        ValueError: If ``head`` is not a known kind, if a ``quantile`` head is given fewer
            than two levels or levels that are not strictly ascending inside (0, 1), or if
            levels are supplied for a head that has none -- the
            latter would otherwise be a config that silently documents a fan the model does
            not emit.
    """
    if head == "point":
        if quantiles:
            raise ValueError(f"a point head carries no quantile levels, got {list(quantiles)}")
        return 1
    if head == "gaussian":
        if quantiles:
            raise ValueError(f"a gaussian head carries no quantile levels, got {list(quantiles)}")
        return 2
    if head == "quantile":
        if len(quantiles) < 2:
            raise ValueError(
                f"a quantile head needs at least 2 levels, got {list(quantiles)}; one level "
                f"is a point forecast wearing a quantile axis"
            )
        if any(not 0.0 < q < 1.0 for q in quantiles):
            raise ValueError(f"quantile levels must lie in (0, 1), got {list(quantiles)}")
        # Strictly ascending, because `sort_quantiles` sorts the fan axis ascending and the
        # levels are then read off it positionally: a descending or duplicated level list
        # would label the sorted fan wrongly and report the 0.05 quantile as the 0.95.
        if any(b <= a for a, b in zip(quantiles, quantiles[1:], strict=False)):
            raise ValueError(f"quantile levels must be strictly ascending, got {list(quantiles)}")
        return len(quantiles)
    raise ValueError(f"unknown head kind {head!r}; expected point, quantile or gaussian")


def sort_quantiles(y: Tensor) -> Tensor:
    """Sort a quantile fan ascending along the quantile axis.

    Pinball loss does not constrain quantile ordering, so a trained quantile head produces
    crossing quantiles on some fraction of inputs -- the 0.6 quantile below the 0.4, say.
    A crossed interval has negative width, which makes PICP and mean interval width
    meaningless. Sorting post-hoc is the standard remedy and is applied unconditionally
    before any probabilistic metric is computed.

    Args:
        y: Quantile forecasts, shape ``(B, H, C, Q)``, dimensionless.

    Returns:
        The same forecasts with the ``Q`` axis sorted ascending, shape ``(B, H, C, Q)``.

    Raises:
        ValueError: If ``y`` is not four-dimensional. A rank check cannot by itself keep a
            Gaussian ``(B, H, C, 2)`` output away from this function -- that is what makes
            the P5-D1 trap a trap -- so the structural guard is
            :class:`PredictiveDistribution`, which calls this only for the ``quantile``
            kind. This check catches the cruder mistake of sorting a point forecast.
    """
    if y.ndim != 4:
        raise ValueError(f"y must have shape (B, H, C, Q), got {tuple(y.shape)}")
    return torch.sort(y, dim=-1).values


@dataclass(frozen=True)
class PredictiveDistribution:
    """A model's raw output, plus what kind of distribution it is.

    Unit-agnostic by construction: every accessor is a selection or an affine map of
    ``raw``, so an instance behaves identically in the model's dimensionless space and in
    corpus units, and :meth:`affine` is the one place the two are related.

    Ascending sorting is applied on construction for the ``quantile`` kind, so a
    constructed instance is never crossed and no caller has to remember to sort. For the
    ``gaussian`` kind the quantile axis does not exist and the P5-D1 sorting trap is
    structurally unreachable rather than merely avoided.

    Attributes:
        raw: The model output. ``(B, H, C)`` for ``point``, ``(B, H, C, Q)`` for
            ``quantile`` (sorted ascending along ``Q``), ``(B, H, C, 2)`` for ``gaussian``
            carrying ``(mean, log_var)``.
        head: The head kind that produced ``raw``.
        quantiles: The fan levels, ascending, empty unless ``head`` is ``quantile``.
    """

    raw: Tensor
    head: HeadKind
    quantiles: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Validate the rank against the head kind and sort a quantile fan.

        Raises:
            ValueError: If ``head`` is unknown, if ``quantiles`` disagrees with ``head``,
                or if ``raw``'s rank or trailing axis does not match the head kind.
        """
        width = n_output_params(self.head, self.quantiles)
        if self.head == "point":
            if self.raw.ndim != 3:
                raise ValueError(
                    f"a point head's output must be (B, H, C), got {tuple(self.raw.shape)}"
                )
            return
        if self.raw.ndim != 4:
            raise ValueError(
                f"a {self.head} head's output must be rank 4, got {tuple(self.raw.shape)}"
            )
        if int(self.raw.shape[-1]) != width:
            raise ValueError(
                f"a {self.head} head's output must have {width} on its trailing axis, got "
                f"{tuple(self.raw.shape)}"
            )
        if self.head == "quantile":
            object.__setattr__(self, "raw", sort_quantiles(self.raw))

    def point(self) -> Tensor:
        """Return the point forecast this distribution reports (P5-D5).

        The choice is recorded rather than left implicit: the **0.5 quantile** for a
        quantile head -- present exactly in :data:`QUANTILE_FAN_9`, so it is read off and
        not interpolated -- and the **mean** for a Gaussian head. CLAUDE.md non-negotiable
        4 requires every probabilistic row to also carry skill against persistence, and
        this is the forecast that skill is computed from.

        Returns:
            Point forecasts, shape ``(B, H, C)``, in whatever units ``raw`` carries.

        Raises:
            ValueError: If a quantile fan does not contain the 0.5 level.
        """
        if self.head == "point":
            return self.raw
        if self.head == "gaussian":
            return self.raw[..., 0]
        return self.raw[..., self._level_index(0.5)]

    def quantiles_at(self, levels: tuple[float, ...]) -> Tensor:
        """Return the predictive quantiles at the requested levels.

        For the ``quantile`` kind the levels are **read off the fan** and an absent level
        raises. Interpolating between fan members would make a reported PICP@90 a property
        of the interpolation rule as much as of the model, which is why
        :data:`QUANTILE_FAN_9` contains 0.05, 0.5 and 0.95 exactly. For the ``gaussian``
        kind it is ``mean + z(level) * sigma`` through the inverse normal CDF.

        Args:
            levels: Quantile levels, each in (0, 1).

        Returns:
            Quantile forecasts, shape ``(B, H, C, len(levels))``, in ``raw``'s units, in
            the order requested.

        Raises:
            ValueError: If ``levels`` is empty, if any level is outside (0, 1), if this is
                a ``point`` distribution, or if a requested level is not in the fan.
        """
        if not levels:
            raise ValueError("levels is empty; there is nothing to return")
        if any(not 0.0 < q < 1.0 for q in levels):
            raise ValueError(f"quantile levels must lie in (0, 1), got {list(levels)}")
        if self.head == "point":
            raise ValueError(
                "a point forecast has no quantiles; build the model with head='quantile' "
                "or head='gaussian' if intervals are wanted"
            )
        if self.head == "quantile":
            index = [self._level_index(q) for q in levels]
            return self.raw[..., index]
        mean, sigma = self.gaussian_params()
        # ndtri is the inverse normal CDF: z(0.95) = 1.6449, so the outermost pair of a
        # gaussian head's 90 percent interval is mean +- 1.6449 sigma.
        z: Tensor = torch.special.ndtri(torch.tensor(levels, dtype=mean.dtype, device=mean.device))
        return mean[..., None] + z * sigma[..., None]

    def interval(self, alpha: float) -> tuple[Tensor, Tensor]:
        """Return the central ``1 - alpha`` prediction interval.

        ``alpha = 0.1`` is the 90 percent interval Gate 5 reads PICP at, and on a
        :data:`QUANTILE_FAN_9` model it returns exactly the 0.05 and 0.95 fan members.

        Args:
            alpha: Total tail mass, in (0, 1).

        Returns:
            Tuple ``(lower, upper)``, each of shape ``(B, H, C)``, in ``raw``'s units.
            Report the width alongside any coverage number: coverage alone is trivially
            achievable by widening an interval until it is useless.

        Raises:
            ValueError: If ``alpha`` is not in (0, 1), if this is a ``point``
                distribution, or if the fan does not contain the required levels.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        pair = self.quantiles_at((alpha / 2.0, 1.0 - alpha / 2.0))
        return pair[..., 0], pair[..., 1]

    def gaussian_params(self) -> tuple[Tensor, Tensor]:
        """Return the Gaussian mean and standard deviation.

        Returns:
            Tuple ``(mean, sigma)``, each of shape ``(B, H, C)``, in ``raw``'s units.
            ``sigma`` is ``exp(0.5 * log_var)`` of the clamped log-variance the head emits.

        Raises:
            ValueError: If this is not a ``gaussian`` distribution. The mean and the
                log-variance are not two quantiles and are not interchangeable with any
                other head's trailing axis (P5-D1).
        """
        if self.head != "gaussian":
            raise ValueError(
                f"gaussian_params() is defined for the gaussian head only, not {self.head!r}"
            )
        return self.raw[..., 0], torch.exp(0.5 * self.raw[..., 1])

    def affine(self, scale: Tensor, offset: Tensor) -> "PredictiveDistribution":
        """Map the distribution through ``y -> y * scale + offset``.

        **This is the single place unit conversion happens**, which is what lets every
        other method be unit-agnostic. Applied with the training-split channel scale and
        the per-window mean of :mod:`dmf.data.normalize`, it takes a dimensionless model
        output into corpus units: **degrees** for roll and pitch, **degrees per second**
        for their rates, **metres** for heave, **metres per second** for heave rate. It is
        the distribution-valued twin of :func:`dmf.data.normalize.invert_norm` and must
        agree with it on the point forecast.

        Per head:

        - ``point``: ``raw * scale + offset``.
        - ``quantile``: the same, broadcast across the trailing fan axis. A **positive**
          scale preserves the ascending order, so the result is still sorted; a negative
          scale would reverse the fan and silently swap every interval's endpoints, so it
          is refused.
        - ``gaussian``: ``mean * scale + offset`` and ``log_var + 2*log(scale)``. The
          offset does **not** touch the variance -- a location shift does not change a
          spread, and adding it would make interval width depend on the window mean.

        Args:
            scale: Per-channel scale, shape ``(1, 1, C)`` or anything broadcastable to
                ``(B, H, C)``. Strictly positive.
            offset: Per-window, per-channel offset, shape ``(B, 1, C)`` or anything
                broadcastable to ``(B, H, C)``.

        Returns:
            A new distribution of the same kind, in the units ``scale`` and ``offset``
            carry.

        Raises:
            ValueError: If any element of ``scale`` is not strictly positive.
        """
        if bool(torch.any(scale <= 0)):
            raise ValueError(
                "scale must be strictly positive: a non-positive scale reverses a quantile "
                "fan and swaps every interval's endpoints without changing its shape"
            )
        if self.head == "point":
            return PredictiveDistribution(self.raw * scale + offset, self.head, self.quantiles)
        if self.head == "quantile":
            fanned = self.raw * scale[..., None] + offset[..., None]
            return PredictiveDistribution(fanned, self.head, self.quantiles)
        mean = self.raw[..., 0] * scale + offset
        log_var = self.raw[..., 1] + 2.0 * torch.log(scale)
        return PredictiveDistribution(
            torch.stack((mean, torch.broadcast_to(log_var, mean.shape)), dim=-1),
            self.head,
            self.quantiles,
        )

    def _level_index(self, level: float) -> int:
        """Return the fan index of ``level``.

        Args:
            level: The requested quantile level.

        Returns:
            Index into the trailing axis of ``raw``.

        Raises:
            ValueError: If no fan member is within :data:`_LEVEL_TOL` of ``level``.
        """
        for i, q in enumerate(self.quantiles):
            if abs(q - level) <= _LEVEL_TOL:
                return i
        raise ValueError(
            f"level {level} is not in the fan {list(self.quantiles)}; levels are read off "
            f"the fan and never interpolated, so a reported coverage is a property of the "
            f"model and not of an interpolation rule"
        )


@runtime_checkable
class IntervalPredictor(Protocol):
    """Anything that turns an input window into a predictive distribution.

    The seam a split-conformal ``ConformalWrapper`` (Project 6) implements: it holds a
    model, delegates to it, and offsets the interval endpoints from held-out residuals. A
    wrapper satisfying this protocol adjusts intervals **without importing or subclassing
    any model**, which is the requirement ``docs/IMPLEMENTATION_PLAN.md`` §Phase 5 states.
    Exercised by a test double rather than only asserted in prose: a seam that is claimed
    in a docstring and never exercised is not a seam.
    """

    def predict(self, x: Tensor) -> PredictiveDistribution:
        """Return the predictive distribution for a batch of input windows.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            The predictive distribution over the horizon.
        """
        ...


class _PointView(nn.Module):
    """A probabilistic model, seen as the point forecaster of :meth:`.point`."""

    def __init__(self, model: "BaseForecaster") -> None:
        """Wrap one model.

        Args:
            model: The model to view. Registered as a submodule, so ``.to(device)`` and
                ``.eval()`` on the view reach the model itself -- which
                :func:`dmf.eval.runner.evaluate_models` relies on.
        """
        super().__init__()
        self.model = model
        self.head_kind: HeadKind = model.head_kind
        self.quantile_levels: tuple[float, ...] = model.quantile_levels

    def forward(self, x: Tensor) -> Tensor:
        """Forecast, reducing the predictive distribution to its point forecast.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Point forecasts, shape ``(B, H, C_out)``, dimensionless.
        """
        return PredictiveDistribution(self.model(x), self.head_kind, self.quantile_levels).point()


def point_view(model: "BaseForecaster") -> nn.Module:
    """View a probabilistic model as a rank-3 point forecaster.

    :func:`dmf.eval.runner.evaluate_models` hard-rejects any model whose output is not
    ``(B, H, C_out)``, and that guarantee must not be widened: it is what makes every row of
    every committed table -- persistence, AR, DLinear, the deep models -- come out of one
    scoring path with one skill denominator. P5-D5 nonetheless requires each probabilistic
    row to carry RMSE, MAE and skill. This wrapper is how both hold at once: the
    probabilistic model's own point forecast (0.5 quantile, or the Gaussian mean) is scored
    through the *exact same* code path as every other row.

    Args:
        model: Any forecaster carrying ``head_kind`` and ``quantile_levels``, including a
            plain point model, for which the view is a pass-through.

    Returns:
        A module whose ``forward(x)`` returns ``(B, H, C_out)``, holding ``model`` as a
        submodule so device moves and eval-mode switches propagate.
    """
    return _PointView(model)
