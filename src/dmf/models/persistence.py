"""Persistence baselines -- the reference every result is measured against.

Persistence is the denominator of the skill score ``1 - MSE_model/MSE_persistence``. A
result reported without it is not a result.

Damped persistence is included so that nobody can call the comparison a straw man. On an
oscillatory, mean-reverting signal, decaying toward the window mean is a genuinely strong
zero-parameter-per-sample forecast, and it is the baseline the deep models are required to
beat at Gate 4.

Two limits bracket the damped model and both are used elsewhere in the project:
``tau -> inf`` reduces it to plain persistence, and ``tau -> 0+`` reduces it to the
**window-mean forecast**, which is the correct null for the shuffle control (a model
trained on time-shuffled targets converges to the conditional mean, which inverts to the
window mean -- and the window mean *beats* persistence at long horizons on a narrowband
signal, so testing the shuffle control against "skill ~ 0" would fire a false alarm).

Units: ``tau`` is carried in **samples** everywhere in this module. Seconds appear only in
the human-readable report string, converted with ``fs_hz``.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from dmf.models.base import BaseForecaster
from dmf.train.registry import register_model
from dmf.typedefs import BoolArray, FloatArray

__all__ = [
    "DEFAULT_TAU_GRID",
    "TAU_WINDOW_MEAN",
    "DampedPersistence",
    "DecayFit",
    "Persistence",
    "decay_factors",
    "decay_sse",
    "fit_decay_constant",
    "solve_decay_tau",
]

_LOG = logging.getLogger(__name__)

#: Search bracket and resolution for the decay constant, **samples**, log-spaced. The lower
#: end is far enough below one sample that the window-mean limit is reachable, the upper end
#: far enough above the 50-sample horizon that the persistence limit is reachable. Both ends
#: being reachable is what makes the boundary flag on :class:`DecayFit` informative.
DEFAULT_TAU_GRID: tuple[float, float, int] = (0.05, 2000.0, 4001)

#: A decay constant small enough that ``exp(-1/tau)`` underflows to exactly 0.0 in float32,
#: so ``DampedPersistence(tau=TAU_WINDOW_MEAN)`` **is** the window-mean forecast. That model
#: is the correct null for the shuffle control, and expressing it as a limit of an existing
#: model avoids adding a fifth file for a forecast that predicts zero.
TAU_WINDOW_MEAN = 1.0e-3


@register_model("persistence")
class Persistence(BaseForecaster):
    """Repeat the last observed value across the whole horizon.

    Zero parameters, zero training. Exact on a constant signal and worst on a signal at
    its steepest, which for a narrowband oscillation means its error grows roughly linearly
    in horizon until it saturates at the signal's own RMS.
    """

    FIT_KIND = "none"

    def forward(self, x: Tensor) -> Tensor:
        """Broadcast the final lookback sample across the horizon.

        The expression is **bitwise identical** to the inline forecast in
        ``dmf.eval.controls._pipeline_persistence_sse``, which is the Gate 2 criterion 5
        control. That is asserted in ``tests/test_models.py`` rather than claimed here: if
        the two ever diverge, every skill denominator in the project silently stops being
        the quantity the control validated.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Point forecasts, shape ``(B, H, C_out)``, dimensionless, every horizon step
            equal to ``x[:, -1, :C_out]``.
        """
        return x[:, -1:, : self.n_target_channels].expand(-1, self.max_horizon, -1)


def decay_factors(tau_samples: FloatArray, max_horizon: int) -> FloatArray:
    """Return the per-horizon decay multipliers.

    **The horizon index is one-based.** Horizon step 0 of the output tensor is the forecast
    for lead time *one* sample, so its factor is ``exp(-1/tau)``, not ``1.0``. Getting this
    wrong shifts the whole fitted curve by one sample and makes ``tau`` meaningless while
    leaving the SSE only slightly worse -- which is why the one-based convention is defined
    in this one function and pinned by
    ``tests/test_models.py::test_decay_horizon_indexing_is_one_based``.

    Args:
        tau_samples: Per-channel decay constants, **samples**, shape ``(C,)``, positive.
        max_horizon: Forecast length ``H``, samples.

    Returns:
        Multipliers ``exp(-(h+1)/tau)`` for ``h`` in ``[0, H)``, shape ``(H, C)``,
        dimensionless.

    Raises:
        ValueError: If any decay constant is not positive, or ``max_horizon`` is not
            positive.
    """
    tau = np.asarray(tau_samples, dtype=np.float64)
    if tau.ndim != 1:
        raise ValueError(f"tau_samples must be 1-D, got shape {tau.shape}")
    if np.any(tau <= 0.0):
        raise ValueError(f"decay constants must be positive samples, got {tau.tolist()}")
    if max_horizon < 1:
        raise ValueError(f"max_horizon must be positive, got {max_horizon}")
    lead = np.arange(1, max_horizon + 1, dtype=np.float64)[:, None]
    return np.exp(-lead / tau[None, :])


def decay_sse(
    tau_samples: FloatArray, sxx: FloatArray, sxy: FloatArray, syy: FloatArray
) -> FloatArray:
    """Evaluate the closed-form training SSE of the damped forecast at a given ``tau``.

    With ``a_n`` the last lookback sample of window ``n`` and ``r_h = exp(-(h+1)/tau)``,

    ``SSE = sum_n sum_h (a_n r_h - y_nh)^2 = sxx * sum_h r_h^2 - 2 sum_h r_h sxy_h + syy``

    so the whole training split collapses to three small sufficient statistics and the
    search over ``tau`` never touches the data again. This is why fitting damped
    persistence costs one dataloader pass rather than one per grid point, and why it can
    share that pass with the AR moments.

    Args:
        tau_samples: Per-channel decay constants, **samples**, shape ``(C,)``.
        sxx: ``sum_n a_n^2`` per channel, shape ``(C,)``, dimensionless.
        sxy: ``sum_n a_n y_nh`` per horizon step and channel, shape ``(H, C)``,
            dimensionless.
        syy: ``sum_n sum_h y_nh^2`` per channel, shape ``(C,)``, dimensionless.

    Returns:
        Per-channel sum of squared errors, shape ``(C,)``, dimensionless.

    Raises:
        ValueError: If the statistics disagree on the channel count.
    """
    sxx = np.asarray(sxx, dtype=np.float64)
    sxy = np.asarray(sxy, dtype=np.float64)
    syy = np.asarray(syy, dtype=np.float64)
    if sxy.ndim != 2 or sxx.shape != (sxy.shape[1],) or syy.shape != (sxy.shape[1],):
        raise ValueError(
            f"inconsistent decay statistics: sxx {sxx.shape}, sxy {sxy.shape}, syy {syy.shape}"
        )
    factors = decay_factors(np.asarray(tau_samples, dtype=np.float64), sxy.shape[0])
    sse = sxx * np.square(factors).sum(axis=0) - 2.0 * (factors * sxy).sum(axis=0) + syy
    return np.asarray(sse, dtype=np.float64)


@dataclass(frozen=True)
class DecayFit:
    """Outcome of fitting the damped-persistence decay constant.

    Attributes:
        tau_samples: Per-channel decay constants, **samples**, shape ``(C,)``.
        tau_seconds: The same constants in seconds, shape ``(C,)``. Stored rather than
            recomputed so that no downstream caller has to remember which unit it holds.
        sse: Training SSE at the fitted ``tau``, shape ``(C,)``, dimensionless.
        at_grid_boundary: Per channel, True if the grid minimum sat on the first or last
            grid point, meaning the reported ``tau`` is a bracket edge rather than an
            interior optimum. Flagged rather than silently returned.
        summary: Human-readable one-line report, quoting ``tau`` in both units.
    """

    tau_samples: FloatArray
    tau_seconds: FloatArray
    sse: FloatArray
    at_grid_boundary: BoolArray
    summary: str


def solve_decay_tau(
    sxx: FloatArray,
    sxy: FloatArray,
    syy: FloatArray,
    fs_hz: float,
    *,
    channels: tuple[str, ...] | None = None,
    grid: tuple[float, float, int] = DEFAULT_TAU_GRID,
) -> DecayFit:
    """Minimise the closed-form decay SSE over ``tau``, per channel.

    Exact grid search in ``log(tau)`` followed by one parabolic refinement through the
    three points bracketing the grid minimum. Deterministic and RNG-free: the same
    statistics always produce the same ``tau``, which is what lets damped persistence emit
    a single results row with ``deterministic=True`` instead of three identical seeds.

    A parabolic step is used rather than a root-find because ``SSE(tau)`` is smooth but not
    convex in ``tau`` -- a Newton step from a bad start can leave the bracket, while the
    grid guarantees the global minimum to grid resolution before refinement begins.

    Args:
        sxx: ``sum_n a_n^2`` per channel, shape ``(C,)``, dimensionless.
        sxy: ``sum_n a_n y_nh``, shape ``(H, C)``, dimensionless.
        syy: ``sum_n sum_h y_nh^2`` per channel, shape ``(C,)``, dimensionless.
        fs_hz: Sampling rate, hertz. Used to report ``tau`` in seconds alongside samples.
        channels: Channel names for the report string. Positional labels if omitted.
        grid: ``(tau_min, tau_max, n_points)`` search bracket in **samples**, log-spaced.

    Returns:
        The fit, including the boundary flags and the report string.

    Raises:
        ValueError: If ``fs_hz`` is not positive, if the grid is degenerate, or if the
            statistics disagree on the channel count.
    """
    if fs_hz <= 0.0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz}")
    tau_min, tau_max, n_points = grid
    if not 0.0 < tau_min < tau_max or n_points < 3:
        raise ValueError(f"degenerate tau grid {grid}")

    sxy = np.asarray(sxy, dtype=np.float64)
    n_channels = sxy.shape[1] if sxy.ndim == 2 else 0
    log_grid = np.linspace(np.log(tau_min), np.log(tau_max), n_points)
    curve = np.stack([decay_sse(np.exp(u) * np.ones(n_channels), sxx, sxy, syy) for u in log_grid])

    best = np.argmin(curve, axis=0)
    tau = np.empty(n_channels, dtype=np.float64)
    boundary = np.zeros(n_channels, dtype=np.bool_)
    for c in range(n_channels):
        i = int(best[c])
        if i in (0, n_points - 1):
            boundary[c] = True
            tau[c] = float(np.exp(log_grid[i]))
            continue
        u0, u1, u2 = log_grid[i - 1 : i + 2]
        f0, f1, f2 = curve[i - 1 : i + 2, c]
        denom = f0 - 2.0 * f1 + f2
        step = 0.0 if denom <= 0.0 else 0.5 * (f0 - f2) / denom * (u2 - u1)
        tau[c] = float(np.exp(u1 + float(np.clip(step, u0 - u1, u2 - u1))))

    sse = decay_sse(tau, sxx, sxy, syy)
    tau_seconds = tau / fs_hz
    names = channels if channels is not None else tuple(f"c{i}" for i in range(n_channels))
    parts = [
        f"{name}: tau={t:.3f} samples ({s:.3f} s at {fs_hz:g} Hz)"
        + ("  [GRID BOUNDARY]" if flag else "")
        for name, t, s, flag in zip(names, tau, tau_seconds, boundary, strict=True)
    ]
    return DecayFit(
        tau_samples=tau,
        tau_seconds=tau_seconds,
        sse=sse,
        at_grid_boundary=boundary,
        summary="damped persistence decay constants -- " + "; ".join(parts),
    )


@register_model("damped_persistence")
class DampedPersistence(BaseForecaster):
    """Decay from the last observed value toward the window mean.

    ``y[h] = mean + (x[-1] - mean) * exp(-h/tau)``, with ``tau`` fitted per channel on the
    training split rather than tuned by hand.
    """

    FIT_KIND = "closed_form"

    def __init__(
        self,
        lookback: int,
        max_horizon: int,
        n_input_channels: int,
        n_target_channels: int,
        tau_samples: FloatArray | None = None,
    ) -> None:
        """Initialise, optionally with pre-fitted decay constants.

        Args:
            lookback: Input window length ``L``, samples.
            max_horizon: Forecast length ``H``, samples.
            n_input_channels: Input channel count ``C_in``.
            n_target_channels: Target channel count ``C_out``.
            tau_samples: Per-channel decay constants, **samples**, shape ``(C_out,)``. If
                None, must be supplied by :func:`fit_decay_constant` before use.

        Raises:
            ValueError: If ``tau_samples`` has the wrong shape or a non-positive entry.
        """
        super().__init__(lookback, max_horizon, n_input_channels, n_target_channels)
        self.register_buffer("factors", torch.zeros(max_horizon, n_target_channels))
        self.register_buffer("tau", torch.zeros(n_target_channels, dtype=torch.float64))
        self._fitted = False
        if tau_samples is not None:
            self.set_tau(tau_samples)

    def set_tau(self, tau_samples: FloatArray) -> None:
        """Install fitted decay constants.

        Args:
            tau_samples: Per-channel decay constants, **samples**, shape ``(C_out,)``.

        Raises:
            ValueError: If the shape is wrong or an entry is not positive.
        """
        tau = np.asarray(tau_samples, dtype=np.float64)
        if tau.shape != (self.n_target_channels,):
            raise ValueError(
                f"tau_samples has shape {tau.shape}, expected ({self.n_target_channels},)"
            )
        factors = decay_factors(tau, self.max_horizon)
        self.factors = torch.from_numpy(factors).to(dtype=torch.float32)
        self.tau = torch.from_numpy(tau.copy())
        self._fitted = True

    @property
    def n_fitted_parameters(self) -> int:
        """One decay constant per target channel."""
        return self.n_target_channels if self._fitted else 0

    def forward(self, x: Tensor) -> Tensor:
        """Forecast an exponential decay toward the per-window mean.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless. Because ``x`` is
                already de-meaned, the decay target is zero in this space; the corpus-unit
                mean is restored downstream by
                :func:`dmf.data.normalize.invert_norm`.

        Returns:
            Point forecasts, shape ``(B, H, C_out)``, dimensionless.

        Raises:
            RuntimeError: If the decay constants have not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "DampedPersistence has no decay constants; call set_tau or "
                "dmf.train.closed_form.fit_damped_persistence first"
            )
        last = x[:, -1:, : self.n_target_channels]
        return last * self.factors.to(dtype=x.dtype)[None, :, :]


def fit_decay_constant(x: FloatArray, y: FloatArray, fs_hz: float) -> FloatArray:
    """Fit the per-channel decay constant by least squares on the training split.

    Array form, kept because the docstring contract and the unit tests use it. It
    materialises nothing beyond the three sufficient statistics, but it does require the
    caller to hold ``x`` and ``y`` in memory; production fitting goes through
    :func:`dmf.train.closed_form.fit_damped_persistence`, which accumulates the same
    statistics streaming over a dataloader (``unseen_vessel/train`` holds 1.88 M windows,
    so the stacked ``y`` alone would be 2.2 GB).

    Args:
        x: Training input windows, shape ``(N, L, C)``, dimensionless.
        y: Training target windows, shape ``(N, H, C)``, dimensionless -- i.e. already put
            through :func:`dmf.data.normalize.normalize_target`, not in corpus units.
        fs_hz: Sampling rate, hertz. Used only to report ``tau`` in seconds alongside the
            returned value in samples.

    Returns:
        Per-channel decay constants, **samples**, shape ``(C,)``.

    Raises:
        ValueError: If ``x`` and ``y`` disagree on ``N`` or ``C``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(
            f"x must be (N, L, C) and y must be (N, H, C), got {x.shape} and {y.shape}"
        )
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError(
            f"x {x.shape} and y {y.shape} must agree on the window count N and channel count C"
        )
    last = x[:, -1, :]
    fit = solve_decay_tau(
        sxx=np.square(last).sum(axis=0),
        sxy=np.einsum("nc,nhc->hc", last, y),
        syy=np.square(y).sum(axis=(0, 1)),
        fs_hz=fs_hz,
    )
    _LOG.info("%s", fit.summary)
    return fit.tau_samples
