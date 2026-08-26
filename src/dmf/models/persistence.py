"""Persistence baselines -- the reference every result is measured against.

Persistence is the denominator of the skill score ``1 - MSE_model/MSE_persistence``. A
result reported without it is not a result.

Damped persistence is included so that nobody can call the comparison a straw man. On an
oscillatory, mean-reverting signal, decaying toward the window mean is a genuinely strong
zero-parameter-per-sample forecast, and it is the baseline the deep models are required to
beat at Gate 4.
"""

from torch import Tensor

from dmf.models.base import BaseForecaster
from dmf.typedefs import FloatArray

__all__ = ["DampedPersistence", "Persistence", "fit_decay_constant"]


class Persistence(BaseForecaster):
    """Repeat the last observed value across the whole horizon.

    Zero parameters, zero training. Exact on a constant signal and worst on a signal at
    its steepest, which for a narrowband oscillation means its error grows roughly linearly
    in horizon until it saturates at the signal's own RMS.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Broadcast the final lookback sample across the horizon.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Point forecasts, shape ``(B, H, C_out)``, dimensionless, every horizon step
            equal to ``x[:, -1, :C_out]``.
        """
        raise NotImplementedError


class DampedPersistence(BaseForecaster):
    """Decay from the last observed value toward the window mean.

    ``y[h] = mean + (x[-1] - mean) * exp(-h/tau)``, with ``tau`` fitted per channel on the
    training split rather than tuned by hand.
    """

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
        """
        raise NotImplementedError

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
        raise NotImplementedError


def fit_decay_constant(x: FloatArray, y: FloatArray, fs_hz: float) -> FloatArray:
    """Fit the per-channel decay constant by least squares on the training split.

    Args:
        x: Training input windows, shape ``(N, L, C)``, dimensionless.
        y: Training target windows, shape ``(N, H, C)``, dimensionless.
        fs_hz: Sampling rate, hertz. Used only to report ``tau`` in seconds alongside the
            returned value in samples.

    Returns:
        Per-channel decay constants, **samples**, shape ``(C,)``.

    Raises:
        ValueError: If ``x`` and ``y`` disagree on ``N`` or ``C``.
    """
    raise NotImplementedError
