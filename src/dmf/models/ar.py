"""Multivariate least-squares AR(p), fitted per training split.

Direct multi-horizon: one coefficient matrix per horizon step, so that a forecast at
``h = 30`` is a single linear map from the lookback rather than thirty compounding
one-step predictions. Fitted on CPU with NumPy; there is no training loop.

Gate 3 uses this model as a difficulty check on the task itself. If AR(p) already reaches
0.8 skill at a 3 s horizon on roll, the task is too easy as specified and the horizon or
observation mode must change before any effort goes into deep models.
"""

from torch import Tensor

from dmf.models.base import BaseForecaster
from dmf.typedefs import FloatArray

__all__ = ["ARForecaster"]


class ARForecaster(BaseForecaster):
    """Least-squares multivariate autoregressive forecaster.

    Attributes are set by :meth:`fit`; the model is not usable before that call.
    """

    def __init__(
        self,
        lookback: int,
        max_horizon: int,
        n_input_channels: int,
        n_target_channels: int,
        order: int = 20,
        ridge: float = 0.0,
    ) -> None:
        """Configure the AR model.

        Args:
            lookback: Input window length ``L``, samples. Must be at least ``order``.
            max_horizon: Forecast length ``H``, samples.
            n_input_channels: Input channel count ``C_in``.
            n_target_channels: Target channel count ``C_out``.
            order: AR order ``p``, samples of history used per prediction. The sweep in
                Phase 3 covers ``p`` in {10, 20, 40}.
            ridge: Tikhonov regularisation strength, dimensionless. Nonzero values guard
                against the ill-conditioning that a narrowband signal produces in the
                lagged design matrix.

        Raises:
            ValueError: If ``order`` exceeds ``lookback`` or is not positive.
        """
        raise NotImplementedError

    def fit(self, x: FloatArray, y: FloatArray) -> None:
        """Solve for one coefficient matrix per horizon step by least squares.

        Must be called with **training-split windows only**.

        Args:
            x: Training input windows, shape ``(N, L, C_in)``, dimensionless.
            y: Training target windows, shape ``(N, H, C_out)``, dimensionless.

        Raises:
            ValueError: If ``x`` and ``y`` disagree on ``N``, or if ``N`` is smaller than
                the number of free parameters per horizon step.
        """
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Apply the fitted coefficient matrices.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Point forecasts, shape ``(B, H, C_out)``, dimensionless.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        raise NotImplementedError
