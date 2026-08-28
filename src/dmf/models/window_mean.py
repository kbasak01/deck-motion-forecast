"""The window-mean forecast: the ``tau -> 0+`` limit of damped persistence.

Predict the lookback window's own mean for every horizon step, whatever the signal has
just done. Zero parameters, no fitting, no seed -- a trivial baseline in the same sense
persistence is.

**It ships as a row because it wins often enough to matter.** On a narrowband,
mean-reverting signal the window mean beats persistence from about 2 s out, and measured on
the committed Phase 3 sweep it also beats the *fitted* ``damped_persistence`` in 54 of 144
(regime, DOF, horizon) cells and the exact-optimum ``ar20`` in 16 (``ideal``) / 19 (``imu``).
Until now that number existed only as the ``skill_null`` column of
``results/baselines_controls.csv``, where it was the shuffle control's null hypothesis. A
trivial baseline that outperforms fitted ones, measured but kept out of the results table,
is CLAUDE.md non-negotiable 6 inverted, so it is now scored in the same pass, over the same
windows, as every other model.

The forecast itself is *not* reimplemented here: this class is
:class:`dmf.models.persistence.DampedPersistence` with ``tau`` pinned to
:data:`dmf.models.persistence.TAU_WINDOW_MEAN`, so the row in ``results/baselines.csv`` is
produced by the same expression that produces the control's null, and
``tests/test_models.py`` asserts the two agree bitwise.
"""

from typing import ClassVar

import numpy as np

from dmf.models.base import FitKind
from dmf.models.persistence import TAU_WINDOW_MEAN, DampedPersistence
from dmf.train.registry import register_model

__all__ = ["WindowMean"]


@register_model("window_mean")
class WindowMean(DampedPersistence):
    """Forecast the per-window mean at every horizon step.

    ``FIT_KIND = "none"``: nothing is estimated from data, so the experiment driver
    instantiates it and scores it without a moments pass, and the row carries
    ``n_params = 0``. That is the honest count -- ``tau`` here is a limit that defines the
    model, not a value fitted on the training split, which is exactly what separates this
    row from ``damped_persistence`` (six fitted decay constants).

    Because ``x`` reaching :meth:`forward` is already de-meaned, the window mean is zero in
    the model's dimensionless space and :func:`dmf.data.normalize.invert_norm` restores it
    to corpus units downstream.
    """

    # The value is inside the ``FitKind`` union ``BaseForecaster`` declares, but mypy
    # re-infers the class variable's type from DampedPersistence's own bare assignment
    # (``FIT_KIND = "closed_form"``) and so rejects any override in a grandchild class.
    # Silenced here rather than fixed by annotating DampedPersistence, which CLAUDE.md forbids
    # editing to accommodate a new model.
    FIT_KIND: ClassVar[FitKind] = "none"  # type: ignore[assignment]

    def __init__(
        self,
        lookback: int,
        max_horizon: int,
        n_input_channels: int,
        n_target_channels: int,
    ) -> None:
        """Configure the model at the ``tau -> 0+`` limit.

        ``tau`` is not a constructor argument on purpose: a configurable one would let a
        config file ship a row labelled ``window_mean`` that is not the window mean.

        Args:
            lookback: Input window length ``L``, samples.
            max_horizon: Forecast length ``H``, samples.
            n_input_channels: Input channel count ``C_in``.
            n_target_channels: Target channel count ``C_out``.
        """
        super().__init__(
            lookback,
            max_horizon,
            n_input_channels,
            n_target_channels,
            tau_samples=np.full(n_target_channels, TAU_WINDOW_MEAN, dtype=np.float64),
        )

    @property
    def n_fitted_parameters(self) -> int:
        """Zero: the decay constant is a defining limit, not a value fitted on train."""
        return 0
