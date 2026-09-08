"""The unconditional interval: a point forecast plus its own empirical residual quantiles.

This is the probabilistic **baseline** ``results/e03/probabilistic.csv`` does not have.
That table holds six learned heads -- three architectures x {quantile, gaussian} -- and
nothing trivial, so "102 of 216 rows within the PICP band" has no floor to be read against.
Persistence plays that role for every point metric (CLAUDE.md non-negotiable 4); nothing
played it for coverage, and a coverage number without a trivial reference is not a result.

**What it is.** A fitted point model -- :class:`dmf.models.dlinear_ols.DLinearOLS`, the
closed-form optimum, so the point forecast underneath the interval is not itself an
optimisation accident -- plus one empirical residual quantile per
``(horizon, target channel, level)``, taken on the **validation** split. The forecast is

    ``q_k(h, c | x) = point(x)[h, c] + r_k(h, c)``

so the interval is *translated* by the point forecast and its **width does not depend on
the input at all**. That is not a defect to be apologised for; it is the whole point.

**Why an unconditional interval is the right null.** A learned head claims to know when it
is uncertain. This object cannot know: it emits the same width on the calmest window of
SS3 and the worst of SS6. So the difference between a learned head's Winkler score and this
one's is exactly the value of the head's *conditional* uncertainty, and a head that cannot
beat it has learned nothing conditional about its own errors -- it has only learned the
marginal error distribution, which costs ``H * C * Q`` numbers and no gradient steps. The
same object therefore serves as the null for ``interval_shuffle_control`` and
``interval_untrained_control``.

**The residual split is validation, and the class refuses anything else.** Fitting the
quantiles on train would measure the point model's *training* residuals, which are smaller
than its test residuals by exactly the amount it overfits, so the intervals would be too
narrow in the direction that flatters coverage. Fitting them on test is the leak itself.
:func:`dmf.train.closed_form.fit_residual_interval` checks
:attr:`dmf.data.dataset.DeckMotionDataset.partition` and raises, rather than trusting the
caller to pass the right dataset -- the same structural guard
:func:`dmf.data.normalize.apply_norm` applies to normalisation statistics.

**Crossing is structurally impossible here** (``docs/protocol.md`` P5-D16): the fan is one
ascending vector of empirical quantiles added to a scalar, so its raw -- pre-sort -- fan is
already non-decreasing at every element, which :meth:`set_residual_quantiles` enforces on
installation rather than assuming. The measured crossing rate of this row is therefore
exactly 0, against a median 6% and a worst-cell 96.6% for ``dlinear_quantile``, and that
contrast is part of what the row is for.

Units: the residual quantiles are dimensionless -- they are fitted in the model's
normalised space, like every other model's output, and reach corpus units only through
:func:`dmf.data.normalize.invert_norm` or
:meth:`dmf.models.heads.PredictiveDistribution.affine`.
"""

from typing import ClassVar

import numpy as np
import torch
from torch import Tensor

from dmf.models.base import BaseForecaster, FitKind
from dmf.models.dlinear_ols import DLinearOLS
from dmf.models.heads import QUANTILE_FAN_9, HeadKind
from dmf.train.registry import register_model
from dmf.typedefs import FloatArray

__all__ = ["EmpiricalResidualInterval"]


@register_model("residual_interval")
class EmpiricalResidualInterval(BaseForecaster):
    """A closed-form point model wrapped in its own empirical residual quantiles.

    ``FIT_KIND = "closed_form"``: both halves are solved rather than trained. The point
    half comes from the same :class:`dmf.train.closed_form.TrainingMoments` pass every other
    closed-form row is solved from, so adding this row to an experiment costs one extra
    solve and one pass over the validation split -- no epochs, no seed, no RNG. It is
    therefore ``deterministic=True`` in the results table and exempt from the three-seed
    rule for the same reason ``ar20`` and ``dlinear_ols`` are.

    The point model is held by **composition, not inheritance**. It is not a
    ``DLinearOLS``: it is a quantile model that has one, and
    :func:`dmf.train.experiment._fit_one` dispatches on ``issubclass``, so subclassing would
    route it into the point-model branch and score it as a point row.
    """

    #: Solved, not trained: the point half from the shared moments pass, the fan from one
    #: streaming pass over the validation split.
    FIT_KIND: ClassVar[FitKind] = "closed_form"

    #: Quantile only. A point-headed build would be ``dlinear_ols`` under a second label,
    #: and a Gaussian one would claim a parametric family the empirical residuals do not
    #: assume -- the freedom from that assumption is the reason this baseline is empirical.
    SUPPORTED_HEADS: ClassVar[tuple[HeadKind, ...]] = ("quantile",)

    #: Declared at class level because ``nn.Module.__getattr__`` types a registered buffer
    #: as ``Tensor | Module``; without these two lines every use below needs a cast.
    residual_quantiles: Tensor
    is_fitted: Tensor

    def __init__(
        self,
        lookback: int,
        max_horizon: int,
        n_input_channels: int,
        n_target_channels: int,
        kernel_size: int = 25,
        ridge: float = 0.0,
        n_quantiles: int = len(QUANTILE_FAN_9),
        head: HeadKind | None = None,
    ) -> None:
        """Configure the wrapper and its (as yet unfitted) point model.

        Args:
            lookback: Input window length ``L``, samples.
            max_horizon: Forecast length ``H``, samples.
            n_input_channels: Input channel count ``C_in``.
            n_target_channels: Target channel count ``C_out``.
            kernel_size: Trend-extraction window of the inner DLinear, samples. Must match
                the ``dlinear_ols`` row's, or the two rows are not the same point forecast
                and the interval stops being that row's interval.
            ridge: Tikhonov strength the inner model is solved under, dimensionless.
            n_quantiles: Fan width ``Q``. Defaults to the project's nine-level fan, so the
                0.05, 0.5 and 0.95 levels PICP@90 and P5-D5 read are present exactly.
            head: Output head kind; only ``"quantile"`` is supported and it is the default
                implied by ``n_quantiles > 0``. Accepted so that
                :func:`dmf.train.registry.build_model` can pass the config's head through
                the same accepted-argument check every other model gets.

        Raises:
            ValueError: If the geometry is invalid, or the head is not ``"quantile"``.
        """
        super().__init__(
            lookback, max_horizon, n_input_channels, n_target_channels, n_quantiles, head
        )
        #: The point forecast the fan is centred on. A submodule, so ``.to(device)``,
        #: ``.eval()`` and ``state_dict()`` reach it.
        self.point_model = DLinearOLS(
            lookback,
            max_horizon,
            n_input_channels,
            n_target_channels,
            kernel_size=kernel_size,
            ridge=ridge,
        )
        self.kernel_size = kernel_size
        self.ridge = ridge
        #: Residual quantiles, ``(H, C_out, Q)``, dimensionless. A **buffer**: nothing
        #: differentiates through them, exactly as for the AR coefficients, and they must
        #: still travel with the state dict so a scored model can be rebuilt from disk.
        self.register_buffer(
            "residual_quantiles",
            torch.zeros(max_horizon, n_target_channels, n_quantiles, dtype=torch.float32),
        )
        #: Fitted flag, as a buffer so it survives ``load_state_dict``. Zero-valued
        #: quantiles are a perfectly well-formed zero-width fan, which would be scored as a
        #: model with perfect sharpness and no coverage rather than as a bug.
        self.register_buffer("is_fitted", torch.zeros((), dtype=torch.bool))

    def set_residual_quantiles(self, quantiles: FloatArray) -> None:
        """Install the fitted residual quantiles.

        Args:
            quantiles: Empirical residual quantiles, shape ``(H, C_out, Q)``,
                dimensionless, non-decreasing along the ``Q`` axis.

        Raises:
            ValueError: If the shape is wrong, if any value is not finite, or if the fan is
                not non-decreasing. The last check is what makes "this row cannot cross"
                a property of the object rather than a claim in its docstring.
        """
        array = np.asarray(quantiles, dtype=np.float64)
        expected = (self.max_horizon, self.n_target_channels, self.n_quantiles)
        if array.shape != expected:
            raise ValueError(f"quantiles has shape {array.shape}, expected {expected}")
        if not np.all(np.isfinite(array)):
            raise ValueError("quantiles contains non-finite values")
        if np.any(np.diff(array, axis=-1) < 0.0):
            worst = float(np.min(np.diff(array, axis=-1)))
            raise ValueError(
                f"residual quantiles must be non-decreasing along the fan axis; the worst "
                f"adjacent step is {worst:.3e}. An empirical quantile vector is ascending "
                f"by construction, so this means the levels or the axes are transposed."
            )
        with torch.no_grad():
            self.residual_quantiles.copy_(torch.from_numpy(array).float())
            self.is_fitted.fill_(True)

    def _predict(self, x: Tensor) -> Tensor:
        """Forecast the point trajectory and translate the residual fan onto it.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Quantile forecasts, shape ``(B, H, C_out, Q)``, dimensionless, non-decreasing
            along ``Q`` **before** any post-hoc sort -- see the module docstring.

        Raises:
            RuntimeError: If the residual quantiles have not been fitted.
        """
        if not bool(self.is_fitted):
            raise RuntimeError(
                "EmpiricalResidualInterval has no residual quantiles; call "
                "dmf.train.closed_form.fit_residual_interval first. An unfitted instance "
                "would emit a zero-width fan, which scores as perfect sharpness."
            )
        point: Tensor = self.point_model(x)
        return point[..., None] + self.residual_quantiles.to(dtype=point.dtype)

    @property
    def n_fitted_parameters(self) -> int:
        """Point-model coefficients plus one number per ``(horizon, channel, level)``.

        The default count would report the inner DLinear's parameters only and miss the
        ``H * C_out * Q`` buffer-held residual quantiles -- 8100 of them at the shipped
        geometry, against 60 300 in the point model -- which is the direction that flatters
        this baseline in the budget column.

        Returns:
            Fitted value count.
        """
        return self.point_model.n_fitted_parameters + int(self.residual_quantiles.numel())
