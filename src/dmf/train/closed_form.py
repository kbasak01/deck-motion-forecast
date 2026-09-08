"""Closed-form fitting of the linear baselines from streaming sufficient statistics.

Three models here are exactly solvable, and all three are solved from the *same* single
pass over the training dataloader:

- **AR(p)** -- least squares over a lag-major design block. One pass at ``P_MAX`` yields
  the exact normal equations for every ``p <= P_MAX``, because
  :func:`dmf.models.ar.lag_features` is prefix-nested in the order.
- **Damped persistence** -- one decay constant per channel, minimised over a closed-form
  ``SSE(tau)`` built from three small statistics (:func:`dmf.models.persistence.decay_sse`).
- **DLinear** (:class:`dmf.models.dlinear_ols.DLinearOLS`) -- least squares over the
  ``(2L)``-column decomposed design ``[trend_lags, remainder_lags]``, stacked over target
  channels because the canonical model shares one ``(L -> H)`` map across them. Fitted
  *beside* the SGD ``dlinear`` row, never instead of it: every ``ar*`` row is at its exact
  optimum while the SGD row is wherever the epoch budget landed, so without this solve a
  DLinear-vs-AR gap conflates optimisation with architecture. The difference between the
  two DLinear rows is the measured optimisation shortfall.

Nothing here ever materialises the design matrix. ``unseen_vessel/train`` holds 1 878 432
windows, so a stacked ``(N, L, C_in)`` float64 array would be 18 GB and the targets alone
2.2 GB. The accumulated moments are 240x240 plus 240x150 for AR and 400x400 plus 400x150
for DLinear -- under two megabytes in total, independent of ``N``.

**One Gram serves all 150 regressions.** The design matrix is identical for every horizon
step and every target channel, so ``H * C_out = 150`` right-hand sides share one Cholesky
factorisation.

**One Gram also serves every information set.** ``lag_features`` is prefix-nested in the
order *and* channel-strided within each lag block, so both a lower order and a leading
subset of the input channels are sub-blocks of the same accumulated moments -- see
:func:`subset_columns`. ``ar_attitude_only`` (AR(40) on roll, pitch and heave only, whose
120 features are budget-matched to ``ar20``'s 20 x 6) is therefore solved from the same
moments as every other AR row and adds **no** pass over the training split, which matters
because that pass, not the solve, is the whole cost of a closed-form fit.

**The solve is centred and whitened.** ``Gc = G/n - outer(mx, mx)`` is rescaled to a
correlation matrix ``R`` with unit diagonal before the ridge penalty is added, which is
what makes ``ridge`` genuinely dimensionless as its docstring claims rather than
dimensionless-by-assertion. The measured ``cond(R)`` is reported: normal equations square
``cond(X)``, and adjacent lags of a narrowband signal are near-degenerate, so the number
belongs in the audit trail rather than in a hope.

Units: inputs and targets reaching this module are **dimensionless** (de-meaned and scaled
by :mod:`dmf.data.normalize`); ``tau`` is in samples; wall-clock times in seconds.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

from dmf.data.dataset import DeckMotionDataset, make_dataloader
from dmf.data.normalize import normalize_target
from dmf.models.ar import ARForecaster, lag_features
from dmf.models.dlinear import series_decompose
from dmf.models.dlinear_ols import DLinearOLS
from dmf.models.heads import quantile_fan
from dmf.models.persistence import DampedPersistence, DecayFit, solve_decay_tau
from dmf.models.residual_interval import EmpiricalResidualInterval
from dmf.typedefs import FloatArray, IntArray

__all__ = [
    "RESIDUAL_QUANTILE_MAX_WINDOWS",
    "ARFitReport",
    "DecayFitReport",
    "DecayMoments",
    "DecompFitReport",
    "DecompMoments",
    "LagMoments",
    "ResidualIntervalFitReport",
    "TrainingMoments",
    "accumulate_training_moments",
    "fit_ar",
    "fit_damped_persistence",
    "fit_dlinear_ols",
    "fit_residual_interval",
    "solve_ar_coefficients",
    "solve_decomp_coefficients",
    "subset_columns",
]

#: Windows the empirical residual quantiles are estimated from, at most. ``id/val`` holds
#: 240 realizations x 1131 windows = 271 440 windows, and the residual array is
#: ``(N, H, C_out)``, so the full split would be 977 MB of float32 for an estimate that is
#: not 30x better than a 25 000-window one: consecutive windows are 0.5 s apart on a signal
#: whose roll period is ~12 s, so the effective sample size is a small fraction of ``N``
#: either way. 25 000 windows still puts ~1250 samples in each 5 percent tail, per
#: ``(horizon, channel)`` cell.
RESIDUAL_QUANTILE_MAX_WINDOWS = 25_000

#: Seed for the target permutation and the batch order used by the shuffle control. Fixed
#: so that the control is reproducible; it is the only RNG in this module, and it is never
#: reached on a real fit -- the honest path uses ``shuffle=False``, whose moments are a
#: plain sum and therefore order-independent anyway.
_SHUFFLE_SEED = 20260826


@dataclass(frozen=True)
class LagMoments:
    """Streaming sufficient statistics for the AR normal equations.

    All entries are float64 sums over the training windows -- not means -- so that batches
    can be added independently of batch size.

    Attributes:
        n: Window count.
        sx: ``sum_n feats_n``, shape ``(d_max,)`` with ``d_max = max_order * C_in``.
        sy: ``sum_n y_n``, shape ``(H * C_out,)``, ``H``-major.
        syy: ``sum_n y_n^2`` elementwise, shape ``(H * C_out,)``. Needed for the residual
            training MSE without a second pass.
        gram: ``sum_n feats_n feats_n^T``, shape ``(d_max, d_max)``.
        cross: ``sum_n feats_n y_n^T``, shape ``(d_max, H * C_out)``.
        max_order: The order the statistics were accumulated at. Any ``p <= max_order`` is
            solvable from the leading sub-blocks.
        n_input_channels: ``C_in``.
        max_horizon: ``H``.
        n_target_channels: ``C_out``.
    """

    n: int
    sx: FloatArray
    sy: FloatArray
    syy: FloatArray
    gram: FloatArray
    cross: FloatArray
    max_order: int
    n_input_channels: int
    max_horizon: int
    n_target_channels: int


@dataclass(frozen=True)
class DecayMoments:
    """Streaming sufficient statistics for the damped-persistence decay constant.

    Attributes:
        n: Window count.
        sxx: ``sum_n a_n^2`` per target channel, shape ``(C_out,)``, where ``a_n`` is the
            last lookback sample.
        sxy: ``sum_n a_n y_nh``, shape ``(H, C_out)``.
        syy: ``sum_n sum_h y_nh^2`` per target channel, shape ``(C_out,)``.
    """

    n: int
    sxx: FloatArray
    sxy: FloatArray
    syy: FloatArray


@dataclass(frozen=True)
class DecompMoments:
    """Streaming sufficient statistics for the closed-form DLinear normal equations.

    The canonical DLinear shares one ``(L -> H)`` map per component across target
    channels, so the regression stacks the channels: one row per (window, target channel)
    pair, ``n_rows = n_windows * C_out``. Every entry is a float64 sum over those rows.

    Attributes:
        n_windows: Windows accumulated over.
        n_rows: Regression rows, ``n_windows * n_target_channels``.
        sx: ``sum_r feats_r``, shape ``(2 * L,)``.
        sy: ``sum_r y_r``, shape ``(H,)``.
        syy: ``sum_r y_r^2`` elementwise, shape ``(H,)``.
        gram: ``sum_r feats_r feats_r^T``, shape ``(2 * L, 2 * L)``. **Singular by
            construction**: with ``trend = A x`` and ``remainder = (I - A) x`` the ``2L``
            columns span an at-most-``L``-dimensional space, so the rank deficiency is a
            property of the DLinear parameterisation and not of the corpus.
        cross: ``sum_r feats_r y_r^T``, shape ``(2 * L, H)``.
        kernel_size: Trend-extraction window the decomposition used, samples. A fit
            solved from these moments must declare the same one or it is a different model.
        lookback: ``L``.
        max_horizon: ``H``.
        n_target_channels: ``C_out``.
    """

    n_windows: int
    n_rows: int
    sx: FloatArray
    sy: FloatArray
    syy: FloatArray
    gram: FloatArray
    cross: FloatArray
    kernel_size: int
    lookback: int
    max_horizon: int
    n_target_channels: int


@dataclass(frozen=True)
class TrainingMoments:
    """Everything the closed-form fitters need, from one pass over the training split.

    Attributes:
        lag: AR normal-equation statistics.
        decay: Damped-persistence statistics.
        fitted_on: Provenance label, ``"<regime>/train"``.
        n_windows: Windows accumulated over.
        shuffled_targets: True if the targets were permuted within each batch, i.e. these
            moments belong to the shuffle **control** and not to a real model.
        wall_time_s: Wall-clock seconds for the pass, dominated by the dataloader.
        decomp: DLinear normal-equation statistics, or None if no ``decompose_kernel`` was
            requested. None rather than zeros, so that asking for a DLinear fit from a pass
            that never accumulated its design fails loudly instead of solving an empty
            system.
    """

    lag: LagMoments
    decay: DecayMoments
    fitted_on: str
    n_windows: int
    shuffled_targets: bool
    wall_time_s: float
    decomp: DecompMoments | None = None


@dataclass(frozen=True)
class ARFitReport:
    """Diagnostics from one AR solve.

    Attributes:
        order: AR order ``p``, samples.
        ridge: Tikhonov strength applied to the whitened correlation matrix,
            dimensionless.
        cond_r: Measured 2-norm condition number of the whitened correlation matrix
            ``R``, dimensionless, **before** the ridge term. Expect 1e8-1e12 on a
            narrowband signal; the normal equations square ``cond(X)``.
        n_fitted_parameters: Coefficients plus intercepts.
        residual_train_mse: Mean squared residual on the training split, dimensionless,
            computed from the moments rather than by a second pass.
        fit_time_s: Wall-clock seconds for the solve alone, excluding the moments pass.
    """

    order: int
    ridge: float
    cond_r: float
    n_fitted_parameters: int
    residual_train_mse: float
    fit_time_s: float


@dataclass(frozen=True)
class DecayFitReport:
    """Diagnostics from one damped-persistence fit.

    Attributes:
        fit: The underlying decay fit, carrying ``tau`` in both samples and seconds and the
            per-channel grid-boundary flags.
        n_fitted_parameters: One decay constant per target channel.
        residual_train_mse: Mean squared residual on the training split, dimensionless.
        fit_time_s: Wall-clock seconds for the grid search, excluding the moments pass.
    """

    fit: DecayFit
    n_fitted_parameters: int
    residual_train_mse: float
    fit_time_s: float


@dataclass(frozen=True)
class DecompFitReport:
    """Diagnostics from one closed-form DLinear solve.

    Attributes:
        kernel_size: Trend-extraction window, samples.
        ridge: Tikhonov strength applied to the whitened correlation matrix,
            dimensionless.
        cond_r: Measured 2-norm condition number of the whitened correlation matrix ``R``,
            dimensionless, **before** the ridge term. Expected to be effectively infinite:
            the decomposed design is exactly rank-deficient by construction, so this number
            reports the parameterisation, not a data pathology.
        n_fitted_parameters: Weights plus intercepts, counted the same way as the SGD twin
            so the two rows share a budget column.
        residual_train_mse: Mean squared residual on the training split, dimensionless,
            computed from the moments rather than by a second pass. Directly comparable to
            :attr:`dmf.train.loop.TrainResult.best_val_loss` in units, though not in
            partition.
        fit_time_s: Wall-clock seconds for the solve alone, excluding the moments pass.
    """

    kernel_size: int
    ridge: float
    cond_r: float
    n_fitted_parameters: int
    residual_train_mse: float
    fit_time_s: float


@dataclass(frozen=True)
class ResidualIntervalFitReport:
    """Diagnostics from one :class:`dmf.models.residual_interval.EmpiricalResidualInterval` fit.

    Attributes:
        point: The inner closed-form DLinear's own report, carried whole so the interval
            row's point half is auditable against the ``dlinear_ols`` row it duplicates.
        fitted_on: Provenance of the residual quantiles, e.g. ``"id/val"``. Recorded on the
            report -- and asserted at fit time -- because the one thing that can make this
            baseline dishonest is fitting its own residuals on the split it is scored on.
        n_windows_total: Windows in the validation partition.
        n_windows_used: Windows the quantiles were actually estimated from.
        window_stride: Sub-sampling stride over the validation window index, windows. 1
            means every window was used.
        residual_val_mse: Mean squared residual of the point model over the sub-sampled
            validation windows, dimensionless. Comparable in units to
            :attr:`DecompFitReport.residual_train_mse`, and on a different partition.
        n_fitted_parameters: Point-model coefficients plus ``H * C_out * Q`` quantiles.
        fit_time_s: Wall-clock seconds, including the validation pass but excluding the
            training moments pass the point half is solved from.
    """

    point: DecompFitReport
    fitted_on: str
    n_windows_total: int
    n_windows_used: int
    window_stride: int
    residual_val_mse: float
    n_fitted_parameters: int
    fit_time_s: float


def accumulate_training_moments(
    dataset: DeckMotionDataset,
    *,
    max_order: int,
    batch_size: int = 4096,
    num_workers: int = 4,
    shuffle_targets: bool = False,
    decompose_kernel: int | None = None,
) -> TrainingMoments:
    """Stream the AR, damped-persistence and DLinear sufficient statistics in one pass.

    Args:
        dataset: The **training** partition to fit on.
        max_order: Largest AR order to be solvable from the result, samples. Every smaller
            order is solvable from the leading sub-blocks, so this is called once at 40.
        batch_size: Windows per batch. Affects speed only.
        num_workers: DataLoader worker processes.
        decompose_kernel: Trend-extraction window of the closed-form DLinear, samples, or
            None to skip its moments entirely. Passed in rather than defaulted because the
            kernel is a property of the model config, and accumulating a design at the
            wrong kernel would silently solve a different model. Adds no pass over the
            training split -- the decomposition rides on the same batches as the AR
            design.
        shuffle_targets: If True, permute the targets within each batch **and** draw the
            batches in shuffled order, destroying the input-target correspondence. This is
            the shuffle **control**: a model fitted on these moments must not beat the
            window-mean forecast.

            The loader is shuffled specifically because permuting within a *contiguous*
            batch is not enough. With ``shuffle=False`` a batch of 4096 windows spans only
            three or four realizations, so a within-batch permutation re-pairs a window
            with a target from the same realization and the same sea state -- and the fit
            can still learn that realization's local mean. Measured on
            ``unseen_heading``, that residual structure gave the "shuffled" model +0.22
            skill over the window-mean null on pitch at 5 s, which reads exactly like a
            leak. Shuffling the loader makes each batch span the whole training split, so
            the permutation destroys the correspondence it is supposed to destroy.

            Note the protocol's stated null ("skill collapses to ~0") is wrong for this
            task -- a model on shuffled targets degenerates to the conditional mean, which
            inverts to the window mean, and on a narrowband signal the window mean *beats*
            persistence at long horizons.

    Returns:
        The accumulated moments, labelled with the dataset's provenance.

    Raises:
        ValueError: If ``dataset`` is not a training partition, if its normalisation
            statistics were not fitted on train, or if ``max_order`` or
            ``decompose_kernel`` exceeds the lookback.
    """
    if dataset.partition != "train":
        raise ValueError(
            f"moments must be accumulated on a training partition, got "
            f"{dataset.regime}/{dataset.partition}. Fitting coefficients on val or test "
            f"windows is a leak that no normalisation label would catch: norm_stats."
            f"fitted_on proves the statistics are train-only, not the windows "
            f"(CLAUDE.md non-negotiable 2)."
        )
    spec = dataset.window_spec
    if not 1 <= max_order <= spec.lookback:
        raise ValueError(f"max_order must be in [1, lookback={spec.lookback}], got {max_order}")
    if decompose_kernel is not None and not 1 <= decompose_kernel <= spec.lookback:
        raise ValueError(
            f"decompose_kernel must be in [1, lookback={spec.lookback}], got {decompose_kernel}"
        )

    stats = dataset.norm_stats.subset(dataset.target_columns)
    n_in = len(dataset.input_columns)
    n_out = len(dataset.target_columns)
    horizon = spec.max_horizon
    d_max = max_order * n_in
    n_outputs = horizon * n_out

    sx = torch.zeros(d_max, dtype=torch.float64)
    sy = torch.zeros(n_outputs, dtype=torch.float64)
    syy = torch.zeros(n_outputs, dtype=torch.float64)
    gram = torch.zeros(d_max, d_max, dtype=torch.float64)
    cross = torch.zeros(d_max, n_outputs, dtype=torch.float64)
    sxx = torch.zeros(n_out, dtype=torch.float64)
    sxy = torch.zeros(horizon, n_out, dtype=torch.float64)
    syy_decay = torch.zeros(n_out, dtype=torch.float64)
    d_decomp = 2 * spec.lookback
    sx_d = torch.zeros(d_decomp, dtype=torch.float64)
    sy_d = torch.zeros(horizon, dtype=torch.float64)
    syy_d = torch.zeros(horizon, dtype=torch.float64)
    gram_d = torch.zeros(d_decomp, d_decomp, dtype=torch.float64)
    cross_d = torch.zeros(d_decomp, horizon, dtype=torch.float64)
    count = 0

    generator = torch.Generator().manual_seed(_SHUFFLE_SEED)
    loader = make_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_targets,
        num_workers=num_workers,
        seed=_SHUFFLE_SEED,
    )
    started = time.perf_counter()
    with torch.no_grad():
        for x, y, window_mean in loader:
            xd = x.double()
            yn = normalize_target(y.double(), stats, window_mean.double())
            if shuffle_targets:
                yn = yn[torch.randperm(yn.shape[0], generator=generator)]
            feats = lag_features(xd, max_order)
            targets = yn.reshape(yn.shape[0], -1)
            count += int(yn.shape[0])
            sx += feats.sum(dim=0)
            sy += targets.sum(dim=0)
            syy += torch.square(targets).sum(dim=0)
            gram += feats.T @ feats
            cross += feats.T @ targets
            last = xd[:, -1, :n_out]
            sxx += torch.square(last).sum(dim=0)
            sxy += torch.einsum("nc,nhc->hc", last, yn)
            syy_decay += torch.square(yn).sum(dim=(0, 1))
            if decompose_kernel is not None:
                # One row per (window, target channel): the canonical DLinear shares a
                # single (L -> H) map across channels, so the channels stack into the same
                # regression rather than getting one each. Row order is window-major with
                # the channel minor, and the target below is reshaped the same way -- if
                # the two ever disagree the fit silently regresses each channel's window on
                # another channel's future.
                trend, remainder = series_decompose(xd[:, :, :n_out], decompose_kernel)
                rows = int(yn.shape[0]) * n_out
                feats_d = torch.cat(
                    [
                        trend.transpose(1, 2).reshape(rows, spec.lookback),
                        remainder.transpose(1, 2).reshape(rows, spec.lookback),
                    ],
                    dim=1,
                )
                targets_d = yn.permute(0, 2, 1).reshape(rows, horizon)
                sx_d += feats_d.sum(dim=0)
                sy_d += targets_d.sum(dim=0)
                syy_d += torch.square(targets_d).sum(dim=0)
                gram_d += feats_d.T @ feats_d
                cross_d += feats_d.T @ targets_d
    elapsed = time.perf_counter() - started

    return TrainingMoments(
        lag=LagMoments(
            n=count,
            sx=sx.numpy(),
            sy=sy.numpy(),
            syy=syy.numpy(),
            gram=gram.numpy(),
            cross=cross.numpy(),
            max_order=max_order,
            n_input_channels=n_in,
            max_horizon=horizon,
            n_target_channels=n_out,
        ),
        decay=DecayMoments(n=count, sxx=sxx.numpy(), sxy=sxy.numpy(), syy=syy_decay.numpy()),
        fitted_on=f"{dataset.regime}/{dataset.partition}",
        n_windows=count,
        shuffled_targets=shuffle_targets,
        wall_time_s=elapsed,
        decomp=None
        if decompose_kernel is None
        else DecompMoments(
            n_windows=count,
            n_rows=count * n_out,
            sx=sx_d.numpy(),
            sy=sy_d.numpy(),
            syy=syy_d.numpy(),
            gram=gram_d.numpy(),
            cross=cross_d.numpy(),
            kernel_size=decompose_kernel,
            lookback=spec.lookback,
            max_horizon=horizon,
            n_target_channels=n_out,
        ),
    )


def subset_columns(*, order: int, n_input_channels: int, n_input_used: int) -> IntArray:
    """Return the design-matrix columns of a leading channel subset, in design order.

    ``lag_features`` puts channel ``c`` at lag ``k`` in column ``k * C_in + c``, so reading
    only the first ``m`` input channels at order ``p`` selects the strided index set
    ``{k * C_in + c : k < p, c < m}``. Enumerating it ``k``-major and ``c``-minor makes the
    result identical, column for column, to ``lag_features(x[..., :m], p)`` -- that
    equality is what allows the Gram and cross moments to be sliced instead of
    re-accumulated, and it is asserted in ``tests/test_models.py`` rather than trusted.

    Args:
        order: AR order ``p``, samples.
        n_input_channels: Channel count ``C_in`` the moments were accumulated over.
        n_input_used: Leading channels to keep, ``m`` in ``[1, C_in]``.

    Returns:
        Column indices, shape ``(order * n_input_used,)``, ascending.

    Raises:
        ValueError: If ``n_input_used`` is outside ``[1, n_input_channels]``.
    """
    if not 1 <= n_input_used <= n_input_channels:
        raise ValueError(
            f"n_input_used must be in [1, n_input_channels={n_input_channels}], got {n_input_used}"
        )
    lags = np.arange(order, dtype=np.int64)[:, None] * n_input_channels
    channels = np.arange(n_input_used, dtype=np.int64)[None, :]
    return (lags + channels).reshape(-1)


def solve_ar_coefficients(
    moments: LagMoments, *, order: int, ridge: float, n_input_used: int | None = None
) -> tuple[FloatArray, FloatArray, float]:
    """Solve the centred, whitened normal equations for one AR order.

    Args:
        moments: Statistics accumulated at ``max_order >= order``.
        order: AR order ``p`` to solve for, samples.
        ridge: Tikhonov strength added to the unit diagonal of the whitened correlation
            matrix, dimensionless.
        n_input_used: Leading input channels the regression may read, or None for all
            ``C_in`` of them. A smaller value slices the Gram and cross moments to that
            channel subset (:func:`subset_columns`) rather than triggering a second
            accumulation pass, so the information-set ablation is free.

    Returns:
        Tuple ``(weight, bias, cond_r)`` with ``weight`` of shape
        ``(order * n_input_used, H * C_out)``, ``bias`` of shape ``(H * C_out,)``, both
        dimensionless, and the measured condition number of ``R`` before regularisation.

    Raises:
        ValueError: If ``order`` exceeds the accumulated ``max_order``, if ``ridge`` is
            negative, if ``n_input_used`` is outside ``[1, C_in]``, or if a feature has
            zero variance on the training split.
    """
    if not 1 <= order <= moments.max_order:
        raise ValueError(f"order must be in [1, {moments.max_order}], got {order}")
    if ridge < 0.0:
        raise ValueError(f"ridge must be non-negative, got {ridge}")
    if moments.n < 2:
        raise ValueError(f"need at least 2 windows to solve, got {moments.n}")

    cols = _design_columns(moments, order=order, n_input_used=n_input_used)
    # Reported as columns of the *accumulated* design, not of the subset: under
    # ``n_input_used`` the two numberings differ, and an index that does not point at the
    # offending lag/channel pair sends the reader to the wrong feature.
    return _solve_centred_whitened(
        n=float(moments.n),
        sx=moments.sx[cols],
        sy=moments.sy,
        gram=moments.gram[np.ix_(cols, cols)],
        cross=moments.cross[cols, :],
        ridge=ridge,
        labels=cols,
        feature_kind="lag features",
    )


def _solve_centred_whitened(
    *,
    n: float,
    sx: FloatArray,
    sy: FloatArray,
    gram: FloatArray,
    cross: FloatArray,
    ridge: float,
    labels: IntArray,
    feature_kind: str,
) -> tuple[FloatArray, FloatArray, float]:
    """Solve one centred, whitened, ridge-regularised least-squares system.

    Shared by every closed-form linear fit in this module, so that "same solver, same
    ridge treatment" is a fact about the code rather than a claim in a config comment. The
    Gram is centred, rescaled to a correlation matrix with unit diagonal, penalised, and
    factorised by Cholesky; a singular system falls back to ``lstsq``, whose minimum-norm
    solution is still an exact minimiser of the squared error.

    Args:
        n: Row count the sums were accumulated over.
        sx: ``sum feats``, shape ``(d,)``.
        sy: ``sum y``, shape ``(k,)``.
        gram: ``sum feats feats^T``, shape ``(d, d)``.
        cross: ``sum feats y^T``, shape ``(d, k)``.
        ridge: Tikhonov strength on the unit diagonal of the correlation matrix,
            dimensionless.
        labels: Column identifiers used in the zero-variance error message, shape ``(d,)``.
        feature_kind: Human-readable name of the design columns, for that same message.

    Returns:
        Tuple ``(weight, bias, cond_r)`` with ``weight`` of shape ``(d, k)``, ``bias`` of
        shape ``(k,)``, both dimensionless, and the condition number of the correlation
        matrix **before** regularisation.

    Raises:
        ValueError: If a design column has zero variance on the training split.
    """
    mx = sx / n
    my = sy / n
    gram_c = gram / n - np.outer(mx, mx)
    cross_c = cross / n - np.outer(mx, my)
    d = int(mx.size)

    scale = np.sqrt(np.diag(gram_c))
    dead = np.flatnonzero(scale <= 0.0)
    if dead.size:
        raise ValueError(
            f"{feature_kind} {labels[dead].tolist()} have zero variance on the training "
            f"split; the normal equations are singular"
        )
    corr = gram_c / np.outer(scale, scale)
    corr = 0.5 * (corr + corr.T)
    cond_r = float(np.linalg.cond(corr))

    rhs = cross_c / scale[:, None]
    regularised = corr + ridge * np.eye(d)
    try:
        z = cho_solve(cho_factor(regularised, lower=True, check_finite=False), rhs)
    except np.linalg.LinAlgError:
        z = np.linalg.lstsq(regularised, rhs, rcond=None)[0]
    weight = z / scale[:, None]
    bias = my - weight.T @ mx
    return weight, bias, cond_r


def solve_decomp_coefficients(
    moments: DecompMoments, *, ridge: float
) -> tuple[FloatArray, FloatArray, float]:
    """Solve the centred, whitened normal equations of the closed-form DLinear.

    Same solver, same centring, same whitening and the same ridge treatment as
    :func:`solve_ar_coefficients` -- the two rows of the results table must not differ in
    how their optimum was found, only in what design it was found over.

    The system is **singular by construction** (see :class:`DecompMoments`), so the
    regularised factorisation is what makes the solve well posed; at ``ridge = 0`` the
    ``lstsq`` fallback returns the minimum-norm exact minimiser instead. Either way the
    resulting *forecast* is unique even though the coefficients are not.

    Args:
        moments: Decomposed-design statistics from :func:`accumulate_training_moments`.
        ridge: Tikhonov strength on the whitened correlation matrix, dimensionless.

    Returns:
        Tuple ``(weight, bias, cond_r)`` with ``weight`` of shape ``(2 * L, H)`` --
        trend rows first, both blocks in window order -- ``bias`` of shape ``(H,)``, and
        the measured condition number before regularisation.

    Raises:
        ValueError: If ``ridge`` is negative or fewer than two rows were accumulated.
    """
    if ridge < 0.0:
        raise ValueError(f"ridge must be non-negative, got {ridge}")
    if moments.n_rows < 2:
        raise ValueError(f"need at least 2 regression rows to solve, got {moments.n_rows}")
    return _solve_centred_whitened(
        n=float(moments.n_rows),
        sx=moments.sx,
        sy=moments.sy,
        gram=moments.gram,
        cross=moments.cross,
        ridge=ridge,
        labels=np.arange(moments.sx.size, dtype=np.int64),
        feature_kind="decomposed design columns",
    )


def _design_columns(moments: LagMoments, *, order: int, n_input_used: int | None) -> IntArray:
    """Return the moment columns one AR fit reads, resolving the None default.

    Args:
        moments: The accumulated statistics.
        order: AR order ``p``, samples.
        n_input_used: Leading input channels to read, or None for all of them.

    Returns:
        Column indices into ``moments.sx``, ``moments.gram`` and ``moments.cross``.
    """
    used = moments.n_input_channels if n_input_used is None else int(n_input_used)
    return subset_columns(order=order, n_input_channels=moments.n_input_channels, n_input_used=used)


def _ar_residual_mse(
    moments: LagMoments, weight: FloatArray, order: int, n_input_used: int | None = None
) -> float:
    """Return the training MSE of a solved AR model, from the moments alone.

    Args:
        moments: The statistics the coefficients were solved from.
        weight: Coefficients, shape ``(order * n_input_used, H * C_out)``.
        order: AR order ``p``, samples.
        n_input_used: Leading input channels the fit read, or None for all of them. Must
            match what :func:`solve_ar_coefficients` was given, or the quadratic form
            below pairs the coefficients with the wrong moment block.

    Returns:
        Mean squared residual over windows, horizon steps and target channels,
        dimensionless.
    """
    cols = _design_columns(moments, order=order, n_input_used=n_input_used)
    return _residual_mse(
        n=float(moments.n),
        sx=moments.sx[cols],
        sy=moments.sy,
        syy=moments.syy,
        gram=moments.gram[np.ix_(cols, cols)],
        cross=moments.cross[cols, :],
        weight=weight,
    )


def _residual_mse(
    *,
    n: float,
    sx: FloatArray,
    sy: FloatArray,
    syy: FloatArray,
    gram: FloatArray,
    cross: FloatArray,
    weight: FloatArray,
) -> float:
    """Return the training MSE of a solved linear model, from its moments alone.

    Second passes over a 1.9 M-window training split are the expensive part of a
    closed-form fit, so the residual comes out of the same sums the solve used.

    Args:
        n: Row count the sums were accumulated over.
        sx: ``sum feats``, shape ``(d,)``.
        sy: ``sum y``, shape ``(k,)``.
        syy: ``sum y^2`` elementwise, shape ``(k,)``.
        gram: ``sum feats feats^T``, shape ``(d, d)``.
        cross: ``sum feats y^T``, shape ``(d, k)``.
        weight: Solved coefficients, shape ``(d, k)``.

    Returns:
        Mean squared residual over rows and outputs, dimensionless.
    """
    mx = sx / n
    my = sy / n
    gram_c = gram / n - np.outer(mx, mx)
    cross_c = cross / n - np.outer(mx, my)
    syy_c = syy / n - np.square(my)
    quad = np.einsum("dj,dk,kj->j", weight, gram_c, weight)
    return float(np.mean(syy_c - 2.0 * np.einsum("dj,dj->j", weight, cross_c) + quad))


def fit_ar(
    moments: TrainingMoments,
    *,
    order: int,
    ridge: float,
    lookback: int,
    n_input_channels: int,
    n_target_channels: int,
    n_input_used: int | None = None,
) -> tuple[ARForecaster, ARFitReport]:
    """Fit an :class:`dmf.models.ar.ARForecaster` from accumulated moments.

    Deterministic: no RNG enters this path, which is why AR emits a single results row with
    ``deterministic=True`` and ``skill_std = NaN`` rather than three identical seeds.

    Args:
        moments: One pass of training moments, at ``max_order >= order``.
        order: AR order ``p``, samples.
        ridge: Tikhonov strength on the whitened correlation matrix, dimensionless.
        lookback: Input window length ``L``, samples.
        n_input_channels: ``C_in`` -- the width of the windows the fitted model will be
            handed at inference, which the moments must have been accumulated over.
        n_target_channels: ``C_out``.
        n_input_used: Leading input channels the regression may read, or None for all of
            them. The moments are sliced, not re-accumulated, so an ablation over the
            information set costs one extra solve and no extra pass over the training
            split.

    Returns:
        Tuple ``(model, report)``; the model is fitted and in eval mode.

    Raises:
        ValueError: If the moments do not match the requested geometry.
    """
    lag = moments.lag
    if lag.n_input_channels != n_input_channels or lag.n_target_channels != n_target_channels:
        raise ValueError(
            f"moments cover C_in={lag.n_input_channels}, C_out={lag.n_target_channels} but "
            f"the model wants C_in={n_input_channels}, C_out={n_target_channels}"
        )
    started = time.perf_counter()
    weight, bias, cond_r = solve_ar_coefficients(
        lag, order=order, ridge=ridge, n_input_used=n_input_used
    )
    residual = _ar_residual_mse(lag, weight, order, n_input_used)
    elapsed = time.perf_counter() - started

    model = ARForecaster(
        lookback=lookback,
        max_horizon=lag.max_horizon,
        n_input_channels=n_input_channels,
        n_target_channels=n_target_channels,
        order=order,
        ridge=ridge,
        n_input_used=n_input_used,
    )
    model.set_coefficients(weight, bias)
    model.eval()
    return model, ARFitReport(
        order=order,
        ridge=ridge,
        cond_r=cond_r,
        n_fitted_parameters=model.n_fitted_parameters,
        residual_train_mse=residual,
        fit_time_s=elapsed,
    )


def fit_damped_persistence(
    moments: TrainingMoments,
    *,
    fs_hz: float,
    lookback: int,
    n_input_channels: int,
    n_target_channels: int,
    channels: tuple[str, ...] | None = None,
) -> tuple[DampedPersistence, DecayFitReport]:
    """Fit a :class:`dmf.models.persistence.DampedPersistence` from accumulated moments.

    Args:
        moments: One pass of training moments.
        fs_hz: Sampling rate, hertz, used to report ``tau`` in seconds as well as samples.
        lookback: Input window length ``L``, samples.
        n_input_channels: ``C_in``.
        n_target_channels: ``C_out``.
        channels: Target channel names, for the report string.

    Returns:
        Tuple ``(model, report)``; the model is fitted and in eval mode.

    Raises:
        ValueError: If the moments do not match the requested channel count.
    """
    decay = moments.decay
    if decay.sxy.shape[1] != n_target_channels:
        raise ValueError(
            f"decay moments cover {decay.sxy.shape[1]} channels, model wants {n_target_channels}"
        )
    started = time.perf_counter()
    fit = solve_decay_tau(decay.sxx, decay.sxy, decay.syy, fs_hz, channels=channels)
    elapsed = time.perf_counter() - started

    model = DampedPersistence(
        lookback=lookback,
        max_horizon=decay.sxy.shape[0],
        n_input_channels=n_input_channels,
        n_target_channels=n_target_channels,
        tau_samples=fit.tau_samples,
    )
    model.eval()
    residual = float(np.sum(fit.sse) / (decay.n * decay.sxy.shape[0] * n_target_channels))
    return model, DecayFitReport(
        fit=fit,
        n_fitted_parameters=model.n_fitted_parameters,
        residual_train_mse=residual,
        fit_time_s=elapsed,
    )


def fit_dlinear_ols(
    moments: TrainingMoments,
    *,
    kernel_size: int,
    ridge: float,
    lookback: int,
    n_input_channels: int,
    n_target_channels: int,
) -> tuple[DLinearOLS, DecompFitReport]:
    """Fit a :class:`dmf.models.dlinear_ols.DLinearOLS` from accumulated moments.

    Deterministic: no RNG enters this path, so the model emits a single results row with
    ``deterministic=True`` -- the same exemption AR gets, for the same reason.

    Args:
        moments: One pass of training moments, accumulated with ``decompose_kernel`` set.
        kernel_size: Trend-extraction window, samples. Must equal the kernel the moments
            were accumulated at.
        ridge: Tikhonov strength on the whitened correlation matrix, dimensionless.
        lookback: Input window length ``L``, samples.
        n_input_channels: ``C_in`` -- the width of the windows the fitted model will be
            handed at inference.
        n_target_channels: ``C_out``.

    Returns:
        Tuple ``(model, report)``; the model is fitted and in eval mode.

    Raises:
        ValueError: If the moments carry no decomposed design, if they were accumulated at
            a different kernel, or if they do not match the requested geometry.
    """
    decomp = moments.decomp
    if decomp is None:
        raise ValueError(
            "these moments carry no decomposed design; accumulate_training_moments must be "
            "called with decompose_kernel set, or the DLinear solve has nothing to read"
        )
    if decomp.kernel_size != kernel_size:
        raise ValueError(
            f"moments were accumulated at kernel_size={decomp.kernel_size} but the model "
            f"wants {kernel_size}; the two would be different models sharing a label"
        )
    if decomp.lookback != lookback or decomp.n_target_channels != n_target_channels:
        raise ValueError(
            f"moments cover L={decomp.lookback}, C_out={decomp.n_target_channels} but the "
            f"model wants L={lookback}, C_out={n_target_channels}"
        )
    started = time.perf_counter()
    weight, bias, cond_r = solve_decomp_coefficients(decomp, ridge=ridge)
    residual = _residual_mse(
        n=float(decomp.n_rows),
        sx=decomp.sx,
        sy=decomp.sy,
        syy=decomp.syy,
        gram=decomp.gram,
        cross=decomp.cross,
        weight=weight,
    )
    elapsed = time.perf_counter() - started

    model = DLinearOLS(
        lookback=lookback,
        max_horizon=decomp.max_horizon,
        n_input_channels=n_input_channels,
        n_target_channels=n_target_channels,
        kernel_size=kernel_size,
        ridge=ridge,
    )
    model.set_coefficients(weight, bias)
    model.eval()
    return model, DecompFitReport(
        kernel_size=kernel_size,
        ridge=ridge,
        cond_r=cond_r,
        n_fitted_parameters=model.n_fitted_parameters,
        residual_train_mse=residual,
        fit_time_s=elapsed,
    )


def fit_residual_interval(
    moments: TrainingMoments,
    val: DeckMotionDataset,
    *,
    kernel_size: int,
    ridge: float,
    lookback: int,
    n_input_channels: int,
    n_target_channels: int,
    n_quantiles: int = len(quantile_fan(9)),
    batch_size: int = 4096,
    num_workers: int = 0,
    max_windows: int = RESIDUAL_QUANTILE_MAX_WINDOWS,
) -> tuple[EmpiricalResidualInterval, ResidualIntervalFitReport]:
    """Fit the unconditional interval baseline: a closed-form point model plus its residuals.

    Two solves, no epochs. The point half comes from ``moments`` -- the same single pass
    every other closed-form row is solved from, so this row adds no pass over the *training*
    split. The fan comes from one streaming pass over the **validation** split, sub-sampled
    at a fixed stride.

    **Validation, structurally.** Train residuals are the point model's own fitting error
    and are narrower than its test error by exactly the amount it overfits, so an interval
    built on them is too tight in the direction that flatters coverage; test residuals are
    the leak itself. The partition is therefore checked here rather than documented.

    Deterministic: no RNG enters, the loader is unshuffled, and the sub-sampling is a fixed
    stride over the window index -- so the row is ``deterministic=True`` and exempt from the
    three-seed rule, like every other closed-form row.

    Args:
        moments: One training-moments pass, accumulated with ``decompose_kernel`` set.
        val: The **validation** partition, built with the training split's normalisation
            statistics.
        kernel_size: Trend-extraction window of the inner DLinear, samples.
        ridge: Tikhonov strength the inner model is solved under, dimensionless.
        lookback: Input window length ``L``, samples.
        n_input_channels: Input channel count ``C_in``.
        n_target_channels: Target channel count ``C_out``.
        n_quantiles: Fan width ``Q``; the levels are :func:`dmf.models.heads.quantile_fan`
            of that width, so this row is scored at exactly the levels every other quantile
            row is.
        batch_size: Windows per batch of the validation pass. Affects speed only.
        num_workers: DataLoader worker processes.
        max_windows: Upper bound on the windows the quantiles are estimated from; the
            stride is chosen to respect it. Recorded on the report.

    Returns:
        Tuple ``(model, report)``; the model is fitted and in eval mode.

    Raises:
        ValueError: If ``val`` is not a validation partition, if its geometry disagrees
            with the requested one, or if ``max_windows`` is not positive. Anything the
            inner solve rejects propagates from :func:`fit_dlinear_ols`.
    """
    if val.partition != "val":
        raise ValueError(
            f"the empirical residual quantiles must be fitted on the validation partition, "
            f"got {val.regime}/{val.partition}. Train residuals understate the point "
            f"model's error by the amount it overfits, and test residuals are the leak "
            f"itself (CLAUDE.md non-negotiable 2)."
        )
    if max_windows < 1:
        raise ValueError(f"max_windows must be positive, got {max_windows}")
    spec = val.window_spec
    if spec.lookback != lookback or len(val.target_columns) != n_target_channels:
        raise ValueError(
            f"the validation partition holds L={spec.lookback}, C_out="
            f"{len(val.target_columns)} but the model wants L={lookback}, "
            f"C_out={n_target_channels}"
        )

    started = time.perf_counter()
    point_model, point_report = fit_dlinear_ols(
        moments,
        kernel_size=kernel_size,
        ridge=ridge,
        lookback=lookback,
        n_input_channels=n_input_channels,
        n_target_channels=n_target_channels,
    )

    n_total = len(val)
    stride = max(1, -(-n_total // max_windows))
    stats = val.norm_stats.subset(val.target_columns)
    loader = make_dataloader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=0
    )
    chunks: list[FloatArray] = []
    position = 0
    with torch.no_grad():
        for x, y, window_mean in loader:
            n = int(x.shape[0])
            # The loader is unshuffled, so `position + i` is the global window index and
            # this offset makes the selection an exact every-`stride`-th window over the
            # whole partition, not a per-batch approximation of one.
            offset = (-position) % stride
            position += n
            if offset >= n:
                continue
            x_sub = x[offset::stride]
            y_sub = y[offset::stride]
            mean_sub = window_mean[offset::stride]
            residual = normalize_target(y_sub, stats, mean_sub) - point_model(x_sub)
            chunks.append(residual.numpy())
    residuals = np.concatenate(chunks, axis=0)

    levels = np.asarray(quantile_fan(n_quantiles), dtype=np.float64)
    # (Q, H, C) -> (H, C, Q). Linear interpolation between order statistics, numpy's
    # default: at 25 000 windows the two neighbouring order statistics of the 5 percent
    # level differ far below the width being reported, so the interpolation rule is not
    # load-bearing -- unlike reading a level off a nine-member fan, which is why
    # QUANTILE_FAN_9 contains 0.05 and 0.95 exactly.
    fan = np.quantile(residuals.astype(np.float64), levels, axis=0).transpose(1, 2, 0)

    model = EmpiricalResidualInterval(
        lookback=lookback,
        max_horizon=spec.max_horizon,
        n_input_channels=n_input_channels,
        n_target_channels=n_target_channels,
        kernel_size=kernel_size,
        ridge=ridge,
        n_quantiles=n_quantiles,
    )
    model.point_model.load_state_dict(point_model.state_dict())
    model.set_residual_quantiles(fan)
    model.eval()
    return model, ResidualIntervalFitReport(
        point=point_report,
        fitted_on=f"{val.regime}/{val.partition}",
        n_windows_total=n_total,
        n_windows_used=int(residuals.shape[0]),
        window_stride=stride,
        residual_val_mse=float(np.mean(np.square(residuals.astype(np.float64)))),
        n_fitted_parameters=model.n_fitted_parameters,
        fit_time_s=time.perf_counter() - started,
    )
