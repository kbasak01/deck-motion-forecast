"""Closed-form fitting of the linear baselines from streaming sufficient statistics.

Two models here are exactly solvable, and both are solved from the *same* single pass over
the training dataloader:

- **AR(p)** -- least squares over a lag-major design block. One pass at ``P_MAX`` yields
  the exact normal equations for every ``p <= P_MAX``, because
  :func:`dmf.models.ar.lag_features` is prefix-nested in the order.
- **Damped persistence** -- one decay constant per channel, minimised over a closed-form
  ``SSE(tau)`` built from three small statistics (:func:`dmf.models.persistence.decay_sse`).

Nothing here ever materialises the design matrix. ``unseen_vessel/train`` holds 1 878 432
windows, so a stacked ``(N, L, C_in)`` float64 array would be 18 GB and the targets alone
2.2 GB. The accumulated moments are 240x240 plus 240x150 -- under a megabyte, independent
of ``N``.

**One Gram serves all 150 regressions.** The design matrix is identical for every horizon
step and every target channel, so ``H * C_out = 150`` right-hand sides share one Cholesky
factorisation.

**One Gram also serves every information set.** ``lag_features`` is prefix-nested in the
order *and* channel-strided within each lag block, so both a lower order and a leading
subset of the input channels are sub-blocks of the same accumulated moments -- see
:func:`subset_columns`. ``ar_attitude_only`` (AR(20) on roll, pitch and heave only) is
therefore solved from the ``ar20`` moments and adds **no** pass over the training split,
which matters because that pass, not the solve, is the whole cost of a closed-form fit.

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
from dmf.models.persistence import DampedPersistence, DecayFit, solve_decay_tau
from dmf.typedefs import FloatArray, IntArray

__all__ = [
    "ARFitReport",
    "DecayFitReport",
    "DecayMoments",
    "LagMoments",
    "TrainingMoments",
    "accumulate_training_moments",
    "fit_ar",
    "fit_damped_persistence",
    "solve_ar_coefficients",
    "subset_columns",
]

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
    """

    lag: LagMoments
    decay: DecayMoments
    fitted_on: str
    n_windows: int
    shuffled_targets: bool
    wall_time_s: float


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


def accumulate_training_moments(
    dataset: DeckMotionDataset,
    *,
    max_order: int,
    batch_size: int = 4096,
    num_workers: int = 4,
    shuffle_targets: bool = False,
) -> TrainingMoments:
    """Stream the AR and damped-persistence sufficient statistics in one pass.

    Args:
        dataset: The **training** partition to fit on.
        max_order: Largest AR order to be solvable from the result, samples. Every smaller
            order is solvable from the leading sub-blocks, so this is called once at 40.
        batch_size: Windows per batch. Affects speed only.
        num_workers: DataLoader worker processes.
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
            statistics were not fitted on train, or if ``max_order`` exceeds the lookback.
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
    n = float(moments.n)
    mx = moments.sx[cols] / n
    my = moments.sy / n
    gram_c = moments.gram[np.ix_(cols, cols)] / n - np.outer(mx, mx)
    cross_c = moments.cross[cols, :] / n - np.outer(mx, my)
    d = int(cols.size)

    scale = np.sqrt(np.diag(gram_c))
    dead = np.flatnonzero(scale <= 0.0)
    if dead.size:
        # Reported as columns of the *accumulated* design, not of the subset: under
        # ``n_input_used`` the two numberings differ, and an index that does not point at
        # the offending lag/channel pair sends the reader to the wrong feature.
        raise ValueError(
            f"lag features {cols[dead].tolist()} have zero variance on the training split; "
            f"the normal equations are singular"
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
    n = float(moments.n)
    mx = moments.sx[cols] / n
    my = moments.sy / n
    gram_c = moments.gram[np.ix_(cols, cols)] / n - np.outer(mx, mx)
    cross_c = moments.cross[cols, :] / n - np.outer(mx, my)
    syy_c = moments.syy / n - np.square(my)
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
