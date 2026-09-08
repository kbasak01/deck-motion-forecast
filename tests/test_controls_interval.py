"""The two interval controls: shuffle on a head, untrained on an interval.

Phase 6 carry-forward item 6 (``docs/protocol.md`` P5-D10) records that no shuffle control
had ever run on a *head* and no untrained control on an *interval*, so every published
coverage number rested on a path no negative control had touched. These tests exercise the
two functions that close that, at fixture scale.

**What a control test has to show is that the control FIRES.** A negative control that
cannot fail is decoration, so every "passes on a clean subject" test below is paired with a
subject handed the answer, and the pairing is the point. The subjects are test doubles
rather than real fitted models for the same reason
:mod:`tests.test_metrics` uses doubles: this file tests what the control does with a
forecast, not how the forecast was produced, and ``src/dmf/models/`` is read-only to this
module's owner.

Units: the fans below are dimensionless, i.e. in the normalised space every model's output
lives in; :func:`dmf.data.normalize.invert_norm` and
:meth:`dmf.models.heads.PredictiveDistribution.affine` are what put them into degrees and
metres, and the scorer does that itself.
"""

import inspect
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor

import dmf.eval.controls as controls_module
from dmf.config import DataConfig
from dmf.data.dataset import DeckMotionDataset
from dmf.data.splits import Regime, build_split
from dmf.data.windows import window_spec_from_config
from dmf.eval.controls import (
    INTERVAL_CONTROL_COLUMNS,
    INTERVAL_CONTROL_METRICS,
    INTERVAL_NULL_PICP_BAND,
    INTERVAL_SHUFFLE_TOL,
    INTERVAL_UNTRAINED_TOL,
    interval_controls_table,
    interval_shuffle_control,
    interval_untrained_control,
)
from dmf.eval.gate import GATE5_PICP_BAND
from dmf.models.base import BaseForecaster
from dmf.models.heads import QUANTILE_FAN_9, HeadKind
from dmf.models.residual_interval import EmpiricalResidualInterval

#: Horizons the fixture-scale controls report, samples. Matches the small window geometry
#: of ``tests/conftest.py`` (0.5, 1 and 2 s at 10 Hz).
SMALL_REPORTED_HORIZONS: tuple[int, ...] = (5, 10, 20)

#: Sampling rate of the corpus, hertz.
FS_HZ = 10.0

#: Fan width the project scores every head at.
N_QUANTILES = 9


def _dataset(
    corpus_root: Path, manifest: pd.DataFrame, cfg: DataConfig, regime: Regime, partition: str
) -> DeckMotionDataset:
    """Build one partition's dataset, propagating the training split's statistics."""
    split = build_split(manifest, regime)
    spec = window_spec_from_config(cfg)
    train = DeckMotionDataset(corpus_root, split, "train", cfg, spec)
    if partition == "train":
        return train
    return DeckMotionDataset(corpus_root, split, partition, cfg, spec, stats=train.norm_stats)


def _normalised_targets(dataset: DeckMotionDataset) -> Tensor:
    """Return every window's true future in normalised space, in loader order.

    Used to hand a model the answer. A negative control must be shown to fire, and the only
    way to be certain it fires is to let a subject cheat.

    Returns:
        Tensor ``(N, H, C_out)``, dimensionless.
    """
    truth = torch.stack([dataset[i][1] for i in range(len(dataset))])
    mean = torch.stack([dataset[i][2] for i in range(len(dataset))])
    scale = torch.as_tensor(
        dataset.norm_stats.subset(dataset.target_columns).scale, dtype=torch.float32
    )
    return (truth - mean) / scale


class _ConstantFan(BaseForecaster):
    """Emit the same fan on every window, centred on the window mean.

    The unconditional interval's shape without its fitting path: width is a constant per
    ``(horizon, channel, level)`` and the forecast is zero in normalised space. Stands in
    for a subject whose interval carries no conditional information at all.
    """

    SUPPORTED_HEADS: ClassVar[tuple[HeadKind, ...]] = ("quantile",)

    def __init__(self, lookback: int, max_horizon: int, n_in: int, n_out: int, fan: Tensor) -> None:
        super().__init__(lookback, max_horizon, n_in, n_out, int(fan.shape[-1]), "quantile")
        self.register_buffer("fan", fan)

    def _predict(self, x: Tensor) -> Tensor:
        fan: Tensor = self.get_buffer("fan")
        return fan.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1, -1)


class _FanOracle(BaseForecaster):
    """A head that cheats: it is handed the true future and emits a hairline fan around it.

    Consumes the answer in loader order through an internal cursor, which is exact because
    :func:`dmf.eval.prob_runner.evaluate_probabilistic_models` builds an unshuffled loader
    and visits every model once per batch. Constructed fresh per control call, so the cursor
    always starts at zero.
    """

    SUPPORTED_HEADS: ClassVar[tuple[HeadKind, ...]] = ("quantile",)

    def __init__(
        self,
        lookback: int,
        max_horizon: int,
        n_in: int,
        n_out: int,
        answer: Tensor,
        half_width: float,
    ) -> None:
        super().__init__(lookback, max_horizon, n_in, n_out, N_QUANTILES, "quantile")
        self.register_buffer("answer", answer)
        self.register_buffer(
            "offsets", torch.linspace(-half_width, half_width, N_QUANTILES, dtype=torch.float32)
        )
        self.cursor = 0

    def _predict(self, x: Tensor) -> Tensor:
        answer: Tensor = self.get_buffer("answer")
        offsets: Tensor = self.get_buffer("offsets")
        batch = int(x.shape[0])
        chunk = answer[self.cursor : self.cursor + batch]
        self.cursor += batch
        assert chunk.shape[0] == batch, "the oracle ran out of answers; loader order changed"
        return chunk.to(dtype=x.dtype, device=x.device)[..., None] + offsets.to(
            dtype=x.dtype, device=x.device
        )


def _calibrated_fan(dataset: DeckMotionDataset, scale: float = 1.0) -> Tensor:
    """Return the empirical residual fan of the window-mean forecast on this partition.

    Every subject and null in this file forecasts zero in normalised space -- the window
    mean -- so its residual *is* the normalised target, and the empirical quantiles of those
    targets are the sharpest fan an unconditional interval could carry here. Taking them
    from the scored partition itself would be a leak in production, and is deliberate in a
    test double: the control's statistic is only informative against a null that is
    calibrated, and a hand-picked constant width is not.

    **A width is not monotonically bad.** The Winkler score has an optimum: at
    ``alpha = 0.1`` a fan narrower than the residuals pays ``2/alpha`` times every miss, and
    one wider than them pays its own width. So a test that assumed "wider is worse" would be
    testing an assumption rather than the control -- ``scale`` moves the fan away from the
    optimum in either direction, which is what makes the sign of ``excess`` predictable.

    Args:
        dataset: The partition to fit on.
        scale: Multiplier on the fitted fan. 1.0 is calibrated; larger over-covers.

    Returns:
        Tensor ``(H, C_out, Q)``, ascending along the last axis, dimensionless.
    """
    residuals = _normalised_targets(dataset).numpy().astype(np.float64)
    levels = np.asarray(QUANTILE_FAN_9, dtype=np.float64)
    fan = np.quantile(residuals, levels, axis=0).transpose(1, 2, 0) * scale
    return torch.from_numpy(np.ascontiguousarray(fan)).float()


def _null_interval(dataset: DeckMotionDataset, scale: float = 1.0) -> EmpiricalResidualInterval:
    """Build a fitted unconditional residual interval sized for a dataset.

    The point half is zeroed rather than solved, so the null forecasts the window mean and
    its fan is exactly :func:`_calibrated_fan`. That makes every ``excess`` in this file a
    statement about widths and coverage rather than about a solver, which is what the
    control's statistic is.

    Args:
        dataset: The partition the null will be scored on.
        scale: Multiplier on the calibrated fan; see :func:`_calibrated_fan`.

    Returns:
        A fitted :class:`dmf.models.residual_interval.EmpiricalResidualInterval`.
    """
    spec = dataset.window_spec
    n_out = len(dataset.target_columns)
    model = EmpiricalResidualInterval(
        lookback=spec.lookback,
        max_horizon=spec.max_horizon,
        n_input_channels=len(dataset.input_columns),
        n_target_channels=n_out,
        kernel_size=25,
        n_quantiles=N_QUANTILES,
    )
    model.point_model.set_coefficients(
        np.zeros((2 * spec.lookback, spec.max_horizon)), np.zeros(spec.max_horizon)
    )
    model.set_residual_quantiles(_calibrated_fan(dataset, scale).numpy().astype(np.float64))
    model.eval()
    return model


# ---------------------------------------------------------------------------------------
# The interval shuffle control.
# ---------------------------------------------------------------------------------------


def test_interval_shuffle_control_passes_when_the_subject_is_the_null(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """A subject identical to the null removes exactly none of its loss.

    The boundary case the ``allow_equality`` rule exists for: ``excess == 0`` is a pass, and
    it must be exactly zero rather than nearly zero, because the two models are scored in
    one pass over identical windows.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    spec = dataset.window_spec
    n_out = len(dataset.target_columns)
    result = interval_shuffle_control(
        dataset,
        shuffled_model=_ConstantFan(
            spec.lookback,
            spec.max_horizon,
            len(dataset.input_columns),
            n_out,
            _calibrated_fan(dataset),
        ),
        null_model=_null_interval(dataset),
        regime="id",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
    )
    assert result.passed
    assert result.worst_excess == pytest.approx(0.0, abs=1e-12)
    assert set(result.rows["control"]) == {"interval_shuffle"}
    assert set(result.rows["metric"]) == set(INTERVAL_CONTROL_METRICS)
    # The shuffle half of the pair IS enforced, which is what makes `enforced` a
    # distinction rather than a constant column.
    assert result.enforced
    assert bool(result.rows["enforced"].all())


def test_interval_shuffle_control_passes_when_the_shuffled_interval_is_wider(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The expected outcome on a clean pipeline: shuffling widens the interval.

    A least-squares fit on shuffled targets degenerates to the window mean, so its residual
    fan is the fan of window-mean residuals -- wider than an honestly fitted one. Wider is
    worse on every scoring rule here, so the excess is negative.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    spec = dataset.window_spec
    n_out = len(dataset.target_columns)
    result = interval_shuffle_control(
        dataset,
        shuffled_model=_ConstantFan(
            spec.lookback,
            spec.max_horizon,
            len(dataset.input_columns),
            n_out,
            _calibrated_fan(dataset, scale=4.0),
        ),
        null_model=_null_interval(dataset),
        regime="id",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
    )
    assert result.passed
    assert result.worst_excess < 0.0
    assert (result.rows["width_subject"] > result.rows["width_null"]).all()


def test_interval_shuffle_control_catches_a_head_that_still_knows_the_future(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The converse: a control that cannot fail is not a control."""
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    spec = dataset.window_spec
    n_out = len(dataset.target_columns)
    with pytest.raises(AssertionError, match="interval shuffle control failed"):
        interval_shuffle_control(
            dataset,
            shuffled_model=_FanOracle(
                spec.lookback,
                spec.max_horizon,
                len(dataset.input_columns),
                n_out,
                _normalised_targets(dataset),
                half_width=1e-3,
            ),
            null_model=_null_interval(dataset),
            regime="id",
            horizons=SMALL_REPORTED_HORIZONS,
            fs_hz=FS_HZ,
            batch_size=len(dataset),
        )


def test_interval_shuffle_control_reports_the_cheat_when_it_is_not_strict(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """``strict=False`` still records the failure rather than swallowing it."""
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    spec = dataset.window_spec
    n_out = len(dataset.target_columns)
    result = interval_shuffle_control(
        dataset,
        shuffled_model=_FanOracle(
            spec.lookback,
            spec.max_horizon,
            len(dataset.input_columns),
            n_out,
            _normalised_targets(dataset),
            half_width=1e-3,
        ),
        null_model=_null_interval(dataset),
        regime="id",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        strict=False,
        batch_size=len(dataset),
    )
    assert not result.passed
    assert result.worst_excess > INTERVAL_SHUFFLE_TOL
    assert not result.enforced
    assert not result.rows["passed"].all()


def test_interval_shuffle_control_rejects_an_unfitted_null(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """An unfitted null emits a zero-width fan, which scores as perfect sharpness.

    Every subject would then appear to remove ~100 percent of its loss, i.e. the control
    would report leakage on a clean pipeline. It is refused at the door instead.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    spec = dataset.window_spec
    n_out = len(dataset.target_columns)
    unfitted = EmpiricalResidualInterval(
        lookback=spec.lookback,
        max_horizon=spec.max_horizon,
        n_input_channels=len(dataset.input_columns),
        n_target_channels=n_out,
        kernel_size=25,
        n_quantiles=N_QUANTILES,
    )
    with pytest.raises(ValueError, match="no residual quantiles"):
        interval_shuffle_control(
            dataset,
            shuffled_model=_ConstantFan(
                spec.lookback,
                spec.max_horizon,
                len(dataset.input_columns),
                n_out,
                _calibrated_fan(dataset),
            ),
            null_model=unfitted,
            regime="id",
            horizons=SMALL_REPORTED_HORIZONS,
            fs_hz=FS_HZ,
            batch_size=256,
        )


# ---------------------------------------------------------------------------------------
# The interval untrained control -- reported, not enforced (docs/protocol.md P3-D9).
# ---------------------------------------------------------------------------------------


def test_interval_untrained_control_passes_when_the_head_is_uninformative(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """A hairline fan around the window mean is priced at 20x its miss distance.

    ``alpha = 0.1`` makes the Winkler penalty ``2/alpha`` times the distance outside the
    interval, so a random-init head that emits almost no width loses badly on a channel that
    carries amplitude. This is the direction the control is meant to see.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    spec = dataset.window_spec
    n_out = len(dataset.target_columns)
    result = interval_untrained_control(
        dataset,
        untrained_model=_ConstantFan(
            spec.lookback,
            spec.max_horizon,
            len(dataset.input_columns),
            n_out,
            _calibrated_fan(dataset, scale=1e-3),
        ),
        null_model=_null_interval(dataset),
        regime="id",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
    )
    assert result.passed
    assert result.worst_excess < INTERVAL_UNTRAINED_TOL
    assert not result.enforced
    # And the rows say so too, not only the object. `interval_controls.csv` is what a
    # reader downstream has; a commitment that lives only on the result object is a
    # commitment the artifact does not carry.
    assert not bool(result.rows["enforced"].any())
    assert bool(result.rows["asserted"].all())


def test_interval_untrained_control_warns_instead_of_raising(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """P3-D9's treatment, applied at the definition site.

    The criterion is kept, the tolerance is kept at zero, the failure is recorded in the
    rows -- and it is warned about rather than raised, because the failing direction is a
    known property of the statistic rather than evidence of leakage. A failure nobody sees
    during a multi-hour sweep is indistinguishable from a control that was never run, so the
    warning is not optional.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    spec = dataset.window_spec
    n_out = len(dataset.target_columns)
    with pytest.warns(RuntimeWarning, match="untrained interval control"):
        result = interval_untrained_control(
            dataset,
            untrained_model=_FanOracle(
                spec.lookback,
                spec.max_horizon,
                len(dataset.input_columns),
                n_out,
                _normalised_targets(dataset),
                half_width=1e-3,
            ),
            null_model=_null_interval(dataset),
            regime="id",
            horizons=SMALL_REPORTED_HORIZONS,
            fs_hz=FS_HZ,
            batch_size=len(dataset),
        )
    assert not result.passed
    assert result.worst_excess > 0.0


def test_interval_untrained_control_raises_when_the_caller_asks_for_strict(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The non-default path still exists, so the criterion is kept rather than removed."""
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    spec = dataset.window_spec
    n_out = len(dataset.target_columns)
    with pytest.raises(AssertionError, match="untrained interval control"):
        interval_untrained_control(
            dataset,
            untrained_model=_FanOracle(
                spec.lookback,
                spec.max_horizon,
                len(dataset.input_columns),
                n_out,
                _normalised_targets(dataset),
                half_width=1e-3,
            ),
            null_model=_null_interval(dataset),
            regime="id",
            horizons=SMALL_REPORTED_HORIZONS,
            fs_hz=FS_HZ,
            strict=True,
            batch_size=len(dataset),
        )


# ---------------------------------------------------------------------------------------
# Schema, and the two properties a reader of `interval_controls.csv` needs.
# ---------------------------------------------------------------------------------------


def test_every_row_carries_its_coverage_and_its_width(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """Coverage without sharpness is meaningless, so no row may state one without the other.

    A maximally wide interval has perfect coverage; the protocol therefore requires PICP and
    width to be read together, and the schema is what makes that structural rather than a
    convention the renderer might drop.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    spec = dataset.window_spec
    n_out = len(dataset.target_columns)
    result = interval_shuffle_control(
        dataset,
        shuffled_model=_ConstantFan(
            spec.lookback,
            spec.max_horizon,
            len(dataset.input_columns),
            n_out,
            _calibrated_fan(dataset, scale=4.0),
        ),
        null_model=_null_interval(dataset),
        regime="id",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
    )
    assert tuple(result.rows.columns) == INTERVAL_CONTROL_COLUMNS
    for column in ("picp_subject", "picp_null", "width_subject", "width_null"):
        assert result.rows[column].notna().all()
    assert (result.rows["alpha"] == 0.1).all()
    table = interval_controls_table([result])
    assert tuple(table.columns) == INTERVAL_CONTROL_COLUMNS
    assert len(table) == len(result.rows)


def test_pinball_and_crps_excess_are_the_same_measurement(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """Two reported columns, one measurement, and the reader is entitled to know.

    :func:`dmf.eval.prob_runner.evaluate_probabilistic_models` accumulates CRPS as the
    finite-fan quadrature of the pinball loss at the same levels for every head kind, so
    ``crps = 2 * mean_q(pinball)`` identically and the two ratios are equal. Both are
    reported because the protocol requires pinball, CRPS, PICP and width together; the
    identity is pinned here so that nobody counts them as two pieces of evidence.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    spec = dataset.window_spec
    n_out = len(dataset.target_columns)
    result = interval_shuffle_control(
        dataset,
        shuffled_model=_ConstantFan(
            spec.lookback,
            spec.max_horizon,
            len(dataset.input_columns),
            n_out,
            _calibrated_fan(dataset, scale=4.0),
        ),
        null_model=_null_interval(dataset),
        regime="id",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
    )
    keys = ["dof", "horizon_samples"]
    crps = result.rows[result.rows["metric"] == "crps"].set_index(keys)["excess"]
    pinball = result.rows[result.rows["metric"] == "pinball"].set_index(keys)["excess"]
    assert np.allclose(crps.to_numpy(), pinball.reindex(crps.index).to_numpy(), atol=1e-12)
    winkler = result.rows[result.rows["metric"] == "winkler"].set_index(keys)["excess"]
    assert not np.allclose(winkler.to_numpy(), crps.to_numpy(), atol=1e-6)


# ---------------------------------------------------------------------------------------
# The floored-cell question: reported, and deliberately NOT narrowed.
#
# `docs/protocol.md` P6-D11 narrowed the POINT shuffle control to unfloored channels because
# its MSE ratio is dominated there by a train/test amplitude mismatch. P6-D12 refused to
# extend that argument to `untrained_control` on inference alone. The interval statistic is
# a ratio of Winkler or CRPS scores, and on a floored cell it degenerates to a ratio of two
# constant fan widths -- both fitted where the channel is excited -- so the mechanism has no
# route into it. The set is therefore derived and REPORTED, and every cell is asserted on.
# ---------------------------------------------------------------------------------------


def test_floored_cells_are_reported_and_still_asserted_on(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """``unseen_heading`` holds out beam seas, where pitch sits on its P1-D2 floor.

    The point shuffle control stops asserting there. This one does not: ``asserted`` is True
    on every row and ``on_residual_floor`` carries the derivation instead, so a future
    investigation starts from the artifact rather than from a rerun.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "unseen_heading", "test")
    spec = dataset.window_spec
    n_out = len(dataset.target_columns)
    result = interval_shuffle_control(
        dataset,
        shuffled_model=_ConstantFan(
            spec.lookback,
            spec.max_horizon,
            len(dataset.input_columns),
            n_out,
            _calibrated_fan(dataset, scale=4.0),
        ),
        null_model=_null_interval(dataset),
        regime="unseen_heading",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
    )
    assert result.floored_dofs == ("pitch", "pitch_rate")
    assert result.rows["asserted"].all()
    floored_rows = result.rows[result.rows["on_residual_floor"]]
    assert set(floored_rows["dof"]) == {"pitch", "pitch_rate"}
    assert floored_rows["asserted"].all()


def test_the_id_regime_floors_nothing(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The derivation follows the partition's headings, not a hard-coded channel pair."""
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    spec = dataset.window_spec
    n_out = len(dataset.target_columns)
    result = interval_shuffle_control(
        dataset,
        shuffled_model=_ConstantFan(
            spec.lookback,
            spec.max_horizon,
            len(dataset.input_columns),
            n_out,
            _calibrated_fan(dataset, scale=4.0),
        ),
        null_model=_null_interval(dataset),
        regime="id",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
    )
    assert result.floored_dofs == ()
    assert not result.rows["on_residual_floor"].any()


# ---------------------------------------------------------------------------------------
# The null-calibration narrowing (`docs/protocol.md` P6-D21).
#
# The interval shuffle control does not assert on a cell whose NULL is itself outside Gate
# 5's registered PICP band. A control whose reference is miscalibrated measures the
# reference: where the null's fan is mis-sized for the partition it meets, a WIDER subject
# covers its misses and scores better without knowing anything.
#
# What these tests have to show is that the narrowing keys off the null's calibration and
# nothing else -- not the subject's excess, not the regime, not a channel list -- and that
# what it excludes still ships with its real numbers. The first two are a pair: the same
# leaking subject must raise against a calibrated null and must NOT raise against a
# miscalibrated one, which is also the plainest statement of what the narrowing costs.
# ---------------------------------------------------------------------------------------

#: Multiplier that makes the fixture null badly under-cover: the fan is a fifth of the
#: residual quantiles, so a nominal 90 percent interval covers roughly a quarter of the
#: windows. Chosen to sit far outside :data:`dmf.eval.controls.INTERVAL_NULL_PICP_BAND`
#: rather than just outside it; the tests assert the resulting coverage rather than assuming
#: it, so a change in the fixture corpus surfaces here instead of silently weakening them.
MISCALIBRATED_NULL_SCALE = 0.2


def _leaking_subject(dataset: DeckMotionDataset) -> _FanOracle:
    """Return a subject handed the true future, i.e. an unambiguous leak.

    Its excess is close to 1.0 on every cell -- two orders of magnitude above
    :data:`dmf.eval.controls.INTERVAL_SHUFFLE_TOL` -- so nothing about whether the control
    fires depends on where the tolerance sits.
    """
    spec = dataset.window_spec
    return _FanOracle(
        spec.lookback,
        spec.max_horizon,
        len(dataset.input_columns),
        len(dataset.target_columns),
        _normalised_targets(dataset),
        half_width=1e-3,
    )


def test_a_leaking_subject_still_raises_when_the_null_is_calibrated(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The narrowing must not have made the control unable to fail.

    Same subject as the test below; the only thing that differs is the null's calibration.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    null = _null_interval(dataset)
    with pytest.raises(AssertionError, match="interval shuffle control failed"):
        interval_shuffle_control(
            dataset,
            shuffled_model=_leaking_subject(dataset),
            null_model=null,
            regime="id",
            horizons=SMALL_REPORTED_HORIZONS,
            fs_hz=FS_HZ,
            batch_size=256,
        )


def test_the_same_leaking_subject_is_not_asserted_on_when_the_null_is_miscalibrated(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """And this is exactly what the narrowing costs, stated as an assertion.

    The subject is identical to the one above and still knows the future; only the null
    changed. The control now declines to judge, because against a null covering ~0.25 at a
    nominal 0.90 the statistic prices the reference. **A real leak confined to such cells
    would not be caught** -- the excess is still written, still enormous, and still visible
    in the CSV, which is the only reason this narrowing is tolerable.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    null = _null_interval(dataset, scale=MISCALIBRATED_NULL_SCALE)
    low, high = INTERVAL_NULL_PICP_BAND
    with pytest.warns(RuntimeWarning, match="asserted on NO cell"):
        result = interval_shuffle_control(
            dataset,
            shuffled_model=_leaking_subject(dataset),
            null_model=null,
            regime="id",
            horizons=SMALL_REPORTED_HORIZONS,
            fs_hz=FS_HZ,
            batch_size=256,
        )
    # The premise of the test, measured rather than assumed.
    assert ((result.rows["picp_null"] < low) | (result.rows["picp_null"] > high)).all()
    assert not result.rows["asserted"].any()
    # Nothing was judged, and the object says so rather than reporting a pass.
    assert np.isnan(result.worst_excess)
    assert result.exclusion_reason
    assert result.worst_excess_reported_only is not None
    assert result.worst_excess_reported_only > INTERVAL_SHUFFLE_TOL
    assert (result.reported_only_rows["excess"] > INTERVAL_SHUFFLE_TOL).any()


def test_the_narrowing_keys_off_the_null_and_not_the_subject(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """Two subjects, one null: the asserted mask is identical.

    If the exclusion were computed from the subject's own excess -- the failure mode that
    would make this a threshold tuned to its result -- a leaking subject and a benign one
    would be asserted on different cells.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    spec = dataset.window_spec
    benign = interval_shuffle_control(
        dataset,
        shuffled_model=_ConstantFan(
            spec.lookback,
            spec.max_horizon,
            len(dataset.input_columns),
            len(dataset.target_columns),
            _calibrated_fan(dataset, scale=4.0),
        ),
        null_model=_null_interval(dataset, scale=MISCALIBRATED_NULL_SCALE),
        regime="id",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        # Not enforced here only so the pair can be compared in one test; `strict` does not
        # enter the narrowing.
        strict=False,
        batch_size=256,
    )
    leaking = interval_shuffle_control(
        dataset,
        shuffled_model=_leaking_subject(dataset),
        null_model=_null_interval(dataset, scale=MISCALIBRATED_NULL_SCALE),
        regime="id",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        strict=False,
        batch_size=256,
    )
    keys = ["dof", "horizon_samples", "metric"]
    left = benign.rows.set_index(keys)["asserted"]
    right = leaking.rows.set_index(keys)["asserted"]
    assert (left == right.reindex(left.index)).all()
    # ... while the excesses they were computed beside are nothing like each other.
    assert leaking.rows["excess"].max() > benign.rows["excess"].max() + 0.5


def test_the_narrowing_cannot_be_reached_through_a_regime_name(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """``regime`` is a label written onto the rows, never a predicate.

    The 12 production cells this narrowing excludes are all ``unseen_seastate``, and the
    cheap way to reproduce that number would have been to name the regime. Scoring the same
    models under two regime labels shows the label does not enter the decision.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    spec = dataset.window_spec
    masks = []
    for label in ("id", "unseen_seastate"):
        result = interval_shuffle_control(
            dataset,
            shuffled_model=_ConstantFan(
                spec.lookback,
                spec.max_horizon,
                len(dataset.input_columns),
                len(dataset.target_columns),
                _calibrated_fan(dataset, scale=4.0),
            ),
            null_model=_null_interval(dataset),
            regime=label,
            horizons=SMALL_REPORTED_HORIZONS,
            fs_hz=FS_HZ,
            batch_size=256,
        )
        assert set(result.rows["regime"]) == {label}
        masks.append(result.rows["asserted"].to_numpy())
    assert np.array_equal(masks[0], masks[1])


def test_the_narrowing_cannot_be_reached_through_a_channel_list_or_a_caller_override() -> None:
    """No channel is named, and no caller may declare a cell unasserted.

    P6-D12 refused a keyword on the point control that would let a caller exclude arbitrary
    channels, because that is a hole in the guard rather than a narrowing of it. The same
    holds here: the only input to the decision is the null's measured coverage, which the
    control computes itself.
    """
    signature = inspect.signature(interval_shuffle_control)
    forbidden = ("band", "picp", "exclude", "not_asserted", "skip", "dofs")
    offenders = [
        name for name in signature.parameters if any(token in name.lower() for token in forbidden)
    ]
    assert offenders == []
    source = inspect.getsource(controls_module._interval_control_rows)
    # The predicate reads the null's coverage column and nothing else.
    assert 'null_scores["picp"]' in source
    for token in ("unseen_seastate", "pitch_rate", "unseen_heading"):
        assert token not in source


def test_excluded_rows_ship_with_their_real_numbers(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """P6-D12's rule, applied to this statistic: nothing is hidden, only unasserted.

    ``passed`` is still computed for an excluded row, so a cell over tolerance that the
    control declined to judge appears as ``asserted=False, passed=False`` in the CSV rather
    than as an absence.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    with pytest.warns(RuntimeWarning, match="asserted on NO cell"):
        result = interval_shuffle_control(
            dataset,
            shuffled_model=_leaking_subject(dataset),
            null_model=_null_interval(dataset, scale=MISCALIBRATED_NULL_SCALE),
            regime="id",
            horizons=SMALL_REPORTED_HORIZONS,
            fs_hz=FS_HZ,
            strict=True,
            batch_size=256,
        )
    table = interval_controls_table([result])
    excluded = table[~table["asserted"]]
    assert len(excluded) == len(table)
    assert not excluded["passed"].any()
    for column in ("excess", "loss_subject", "loss_null", "picp_null", "width_null"):
        assert excluded[column].notna().all()
    # The rows are still enforced-by-this-control rows: `asserted` and `enforced` answer
    # different questions and are not collapsed (P6-D15).
    assert excluded["enforced"].all()
    assert result.excluded_cells
    assert len(result.excluded_cells) == len(table) // len(INTERVAL_CONTROL_METRICS)


def test_the_untrained_interval_control_is_not_narrowed(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """P6-D12's refusal, restated: an argument is not extended to a second statistic for free.

    The untrained control is reported rather than enforced, so a miscalibrated null makes
    its rows harder to read rather than making a run harder to stop. Every cell stays
    asserted, and the failing rows ship (P3-D9).
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    spec = dataset.window_spec
    result = interval_untrained_control(
        dataset,
        untrained_model=_ConstantFan(
            spec.lookback,
            spec.max_horizon,
            len(dataset.input_columns),
            len(dataset.target_columns),
            _calibrated_fan(dataset, scale=1e-3),
        ),
        null_model=_null_interval(dataset, scale=MISCALIBRATED_NULL_SCALE),
        regime="id",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
    )
    assert result.rows["asserted"].all()
    assert result.excluded_cells == ()
    assert result.exclusion_reason == ""


# ---------------------------------------------------------------------------------------
# The calibrated tolerance (P6-D21), and the run it was calibrated from.
# ---------------------------------------------------------------------------------------

#: The interval shuffle control's worst **in-distribution** excess on the production corpus,
#: from ``results/e04/interval_shuffle_calibration.csv``: `pitch_rate` at 150 samples on
#: `id`, Winkler. The number :data:`dmf.eval.controls.INTERVAL_SHUFFLE_TOL` was derived from.
MEASURED_ID_WORST_EXCESS = 0.024586

#: Margin P3-D8 used when it calibrated the point control's tolerance (0.53 percent measured,
#: 2 percent chosen). P6-D21 follows the same convention, so the interval tolerance must not
#: sit closer to its measured worst case than the point one did.
P3D8_MARGIN = 3.8

#: The calibration run, all four regimes, both interval controls.
CALIBRATION_CSV = (
    Path(__file__).resolve().parents[1] / "results/e04/interval_shuffle_calibration.csv"
)


def test_the_band_the_narrowing_uses_is_gate_5s_own_band() -> None:
    """Not a new threshold: the same object, so the two cannot drift apart.

    P6-D21 rests on the exclusion being Gate 5's pre-registered [0.85, 0.95] (P5-D2) applied
    to the null rather than to a head. A copied literal would let that claim quietly stop
    being true.
    """
    assert INTERVAL_NULL_PICP_BAND is GATE5_PICP_BAND
    assert INTERVAL_NULL_PICP_BAND == (0.85, 0.95)


def test_the_shuffle_tolerance_is_the_calibrated_value_at_its_stated_margin() -> None:
    """The tolerance is 0.10, and 0.10 is what P6-D21's derivation gives.

    Pinned so that the docstring's arithmetic is checked rather than trusted: 0.10 against a
    measured 2.46 percent is a 4.1x margin, which is at least the 3.8x P3-D8 used for the
    point control. This is a post-hoc value whose legitimacy rests on P6-D16 assumption 1
    having registered this control's first production run as a calibration.
    """
    assert INTERVAL_SHUFFLE_TOL == 0.10
    assert INTERVAL_SHUFFLE_TOL >= P3D8_MARGIN * MEASURED_ID_WORST_EXCESS
    # And it is not so wide that the control stops meaning anything: the honest excesses of
    # the same run are two orders of magnitude below it.
    assert INTERVAL_SHUFFLE_TOL < 0.2


@pytest.mark.skipif(
    not CALIBRATION_CSV.exists(), reason=f"{CALIBRATION_CSV} is the P6-D21 calibration run"
)
def test_the_three_positive_id_cells_of_the_calibration_stay_visible() -> None:
    """The three in-distribution cells with a positive excess must not be hidden by either change.

    P6-D21 reports them beside the table rather than behind the wider tolerance. They are
    `pitch_rate` at 150 samples on `id`, one per scoring rule, and the widened tolerance is
    the only thing that lets `id` pass: the narrowing must NOT be what excuses them, because
    the null is well calibrated there (PICP 0.904).
    """
    frame = pd.read_csv(CALIBRATION_CSV)
    shuffle = frame[frame["control"] == "interval_shuffle"]
    in_distribution = shuffle[shuffle["regime"] == "id"]
    positive = in_distribution[in_distribution["excess"] > 0.0]
    assert len(positive) == 3
    assert set(positive["dof"]) == {"pitch_rate"}
    assert set(positive["horizon_samples"]) == {150}
    assert set(positive["metric"]) == set(INTERVAL_CONTROL_METRICS)
    low, high = INTERVAL_NULL_PICP_BAND
    # Still asserted on under the narrowing: the null is calibrated in these cells.
    assert ((positive["picp_null"] >= low) & (positive["picp_null"] <= high)).all()
    worst = float(positive["excess"].max())
    assert worst == pytest.approx(MEASURED_ID_WORST_EXCESS, abs=5e-6)
    # It exceeds the tolerance the control was inherited with, and passes the calibrated one.
    assert worst > 0.02
    assert worst <= INTERVAL_SHUFFLE_TOL
    # Every other in-distribution cell is negative, which is what makes 2.46 percent the
    # worst case the tolerance was derived from rather than one point of many.
    assert len(in_distribution) == 108
    assert (in_distribution["excess"] <= worst).all()


@pytest.mark.skipif(
    not CALIBRATION_CSV.exists(), reason=f"{CALIBRATION_CSV} is the P6-D21 calibration run"
)
def test_the_narrowing_would_exclude_the_production_cells_it_claims_to() -> None:
    """Applied to the committed calibration rows, the predicate reproduces P6-D21's counts.

    13 shuffle cells exceed 0.02: one on `id` and twelve on `unseen_seastate`. The band
    excludes the twelve **on the null's coverage alone**, leaves the `id` one asserted, and
    leaves every `id` cell asserted -- 108 of 108.
    """
    frame = pd.read_csv(CALIBRATION_CSV)
    shuffle = frame[frame["control"] == "interval_shuffle"]
    low, high = INTERVAL_NULL_PICP_BAND
    asserted = (shuffle["picp_null"] >= low) & (shuffle["picp_null"] <= high)
    over_old_tol = shuffle["excess"] > 0.02
    assert int(over_old_tol.sum()) == 13
    assert dict(shuffle.loc[over_old_tol, "regime"].value_counts()) == {
        "unseen_seastate": 12,
        "id": 1,
    }
    assert int((over_old_tol & asserted).sum()) == 1
    assert bool(asserted[shuffle["regime"] == "id"].all())
    # And the regime whose null is miscalibrated everywhere ends up with nothing asserted,
    # which is the cost the control warns about at run time rather than a clean pass.
    assert not bool(asserted[shuffle["regime"] == "unseen_seastate"].any())


def test_the_control_refuses_an_alpha_the_band_was_not_registered_at(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The band is Gate 5's PICP@90 band, so it may only judge a nominal 90 percent interval.

    Read against a nominal 80 percent fan, `[0.85, 0.95]` would call a perfectly calibrated
    null miscalibrated and leave the control asserting on nothing -- a narrowing to zero
    that nothing in the output would explain. It refuses instead.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    spec = dataset.window_spec
    with pytest.raises(ValueError, match="Gate 5's PICP@90"):
        interval_shuffle_control(
            dataset,
            shuffled_model=_ConstantFan(
                spec.lookback,
                spec.max_horizon,
                len(dataset.input_columns),
                len(dataset.target_columns),
                _calibrated_fan(dataset, scale=4.0),
            ),
            null_model=_null_interval(dataset),
            regime="id",
            horizons=SMALL_REPORTED_HORIZONS,
            fs_hz=FS_HZ,
            alpha=0.2,
            batch_size=256,
        )
