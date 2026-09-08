"""The producer for the two control artifacts that had none.

``docs/protocol.md`` P6-D15 records both gaps, and they are opposite defects of the same kind.

- :func:`dmf.eval.controls.interval_shuffle_control` and
  :func:`dmf.eval.controls.interval_untrained_control` are implemented and tested, and
  **nothing called them**. Phase 6 carry-forward item 6 was therefore closed in code and open
  in artifacts: every Phase 5 coverage claim rested on an interval path with no interval
  control, and no ``interval_controls.csv`` existed for the renderer to read.
- :func:`dmf.eval.controls.persistence_pipeline_sanity` is *enforced* -- it raises inside the
  driver -- and **unreported**: it writes no row anywhere. The mirror image of the untrained
  control, which is reported and unenforced.

This module runs all three from committed artifacts and writes the two files. It is a
sibling of :mod:`dmf.eval.scoring` rather than part of it: scoring answers "how good is each
model", this answers "is the pipeline that produced those numbers sound", and mixing them
would let a scoring pass that skipped its controls look identical to one that ran them.

**Nothing here is trained.** The null and the shuffled subject are closed-form solves --
:func:`dmf.train.closed_form.fit_residual_interval`, from a moments pass over the training
split -- and the untrained subject is a freshly constructed head that is *supposed* to be
unfitted. No checkpoint is read and none is written.

**What ``enforced`` means on each row, and why it differs.**

============================  ==========  ==========  ==========================================
control                       asserted    enforced    why
============================  ==========  ==========  ==========================================
``interval_shuffle``          True        **True**    information surviving a target shuffle on
                                                      the interval path can only be leakage, so
                                                      it stops the run
``interval_untrained``        True        False       P3-D9: the literal criterion is known to
                                                      be wrong for this signal, and its failing
                                                      direction is analysed at the definition
                                                      site. Reported, never tuned away
``pipeline_sanity``           True        **True**    Gate 2 criterion 5; it raises, so a
                                                      failing row cannot reach the CSV at all
============================  ==========  ==========  ==========================================

The ``strict`` keyword at each call site below is the *only* source of those values --
:data:`dmf.eval.assemble.ENFORCED_BY_CONTROL` claims to describe them, and
``tests/test_assemble.py`` parses this module's source and asserts the two agree, so the
claim is checked against the driver rather than trusted to stay in step with it.

**One decision worth naming, because it is not the docstring's literal description.**
:func:`dmf.eval.controls.interval_shuffle_control` describes its subject as an interval whose
point half is fitted on shuffled targets *and* whose fan is taken against the same shuffled
targets. Here the subject's point half is solved from shuffled moments and its fan is fitted
by the ordinary :func:`dmf.train.closed_form.fit_residual_interval` pass, i.e. against the
**honest** validation targets, for two reasons. First, it reuses the tested fitter rather
than reimplementing the residual-quantile pass, which is where an unaudited difference
between the subject and the null could hide. Second, it is the conservative direction: an
honestly calibrated fan is the *best* fan the shuffled point model could carry, so it makes
the control harder to pass, not easier. A control given to its subject in the generous
direction is the one worth running.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from dmf.config import ExperimentConfig, ModelConfig
from dmf.data.dataset import DeckMotionDataset
from dmf.data.splits import REGIMES, Regime, build_split, load_manifest
from dmf.data.windows import window_spec_from_config
from dmf.eval.assemble import NOT_AN_ARM
from dmf.eval.controls import (
    IntervalControlResult,
    PipelineSanityResult,
    interval_controls_table,
    interval_shuffle_control,
    interval_untrained_control,
    persistence_pipeline_sanity,
    pipeline_sanity_table,
)
from dmf.eval.report import write_table
from dmf.models.base import BaseForecaster
from dmf.models.residual_interval import EmpiricalResidualInterval
from dmf.train.closed_form import accumulate_training_moments, fit_residual_interval
from dmf.train.experiment import GATE5_ALPHA, PERSISTENCE_LABEL, _decompose_kernel
from dmf.train.loop import set_seed
from dmf.train.registry import MODEL_REGISTRY, build_model

__all__ = [
    "CONTROL_ARTIFACTS",
    "PIPELINE_SANITY_RTOL",
    "ControlArtifacts",
    "run_controls",
    "run_regime_controls",
]

#: Table name -> file name, for everything this module writes. The third member of the
#: family with :data:`dmf.eval.scoring.SCORING_ARTIFACTS` and
#: :data:`dmf.eval.assemble.ASSEMBLED_ARTIFACTS`; no two of the three share a file name.
CONTROL_ARTIFACTS: dict[str, str] = {
    "interval_controls": "interval_controls.csv",
    "pipeline_sanity": "pipeline_sanity.csv",
}

#: Relative tolerance the pipeline-sanity control is run at. The default of
#: :func:`dmf.eval.controls.persistence_pipeline_sanity`, restated here because it is now
#: written into a committed column and a tolerance a reader cannot see is a tolerance nobody
#: can check. Measured disagreement on the production ``id/test`` partition is 5.006e-08 --
#: the float32 storage floor, two orders of magnitude below this.
PIPELINE_SANITY_RTOL: float = 1e-6


@dataclass
class ControlArtifacts:
    """The control tables one run produced, and every control it could not run.

    Attributes:
        interval_controls: Rows on :data:`dmf.eval.controls.INTERVAL_CONTROL_COLUMNS`, with
            ``experiment`` and ``arm`` prepended so the renderer can group them exactly as it
            groups the point controls.
        pipeline_sanity: Rows on :data:`dmf.eval.controls.PIPELINE_SANITY_COLUMNS`.
        skipped: ``"<regime>/<control>"`` -> why it did not run. A control that could not run
            is recorded; it never silently becomes an absent row that reads as a pass.
        paths: Written files, keyed as :data:`CONTROL_ARTIFACTS`.
    """

    interval_controls: pd.DataFrame
    pipeline_sanity: pd.DataFrame
    skipped: dict[str, str] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)


def _as_regime(name: str) -> Regime:
    """Narrow a regime name to the :data:`dmf.data.splits.Regime` literal.

    Args:
        name: Regime name from a config.

    Returns:
        The same name, typed.

    Raises:
        ValueError: If it is not one of the four regimes.
    """
    if name not in REGIMES:
        raise ValueError(f"unknown regime {name!r}, expected one of {list(REGIMES)}")
    return name


def _interval_heads(cfg: ExperimentConfig) -> tuple[ModelConfig, ...]:
    """Return every configured model that emits an interval, in configuration order.

    Args:
        cfg: The experiment config.

    Returns:
        The non-point rows. Empty means the experiment publishes no interval, and an interval
        control on a run with no interval claim audits nothing -- so both controls are
        skipped there rather than run against a null nobody's rows sit above.
    """
    return tuple(model for model in cfg.models if model.head != "point")


def _untrained_head_config(cfg: ExperimentConfig) -> ModelConfig | None:
    """Return the head the untrained control is run on, or None.

    The first **SGD** head in configuration order -- which is table order, so the subject is
    the head a reader meets first rather than an arbitrary one. Restricted to SGD because
    "randomly initialised" is only meaningful for a model that *has* a random initialisation:
    an unfitted :class:`dmf.models.residual_interval.EmpiricalResidualInterval` is not a
    random-init model, it is an empty one, and it raises rather than emitting the zero-width
    fan that would score as perfect sharpness.

    Args:
        cfg: The experiment config.

    Returns:
        The model config, or None if the experiment configures no SGD head.
    """
    return next(
        (model for model in _interval_heads(cfg) if MODEL_REGISTRY[model.name].FIT_KIND == "sgd"),
        None,
    )


def _residual_interval_params(cfg: ExperimentConfig) -> tuple[int, float, int] | None:
    """Return ``(kernel_size, ridge, n_quantiles)`` for the unconditional null, or None.

    Taken from the config's own ``residual_interval`` row when it has one, and otherwise from
    its ``dlinear_ols`` row -- which is the same thing, because the null's point half *is*
    ``dlinear_ols`` and ``configs/model/residual_interval.yaml`` requires the two to carry
    identical ``kernel_size`` and ``ridge`` (P5-D5: an interval around a different forecast
    is a different object). Nothing is defaulted: a null solved at a kernel no configured
    model uses would be a floor nobody's rows sit above.

    Args:
        cfg: The experiment config.

    Returns:
        The parameters, or None if the config carries neither row.
    """
    for model in cfg.models:
        if issubclass(MODEL_REGISTRY[model.name], EmpiricalResidualInterval):
            return (
                int(model.params["kernel_size"]),
                float(model.params.get("ridge", 0.0)),
                len(model.quantiles),
            )
    for model in cfg.models:
        if model.label == "dlinear_ols" and "kernel_size" in model.params:
            return (
                int(model.params["kernel_size"]),
                float(model.params.get("ridge", 0.0)),
                len(_QUANTILE_FAN_DEFAULT),
            )
    return None


#: The nine levels every quantile row in this project is scored at
#: (:func:`dmf.models.heads.quantile_fan`), used when the null's parameters come from a
#: ``dlinear_ols`` row, which carries no quantiles of its own.
_QUANTILE_FAN_DEFAULT: tuple[float, ...] = (
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


def run_regime_controls(
    cfg: ExperimentConfig,
    corpus_root: Path,
    regime: str,
    *,
    device: str = "cpu",
    batch_size: int = 4096,
    alpha: float = GATE5_ALPHA,
    rtol: float = PIPELINE_SANITY_RTOL,
    with_interval_controls: bool = True,
) -> tuple[list[IntervalControlResult], list[PipelineSanityResult], dict[str, str]]:
    """Run every control this module owns on one (experiment, regime).

    Args:
        cfg: The experiment config. Its ``data`` block defines the arm; its model list
            supplies the null's parameters and the untrained subject.
        corpus_root: Corpus dataset root, re-read independently by the raw persistence path.
        regime: One of the four evaluation regimes.
        device: Torch device the scoring passes run on.
        batch_size: Windows per batch.
        alpha: Nominal miscoverage the interval controls are read at. 0.1 is PICP@90.
        rtol: Relative tolerance of the pipeline-sanity control.
        with_interval_controls: Whether to run the two interval controls. False leaves the
            pipeline-sanity control, which every run must have.

    Returns:
        ``(interval results, pipeline-sanity results, skipped)``.

    Raises:
        AssertionError: If the pipeline-sanity control or the interval shuffle control fails.
            Both are enforced: the first invalidates every skill denominator in the run, the
            second every interval claim in it.
        ValueError: If the regime is unknown, or the config carries no persistence row.
    """
    labels = [model.label for model in cfg.models]
    if PERSISTENCE_LABEL not in labels:
        raise ValueError(
            f"experiment {cfg.name!r} has no {PERSISTENCE_LABEL!r} model, so the "
            f"pipeline-sanity control has no subject and every skill denominator in the run "
            f"is unchecked (CLAUDE.md non-negotiable 4). Configured models: {labels}"
        )
    spec = window_spec_from_config(cfg.data)
    split = build_split(load_manifest(corpus_root), _as_regime(regime))
    train = DeckMotionDataset(corpus_root, split, "train", cfg.data, spec)
    val = DeckMotionDataset(corpus_root, split, "val", cfg.data, spec, stats=train.norm_stats)
    test = DeckMotionDataset(corpus_root, split, "test", cfg.data, spec, stats=train.norm_stats)
    n_in = len(train.input_columns)
    n_out = len(train.target_columns)
    skipped: dict[str, str] = {}

    persistence_cfg = next(model for model in cfg.models if model.label == PERSISTENCE_LABEL)
    persistence = build_model(persistence_cfg, spec, n_in, n_out, revin=cfg.data.revin)
    persistence.eval()
    # Enforced, exactly as in the training driver: it raises. The rows it now also produces
    # do not change that -- they record the number the assertion compared.
    sanity = persistence_pipeline_sanity(
        test, corpus_root, rtol=rtol, batch_size=batch_size, model=persistence
    )

    interval: list[IntervalControlResult] = []
    if not with_interval_controls:
        skipped[f"{regime}/interval"] = "not requested"
        return interval, [sanity], skipped

    if not _interval_heads(cfg):
        skipped[f"{regime}/interval"] = (
            f"experiment {cfg.name!r} publishes no interval -- every configured model has a "
            f"point head -- so there is no coverage claim for an interval control to audit. "
            f"Running one anyway would re-measure the unconditional floor under a second "
            f"experiment label, which reads as two measurements of one thing"
        )
        return interval, [sanity], skipped

    params = _residual_interval_params(cfg)
    head_cfg = _untrained_head_config(cfg)
    if params is None:
        skipped[f"{regime}/interval"] = (
            f"experiment {cfg.name!r} carries neither a residual_interval nor a dlinear_ols "
            f"row, so the unconditional null (P6-D6) cannot be solved at the kernel and ridge "
            f"the run's own point forecast uses. Solving it at a default would compare every "
            f"head against a floor no configured model shares"
        )
        return interval, [sanity], skipped
    kernel_size, ridge, n_quantiles = params

    honest_moments = accumulate_training_moments(
        train,
        max_order=1,
        num_workers=cfg.train.num_workers,
        decompose_kernel=_decompose_kernel(cfg) or kernel_size,
    )
    null_model, _ = fit_residual_interval(
        honest_moments,
        val,
        kernel_size=kernel_size,
        ridge=ridge,
        lookback=spec.lookback,
        n_input_channels=n_in,
        n_target_channels=n_out,
        n_quantiles=n_quantiles,
        batch_size=batch_size,
        num_workers=cfg.train.num_workers,
    )

    shuffled_moments = accumulate_training_moments(
        train,
        max_order=1,
        num_workers=cfg.train.num_workers,
        decompose_kernel=_decompose_kernel(cfg) or kernel_size,
        shuffle_targets=True,
    )
    shuffled_model, _ = fit_residual_interval(
        shuffled_moments,
        val,
        kernel_size=kernel_size,
        ridge=ridge,
        lookback=spec.lookback,
        n_input_channels=n_in,
        n_target_channels=n_out,
        n_quantiles=n_quantiles,
        batch_size=batch_size,
        num_workers=cfg.train.num_workers,
    )
    interval.append(
        interval_shuffle_control(
            test,
            shuffled_model=shuffled_model,
            null_model=null_model,
            regime=regime,
            horizons=cfg.data.horizons,
            fs_hz=cfg.data.fs_hz,
            alpha=alpha,
            # Enforced. Information surviving a shuffle of the targets cannot reach an
            # interval by any path but leakage, so a failure stops the run rather than
            # being recorded beside the coverage table it would invalidate.
            strict=True,
            batch_size=batch_size,
            num_workers=cfg.train.num_workers,
            device=device,
        )
    )

    if head_cfg is None:
        skipped[f"{regime}/interval_untrained"] = (
            f"experiment {cfg.name!r} configures no SGD interval head, and 'randomly "
            f"initialised' is meaningless for a closed-form one -- an unfitted solve is an "
            f"empty object, not an untrained model. The shuffle control above still ran, "
            f"against the same null"
        )
        return interval, [sanity], skipped
    untrained = _untrained_head(head_cfg, cfg, n_in=n_in, n_out=n_out)
    interval.append(
        interval_untrained_control(
            test,
            untrained_model=untrained,
            null_model=null_model,
            regime=regime,
            horizons=cfg.data.horizons,
            fs_hz=cfg.data.fs_hz,
            alpha=alpha,
            # Reported, NOT enforced (P3-D9, and P6-D12 for why the P6-D11 narrowing is not
            # extended to it): on a channel sitting on its P1-D2 residual floor an untrained
            # near-zero-width fan can remove a large fraction of the null's loss without
            # having seen a datum. That failure ships as a row rather than being tuned away.
            strict=False,
            batch_size=batch_size,
            num_workers=cfg.train.num_workers,
            device=device,
        )
    )
    return interval, [sanity], skipped


def _untrained_head(
    head_cfg: ModelConfig, cfg: ExperimentConfig, *, n_in: int, n_out: int
) -> BaseForecaster:
    """Construct the untrained interval subject, seeded so the row is reproducible.

    Args:
        head_cfg: The head's model config.
        cfg: The experiment config, for the window geometry, the RevIN setting and the seed
            the initialisation is drawn at.
        n_in: Input channel count ``C_in``.
        n_out: Target channel count ``C_out``.

    Returns:
        The freshly constructed, unfitted head, in eval mode.
    """
    set_seed(cfg.seeds[0])
    model = build_model(
        head_cfg,
        window_spec_from_config(cfg.data),
        n_in,
        n_out,
        revin=cfg.data.revin,
    )
    model.eval()
    return model


def run_controls(
    configs: Sequence[ExperimentConfig],
    corpus_root: Path,
    *,
    results_dir: Path,
    experiments: Sequence[str] | None = None,
    arms: Sequence[str] | None = None,
    regimes: Sequence[str] | None = None,
    device: str = "cpu",
    batch_size: int = 4096,
    alpha: float = GATE5_ALPHA,
    rtol: float = PIPELINE_SANITY_RTOL,
    with_interval_controls: bool = True,
    write: bool = True,
) -> ControlArtifacts:
    """Run the controls for every requested (experiment, regime) and write both tables.

    Args:
        configs: Experiment configs to control.
        corpus_root: Corpus dataset root.
        results_dir: Directory the CSVs are written to. **Named by the caller**: ``results/``,
            ``results/imu/``, ``results/e02/`` and ``results/e03/`` are the Gate 3-5 audit
            trails and are read, never regenerated.
        experiments: Provenance label per config, defaulting to each config's own ``name``.
            The caller passes the config **stem** where the two differ --
            ``configs/experiment/e02_deep.yaml`` is named ``deep`` inside -- so that these
            rows key on the same experiment names
            :func:`dmf.eval.assemble.build_controls_table` writes into ``controls.csv``.
        arms: Arm label per config, defaulting to
            :data:`dmf.eval.assemble.NOT_AN_ARM` for each. A non-empty sentinel and not an
            empty string: the renderer groups on this column, an empty value round-trips
            through CSV as NaN, and pandas drops NaN group keys -- which once deleted 288
            rows from a rendered summary while leaving them in the file (P6-D15 defect 4).
        regimes: Regimes to control, defaulting to each config's own.
        device: Torch device.
        batch_size: Windows per batch.
        alpha: Nominal miscoverage the interval controls are read at.
        rtol: Relative tolerance of the pipeline-sanity control.
        with_interval_controls: Whether to run the interval controls.
        write: Whether to write the CSVs.

    Returns:
        The combined artifacts.

    Raises:
        ValueError: If ``configs`` is empty, or if ``arms`` is given at a different length.
        AssertionError: As :func:`run_regime_controls` raises.
    """
    if not configs:
        raise ValueError("no experiment configs to control; a run with no controls is not audited")
    labels = tuple(arms) if arms is not None else tuple(NOT_AN_ARM for _ in configs)
    names = tuple(experiments) if experiments is not None else tuple(cfg.name for cfg in configs)
    if len(labels) != len(configs) or len(names) != len(configs):
        raise ValueError(
            f"{len(labels)} arm label(s) and {len(names)} experiment label(s) for "
            f"{len(configs)} config(s); every table's rows must say which run and which arm "
            f"they describe"
        )
    blank = [value for value in (*labels, *names) if not str(value).strip()]
    if blank:
        raise ValueError(
            "an empty experiment or arm label round-trips through CSV as NaN and pandas "
            "drops NaN group keys, so those rows would be in the file and absent from the "
            f"rendered summary (P6-D15 defect 4). Use {NOT_AN_ARM!r} rather than an empty "
            "string"
        )

    interval_parts: list[pd.DataFrame] = []
    sanity_parts: list[pd.DataFrame] = []
    skipped: dict[str, str] = {}
    for cfg, arm, name in zip(configs, labels, names, strict=True):
        # Per config, never pooled: `horizons` and `fs_hz` belong to a config's own data
        # block, and two arms of different geometry tabulated under one of them would print
        # one arm's rows at the other's lead times.
        sanity_rows: list[tuple[PipelineSanityResult, str, str, str]] = []
        chosen = tuple(regimes) if regimes is not None else tuple(cfg.regimes)
        for regime in chosen:
            if regime not in cfg.regimes:
                skipped[f"{name}/{regime}"] = (
                    f"experiment {cfg.name!r} was not fitted on regime {regime!r}; it carries "
                    f"{list(cfg.regimes)}"
                )
                continue
            interval, sanity, reasons = run_regime_controls(
                cfg,
                corpus_root,
                regime,
                device=device,
                batch_size=batch_size,
                alpha=alpha,
                rtol=rtol,
                with_interval_controls=with_interval_controls,
            )
            skipped.update({f"{name}/{key}": value for key, value in reasons.items()})
            sanity_rows += [(result, name, arm, regime) for result in sanity]
            if interval:
                frame = interval_controls_table(interval)
                frame.insert(0, "arm", arm)
                frame.insert(0, "experiment", name)
                interval_parts.append(frame)
        if sanity_rows:
            sanity_parts.append(
                pipeline_sanity_table(
                    sanity_rows,
                    horizons=tuple(cfg.data.horizons),
                    fs_hz=cfg.data.fs_hz,
                    rtol=rtol,
                )
            )

    artifacts = ControlArtifacts(
        interval_controls=(
            pd.concat(interval_parts, ignore_index=True) if interval_parts else pd.DataFrame()
        ),
        pipeline_sanity=(
            pd.concat(sanity_parts, ignore_index=True) if sanity_parts else pd.DataFrame()
        ),
        skipped=skipped,
    )
    if write:
        for name, filename in CONTROL_ARTIFACTS.items():
            frame = getattr(artifacts, name)
            # No empty-table placeholder, as in dmf.eval.scoring: an empty CSV in results/
            # cannot be told from a table with nothing to report, and `skipped` is where an
            # absence is explained.
            if not frame.empty:
                artifacts.paths[name] = write_table(frame, results_dir / filename)
    return artifacts
