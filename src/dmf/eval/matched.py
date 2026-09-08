"""Re-scoring the lookback arms on the forecast origins they share.

``docs/protocol.md`` P6-D4 item 1 pre-registered that the lookback contrast is matched on
the **forecast origin** ``t = s + L - 1`` and never on the window start, and P6-D15 finding 1
measured that the committed arms are not: each run scored its own arm's full window set,
1151 / 1131 / 1091 windows per realization at L=100/200/400. Differencing those tables
compares forecasts of *different absolute times* -- the exact error the pre-registration
exists to prevent, and one that leaves both tables looking perfectly well formed.
:mod:`dmf.eval.assemble` therefore refuses the whole ``lookback`` ablation, which is the one
Gate 6 blocker that is not "the sweep has not finished".

This module is the missing producer. **It re-scores; it does not retrain.** Every model comes
back through :func:`dmf.train.experiment.load_or_fit`: SGD rows load their committed
``state_dict`` and raise rather than fall back to a random initialisation, closed-form rows
are re-solved from the same full training split the sweep used. The only thing that differs
from the arm's own committed run is the **test** window population, which is restricted to
:func:`dmf.eval.ablations.matched_origins` -- the L=400 arm's 1091 origins, 399..5849, which
the other two arms' origin sets both contain.

**Why the matching happens here and not in the sweep.** The intersection is a property of
three arms, and no single training run knows about the other two; imposing it at fit time
would also have thrown away windows the models were entitled to learn from. Matching is a
*scoring* operation, so it belongs to a scoring driver, and confining it to the test
partition is what makes a matched row and the arm's committed row differ in their window set
and in nothing else.

**Three things this module deliberately does not do.**

- It does not score the quiescence or phase-lag passes. Neither is defined on a thinned
  origin set -- one scores sustained intervals over consecutive windows, the other builds its
  own stride-1 dataset -- and :func:`dmf.eval.scoring.score_regime` refuses the combination
  rather than reporting them beside a matched accuracy row.
- It does not write into any committed directory. The reference arm's rows here are
  ``results/e02/``'s *checkpoints* re-scored, never ``results/e02/``'s rows re-read: those
  were scored on 1131 origins per realization and are not the matched quantity. ``results/``,
  ``results/imu/``, ``results/e02/``, ``results/e03/`` and the per-arm ``results/e04/*``
  directories are read and never written.
- It does not decide anything. It writes tables that carry
  :data:`dmf.eval.ablations.MATCHED_ORIGIN_COLUMNS` on every row, and
  :func:`dmf.eval.ablations.unmatched_origin_reason` re-derives the intersection from those
  columns when the assembler reads them back. A table without them -- an arm's own
  ``baselines_by_seed.csv``, say, copied into place -- is refused there, so this module being
  the only writer is enforced by the reader rather than by convention.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from dmf.config import ExperimentConfig, load_experiment
from dmf.data.splits import load_manifest
from dmf.data.windows import WindowSpec, window_spec_from_config
from dmf.eval.ablations import (
    ABLATIONS,
    ARMS,
    matched_origin_provenance,
    matched_origins,
    matched_window_counts,
)
from dmf.eval.assemble import (
    ARM_CELLS_FILE,
    ARM_ROWS_FILE,
    MATCHED_ORIGIN_ARMS,
    MATCHED_ORIGIN_SUBDIR,
)
from dmf.eval.report import PHASE6_SUBDIR, write_table
from dmf.eval.scoring import score_regime
from dmf.typedefs import IntArray

__all__ = [
    "CONFIG_DIR",
    "LOOKBACK_ABLATION",
    "SOURCE_EXPERIMENT_COLUMN",
    "WRITES_ARMS",
    "MatchedOriginArtifacts",
    "corpus_n_samples",
    "default_out_root",
    "lookback_arm_specs",
    "matched_lookback_plan",
    "score_matched_lookback",
]

#: The ablation this module re-scores. Named rather than hard-coded to the three arm keys, so
#: that a fourth lookback arm added to :data:`dmf.eval.ablations.ARMS` is picked up here and
#: -- more importantly -- changes the intersection every existing arm is matched on.
LOOKBACK_ABLATION: str = "lookback"

#: Where the experiment configs live, relative to the repository root.
CONFIG_DIR: Path = Path("configs/experiment")

#: Column naming the config each row was scored from. **Not** ``experiment``: a matched table
#: is read by :func:`dmf.eval.assemble.read_arm_frames`, which labels the arm's rows with the
#: arm key by inserting a column of that name, and pandas refuses to insert a column that
#: already exists. The provenance is still on the row, under a name that cannot collide.
SOURCE_EXPERIMENT_COLUMN: str = "source_experiment"


@dataclass
class MatchedOriginArtifacts:
    """Everything one matched-origin re-scoring produced.

    Attributes:
        rows: Arm key -> its matched per-seed accuracy rows, on the
            ``baselines_by_seed.csv`` schema plus
            :data:`dmf.eval.ablations.MATCHED_ORIGIN_COLUMNS`.
        origins: The origin set every arm was scored on, samples, ascending.
        n_samples: Realization length the intersection was computed at, samples.
        window_counts: Lookback -> ``(windows per realization before matching, after)``, the
            pair a lookback table must print beside its rows.
        cells: Arm key -> its matched **per grid cell** rows, on the
            ``baselines_by_cell.csv`` schema plus the same provenance columns. Written
            because it is the only granularity a paired interval can be resampled from
            (:func:`dmf.eval.ablations.contrast_uncertainty`): without it the lookback
            ablation is the one ablation whose contrasts ship with no uncertainty at all,
            and an arm's *unmatched* committed per-cell table is not a substitute -- it
            covers 1151 or 1131 origins per realization rather than the 1091 the contrast
            was taken over.
        paths: Arm key -> the per-seed file written for it.
        cell_paths: Arm key -> the per-cell file written for it.
        skipped: Arm key or ``"<arm>/<experiment>"`` -> why nothing was produced for it. A
            missing checkpoint tree is the usual reason, and it is recorded rather than
            raised so that a partially finished sweep still yields the arms that are done.
    """

    rows: dict[str, pd.DataFrame]
    origins: IntArray
    n_samples: int
    window_counts: dict[int, tuple[int, int]]
    cells: dict[str, pd.DataFrame] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)
    cell_paths: dict[str, Path] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


def corpus_n_samples(corpus_root: Path) -> int:
    """Return the realization length the whole corpus shares.

    Read from the manifest's ``n_rows`` rather than by opening a Parquet file, and required
    to be uniform: the matched origin set is a single array applied to every realization, so
    a corpus of two record lengths has no single intersection and the arms could not be
    paired at all.

    Args:
        corpus_root: Corpus dataset root.

    Returns:
        Samples per realization.

    Raises:
        ValueError: If the manifest carries more than one record length.
    """
    lengths = sorted({int(value) for value in load_manifest(corpus_root)["n_rows"]})
    if len(lengths) != 1:
        raise ValueError(
            f"the corpus at {corpus_root} holds realizations of {lengths} samples. A matched "
            f"origin set is one array applied to every realization, so records of differing "
            f"length have no single intersection and the lookback arms cannot be paired"
        )
    return lengths[0]


def lookback_arm_specs(
    config_dir: Path = CONFIG_DIR, *, ablation: str = LOOKBACK_ABLATION
) -> dict[str, tuple[WindowSpec, tuple[ExperimentConfig, ...]]]:
    """Load every arm of one ablation, with the geometry its configs actually define.

    The geometry comes from the ``data`` block of the arm's own experiment configs, not from
    :data:`dmf.eval.ablations.ARMS`, and the two are then checked against each other. The
    registry's ``lookback`` is a claim about the config; this is where it stops being one.

    Args:
        config_dir: Directory holding ``<experiment>.yaml``.
        ablation: Ablation key.

    Returns:
        Arm key -> ``(window geometry, the configs that produce it)``, in arm-key order.

    Raises:
        KeyError: If ``ablation`` is not in :data:`dmf.eval.ablations.ABLATIONS`.
        FileNotFoundError: If a declared config is absent.
        ValueError: If an arm's configs disagree with each other or with the registry on the
            window geometry -- a lookback contrast whose arms are not the lookbacks they are
            labelled with is not a contrast, and every number in it would be mislabelled.
    """
    out: dict[str, tuple[WindowSpec, tuple[ExperimentConfig, ...]]] = {}
    for arm in ABLATIONS[ablation]:
        configs: list[ExperimentConfig] = []
        spec: WindowSpec | None = None
        for stem in ARMS[arm].experiments:
            path = config_dir / f"{stem}.yaml"
            if not path.is_file():
                raise FileNotFoundError(
                    f"arm {arm!r} declares experiment {stem!r}, absent at {path}"
                )
            cfg = load_experiment(path)
            current = window_spec_from_config(cfg.data)
            if spec is not None and current != spec:
                raise ValueError(
                    f"arm {arm!r} is fed by configs of two different window geometries "
                    f"({spec} and {current}); its rows would not share an origin set even "
                    f"with itself"
                )
            spec = current
            configs.append(cfg)
        if spec is None:
            raise ValueError(f"arm {arm!r} declares no experiment config")
        if spec.lookback != ARMS[arm].lookback:
            raise ValueError(
                f"arm {arm!r} is registered at lookback {ARMS[arm].lookback} but its configs "
                f"define {spec.lookback}. The registry is what the ablation table's "
                f"`lookback` column is written from, so the two disagreeing would publish "
                f"the contrast under the wrong lookbacks"
            )
        out[arm] = (spec, tuple(configs))
    return out


def matched_lookback_plan(
    corpus_root: Path, config_dir: Path = CONFIG_DIR, *, ablation: str = LOOKBACK_ABLATION
) -> tuple[IntArray, int, dict[str, tuple[WindowSpec, tuple[ExperimentConfig, ...]]]]:
    """Compute the origin set the arms of one ablation share, and what defines it.

    Args:
        corpus_root: Corpus dataset root.
        config_dir: Directory holding the experiment configs.
        ablation: Ablation key.

    Returns:
        ``(origins, n_samples, arms)``: the intersection, the realization length it was taken
        at, and the per-arm geometry and configs.

    Raises:
        ValueError: As :func:`lookback_arm_specs` and
            :func:`dmf.eval.ablations.matched_origins` raise.
    """
    arms = lookback_arm_specs(config_dir, ablation=ablation)
    n_samples = corpus_n_samples(corpus_root)
    origins = matched_origins([spec for spec, _ in arms.values()], n_samples)
    return origins, n_samples, arms


def score_matched_lookback(
    corpus_root: Path,
    *,
    out_root: Path,
    config_dir: Path = CONFIG_DIR,
    checkpoint_root: Path = Path("artifacts/checkpoints"),
    arms: Sequence[str] | None = None,
    regimes: Sequence[str] | None = None,
    device: str = "cpu",
    batch_size: int = 4096,
    write: bool = True,
) -> MatchedOriginArtifacts:
    """Re-score every lookback arm on the shared origin set, from committed artifacts.

    One file per arm, ``<out_root>/<arm>/baselines_by_seed.csv``, which is where
    :func:`dmf.eval.assemble.read_arm_frames` looks. The reference arm ``lookback_20s`` is
    scored here like any other: its rows are ``results/e02/``'s *checkpoints* re-scored on
    1091 origins, not ``results/e02/``'s committed rows, which were scored on 1131 (P6-D4,
    the arm registry's own note).

    Args:
        corpus_root: Corpus dataset root.
        out_root: Directory the per-arm subdirectories are written under, normally
            ``results/e04/matched_origins``. **Named by the caller**: this function never
            derives a path inside a committed audit trail.
        config_dir: Directory holding the experiment configs.
        checkpoint_root: Root of the committed checkpoints. Read, never written.
        arms: Arms to score, default every arm of the lookback ablation. Restricting this
            does **not** change the origin set: it is still the intersection over every arm,
            so a one-arm run produces rows that pair with the others' (see
            :func:`matched_lookback_plan`).
        regimes: Regimes to score, default each config's own. A regime a config was not
            fitted on is skipped with that reason rather than scored from absent checkpoints.
        device: Torch device the forward passes run on.
        batch_size: Windows per batch. Speed and memory only.
        write: Whether to write the CSVs.

    Returns:
        The artifacts, with a ``skipped`` entry for every arm or experiment that produced no
        rows.

    Raises:
        ValueError: If an unknown arm is requested, or as :func:`matched_lookback_plan`
            raises.
    """
    origins, n_samples, plan = matched_lookback_plan(corpus_root, config_dir)
    chosen = tuple(arms) if arms is not None else tuple(plan)
    unknown = [arm for arm in chosen if arm not in plan]
    if unknown:
        raise ValueError(
            f"arms {unknown} are not arms of the {LOOKBACK_ABLATION!r} ablation, which has "
            f"{sorted(plan)}"
        )
    counts = matched_window_counts([spec for spec, _ in plan.values()], n_samples)

    artifacts = MatchedOriginArtifacts(
        rows={}, origins=origins, n_samples=n_samples, window_counts=counts
    )
    for arm in chosen:
        spec, configs = plan[arm]
        provenance = matched_origin_provenance(
            spec, [other for other, _ in plan.values()], n_samples
        )
        parts: list[pd.DataFrame] = []
        cell_parts: list[pd.DataFrame] = []
        for cfg in configs:
            wanted = tuple(regimes) if regimes is not None else tuple(cfg.regimes)
            scoreable = [name for name in wanted if name in cfg.regimes]
            for name in wanted:
                if name not in cfg.regimes:
                    artifacts.skipped[f"{arm}/{cfg.name}/{name}"] = (
                        f"experiment {cfg.name!r} was not fitted on regime {name!r}; it "
                        f"carries {list(cfg.regimes)}, and scoring it would need checkpoints "
                        f"that do not exist"
                    )
            for name in scoreable:
                scored = score_regime(
                    cfg,
                    corpus_root,
                    name,
                    device=device,
                    checkpoint_root=checkpoint_root,
                    # Neither pass is defined on a thinned origin set; `score_regime` refuses
                    # the combination, and this is the call site that respects it.
                    with_quiescence=False,
                    with_phase_lag=False,
                    batch_size=batch_size,
                    test_origins=origins,
                )
                frame = scored.metrics_full
                if frame.empty:
                    artifacts.skipped[f"{arm}/{cfg.name}/{name}"] = (
                        "the scoring pass produced no row"
                    )
                    continue
                frame = frame.copy()
                frame.insert(0, SOURCE_EXPERIMENT_COLUMN, cfg.name)
                for column, value in provenance.items():
                    frame[column] = value
                parts.append(frame)
                cells = scored.metrics_by_cell
                if not cells.empty:
                    cells = cells.copy()
                    cells.insert(0, SOURCE_EXPERIMENT_COLUMN, cfg.name)
                    for column, value in provenance.items():
                        cells[column] = value
                    cell_parts.append(cells)
        if not parts:
            artifacts.skipped[arm] = (
                f"no matched rows for arm {arm!r}: none of its experiments "
                f"{[cfg.name for cfg in configs]} produced a scored regime"
            )
            continue
        artifacts.rows[arm] = pd.concat(parts, ignore_index=True)
        if cell_parts:
            artifacts.cells[arm] = pd.concat(cell_parts, ignore_index=True)
        else:
            artifacts.skipped[f"cells/{arm}"] = (
                f"arm {arm!r} produced no per-cell rows, so its contrast will ship with no "
                f"paired interval"
            )
        if write:
            artifacts.paths[arm] = write_table(artifacts.rows[arm], out_root / arm / ARM_ROWS_FILE)
            if arm in artifacts.cells:
                artifacts.cell_paths[arm] = write_table(
                    artifacts.cells[arm], out_root / arm / ARM_CELLS_FILE
                )
    return artifacts


def default_out_root(results_root: Path) -> Path:
    """Return where a matched re-scoring is written under a results root.

    Args:
        results_root: The results root, normally ``results/``, or its Phase 6 subdirectory.

    Returns:
        ``<root>/e04/matched_origins``, the path
        :func:`dmf.eval.assemble.read_arm_frames` reads each arm from.
    """
    base = results_root if results_root.name == PHASE6_SUBDIR else results_root / PHASE6_SUBDIR
    return base / MATCHED_ORIGIN_SUBDIR


#: The arms the assembler will only read from a matched table, restated here as the writer's
#: obligation: every one of them must be produced by this module or the ``lookback`` ablation
#: stays skipped. Imported rather than re-listed so the two cannot drift.
WRITES_ARMS: frozenset[str] = MATCHED_ORIGIN_ARMS
