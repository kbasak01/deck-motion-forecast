"""Torch ``Dataset`` and ``DataLoader`` construction over the simulated corpus.

The dataset is constructed **from an already-decided split**. It takes a
:class:`dmf.data.splits.Split` and a partition name, not a corpus and a fraction, so that
there is no code path in which this module could shuffle-then-split.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from dmf.config import DataConfig, ObservationMode
from dmf.data.channels import IMU_BY_IDEAL
from dmf.data.normalize import NormStats, build_norm_stats
from dmf.data.splits import RealizationKey, Split, load_manifest
from dmf.data.windows import WindowSpec, n_windows, window_origins, window_start_indices
from dmf.sim.generate import RealizationSpec, realization_path
from dmf.sim.imu import IDEAL_COLUMNS
from dmf.typedefs import FloatArray, IntArray

__all__ = [
    "PARTITIONS",
    "SEA_STATE_CHANNEL_PREFIX",
    "DeckMotionDataset",
    "make_dataloader",
    "resolve_columns",
    "sea_state_channels",
]

#: Prefix of the synthetic one-hot sea-state channel names, e.g. ``"ss_onehot_SS5"``.
#: Not a corpus column: :func:`resolve_columns` would reject it, which is the point --
#: the indicator is exogenous metadata, not a measured channel.
SEA_STATE_CHANNEL_PREFIX: str = "ss_onehot_"

#: Valid partition names. ``val`` and ``test`` always require training statistics.
PARTITIONS: tuple[str, ...] = ("train", "val", "test")


def resolve_columns(names: tuple[str, ...], mode: ObservationMode) -> tuple[str, ...]:
    """Map logical channel names onto the corpus columns of one observation mode.

    In ``ideal`` mode the names pass through. In ``imu`` mode each name that has a noisy
    twin is replaced by it (``roll`` -> ``roll_imu``); ``heave_acc`` has no twin and passes
    through, because it *is* the accelerometer channel.

    Callers must resolve ``input_channels`` and ``target_dofs`` from the same call site.
    Mixing an ``imu`` input with an ``ideal`` target is not merely untidy: per
    ``docs/protocol.md`` P1-D6 the reconstructed ``heave_imu`` channel leads the truth by
    about 1.3 s at the SS5 spectral peak, so persistence on an ``imu`` input scored against
    an ``ideal`` target is roughly twice as strong at a 1 s horizon. That silently changes
    the denominator of every skill score, and it is invisible in a loss curve.

    Args:
        names: Logical channel names, e.g. ``("roll", "pitch", "heave")``.
        mode: Observation mode.

    Returns:
        Corpus column names, in the same order as ``names``.

    Raises:
        ValueError: If a name is not a corpus motion channel.
    """
    unknown = [n for n in names if n not in IDEAL_COLUMNS]
    if unknown:
        raise ValueError(f"unknown channel(s) {unknown}; corpus channels are {list(IDEAL_COLUMNS)}")
    if mode == "ideal":
        return tuple(names)
    return tuple(IMU_BY_IDEAL.get(n, n) for n in names)


def sea_state_channels(sea_states: tuple[str, ...]) -> tuple[str, ...]:
    """Name the one-hot sea-state indicator channels.

    Args:
        sea_states: Sea-state labels, in the order the indicator columns take.

    Returns:
        Channel names, one per sea state, prefixed with
        :data:`SEA_STATE_CHANNEL_PREFIX`.
    """
    return tuple(f"{SEA_STATE_CHANNEL_PREFIX}{name}" for name in sea_states)


def _key_to_path(key: RealizationKey) -> Path:
    """Return the corpus-relative Parquet path of one realization key.

    Args:
        key: ``(ss, heading_deg, speed_kn, vessel, seed)``.

    Returns:
        The relative path, via :func:`dmf.sim.generate.realization_path` so that the layout
        is defined in exactly one place.
    """
    ss, heading, speed, vessel, seed = key
    return realization_path(
        RealizationSpec(seed=seed, sea_state=ss, heading_deg=heading, speed_kn=speed, vessel=vessel)
    )


class DeckMotionDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Windows drawn from one partition of one split.

    Yields ``(x, y, window_mean)``: the de-meaned input window, the target window in
    corpus units, and the per-window mean needed to map a prediction back into corpus
    units. The mean is returned rather than re-derived downstream so that the inverse
    transform cannot drift out of step with the forward one.

    With ``DataConfig.condition_on_sea_state`` set, ``x`` carries a one-hot sea-state
    indicator **appended after every motion channel**, so ``C_in`` becomes
    ``len(input_channels) + n_sea_states`` while positions ``0 .. C_out - 1`` remain
    ``target_dofs``. That ordering is load-bearing, not stylistic:
    :class:`dmf.models.persistence.Persistence` forecasts by slicing the first ``C_out``
    input channels (P2-D4), so an interleaved or prepended indicator would make every
    baseline forecast an indicator column.

    With ``origins`` given, the dataset holds a **subset** of the windows the geometry
    admits: only those whose forecast origin ``s + L - 1`` is in the given set, in every
    realization alike. That is the mechanism the lookback ablation needs, and the argument
    is the *origin* rather than the start deliberately -- two arms matched on the window
    start forecast absolute times differing by ``L_a - L_b``, up to 300 samples at the
    production geometry, and both tables would look perfectly well formed
    (``docs/protocol.md`` P6-D4 item 1;
    :func:`dmf.eval.ablations.start_matching_compares_different_times`). The subset is
    applied identically to every realization, so windows stay realization-major and
    uniformly counted, which is what every consumer of
    :attr:`windows_per_realization` assumes.
    """

    def __init__(
        self,
        corpus_root: Path,
        split: Split,
        partition: str,
        cfg: DataConfig,
        window_spec: WindowSpec,
        stats: NormStats | None = None,
        *,
        origins: IntArray | Sequence[int] | None = None,
    ) -> None:
        """Build the dataset for one partition.

        The partition's realizations are loaded eagerly, projected to the configured
        columns only, into one contiguous ``float32`` array of shape
        ``(n_realizations, n_samples, C_in)``. The largest partition in the project,
        ``unseen_vessel/train``, is 1632 x 6000 x 6 x 4 B = 235 MB. Realizations are visited
        in sorted key order, so window indices are reproducible across runs and machines.

        Args:
            corpus_root: Path to the Parquet corpus produced by
                :func:`dmf.sim.generate.generate_corpus`.
            split: The realization-level split to draw from.
            partition: One of ``"train"``, ``"val"``, ``"test"``.
            cfg: Channel selection, observation mode, and sampling settings.
            window_spec: Window geometry.
            stats: Normalisation statistics fitted on the training split. Required for
                ``"val"`` and ``"test"``; if omitted for ``"train"``, they are fitted here
                from the training realizations. They cover the **motion** channels only;
                the sea-state indicator is not normalised (see :meth:`__getitem__`).
            origins: Forecast origins to keep, samples, ascending and unique, or None for
                every window the geometry admits. Each must be an origin this geometry
                actually scores, i.e. a member of
                :func:`dmf.data.windows.window_origins`; anything else is refused rather
                than rounded to the nearest window, because a silently relocated origin is
                the failure the matching exists to prevent.

                **The subset does not change the normalisation statistics.** When they are
                fitted here they are still fitted over the whole training series, exactly
                as they are without a subset, so a matched re-scoring of a test partition
                is scored under the same scale the unmatched run used and the two tables
                differ in their window set and in nothing else.

        Raises:
            ValueError: If ``partition`` is unrecognised, or if ``partition`` is not
                ``"train"`` and ``stats`` is None -- constructing a validation or test set
                without training statistics is a leak, so it is refused rather than
                silently recomputed. Also if the partition is empty, if its realizations
                differ in length, if a realization's Parquet file is missing, if
                ``cfg.condition_on_sea_state`` is set and a realization carries a sea state
                absent from the corpus manifest, or if ``origins`` is empty, unsorted,
                repeated, or names an origin this geometry does not score.
        """
        if partition not in PARTITIONS:
            raise ValueError(f"unknown partition {partition!r}, expected one of {list(PARTITIONS)}")
        if partition != "train" and stats is None:
            raise ValueError(
                f"partition {partition!r} requires normalisation statistics fitted on the "
                f"training split; refitting them here would leak held-out variance "
                f"(CLAUDE.md non-negotiable 3)"
            )

        keys: frozenset[RealizationKey] = {
            "train": split.train_keys,
            "val": split.val_keys,
            "test": split.test_keys,
        }[partition]
        if not keys:
            raise ValueError(f"partition {partition!r} of regime {split.regime!r} is empty")

        self._corpus_root = corpus_root
        self._cfg = cfg
        self._spec = window_spec
        self._partition = partition
        self._regime = split.regime
        self._keys: tuple[RealizationKey, ...] = tuple(sorted(keys))
        self._motion_columns = resolve_columns(cfg.input_channels, cfg.observation_mode)
        self._target_columns = resolve_columns(cfg.target_dofs, cfg.observation_mode)
        self._target_index = [self._motion_columns.index(c) for c in self._target_columns]
        self._sea_states = self._corpus_sea_states() if cfg.condition_on_sea_state else ()
        self._sea_state_columns = sea_state_channels(self._sea_states)
        self._sea_state_onehot = self._build_onehot()

        self._data = self._load(corpus_root)
        self._n_samples = int(self._data.shape[1])
        self._per_realization = n_windows(self._n_samples, window_spec)
        if self._per_realization == 0:
            raise ValueError(
                f"realizations of {self._n_samples} samples yield no windows at "
                f"lookback {window_spec.lookback} + max horizon {window_spec.max_horizon}"
            )
        self._starts = window_start_indices(self._n_samples, window_spec)
        self._all_origins = window_origins(self._n_samples, window_spec)
        self._origins = self._all_origins
        self._is_subset = origins is not None
        if origins is not None:
            self._origins = self._validate_origins(origins)
            self._starts = self._origins - window_spec.lookback + 1
            self._per_realization = int(self._origins.size)
        self._stats = stats if stats is not None else self._fit_stats()

    def _validate_origins(self, origins: IntArray | Sequence[int]) -> IntArray:
        """Narrow a requested origin set to this geometry's own, or refuse it.

        Args:
            origins: Requested forecast origins, samples.

        Returns:
            The origins as an ascending int64 array.

        Raises:
            ValueError: If the set is empty, not one-dimensional, not strictly ascending,
                or contains an origin this geometry does not score.
        """
        wanted = np.asarray(origins, dtype=np.int64)
        if wanted.ndim != 1:
            raise ValueError(f"origins must be one-dimensional, got shape {wanted.shape}")
        if wanted.size == 0:
            raise ValueError(
                "origins is empty; a dataset holding no window would report metrics over "
                "nothing while looking like a table"
            )
        if not bool(np.all(np.diff(wanted) > 0)):
            raise ValueError(
                "origins must be strictly ascending and unique, so that the window index "
                "stays reproducible and each origin is scored exactly once"
            )
        unknown = np.setdiff1d(wanted, self._all_origins)
        if unknown.size:
            spec = self._spec
            raise ValueError(
                f"{unknown.size} requested origin(s), e.g. {unknown[:5].tolist()}, are not "
                f"scored at lookback {spec.lookback} with stride {spec.stride} on a "
                f"{self._n_samples}-sample realization, whose origins run "
                f"{int(self._all_origins[0])}..{int(self._all_origins[-1])}. An origin set "
                f"must be the intersection of every arm's origins "
                f"(dmf.eval.ablations.matched_origins); relocating one to the nearest "
                f"window would silently compare a forecast of a different absolute time"
            )
        return wanted

    def _corpus_sea_states(self) -> tuple[str, ...]:
        """Read the corpus sea-state vocabulary from the manifest.

        Derived rather than hard-coded, and derived from the **manifest** rather than from
        this partition's own keys, because the indicator must have the same width and the
        same column order in every partition of every regime. Taking it from the partition
        would give ``unseen_seastate`` a 3-wide training indicator and a 4-wide test one.

        What this does and does not use: the vocabulary is the set of sea states the corpus
        generator was configured to simulate -- corpus metadata, the same kind of fact as
        the channel list -- and no held-out sample value enters it. On
        ``unseen_seastate`` the SS6 column is simply never hot during training, which is
        extrapolation to an unseen category rather than leakage, and the shuffle control on
        that arm is what demonstrates the difference.

        Returns:
            Sea-state labels in sorted order, e.g. ``("SS3", "SS4", "SS5", "SS6")``.

        Raises:
            ValueError: If the manifest carries no sea states.
        """
        manifest = load_manifest(self._corpus_root)
        labels = tuple(sorted({str(v) for v in manifest["ss"].tolist()}))
        if not labels:
            raise ValueError(f"corpus manifest at {self._corpus_root} carries no sea states")
        return labels

    def _build_onehot(self) -> npt.NDArray[np.float32]:
        """Build the per-realization one-hot sea-state indicator.

        Returns:
            Array of shape ``(n_realizations, n_sea_states)``, dimensionless, one hot row
            per realization in the sorted key order the window index follows. Shape
            ``(n_realizations, 0)`` when conditioning is off.

        Raises:
            ValueError: If a realization's sea state is absent from the vocabulary.
        """
        onehot = np.zeros((len(self._keys), len(self._sea_states)), dtype=np.float32)
        if not self._sea_states:
            return onehot
        index = {name: i for i, name in enumerate(self._sea_states)}
        for row, key in enumerate(self._keys):
            label = key[0]
            if label not in index:
                raise ValueError(
                    f"realization {key} has sea state {label!r}, which is absent from the "
                    f"corpus manifest vocabulary {list(self._sea_states)}"
                )
            onehot[row, index[label]] = 1.0
        return onehot

    def _load(self, corpus_root: Path) -> npt.NDArray[np.float32]:
        """Read the partition's realizations into one contiguous float32 array.

        Args:
            corpus_root: Dataset root.

        Returns:
            Array of shape ``(n_realizations, n_samples, C_motion)``, corpus units. The
            sea-state indicator is not stored here: it is constant along the time axis, so
            materialising it per sample would cost ``n_realizations * n_samples *
            n_sea_states`` floats to carry ``n_realizations`` bits.

        Raises:
            ValueError: If a file is missing or realizations differ in length.
        """
        columns = list(self._motion_columns)
        blocks: list[npt.NDArray[np.float32]] = []
        n_samples = -1
        for key in self._keys:
            path = corpus_root / _key_to_path(key)
            if not path.exists():
                raise ValueError(f"realization {key} has no Parquet file at {path}")
            frame = pd.read_parquet(path, columns=columns)
            block = frame[columns].to_numpy(dtype=np.float32)
            if n_samples < 0:
                n_samples = block.shape[0]
            elif block.shape[0] != n_samples:
                raise ValueError(
                    f"realization {key} has {block.shape[0]} samples but the partition's "
                    f"first realization has {n_samples}; the window index assumes a uniform "
                    f"record length"
                )
            blocks.append(block)
        return np.stack(blocks, axis=0)

    def _fit_stats(self) -> NormStats:
        """Fit per-channel scales over this (training) partition.

        Accumulates first and second moments per realization in float64 rather than
        materialising a float64 copy of the whole partition. Covers the **motion** channels
        only -- the sea-state indicator has no scale to fit, and a one-hot column is
        degenerate on any partition confined to a single sea state, which
        :func:`dmf.data.normalize.build_norm_stats` would (correctly) refuse as
        zero-variance.

        Returns:
            Statistics labelled ``"<regime>/train"``, over the motion channels.
        """
        count = 0
        total = np.zeros(self._data.shape[2], dtype=np.float64)
        total_sq = np.zeros(self._data.shape[2], dtype=np.float64)
        for block in self._data:
            values = block.astype(np.float64)
            count += values.shape[0]
            total += values.sum(axis=0)
            total_sq += np.square(values).sum(axis=0)
        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 0.0)
        return build_norm_stats(
            scale=np.sqrt(variance),
            channels=self._motion_columns,
            fitted_on=f"{self._regime}/train",
            n_realizations=len(self._keys),
        )

    def __len__(self) -> int:
        """Return the number of windows in this partition.

        Returns:
            Window count, summed over the partition's realizations.
        """
        return len(self._keys) * self._per_realization

    def describe_window(self, index: int) -> tuple[RealizationKey, int]:
        """Return the provenance of one window.

        Every window comes from exactly one realization: the index arithmetic here is what
        makes "windows never span a realization boundary" checkable rather than asserted in
        a docstring, and it lets any result be broken down per grid cell later.

        Args:
            index: Window index in ``[0, len(self))``.

        Returns:
            Tuple ``(realization_key, start_sample)``, where ``start_sample`` is the index
            of the first lookback sample within that realization.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"window index {index} out of range for {len(self)} windows")
        realization, offset = divmod(index, self._per_realization)
        return self._keys[realization], int(self._starts[offset])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return one window triple.

        Args:
            index: Window index in ``[0, len(self))``.

        With ``condition_on_sea_state`` set, the one-hot indicator is appended after the
        motion channels **after** de-meaning and scaling, and is therefore left at exactly
        0.0 or 1.0. Two reasons, and the first is decisive: the indicator is constant within
        a window, so per-window de-meaning would subtract it from itself and hand the model
        an all-zero block -- the feature would be destroyed by the very transform meant to
        prepare it. Second, a categorical indicator has no physical scale for
        :class:`dmf.data.normalize.NormStats` to fit; its statistics stay over the motion
        channels, where "fitted on train only" is a meaningful claim.

        Returns:
            Tuple ``(x, y, window_mean)`` with shapes ``(L, C_in)``, ``(H, C_out)`` and
            ``(1, C_out)``, where ``C_in`` is ``len(input_columns)`` and so includes the
            sea-state indicator when it is enabled. ``x``'s motion channels are de-meaned
            and scaled (dimensionless) and its indicator channels are raw 0/1; ``y`` and
            ``window_mean`` are in corpus units (degrees for angles, metres for heave).

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"window index {index} out of range for {len(self)} windows")
        realization, offset = divmod(index, self._per_realization)
        start = int(self._starts[offset])
        spec = self._spec
        block = self._data[realization]
        x_raw = block[start : start + spec.lookback, :].astype(np.float64)
        y_raw = block[start + spec.lookback : start + spec.total_length, :][:, self._target_index]
        mean = x_raw.mean(axis=0, keepdims=True)
        x = (x_raw - mean) / self._stats.scale[None, :]
        if self._sea_states:
            indicator = np.broadcast_to(
                self._sea_state_onehot[realization][None, :].astype(np.float64),
                (spec.lookback, len(self._sea_states)),
            )
            x = np.concatenate((x, indicator), axis=1)
        window_mean = mean[:, self._target_index]
        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(y_raw, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(window_mean, dtype=np.float32)),
        )

    @property
    def norm_stats(self) -> NormStats:
        """Normalisation statistics in use, for propagation to val and test sets."""
        return self._stats

    @property
    def partition(self) -> str:
        """Which partition this dataset holds: ``"train"``, ``"val"`` or ``"test"``.

        Exposed so that a fitter can refuse anything but training data. ``norm_stats``
        already carries a ``fitted_on`` label, but that proves only that the *statistics*
        are train-only -- it says nothing about the windows the caller is about to fit
        coefficients on, and a val or test dataset built with train statistics carries a
        perfectly valid label. The two guards check different things and both are needed.
        """
        return self._partition

    @property
    def regime(self) -> str:
        """The evaluation regime this partition was cut from, e.g. ``"id"``."""
        return self._regime

    @property
    def realization_keys(self) -> tuple[RealizationKey, ...]:
        """Realization keys in this partition, in the sorted order the index follows."""
        return self._keys

    @property
    def window_spec(self) -> WindowSpec:
        """Window geometry this dataset was cut with."""
        return self._spec

    @property
    def target_columns(self) -> tuple[str, ...]:
        """Corpus column names of the target channels, after observation-mode resolution."""
        return self._target_columns

    @property
    def input_columns(self) -> tuple[str, ...]:
        """Model-facing input channel names, in tensor-column order.

        The motion channels, after observation-mode resolution, followed by the one-hot
        sea-state indicator channels when ``condition_on_sea_state`` is set. Its length is
        the ``C_in`` every model is constructed with, which is why the indicator appears
        here and not only in a separate property: a model built from
        ``len(motion_columns)`` would be silently the wrong width.
        """
        return self._motion_columns + self._sea_state_columns

    @property
    def motion_columns(self) -> tuple[str, ...]:
        """Corpus column names of the motion input channels only.

        Distinct from :attr:`input_columns` exactly when the sea-state indicator is on.
        This is the set that is read from Parquet, that :attr:`norm_stats` covers, and that
        ``target_columns`` is a prefix of.
        """
        return self._motion_columns

    @property
    def sea_state_columns(self) -> tuple[str, ...]:
        """Names of the appended one-hot sea-state channels; empty when conditioning is off."""
        return self._sea_state_columns

    @property
    def sea_states(self) -> tuple[str, ...]:
        """Corpus sea-state vocabulary the indicator is built over, in column order."""
        return self._sea_states

    @property
    def corpus_root(self) -> Path:
        """Dataset root the realizations were read from."""
        return self._corpus_root

    @property
    def windows_per_realization(self) -> int:
        """Windows cut from each realization, identical across the partition."""
        return self._per_realization

    @property
    def n_samples(self) -> int:
        """Samples per realization, uniform across the partition by construction."""
        return self._n_samples

    @property
    def window_starts(self) -> IntArray:
        """Start sample of each window of one realization, ascending.

        The subset when one was requested, the full set otherwise. Exposed so that a check
        which re-derives the windows from the raw Parquet -- ``_raw_persistence_sse`` in
        :mod:`dmf.eval.controls` is the one -- can follow the same population without
        importing this class's internals.
        """
        return self._starts

    @property
    def window_origins(self) -> IntArray:
        """Forecast origin ``s + L - 1`` of each window of one realization, ascending.

        The quantity two arms of different lookback must share for their rows to describe
        the same forecast (P6-D4 item 1).
        """
        return self._origins

    @property
    def is_origin_subset(self) -> bool:
        """Whether this dataset holds a chosen subset of the origins the geometry admits.

        Carried into the provenance of any table scored from it: a matched re-scoring and
        an unmatched one produce identically shaped frames, and only this says which is
        which.
        """
        return self._is_subset

    @property
    def training_series(self) -> FloatArray:
        """Concatenated samples of this partition, shape ``(n_realizations * N, C_motion)``.

        In corpus units, motion channels only -- the sea-state indicator is not part of the
        series and has no statistics to refit. Provided for tests that need to refit
        statistics independently; materialises a float64 copy, so it is not used on the hot
        path.
        """
        return self._data.reshape(-1, self._data.shape[2]).astype(np.float64)


def _worker_init(worker_id: int) -> None:
    """Seed a DataLoader worker deterministically from the loader's base seed.

    Args:
        worker_id: Worker ordinal supplied by torch.
    """
    info = torch.utils.data.get_worker_info()
    base = 0 if info is None else int(info.seed)
    np.random.seed((base + worker_id) % (2**32))


def make_dataloader(
    dataset: DeckMotionDataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    """Wrap a dataset in a seeded DataLoader.

    Shuffling here is safe and expected: it shuffles windows *within* an already-split
    partition, which is not the same operation as shuffling before splitting.

    Args:
        dataset: The dataset to wrap.
        batch_size: Windows per minibatch.
        shuffle: Whether to shuffle within the partition. True for training, False
            elsewhere so that evaluation order is reproducible.
        num_workers: Worker process count.
        seed: Seed for the shuffling generator and worker seeding, so that a run is
            reproducible from its recorded seed.

    Returns:
        The configured DataLoader. ``drop_last`` is False: dropping a partial final batch
        would silently change the window population a metric is averaged over.

    Raises:
        ValueError: If ``batch_size`` is not positive or ``num_workers`` is negative.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        generator=generator,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
