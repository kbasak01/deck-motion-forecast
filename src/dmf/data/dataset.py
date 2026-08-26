"""Torch ``Dataset`` and ``DataLoader`` construction over the simulated corpus.

The dataset is constructed **from an already-decided split**. It takes a
:class:`dmf.data.splits.Split` and a partition name, not a corpus and a fraction, so that
there is no code path in which this module could shuffle-then-split.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from dmf.config import DataConfig, ObservationMode
from dmf.data.normalize import NormStats, build_norm_stats
from dmf.data.splits import RealizationKey, Split
from dmf.data.windows import WindowSpec, n_windows, window_start_indices
from dmf.sim.generate import RealizationSpec, realization_path
from dmf.sim.imu import IDEAL_COLUMNS, IMU_COLUMNS
from dmf.typedefs import FloatArray

__all__ = ["DeckMotionDataset", "PARTITIONS", "make_dataloader", "resolve_columns"]

#: Valid partition names. ``val`` and ``test`` always require training statistics.
PARTITIONS: tuple[str, ...] = ("train", "val", "test")

#: Logical channel name -> ``imu`` corpus column, for the six channels that have a noisy
#: twin. ``heave_acc`` is deliberately absent: it is the raw accelerometer channel and is
#: shared by both observation modes (``dmf.sim.imu``).
_IMU_BY_IDEAL: dict[str, str] = dict(zip(IDEAL_COLUMNS[:6], IMU_COLUMNS, strict=True))


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
    return tuple(_IMU_BY_IDEAL.get(n, n) for n in names)


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
    """

    def __init__(
        self,
        corpus_root: Path,
        split: Split,
        partition: str,
        cfg: DataConfig,
        window_spec: WindowSpec,
        stats: NormStats | None = None,
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
                from the training realizations.

        Raises:
            ValueError: If ``partition`` is unrecognised, or if ``partition`` is not
                ``"train"`` and ``stats`` is None -- constructing a validation or test set
                without training statistics is a leak, so it is refused rather than
                silently recomputed. Also if the partition is empty, if its realizations
                differ in length, or if a realization's Parquet file is missing.
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
        self._input_columns = resolve_columns(cfg.input_channels, cfg.observation_mode)
        self._target_columns = resolve_columns(cfg.target_dofs, cfg.observation_mode)
        self._target_index = [self._input_columns.index(c) for c in self._target_columns]

        self._data = self._load(corpus_root)
        self._n_samples = int(self._data.shape[1])
        self._per_realization = n_windows(self._n_samples, window_spec)
        if self._per_realization == 0:
            raise ValueError(
                f"realizations of {self._n_samples} samples yield no windows at "
                f"lookback {window_spec.lookback} + max horizon {window_spec.max_horizon}"
            )
        self._starts = window_start_indices(self._n_samples, window_spec)
        self._stats = stats if stats is not None else self._fit_stats()

    def _load(self, corpus_root: Path) -> npt.NDArray[np.float32]:
        """Read the partition's realizations into one contiguous float32 array.

        Args:
            corpus_root: Dataset root.

        Returns:
            Array of shape ``(n_realizations, n_samples, C_in)``, corpus units.

        Raises:
            ValueError: If a file is missing or realizations differ in length.
        """
        columns = list(self._input_columns)
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
        materialising a float64 copy of the whole partition.

        Returns:
            Statistics labelled ``"<regime>/train"``.
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
            channels=self._input_columns,
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

        Returns:
            Tuple ``(x, y, window_mean)`` with shapes ``(L, C_in)``, ``(H, C_out)`` and
            ``(1, C_out)``. ``x`` is de-meaned and scaled (dimensionless); ``y`` and
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
        """Corpus column names of the input channels, after observation-mode resolution."""
        return self._input_columns

    @property
    def corpus_root(self) -> Path:
        """Dataset root the realizations were read from."""
        return self._corpus_root

    @property
    def windows_per_realization(self) -> int:
        """Windows cut from each realization, identical across the partition."""
        return self._per_realization

    @property
    def training_series(self) -> FloatArray:
        """Concatenated samples of this partition, shape ``(n_realizations * N, C_in)``.

        In corpus units. Provided for tests that need to refit statistics independently;
        materialises a float64 copy, so it is not used on the hot path.
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
