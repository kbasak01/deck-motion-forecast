"""Torch ``Dataset`` and ``DataLoader`` construction over the simulated corpus.

The dataset is constructed **from an already-decided split**. It takes a
:class:`dmf.data.splits.Split` and a partition name, not a corpus and a fraction, so that
there is no code path in which this module could shuffle-then-split.
"""

from pathlib import Path

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from dmf.config import DataConfig
from dmf.data.normalize import NormStats
from dmf.data.splits import Split
from dmf.data.windows import WindowSpec

__all__ = ["DeckMotionDataset", "make_dataloader"]


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
                silently recomputed.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of windows in this partition.

        Returns:
            Window count, summed over the partition's realizations.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @property
    def norm_stats(self) -> NormStats:
        """Normalisation statistics in use, for propagation to val and test sets."""
        raise NotImplementedError


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
        The configured DataLoader.
    """
    raise NotImplementedError
