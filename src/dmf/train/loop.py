"""The training loop: AMP, cosine schedule with warmup, early stopping, seed control.

Defaults, from the implementation plan: AdamW at lr 1e-3, cosine schedule with 5 percent
warmup, batch 256, bf16 autocast, gradient clip 1.0, early stopping on validation loss
with patience 15.

Early stopping watches the **validation** partition, which is drawn from training seeds.
The test partition is touched once, after training has finished, by the evaluation code.
"""

from dataclasses import dataclass
from pathlib import Path

from torch import Tensor
from torch.utils.data import DataLoader

from dmf.config import TrainConfig
from dmf.models.base import BaseForecaster

__all__ = ["EarlyStopper", "TrainResult", "fit", "set_seed", "train_one_epoch", "validate"]


@dataclass(frozen=True)
class TrainResult:
    """Outcome of one training run, for one model at one seed.

    Attributes:
        seed: The training seed. Recorded so that a results row can always be traced back
            to the run that produced it.
        best_val_loss: Lowest validation loss reached, dimensionless.
        best_epoch: Epoch at which ``best_val_loss`` occurred, zero-indexed.
        epochs_run: Total epochs completed before early stopping or exhaustion.
        train_losses: Per-epoch training loss, dimensionless.
        val_losses: Per-epoch validation loss, dimensionless.
        checkpoint_path: Path to the saved best-epoch weights.
        wall_time_s: Total training wall time, seconds.
    """

    seed: int
    best_val_loss: float
    best_epoch: int
    epochs_run: int
    train_losses: tuple[float, ...]
    val_losses: tuple[float, ...]
    checkpoint_path: Path
    wall_time_s: float


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed every generator that affects a training run.

    Seeds Python's ``random``, NumPy, and torch (CPU and CUDA).

    Args:
        seed: The seed value.
        deterministic: If True, also enable ``torch.use_deterministic_algorithms`` and the
            cuDNN deterministic path. This costs throughput and makes a few kernels
            unavailable, but these models train in minutes, so exact reproducibility is
            the better trade here.
    """
    raise NotImplementedError


class EarlyStopper:
    """Stop training when validation loss has not improved for ``patience`` epochs."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        """Configure the stopper.

        Args:
            patience: Epochs without improvement to tolerate before stopping.
            min_delta: Minimum decrease in validation loss that counts as an improvement,
                dimensionless.
        """
        raise NotImplementedError

    def update(self, val_loss: float) -> bool:
        """Record an epoch's validation loss and report whether to stop.

        Args:
            val_loss: This epoch's validation loss, dimensionless.

        Returns:
            True if training should stop now.
        """
        raise NotImplementedError

    @property
    def best_loss(self) -> float:
        """Lowest validation loss seen so far, dimensionless."""
        raise NotImplementedError


def train_one_epoch(
    model: BaseForecaster,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    cfg: TrainConfig,
    epoch: int,
    total_epochs: int,
) -> float:
    """Run one training epoch.

    Args:
        model: The model to update, already on the target device.
        loader: Training minibatches of ``(x, y, window_mean)``.
        cfg: Optimisation settings.
        epoch: Zero-indexed epoch number, used to position the cosine schedule.
        total_epochs: Total planned epochs, used to position the cosine schedule.

    Returns:
        Mean training loss over the epoch, dimensionless.
    """
    raise NotImplementedError


def validate(
    model: BaseForecaster,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    cfg: TrainConfig,
) -> float:
    """Evaluate the model on the validation partition.

    Args:
        model: The model to evaluate.
        loader: Validation minibatches of ``(x, y, window_mean)``.
        cfg: Optimisation settings, for the autocast dtype.

    Returns:
        Mean validation loss, dimensionless.
    """
    raise NotImplementedError


def fit(
    model: BaseForecaster,
    train_loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    cfg: TrainConfig,
    seed: int,
    checkpoint_dir: Path,
) -> TrainResult:
    """Train one model at one seed to convergence or early stop.

    Args:
        model: The model to train.
        train_loader: Training minibatches.
        val_loader: Validation minibatches.
        cfg: Optimisation settings.
        seed: Training seed, passed to :func:`set_seed` and recorded in the result.
        checkpoint_dir: Directory for best-epoch weights. Lives under ``artifacts/`` and
            is gitignored.

    Returns:
        The run's outcome, including the loss curves and checkpoint path.
    """
    raise NotImplementedError
