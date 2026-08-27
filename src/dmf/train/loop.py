"""The training loop: AMP, cosine schedule with warmup, early stopping, seed control.

Defaults, from the implementation plan: AdamW at lr 1e-3, cosine schedule with 5 percent
warmup, batch 256, bf16 autocast, gradient clip 1.0, early stopping on validation loss
with patience 15.

Early stopping watches the **validation** partition, which is drawn from training seeds.
The test partition is touched once, after training has finished, by the evaluation code.

**The loss is dimensionless; the loader is not.** ``DeckMotionDataset`` yields ``y`` in
corpus units and the per-window mean beside it, because metrics must be computed in degrees
and metres. A model's output is dimensionless. The conversion therefore has to happen
inside the loop, and the loop has no ``NormStats`` in its signature -- so :func:`_target_scale`
recovers them from the loader's dataset. Doing it the other way round (converting the
*prediction* into corpus units and taking the loss there) would weight roll, pitch and heave
by their physical variances, making the loss a different objective for every channel set.
"""

import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from dmf.config import TrainConfig
from dmf.data.dataset import DeckMotionDataset
from dmf.data.normalize import NormStats, normalize_target
from dmf.models.base import BaseForecaster
from dmf.train.losses import mse_loss

__all__ = ["EarlyStopper", "TrainResult", "fit", "set_seed", "train_one_epoch", "validate"]

#: Autocast dtype by config name. ``off`` disables autocast entirely.
_AMP_DTYPES: dict[str, torch.dtype | None] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "off": None,
}


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
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuBLAS needs this set before the first handle is created, or the deterministic
        # GEMM path raises at the first matmul rather than at the switch.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopper:
    """Stop training when validation loss has not improved for ``patience`` epochs."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        """Configure the stopper.

        Args:
            patience: Epochs without improvement to tolerate before stopping.
            min_delta: Minimum decrease in validation loss that counts as an improvement,
                dimensionless.

        Raises:
            ValueError: If ``patience`` is negative or ``min_delta`` is negative.
        """
        if patience < 0:
            raise ValueError(f"patience must be non-negative, got {patience}")
        if min_delta < 0.0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")
        self.patience = patience
        self.min_delta = min_delta
        self._best = math.inf
        self._since_improvement = 0
        self.best_epoch = -1
        self._epoch = -1

    def update(self, val_loss: float) -> bool:
        """Record an epoch's validation loss and report whether to stop.

        Args:
            val_loss: This epoch's validation loss, dimensionless.

        Returns:
            True if training should stop now.
        """
        self._epoch += 1
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self.best_epoch = self._epoch
            self._since_improvement = 0
            return False
        self._since_improvement += 1
        return self._since_improvement > self.patience

    @property
    def best_loss(self) -> float:
        """Lowest validation loss seen so far, dimensionless."""
        return self._best


def _target_scale(loader: DataLoader[tuple[Tensor, Tensor, Tensor]]) -> NormStats | None:
    """Recover the target-channel normalisation statistics from a loader.

    ``train_one_epoch`` and ``validate`` are handed a loader, not a ``NormStats``, but the
    loader yields ``y`` in corpus units while the loss must be dimensionless. Rather than
    widen the public signature -- which every future model's training path would then have
    to thread -- the statistics are read back off the dataset, behind an ``isinstance``
    check so that a plain tensor dataset in a unit test still works.

    Args:
        loader: The loader to inspect.

    Returns:
        Statistics over the **target** channels, or None if the dataset does not carry
        any, in which case the targets are taken to be dimensionless already.
    """
    dataset = loader.dataset
    if isinstance(dataset, DeckMotionDataset):
        return dataset.norm_stats.subset(dataset.target_columns)
    return None


def _normalised_target(y: Tensor, window_mean: Tensor, stats: NormStats | None) -> Tensor:
    """Put a loader's target into the model's dimensionless space.

    Args:
        y: Targets, shape ``(B, H, C_out)``, corpus units if ``stats`` is given.
        window_mean: Per-window means, shape ``(B, 1, C_out)``, corpus units.
        stats: Target-channel statistics, or None if ``y`` is already dimensionless.

    Returns:
        Targets, shape ``(B, H, C_out)``, dimensionless.
    """
    if stats is None:
        return y
    return normalize_target(y, stats, window_mean)


def _lr_at(step: int, total_steps: int, cfg: TrainConfig) -> float:
    """Return the learning rate at a global optimisation step.

    Linear warmup over ``warmup_frac`` of the run, then cosine decay to zero.

    Args:
        step: Zero-indexed global step.
        total_steps: Total planned steps across the whole run.
        cfg: Optimisation settings.

    Returns:
        Learning rate for this step.
    """
    total = max(total_steps, 1)
    warmup = int(cfg.warmup_frac * total)
    if warmup > 0 and step < warmup:
        return cfg.lr * float(step + 1) / float(warmup)
    progress = (step - warmup) / max(total - warmup, 1)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def _autocast(device: torch.device, cfg: TrainConfig) -> torch.autocast:
    """Return the autocast context for a device and config.

    Args:
        device: The device the model lives on.
        cfg: Optimisation settings, supplying ``amp_dtype``.

    Returns:
        An autocast context manager; disabled when ``amp_dtype`` is ``"off"`` or the
        device is CPU (CPU autocast buys nothing here and changes the numbers).
    """
    dtype = _AMP_DTYPES[cfg.amp_dtype]
    enabled = dtype is not None and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=dtype or torch.float32, enabled=enabled)


def train_one_epoch(
    model: BaseForecaster,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    cfg: TrainConfig,
    epoch: int,
    total_epochs: int,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Run one training epoch.

    Args:
        model: The model to update, already on the target device.
        loader: Training minibatches of ``(x, y, window_mean)``.
        cfg: Optimisation settings.
        epoch: Zero-indexed epoch number, used to position the cosine schedule.
        total_epochs: Total planned epochs, used to position the cosine schedule.
        optimizer: The optimiser to step. Keyword-only and defaulted so the documented
            positional signature still calls, but :func:`fit` always supplies one:
            constructing a fresh AdamW here would reset the moment estimates every epoch,
            which looks like a mildly worse learning curve rather than like a bug.
        scaler: Gradient scaler, required only for ``fp16`` autocast.

    Returns:
        Mean training loss over the epoch, dimensionless.
    """
    device = next(model.parameters()).device
    opt = (
        optimizer
        if optimizer is not None
        else torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    )
    stats = _target_scale(loader)
    steps_per_epoch = max(len(loader), 1)
    total_steps = steps_per_epoch * max(total_epochs, 1)

    model.train()
    total_loss = 0.0
    n_batches = 0
    for i, (x, y, window_mean) in enumerate(loader):
        lr = _lr_at(epoch * steps_per_epoch + i, total_steps, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        window_mean = window_mean.to(device, non_blocking=True)
        target = _normalised_target(y, window_mean, stats)
        opt.zero_grad(set_to_none=True)
        with _autocast(device, cfg):
            loss = mse_loss(model(x).float(), target.float())
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        total_loss += float(loss.detach())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def validate(
    model: BaseForecaster,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    cfg: TrainConfig,
) -> float:
    """Evaluate the model on the validation partition.

    The reported loss is a **window-weighted** mean, not a mean over batches: the loader
    does not drop its partial final batch, so a batch mean would silently over-weight the
    tail. Early stopping compares these numbers across epochs, so the weighting has to be
    stable.

    Args:
        model: The model to evaluate.
        loader: Validation minibatches of ``(x, y, window_mean)``.
        cfg: Optimisation settings, for the autocast dtype.

    Returns:
        Mean validation loss, dimensionless.
    """
    device = next(model.parameters()).device
    stats = _target_scale(loader)
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, y, window_mean in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            window_mean = window_mean.to(device, non_blocking=True)
            target = _normalised_target(y, window_mean, stats)
            with _autocast(device, cfg):
                loss = mse_loss(model(x).float(), target.float())
            n = int(y.shape[0])
            total += float(loss.detach()) * n
            count += n
    return total / max(count, 1)


def fit(
    model: BaseForecaster,
    train_loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    cfg: TrainConfig,
    seed: int,
    checkpoint_dir: Path,
) -> TrainResult:
    """Train one model at one seed to convergence or early stop.

    The early-stopping criterion -- validation MSE in normalised space, patience from
    ``cfg`` -- is deliberately identical for every model in the project. Gate 4 compares
    architectures, and a comparison in which one model was stopped on a different rule is
    not a comparison of architectures.

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
    set_seed(seed)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(
        device.type, enabled=cfg.amp_dtype == "fp16" and device.type == "cuda"
    )
    stopper = EarlyStopper(cfg.patience)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{type(model).__name__.lower()}_seed{seed}.pt"

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    started = time.perf_counter()
    for epoch in range(cfg.epochs):
        train_losses.append(
            train_one_epoch(
                model, train_loader, cfg, epoch, cfg.epochs, optimizer=optimizer, scaler=scaler
            )
        )
        val_loss = validate(model, val_loader, cfg)
        val_losses.append(val_loss)
        improved = val_loss < stopper.best_loss
        should_stop = stopper.update(val_loss)
        if improved:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if should_stop:
            break
    elapsed = time.perf_counter() - started

    model.load_state_dict(best_state)
    torch.save(best_state, checkpoint_path)
    model.eval()
    return TrainResult(
        seed=seed,
        best_val_loss=stopper.best_loss,
        best_epoch=stopper.best_epoch,
        epochs_run=len(val_losses),
        train_losses=tuple(train_losses),
        val_losses=tuple(val_losses),
        checkpoint_path=checkpoint_path,
        wall_time_s=elapsed,
    )
