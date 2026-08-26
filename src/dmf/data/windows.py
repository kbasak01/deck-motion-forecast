"""Lookback/horizon windowing of realization time series.

Windows are always cut *within* a single realization. A window never spans a realization
boundary, since consecutive realizations are independent wave fields and a window
straddling them would present a discontinuity as if it were dynamics.
"""

from dataclasses import dataclass

import numpy as np

from dmf.config import DataConfig
from dmf.typedefs import FloatArray, IntArray

__all__ = [
    "WindowSpec",
    "make_windows",
    "n_windows",
    "window_spec_from_config",
    "window_start_indices",
]


@dataclass(frozen=True)
class WindowSpec:
    """Geometry of the supervised windowing task.

    Attributes:
        lookback: Input window length, samples. At ``fs = 10 Hz`` the default 200 is 20 s.
        horizons: Forecast horizons to report, samples, ascending. The model emits
            ``max(horizons)`` steps in one direct multi-horizon output; shorter horizons
            are sliced from that tensor rather than produced by autoregressive rollout,
            which avoids error compounding and keeps the ONNX graph a single fixed shape.
        stride: Step between consecutive window start indices, samples.
    """

    lookback: int
    horizons: tuple[int, ...]
    stride: int

    @property
    def max_horizon(self) -> int:
        """Longest forecast horizon, samples.

        Raises:
            ValueError: If ``horizons`` is empty.
        """
        if not self.horizons:
            raise ValueError("WindowSpec.horizons must be non-empty")
        return max(self.horizons)

    @property
    def total_length(self) -> int:
        """Samples consumed by one window, ``lookback + max_horizon``."""
        return self.lookback + self.max_horizon


def window_spec_from_config(cfg: DataConfig) -> WindowSpec:
    """Derive the window geometry from a task configuration.

    One construction site for the geometry, so the dataset, the split integrity checks and
    :func:`dmf.train.registry.build_model` cannot drift apart on the lookback or the
    horizon set.

    Args:
        cfg: Task configuration. ``lookback``, ``horizons`` and ``stride`` are in samples.

    Returns:
        The window geometry described by ``cfg``.
    """
    return WindowSpec(lookback=cfg.lookback, horizons=tuple(cfg.horizons), stride=cfg.stride)


def n_windows(n_samples: int, spec: WindowSpec) -> int:
    """Count the windows obtainable from a single realization.

    Args:
        n_samples: Length of the realization, samples.
        spec: Window geometry.

    Returns:
        Number of complete windows, zero if the realization is shorter than
        ``spec.total_length``.

    Raises:
        ValueError: If ``spec.stride`` is not positive.
    """
    if spec.stride <= 0:
        raise ValueError(f"window stride must be positive, got {spec.stride}")
    if n_samples < spec.total_length:
        return 0
    return (n_samples - spec.total_length) // spec.stride + 1


def window_start_indices(n_samples: int, spec: WindowSpec) -> IntArray:
    """Compute the start sample index of every window in a realization.

    Returned so that split integrity can be checked at the level of concrete time indices:
    ``tests/test_splits.py`` asserts that no time index appears in both a train window and
    a test window for the same realization.

    Args:
        n_samples: Length of the realization, samples.
        spec: Window geometry.

    Returns:
        Start indices, shape ``(n_windows,)``, ascending.
    """
    count = n_windows(n_samples, spec)
    return np.arange(count, dtype=np.int64) * spec.stride


def make_windows(series: FloatArray, spec: WindowSpec) -> tuple[FloatArray, FloatArray]:
    """Cut one realization into input/target window pairs.

    Args:
        series: One realization's channels, shape ``(n_samples, n_channels)``. Units are
            per-channel as documented in the corpus schema: angles in degrees, angular
            rates in degrees per second, heave in metres.
        spec: Window geometry.

    Returns:
        Tuple ``(x, y)`` where ``x`` has shape ``(n_windows, lookback, n_channels)`` and
        ``y`` has shape ``(n_windows, max_horizon, n_channels)``, both in the input units.
        Target-channel selection happens later, in the dataset, not here.

    Raises:
        ValueError: If ``series`` is not two-dimensional or is shorter than
            ``spec.total_length``.
    """
    if series.ndim != 2:
        raise ValueError(f"series must be 2-D (n_samples, n_channels), got shape {series.shape}")
    n_samples = series.shape[0]
    if n_samples < spec.total_length:
        raise ValueError(
            f"realization of {n_samples} samples is shorter than the window total length "
            f"{spec.total_length} (lookback {spec.lookback} + max horizon {spec.max_horizon})"
        )
    starts = window_start_indices(n_samples, spec)
    lookback_offsets = np.arange(spec.lookback, dtype=np.int64)
    horizon_offsets = spec.lookback + np.arange(spec.max_horizon, dtype=np.int64)
    x: FloatArray = series[starts[:, None] + lookback_offsets[None, :], :]
    y: FloatArray = series[starts[:, None] + horizon_offsets[None, :], :]
    return x, y
