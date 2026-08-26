"""IMU observability model.

Real deck sensors report angular rates and linear accelerations, not absolute heave. A
forecaster evaluated only on clean roll/pitch/heave is being handed information that no
shipboard system actually has, so the corpus supports two observation modes:

- ``ideal`` -- clean ``roll``, ``pitch``, ``heave`` straight from the simulator.
- ``imu`` -- attitude corrupted by white noise plus a slow bias random walk, and heave
  reconstructed by double-integrating vertical acceleration through a second-order
  high-pass at 0.03 Hz. That high-pass is what real heave-compensation systems use, and it
  distorts exactly the low-frequency content a forecaster would most like to have.

Phase 6 runs both modes as an ablation. The gap between them is the honest statement of
how much of the ``ideal`` result survives contact with real instrumentation.
"""

import numpy as np
import pandas as pd

from dmf.config import ObservationMode
from dmf.typedefs import FloatArray

__all__ = [
    "add_attitude_noise",
    "apply_observation_model",
    "bias_random_walk",
    "heave_from_vertical_acc",
    "highpass_biquad",
]


def add_attitude_noise(
    angle_deg: FloatArray,
    sigma_deg: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Add zero-mean white measurement noise to an attitude channel.

    Args:
        angle_deg: Clean attitude series, **degrees**, shape ``(n_samples,)``.
        sigma_deg: Noise standard deviation, degrees. The corpus default is 0.02, which is
            representative of a tactical-grade AHRS in a benign vibration environment.
        rng: Seeded generator.

    Returns:
        Noisy attitude series, degrees, shape ``(n_samples,)``.

    Raises:
        ValueError: If ``sigma_deg`` is negative.
    """
    raise NotImplementedError


def bias_random_walk(
    n_samples: int,
    sigma_deg_per_sqrt_s: float,
    fs_hz: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Generate a slow bias random walk to superimpose on an attitude channel.

    Args:
        n_samples: Number of samples to generate.
        sigma_deg_per_sqrt_s: Random-walk intensity, degrees per square-root second.
        fs_hz: Sampling rate, hertz.
        rng: Seeded generator.

    Returns:
        Bias series, **degrees**, shape ``(n_samples,)``, starting at zero.

    Raises:
        ValueError: If ``n_samples`` is not positive or ``fs_hz`` is not strictly
            positive.
    """
    raise NotImplementedError


def highpass_biquad(x: FloatArray, cutoff_hz: float, fs_hz: float) -> FloatArray:
    """Apply a second-order Butterworth high-pass filter.

    Applied forward-only (not zero-phase), because a causal filter is what a shipboard
    heave-compensation unit can actually run. Using ``filtfilt`` here would remove the
    filter's phase lag and quietly make the ``imu`` mode easier than reality.

    Args:
        x: Input series, shape ``(n_samples,)``. Units are arbitrary and preserved.
        cutoff_hz: Cutoff frequency, hertz. The corpus default is 0.03.
        fs_hz: Sampling rate, hertz.

    Returns:
        Filtered series, same shape and units as ``x``.

    Raises:
        ValueError: If ``cutoff_hz`` is not in ``(0, fs_hz/2)``.
    """
    raise NotImplementedError


def heave_from_vertical_acc(
    acc_m_s2: FloatArray,
    fs_hz: float,
    cutoff_hz: float,
) -> FloatArray:
    """Reconstruct heave displacement from vertical acceleration.

    Double-integrates with a high-pass applied at each integration stage to suppress the
    drift that unbounded integration of a biased accelerometer produces. The resulting
    series has attenuated and phase-shifted low-frequency content relative to true heave;
    that distortion is the point of the ``imu`` mode, not a defect to be tuned away.

    Args:
        acc_m_s2: Vertical acceleration series, metres per second squared, shape
            ``(n_samples,)``.
        fs_hz: Sampling rate, hertz.
        cutoff_hz: High-pass cutoff applied at each integration stage, hertz.

    Returns:
        Reconstructed heave displacement, **metres**, shape ``(n_samples,)``.

    Raises:
        ValueError: If ``cutoff_hz`` is not in ``(0, fs_hz/2)``.
    """
    raise NotImplementedError


def apply_observation_model(
    df: pd.DataFrame,
    mode: ObservationMode,
    fs_hz: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Apply the requested observation model to one realization.

    Args:
        df: One realization's clean trajectory, with columns ``t``, ``roll``, ``pitch``,
            ``heave``, ``roll_rate``, ``pitch_rate``, ``heave_rate``, ``heave_acc``.
            Angles in degrees, angular rates in degrees per second, heave in metres,
            heave rate in metres per second, heave acceleration in metres per second
            squared.
        mode: ``"ideal"`` returns ``df`` unchanged; ``"imu"`` applies the noise, bias and
            high-pass reconstruction described in the module docstring.
        fs_hz: Sampling rate, hertz.
        rng: Seeded generator, so that the observation model is reproducible from the
            realization seed.

    Returns:
        A new DataFrame with the same columns and units. The original is not modified.

    Raises:
        ValueError: If a required column is missing or ``mode`` is unrecognised.
    """
    raise NotImplementedError
