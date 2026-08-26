"""IMU observability model.

Real deck sensors report angular rates and linear accelerations, not absolute heave. A
forecaster evaluated only on clean roll/pitch/heave is being handed information that no
shipboard system actually has, so the corpus supports two observation modes:

- ``ideal`` -- clean ``roll``, ``pitch``, ``heave`` straight from the simulator.
- ``imu`` -- attitude corrupted by white noise plus a slow bias random walk, and heave
  reconstructed by double-integrating vertical acceleration through a second-order
  high-pass at 0.03 Hz. That high-pass is what real heave-compensation systems use, and it
  distorts exactly the low-frequency content a forecaster would most like to have. The
  distortion is mostly phase, not amplitude: see the table in
  :func:`heave_from_vertical_acc`, and the warning there that an ``imu`` input must never
  be scored against an ``ideal`` target.

Phase 6 runs both modes as an ablation. The gap between them is the honest statement of
how much of the ``ideal`` result survives contact with real instrumentation.

Units: attitude in **degrees**, angular rate in **degrees per second**, heave in
**metres**, heave rate in metres per second, vertical acceleration in metres per second
squared, sampling rate and filter cutoffs in **hertz** (this module and
:attr:`dmf.config.SimConfig.fs_hz` are the only places Hz appears; the wave physics is
rad/s throughout).

The module is pure NumPy: the second-order high-pass is a hand-rolled direct-form-I biquad
rather than a ``scipy.signal`` call, and ``tests/test_generate.py`` checks its coefficients
and its output against ``scipy.signal.butter``/``lfilter`` to machine precision.
"""

import numpy as np
import pandas as pd

from dmf.config import ObservationMode
from dmf.typedefs import FloatArray

__all__ = [
    "ATTITUDE_BIAS_RW_DEG_PER_SQRT_S",
    "ATTITUDE_NOISE_SIGMA_DEG",
    "GYRO_NOISE_SIGMA_DPS",
    "HEAVE_HIGHPASS_HZ",
    "IDEAL_COLUMNS",
    "IMU_COLUMNS",
    "add_attitude_noise",
    "apply_observation_model",
    "bias_random_walk",
    "heave_from_vertical_acc",
    "highpass_biquad",
]

#: White measurement-noise standard deviation on roll and pitch, **degrees**. The plan's
#: figure, representative of a tactical-grade AHRS in a benign vibration environment.
ATTITUDE_NOISE_SIGMA_DEG: float = 0.02

#: Attitude bias random-walk intensity, **degrees per square-root second**. Not given in
#: the plan, which says only "slow bias random walk". Chosen so the bias wanders by about
#: 0.05 deg over one 600 s record (``0.002*sqrt(600) = 0.049``): the same order as the
#: white noise, but concentrated at frequencies far below the wave band, which is what
#: makes it a distinct error source rather than more white noise.
ATTITUDE_BIAS_RW_DEG_PER_SQRT_S: float = 0.002

#: White noise standard deviation on the gyro rate channels, **degrees per second**. Also
#: not specified in the plan. Set equal in magnitude to the attitude noise figure; a real
#: MEMS gyro's angle random walk is smaller than this, so the rate channels are, if
#: anything, treated slightly pessimistically.
GYRO_NOISE_SIGMA_DPS: float = 0.02

#: High-pass cutoff used at each integration stage of the heave reconstruction, **hertz**.
HEAVE_HIGHPASS_HZ: float = 0.03

#: Clean motion channels required on an input frame, in corpus order.
IDEAL_COLUMNS: tuple[str, ...] = (
    "roll",
    "pitch",
    "heave",
    "roll_rate",
    "pitch_rate",
    "heave_rate",
    "heave_acc",
)

#: Channels the ``imu`` observation mode produces, aligned one-to-one with the first six
#: entries of :data:`IDEAL_COLUMNS`. ``heave_acc`` has no ``_imu`` counterpart: it is what
#: the accelerometer measures directly, so both modes see the same channel.
IMU_COLUMNS: tuple[str, ...] = (
    "roll_imu",
    "pitch_imu",
    "heave_imu",
    "roll_rate_imu",
    "pitch_rate_imu",
    "heave_rate_imu",
)


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
    if sigma_deg < 0.0:
        raise ValueError(f"sigma_deg must be non-negative, got {sigma_deg}")
    clean = np.asarray(angle_deg, dtype=np.float64)
    if sigma_deg == 0.0:
        return np.array(clean, dtype=np.float64)
    return np.asarray(clean + rng.normal(0.0, sigma_deg, clean.shape), dtype=np.float64)


def bias_random_walk(
    n_samples: int,
    sigma_deg_per_sqrt_s: float,
    fs_hz: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Generate a slow bias random walk to superimpose on an attitude channel.

    The increment per sample has standard deviation ``sigma_deg_per_sqrt_s*sqrt(dt)`` with
    ``dt = 1/fs_hz`` seconds, so the accumulated bias has standard deviation
    ``sigma_deg_per_sqrt_s*sqrt(t)`` degrees at elapsed time ``t`` seconds, independently
    of the sampling rate.

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
    if n_samples < 1:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if fs_hz <= 0.0:
        raise ValueError(f"fs_hz must be strictly positive, got {fs_hz}")
    if sigma_deg_per_sqrt_s < 0.0:
        raise ValueError(f"sigma_deg_per_sqrt_s must be non-negative, got {sigma_deg_per_sqrt_s}")
    steps = rng.normal(0.0, sigma_deg_per_sqrt_s * np.sqrt(1.0 / fs_hz), n_samples)
    # The first sample is the zero starting bias; the walk begins on the second sample.
    steps[0] = 0.0
    return np.asarray(np.cumsum(steps), dtype=np.float64)


def highpass_biquad(x: FloatArray, cutoff_hz: float, fs_hz: float) -> FloatArray:
    """Apply a second-order Butterworth high-pass filter.

    Applied forward-only (not zero-phase), because a causal filter is what a shipboard
    heave-compensation unit can actually run. Using ``filtfilt`` here would remove the
    filter's phase lag and quietly make the ``imu`` mode easier than reality.

    The coefficients are the bilinear transform of the analogue second-order Butterworth
    high-pass with pre-warped cutoff ``K = tan(pi*cutoff_hz/fs_hz)``::

        norm = 1 / (1 + sqrt(2)*K + K**2)
        b = [norm, -2*norm, norm]
        a = [1, 2*(K**2 - 1)*norm, (1 - sqrt(2)*K + K**2)*norm]

    which reproduces ``scipy.signal.butter(2, cutoff_hz/(fs_hz/2), "highpass")`` exactly;
    the recursion below reproduces ``scipy.signal.lfilter`` on those coefficients to
    machine precision. Both are asserted in ``tests/test_generate.py``. It is written out
    here so that :mod:`dmf.sim` keeps its pure-NumPy, dependency-light contract.

    Initial conditions are zero, so the output carries a startup transient lasting a few
    time constants (``1/(2*pi*0.03) = 5.3 s`` at the corpus cutoff). The corpus applies the
    observation model to the full synthesised record and discards the 120 s spin-up
    afterwards, so no retained sample is inside that transient.

    Args:
        x: Input series, shape ``(n_samples,)``. Units are arbitrary and preserved.
        cutoff_hz: Cutoff frequency, hertz. The corpus default is 0.03.
        fs_hz: Sampling rate, hertz.

    Returns:
        Filtered series, same shape and units as ``x``.

    Raises:
        ValueError: If ``cutoff_hz`` is not in ``(0, fs_hz/2)``.
    """
    if fs_hz <= 0.0:
        raise ValueError(f"fs_hz must be strictly positive, got {fs_hz}")
    if not 0.0 < cutoff_hz < fs_hz / 2.0:
        raise ValueError(
            f"cutoff_hz must lie strictly inside (0, fs_hz/2) = (0, {fs_hz / 2.0}), got {cutoff_hz}"
        )
    series = np.asarray(x, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {series.shape}")

    k = float(np.tan(np.pi * cutoff_hz / fs_hz))
    norm = 1.0 / (1.0 + np.sqrt(2.0) * k + k * k)
    b0 = norm
    b1 = -2.0 * norm
    b2 = norm
    a1 = 2.0 * (k * k - 1.0) * norm
    a2 = (1.0 - np.sqrt(2.0) * k + k * k) * norm

    out = np.empty(series.size, dtype=np.float64)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(series.size):
        xi = float(series[i])
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, xi
        y2, y1 = y1, yi
        out[i] = yi
    return out


def _integrate_highpass(x: FloatArray, fs_hz: float, cutoff_hz: float) -> FloatArray:
    """Trapezoidally integrate one stage and high-pass the result.

    The high-pass follows the integration rather than preceding it, because it is the
    integration that manufactures the drift: any residual offset in the integrand becomes
    a ramp, and the ramp is what has to be removed.

    Args:
        x: Integrand, shape ``(n_samples,)``, arbitrary units ``U``.
        fs_hz: Sampling rate, hertz.
        cutoff_hz: High-pass cutoff, hertz.

    Returns:
        The high-passed integral, units ``U`` seconds, shape ``(n_samples,)``, starting at
        the filter's zero initial condition.
    """
    series = np.asarray(x, dtype=np.float64)
    dt = 1.0 / fs_hz
    integral = np.empty(series.size, dtype=np.float64)
    integral[0] = 0.0
    np.cumsum(0.5 * (series[1:] + series[:-1]) * dt, out=integral[1:])
    return highpass_biquad(integral, cutoff_hz, fs_hz)


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

    Concretely, ``disp = HP(integral(HP(integral(acc))))``. The two high-pass stages are
    nearly transparent in **magnitude** across the wave band, but not in **phase**, and the
    phase is the larger effect:

    ======== ================ =====================  ==================
    f (Hz)    combined |H|      combined phase         equivalent lead
    ======== ================ =====================  ==================
    0.050     0.885            +106 deg               +5.9 s
    0.080     0.981            +63 deg                +2.2 s
    0.103     0.993            +48 deg                +1.3 s
    0.200     1.000            +24 deg                +0.3 s
    ======== ================ =====================  ==================

    So at the SS5 spectral peak (0.103 Hz) the reconstruction has essentially the right
    amplitude while **leading** true heave by about 1.3 s, and the lead grows towards low
    frequency. That is what a causal high-pass does and is exactly the distortion the
    ``imu`` mode exists to expose; it is not tuned away.

    Two consequences worth carrying forward. First, the lead is frequency-dependent, so it
    is a dispersive distortion rather than a clean time shift and cannot be undone by
    shifting the series. Second, and more important for evaluation: because
    ``heave_imu`` leads ``heave``, a model given ``heave_imu`` as an **input** while scored
    against clean ``heave`` as a **target** would be handed part of the answer. The
    ``imu`` ablation must forecast the ``imu`` channels themselves, exactly as a shipboard
    system would, and never mix an ``imu`` input with an ``ideal`` target.

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
    velocity = _integrate_highpass(np.asarray(acc_m_s2, dtype=np.float64), fs_hz, cutoff_hz)
    return _integrate_highpass(velocity, fs_hz, cutoff_hz)


def apply_observation_model(
    df: pd.DataFrame,
    mode: ObservationMode,
    fs_hz: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Apply the requested observation model to one realization.

    In ``imu`` mode the six :data:`IMU_COLUMNS` **replace** the corresponding clean
    channels in the returned frame, so that the two modes are interchangeable at the
    column level. The corpus writer keeps both sets side by side by calling this function
    once and renaming, which is why this is the single source of truth for the ``_imu``
    channels: there is no second code path that could drift from it.

    What each channel gets:

    ============= ==================================================================
    Channel        Corruption
    ============= ==================================================================
    roll, pitch    white noise (0.02 deg) + attitude bias random walk
    heave          discarded and rebuilt from ``heave_acc``, see
                   :func:`heave_from_vertical_acc`
    roll_rate,     white gyro noise, no bias walk -- the attitude bias walk already
    pitch_rate     stands in for the integrated gyro bias
    heave_rate     the intermediate stage of the same double integration, so heave and
                   heave rate stay mutually consistent
    heave_acc      unchanged; it is the raw accelerometer channel
    ============= ==================================================================

    The accelerometer channel itself is left noise-free. The plan specifies noise on
    attitude only, and adding an accelerometer noise floor here would double-count: its
    dominant effect after two integrations is precisely the low-frequency drift the
    high-pass is already there to remove.

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
    if mode not in ("ideal", "imu"):
        raise ValueError(f"unknown observation mode {mode!r}, expected 'ideal' or 'imu'")
    missing = [name for name in ("t", *IDEAL_COLUMNS) if name not in df.columns]
    if missing:
        raise ValueError(f"observation model needs missing columns {missing}")

    out = df.copy()
    if mode == "ideal":
        return out

    n_samples = len(out)
    for channel in ("roll", "pitch"):
        clean = np.asarray(out[channel], dtype=np.float64)
        noisy = add_attitude_noise(clean, ATTITUDE_NOISE_SIGMA_DEG, rng)
        bias = bias_random_walk(n_samples, ATTITUDE_BIAS_RW_DEG_PER_SQRT_S, fs_hz, rng)
        out[channel] = noisy + bias
    for channel in ("roll_rate", "pitch_rate"):
        clean = np.asarray(out[channel], dtype=np.float64)
        out[channel] = add_attitude_noise(clean, GYRO_NOISE_SIGMA_DPS, rng)

    acc = np.asarray(out["heave_acc"], dtype=np.float64)
    velocity = _integrate_highpass(acc, fs_hz, HEAVE_HIGHPASS_HZ)
    out["heave_rate"] = velocity
    out["heave"] = _integrate_highpass(velocity, fs_hz, HEAVE_HIGHPASS_HZ)
    return out
