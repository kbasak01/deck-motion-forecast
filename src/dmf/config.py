"""Configuration dataclasses and YAML loading.

Every tunable in this project lives in a YAML file under ``configs/`` and is loaded into
one of the frozen dataclasses below. Model and simulation code must never contain a magic
number; if a value could plausibly be changed in an experiment, it belongs here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "ObservationMode",
    "SeaState",
    "SimConfig",
    "TrainConfig",
    "load_experiment",
    "load_yaml",
]

#: Observation model applied to the simulated motion before it reaches the forecaster.
#: ``ideal`` exposes clean roll/pitch/heave; ``imu`` exposes noisy attitude plus heave
#: reconstructed from vertical acceleration through a high-pass filter.
ObservationMode = Literal["ideal", "imu"]


@dataclass(frozen=True)
class SeaState:
    """A single JONSWAP sea state.

    Attributes:
        name: Short label, e.g. ``"SS5"``. Used as a Parquet metadata column value.
        hs_m: Significant wave height, metres.
        tp_s: Spectral peak period, seconds.
        gamma: JONSWAP peak-enhancement factor, dimensionless (1.0 recovers
            Pierson-Moskowitz; 3.3 is the standard North Sea value).
    """

    name: str
    hs_m: float
    tp_s: float
    gamma: float


@dataclass(frozen=True)
class SimConfig:
    """Corpus-generation settings for the wave and vessel-response simulator.

    Attributes:
        sea_states: Sea states to simulate, one per corpus grid cell axis value.
        headings_deg: Encounter angles, degrees, where 180 is head seas, 90 beam seas,
            and 0 following seas.
        speeds_kn: Forward speeds, knots.
        vessels: Vessel config file stems under ``configs/sim/vessels/``.
        seeds_per_cell: Number of independent realizations per
            (sea state, heading, speed, vessel) cell.
        duration_s: Retained record length per realization, seconds, after spin-up.
        spinup_s: Leading transient discarded from each record, seconds.
        fs_hz: Sampling rate of the stored record, hertz.
        n_components: Number of wave components in the random-phase synthesis.
        w_min_rad_s: Lower bound of the synthesis frequency band, radians per second.
        w_max_rad_s: Upper bound of the synthesis frequency band, radians per second.
        jitter_frequencies: If True, draw each component frequency uniformly inside its
            bin instead of using the bin centre. Must be True for any corpus used to
            train or evaluate a forecaster: a uniform grid makes the record periodic with
            period ``2*pi/dw``, which the forecaster memorises.
    """

    sea_states: tuple[SeaState, ...]
    headings_deg: tuple[float, ...]
    speeds_kn: tuple[float, ...]
    vessels: tuple[str, ...]
    seeds_per_cell: int
    duration_s: float
    spinup_s: float
    fs_hz: float
    n_components: int
    w_min_rad_s: float
    w_max_rad_s: float
    jitter_frequencies: bool


@dataclass(frozen=True)
class DataConfig:
    """Windowing, channel selection, and observation settings for the learning task.

    Attributes:
        fs_hz: Sampling rate of the corpus, hertz. Must match ``SimConfig.fs_hz``.
        lookback: Input window length, samples.
        horizons: Forecast horizons to report, samples. The model always emits
            ``max(horizons)`` steps; shorter horizons are sliced from that tensor.
        target_dofs: Names of the forecast target channels, e.g.
            ``("roll", "pitch", "heave")``. Angles in degrees, heave in metres.
        input_channels: Names of the model input channels. Angles in degrees, angular
            rates in degrees per second, heave in metres, heave rate in metres per second.
        stride: Step between consecutive window start indices, samples.
        observation_mode: Observation model applied before windowing.
        revin: If True, apply RevIN instead of plain per-window de-meaning.
    """

    fs_hz: float
    lookback: int
    horizons: tuple[int, ...]
    target_dofs: tuple[str, ...]
    input_channels: tuple[str, ...]
    stride: int
    observation_mode: ObservationMode
    revin: bool


@dataclass(frozen=True)
class ModelConfig:
    """Model selection and architecture hyperparameters.

    Attributes:
        name: Registry key, e.g. ``"tcn"``. Resolved by
            :func:`dmf.train.registry.build_model`.
        head: Output head type, one of ``"point"``, ``"quantile"``, ``"gaussian"``.
        quantiles: Quantile levels for a quantile head, each in (0, 1), ascending.
            Ignored for other head types.
        params: Architecture keyword arguments passed to the model constructor. Units are
            model-specific and documented on each model class.
    """

    name: str
    head: Literal["point", "quantile", "gaussian"]
    quantiles: tuple[float, ...]
    params: dict[str, Any]


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings for the training loop.

    Attributes:
        epochs: Maximum number of epochs.
        batch_size: Minibatch size, windows.
        lr: Peak AdamW learning rate.
        weight_decay: AdamW weight decay coefficient.
        warmup_frac: Fraction of total steps spent in linear warmup, in [0, 1).
        grad_clip: Global gradient-norm clip value.
        patience: Early-stopping patience, epochs without validation improvement.
        amp_dtype: Autocast dtype, one of ``"bf16"``, ``"fp16"``, ``"off"``.
        num_workers: DataLoader worker processes.
    """

    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_frac: float
    grad_clip: float
    patience: int
    amp_dtype: Literal["bf16", "fp16", "off"]
    num_workers: int


@dataclass(frozen=True)
class ExperimentConfig:
    """A complete, reproducible experiment specification.

    Attributes:
        name: Experiment identifier, used as the results subdirectory name.
        data: Windowing and observation settings.
        model: Model and head settings.
        train: Optimisation settings.
        seeds: Training seeds. Must contain at least three entries: any model-vs-model
            comparison is reported as mean +/- std over these seeds.
        regimes: Evaluation regimes to score, e.g.
            ``("id", "unseen_seastate", "unseen_heading", "unseen_vessel")``.
    """

    name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    seeds: tuple[int, ...]
    regimes: tuple[str, ...]


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a plain dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed top-level mapping. Scalar units are whatever the individual config
        schema documents; this function performs no unit conversion.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the document's top level is not a mapping.
    """
    raise NotImplementedError


def load_experiment(path: Path) -> ExperimentConfig:
    """Load an experiment YAML into a fully populated :class:`ExperimentConfig`.

    Resolves any ``data:``, ``model:`` and ``train:`` entries that are given as paths to
    other config files, so that experiment files stay small.

    Args:
        path: Path to a file under ``configs/experiment/``.

    Returns:
        The assembled experiment configuration.

    Raises:
        ValueError: If fewer than three training seeds are specified, or if a referenced
            sub-config cannot be resolved.
    """
    raise NotImplementedError
