"""Configuration dataclasses and YAML loading.

Every tunable in this project lives in a YAML file under ``configs/`` and is loaded into
one of the frozen dataclasses below. Model and simulation code must never contain a magic
number; if a value could plausibly be changed in an experiment, it belongs here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "ObservationMode",
    "SeaState",
    "SimConfig",
    "TrainConfig",
    "load_experiment",
    "load_sim",
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
        seeds_per_cell: Default number of independent realizations per
            (sea state, heading, speed, vessel) cell.
        seeds_per_cell_by_vessel: Per-vessel overrides of ``seeds_per_cell``, as
            ``(vessel_stem, n_seeds)`` pairs. A tuple rather than a mapping so that the
            dataclass stays frozen and hashable. The corpus uses 40 seeds per cell for the
            primary ``frigate`` hull and 8 for ``s175``: ``s175`` is the held-out
            ``unseen_vessel`` test hull, is never trained on, and so needs only enough
            realizations to make its test-set mean stable.
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
    seeds_per_cell_by_vessel: tuple[tuple[str, int], ...] = ()

    def seeds_for(self, vessel: str) -> int:
        """Return the number of realizations per grid cell for one vessel.

        Args:
            vessel: Vessel config stem, e.g. ``"frigate"``.

        Returns:
            The override from :attr:`seeds_per_cell_by_vessel` if one is present for
            ``vessel``, otherwise :attr:`seeds_per_cell`. Dimensionless count.
        """
        for name, count in self.seeds_per_cell_by_vessel:
            if name == vessel:
                return count
        return self.seeds_per_cell


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
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level of a config file must be a mapping")
    return {str(key): value for key, value in raw.items()}


def load_sim(path: Path) -> SimConfig:
    """Load a corpus-generation config into a :class:`SimConfig`.

    The ``sea_states`` and ``headings`` entries may each be either an inline mapping or a
    path to another YAML file, resolved relative to ``path``'s directory, so that the sea
    state and heading tables stay in one place and are shared by every corpus variant.

    Units are those documented on :class:`SimConfig` and :class:`SeaState`: metres,
    seconds, hertz, knots, degrees, and radians per second for the synthesis band. No unit
    conversion happens here; knots reach metres per second only at the physics boundary in
    :func:`dmf.sim.encounter.knots_to_m_s`.

    Args:
        path: Path to a file under ``configs/sim/``.

    Returns:
        The assembled simulation configuration.

    Raises:
        FileNotFoundError: If ``path`` or a referenced sub-config does not exist.
        ValueError: If a required key is missing, if a referenced sub-config does not
            contain the key it is supposed to supply, or if a seed count is not positive.
    """
    raw = load_yaml(path)

    def _resolve(key: str) -> dict[str, Any]:
        """Return the mapping for ``key``, following a file reference if given one."""
        if key not in raw:
            raise ValueError(f"{path}: missing required key {key!r}")
        entry = raw[key]
        if isinstance(entry, str):
            return load_yaml(path.parent / entry)
        if isinstance(entry, dict):
            return {str(k): v for k, v in entry.items()}
        raise ValueError(f"{path}: {key!r} must be a mapping or a path to one")

    sea_raw = _resolve("sea_states").get("sea_states")
    if not isinstance(sea_raw, list) or not sea_raw:
        raise ValueError(f"{path}: 'sea_states' must resolve to a non-empty list")
    sea_states = tuple(
        SeaState(
            name=str(entry["name"]),
            hs_m=float(entry["hs_m"]),
            tp_s=float(entry["tp_s"]),
            gamma=float(entry.get("gamma", 3.3)),
        )
        for entry in sea_raw
    )

    headings_raw = _resolve("headings")
    for key in ("headings_deg", "speeds_kn"):
        if key not in headings_raw:
            raise ValueError(f"{path}: heading config supplies no {key!r}")
    headings_deg = tuple(float(v) for v in headings_raw["headings_deg"])
    speeds_kn = tuple(float(v) for v in headings_raw["speeds_kn"])

    missing = {
        "vessels",
        "seeds_per_cell",
        "duration_s",
        "spinup_s",
        "fs_hz",
        "n_components",
        "w_min_rad_s",
        "w_max_rad_s",
        "jitter_frequencies",
    } - set(raw)
    if missing:
        raise ValueError(f"{path}: missing required keys {sorted(missing)}")

    overrides_raw = raw.get("seeds_per_cell_by_vessel", {}) or {}
    if not isinstance(overrides_raw, dict):
        raise ValueError(f"{path}: 'seeds_per_cell_by_vessel' must be a mapping")
    overrides = tuple((str(k), int(v)) for k, v in sorted(overrides_raw.items()))
    for name, count in ((("<default>"), int(raw["seeds_per_cell"])), *overrides):
        if count < 1:
            raise ValueError(f"{path}: seeds per cell for {name} must be >= 1, got {count}")

    return SimConfig(
        sea_states=sea_states,
        headings_deg=headings_deg,
        speeds_kn=speeds_kn,
        vessels=tuple(str(v) for v in raw["vessels"]),
        seeds_per_cell=int(raw["seeds_per_cell"]),
        duration_s=float(raw["duration_s"]),
        spinup_s=float(raw["spinup_s"]),
        fs_hz=float(raw["fs_hz"]),
        n_components=int(raw["n_components"]),
        w_min_rad_s=float(raw["w_min_rad_s"]),
        w_max_rad_s=float(raw["w_max_rad_s"]),
        jitter_frequencies=bool(raw["jitter_frequencies"]),
        seeds_per_cell_by_vessel=overrides,
    )


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
