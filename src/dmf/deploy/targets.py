"""Which trained models are exported and benchmarked, and how one is rebuilt from disk.

``make bench`` takes no arguments, so the set of deployable models has to be declared
somewhere rather than passed in. It is declared here, with the reason for each entry, so
that the benchmark's scope is reviewable in one place instead of being implied by whatever
was on the command line the day the numbers were produced.

**What is deliberately not here.** The sea-state-conditioned arm is excluded: it consumes a
one-hot ground-truth sea state, which at deployment is estimated online from the same
motion record the forecaster consumes, so it is privileged information and not a deployable
configuration (``docs/protocol.md`` P6-D18). Benchmarking it would put a latency number
next to a model that cannot be fielded. The RevIN arm is excluded for a duller reason --
its checkpoints differ from the reference arm only for the SGD rows and it adds no distinct
graph shape.

Units: ``lookback`` and ``max_horizon`` in samples; the corpus is 10 Hz, so the shipped
geometry is a 20 s window in and a 15 s forecast out.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from dmf.config import ExperimentConfig, ModelConfig, load_experiment
from dmf.data.windows import window_spec_from_config
from dmf.models.base import BaseForecaster
from dmf.train.experiment import _instantiate, resolve_checkpoint

__all__ = [
    "DEFAULT_CHECKPOINT_ROOT",
    "DEFAULT_ONNX_DIR",
    "DEFAULT_TARGETS",
    "ModelTarget",
    "load_target",
    "onnx_path_for",
    "target_by_key",
]

#: Where :func:`dmf.train.loop.fit` wrote the committed best-epoch weights.
DEFAULT_CHECKPOINT_ROOT: Path = Path("artifacts/checkpoints")

#: Where exported graphs are written. Under ``artifacts/`` and gitignored: an ONNX file is
#: a derived artifact of a checkpoint plus this module, and committing one invites the two
#: to drift.
DEFAULT_ONNX_DIR: Path = Path("artifacts/onnx")


@dataclass(frozen=True)
class ModelTarget:
    """One (trained model, regime, seed) triple to export and benchmark.

    Attributes:
        key: Short identifier, used as the ONNX filename stem and as the benchmark row's
            model column. Equal to the model config's label.
        experiment: Experiment config the model was fitted under. Supplies the task
            geometry -- lookback, horizon and channel set -- as well as the architecture,
            so that the exported graph's fixed axes are the ones the weights were trained
            with rather than a default restated here.
        label: ``ModelConfig.label`` within that experiment.
        regime: Split regime the checkpoint was fitted on. ``id`` throughout: latency is a
            property of the graph and not of the split, so exporting all four regimes
            would produce four identical timing rows.
        seed: Training seed. One seed only, and this is a deliberate asymmetry with
            ``CLAUDE.md`` non-negotiable 5 -- three seeds are required for an *accuracy*
            comparison, where the seed moves the number. Seeds share an architecture, so
            they share a graph and a latency to within run-to-run noise; the accuracy
            column any Pareto plot joins against still comes from the three-seed tables.
        note: Why this model is in the deployable set.
    """

    key: str
    experiment: Path
    label: str
    regime: str = "id"
    seed: int = 0
    note: str = ""


#: The four graphs benchmarked by default: two point models and their quantile twins.
#:
#: Two architectures rather than one because they stress different kernels -- the TCN is
#: six dilated convolution blocks, the LSTM is a 200-step recurrence that cannot be
#: parallelised along time, and "which runtime is faster" has no architecture-independent
#: answer. Both heads rather than one because the quantile graph emits nine times the
#: output volume and carries the sort node, and the interval rule of the operational
#: evaluation is what a landing decision would actually be taken from.
#:
#: ``dlinear``/``dlinear_ols`` are not here: they are the accuracy baselines, and their
#: latency is a matrix multiply that nothing in this study turns on. ``transformer`` is not
#: here because Phase 5 already dropped it (P5-D3).
DEFAULT_TARGETS: tuple[ModelTarget, ...] = (
    ModelTarget(
        key="tcn",
        experiment=Path("configs/experiment/e02_deep.yaml"),
        label="tcn",
        note="best point model outside unseen_heading (P4-D14); dilated-convolution graph",
    ),
    ModelTarget(
        key="lstm",
        experiment=Path("configs/experiment/e02_deep.yaml"),
        label="lstm",
        note="best point model at the gate cell on id (P5-D3); sequential recurrence",
    ),
    ModelTarget(
        key="tcn_quantile",
        experiment=Path("configs/experiment/e03_probabilistic.yaml"),
        label="tcn_quantile",
        note="interval head on the TCN backbone; 9x the output volume plus the sort node",
    ),
    ModelTarget(
        key="lstm_quantile",
        experiment=Path("configs/experiment/e03_probabilistic.yaml"),
        label="lstm_quantile",
        note="interval head on the LSTM backbone; the largest graph in the study",
    ),
)


def target_by_key(key: str) -> ModelTarget:
    """Look up a default target by its key.

    Args:
        key: The target key, e.g. ``"tcn_quantile"``.

    Returns:
        The target.

    Raises:
        KeyError: If no default target carries that key.
    """
    for target in DEFAULT_TARGETS:
        if target.key == key:
            return target
    raise KeyError(f"unknown target {key!r}; known targets are {[t.key for t in DEFAULT_TARGETS]}")


def onnx_path_for(target: ModelTarget, onnx_dir: Path = DEFAULT_ONNX_DIR) -> Path:
    """Return the path an exported target is written to.

    Args:
        target: The target.
        onnx_dir: Directory holding exported graphs.

    Returns:
        ``<onnx_dir>/<key>_<regime>_seed<seed>.onnx``. The regime and seed are in the name
        because the file is weights, not just an architecture, and a name that omitted them
        would let a re-export from another split overwrite a benchmarked graph in place.
    """
    return onnx_dir / f"{target.key}_{target.regime}_seed{target.seed}.onnx"


def _model_config(experiment: ExperimentConfig, label: str) -> ModelConfig:
    """Find one model config within an experiment by label.

    Args:
        experiment: The loaded experiment config.
        label: ``ModelConfig.label`` to find.

    Returns:
        The matching model config.

    Raises:
        KeyError: If the experiment carries no model with that label.
    """
    for cfg in experiment.models:
        if cfg.label == label:
            return cfg
    raise KeyError(
        f"experiment {experiment.name!r} carries no model labelled {label!r}; it has "
        f"{[cfg.label for cfg in experiment.models]}"
    )


def load_target(
    target: ModelTarget,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    checkpoint: Path | None = None,
) -> BaseForecaster:
    """Rebuild one trained model from its config and committed weights.

    The state dict is loaded ``strict=True`` and no fallback exists: a missing or
    mismatched checkpoint raises rather than leaving a randomly initialised model to be
    exported, parity-checked against itself and benchmarked. That failure is invisible in
    every output this module produces -- the shapes are right, the parity passes, the
    latency is correct -- which is exactly why it is refused here.

    Args:
        target: Which model to load.
        checkpoint_root: Root of the checkpoint tree, ``<root>/<experiment>/<regime>/``.
        checkpoint: Explicit checkpoint file, overriding the resolved one. For loading
            weights that are not in the committed layout; the file must still be a state
            dict for exactly this architecture.

    Returns:
        The model, on CPU, in eval mode, with its committed weights.

    Raises:
        FileNotFoundError: If no checkpoint is found for the triple.
        ValueError: If the experiment conditions on sea state. That arm consumes a one-hot
            ground-truth sea state, which is privileged information at deployment
            (``docs/protocol.md`` P6-D18), so it is refused here rather than shipped with a
            latency number attached.
    """
    experiment = load_experiment(target.experiment)
    if experiment.data.condition_on_sea_state:
        raise ValueError(
            f"{target.experiment} conditions on sea state; that arm is an upper bound and "
            f"not a deployable configuration (docs/protocol.md P6-D18), so it is not "
            f"exported"
        )
    cfg = _model_config(experiment, target.label)
    spec = window_spec_from_config(experiment.data)
    n_in = len(experiment.data.input_channels)
    n_out = len(experiment.data.target_dofs)
    model = _instantiate(cfg, spec, n_in, n_out, target.seed, revin=experiment.data.revin)
    path = checkpoint or resolve_checkpoint(
        checkpoint_root / experiment.name / target.regime, cfg, target.seed
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
