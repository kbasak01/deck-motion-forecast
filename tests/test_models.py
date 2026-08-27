"""Baseline models, closed-form fitting, and the training loop -- Gate 3.

Gate 3 asks for ``results/baselines.csv`` with skill-score-vs-persistence for every
baseline, horizon, DOF and regime, and then for that file to be *read*: if AR(p) already
reaches 0.8 skill at 3 s on roll, the task is too easy as specified. This module owns the
correctness of the numbers that go into it.

What is asserted here, and why each one is load-bearing:

- **Lag features are prefix-nested.** The entire one-pass, three-order AR fit rests on
  ``lag_features(x, 10) == lag_features(x, 40)[:, :10*C_in]``. If that stops holding, AR(10)
  and AR(20) are silently solved from the wrong normal equations and still return plausible
  numbers.
- **Streaming normal equations match an explicit ``lstsq``.** Positive control for the whole
  accumulation path, on a corpus small enough to stack.
- **AR recovers a planted VAR process**, and **its fit is bitwise identical under different
  seeds** -- the evidence behind the seed exemption, which is strictly stronger than
  observing three identical result rows because it shows the seed *cannot* enter.
- **Persistence is bitwise equal to the inline expression in ``eval/controls.py``**, which
  is the expression the Gate 2 criterion-5 control validated. If the two diverge, every
  skill denominator in the project stops being the quantity that control checked.
- **The damped-persistence horizon index is one-based**: the 1-step-ahead factor is
  ``exp(-1/tau)``, not ``1.0``. An off-by-one here leaves the SSE barely worse and makes
  ``tau`` meaningless.
- **Both limits of damped persistence**: ``tau -> inf`` is persistence, ``tau -> 0+`` is the
  window-mean forecast, which is the correct null for the shuffle control.
- **SGD reaches the closed-form optimum** for the one model whose optimum is computable,
  which converts "the training loop works" from a vibe into an assertion before Phase 4
  inherits it.
- **Negative controls beside the positive ones**: an untrained DLinear must lose to
  persistence; AR before ``fit`` must raise; an order beyond the lookback must raise; a
  non-train partition must be refused; a duplicate registry key must raise.

Two geometries appear here and they are kept distinct on purpose. Tests whose subject is
*the shipped task* derive it from ``configs/data/default.yaml`` (:data:`PRODUCTION_CFG`,
:data:`PRODUCTION_SPEC`, :data:`N_IN`, :data:`N_OUT`) or from the dataset they are handed,
so a task revision moves them instead of breaking them. Tests whose subject is arithmetic on
synthetic tensors pin their own small geometry (:data:`SYNTH_C_IN`, :data:`SYNTH_C_OUT` and
local lookback/horizon literals) so that the identity under test stays hand-checkable.

Units: samples for lookback, horizons, strides and ``tau``; degrees for roll and pitch and
degrees per second for their rates, metres for heave and metres per second for heave rate;
hertz for ``fs_hz``. Model inputs and outputs are dimensionless.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import SMALL_LOOKBACK
from dmf.config import DataConfig, ModelConfig, load_data, load_experiment, load_model
from dmf.data.dataset import DeckMotionDataset, make_dataloader
from dmf.data.normalize import invert_norm, normalize_target
from dmf.data.splits import REGIMES, Regime, build_split
from dmf.data.windows import window_spec_from_config
from dmf.models.ar import ARForecaster, lag_features
from dmf.models.base import BaseForecaster
from dmf.models.dlinear import DLinear, moving_average, series_decompose
from dmf.models.persistence import (
    TAU_WINDOW_MEAN,
    DampedPersistence,
    Persistence,
    decay_factors,
    decay_sse,
    fit_decay_constant,
    solve_decay_tau,
)
from dmf.train.closed_form import (
    LagMoments,
    accumulate_training_moments,
    fit_ar,
    fit_damped_persistence,
    solve_ar_coefficients,
    subset_columns,
)
from dmf.train.loop import EarlyStopper, fit, set_seed, validate
from dmf.train.losses import mse_loss
from dmf.train.registry import MODEL_REGISTRY, build_model, register_model

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"

#: The current task definition, **loaded** from ``configs/data/default.yaml`` rather than
#: mirrored into constants here. The mirror this replaces (``lookback=200,
#: horizons=(10, 20, 30, 50)``, three targets) is exactly why the Gate 3 revision of the
#: task -- horizons out to 15 s, all six DOFs as targets (``docs/protocol.md`` P3) -- broke
#: this module instead of being picked up by it.
PRODUCTION_CFG = load_data(CONFIG_ROOT / "data" / "default.yaml")

#: Window geometry of the current task.
PRODUCTION_SPEC = window_spec_from_config(PRODUCTION_CFG)

#: Channel counts of the current task, derived, never restated.
N_IN = len(PRODUCTION_CFG.input_channels)
N_OUT = len(PRODUCTION_CFG.target_dofs)

#: Channel counts for the tests whose subject is arithmetic on synthetic tensors rather than
#: the shipped task: a planted VAR process, the decay-SSE closed form, the DLinear oracle.
#: Fixed on purpose, and fixed with ``C_out < C_in`` on purpose -- a target set that is a
#: *proper* prefix of the inputs is the case that can detect a wrong slice in
#: :class:`dmf.models.persistence.Persistence` or a channel-mixing DLinear, and the shipped
#: task no longer supplies it (``target_dofs == input_channels`` since P3).
SYNTH_C_IN = 6
SYNTH_C_OUT = 3

#: The four baseline registry keys Phase 3 must supply.
BASELINE_KEYS: tuple[str, ...] = (
    "persistence",
    "damped_persistence",
    "ar",
    "dlinear",
)


def _small_dataset(
    corpus: Path, cfg: DataConfig, regime: Regime, partition: str
) -> DeckMotionDataset:
    """Build one partition of the small corpus, with train-fitted statistics."""
    spec = window_spec_from_config(cfg)
    split = build_split(_manifest(corpus), regime)
    train = DeckMotionDataset(corpus, split, "train", cfg, spec)
    if partition == "train":
        return train
    return DeckMotionDataset(corpus, split, partition, cfg, spec, stats=train.norm_stats)


def _manifest(corpus: Path):  # type: ignore[no-untyped-def]
    """Load the manifest of a corpus root."""
    from dmf.data.splits import load_manifest

    return load_manifest(corpus)


def _stack(dataset: DeckMotionDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise a whole (small) partition as ``(x, y_normalised)``.

    Only ever called on the fixture corpus; the production partitions are far too large,
    which is the entire reason the streaming path exists.
    """
    loader = make_dataloader(dataset, batch_size=512, shuffle=False, num_workers=0, seed=0)
    stats = dataset.norm_stats.subset(dataset.target_columns)
    xs, ys = [], []
    for x, y, window_mean in loader:
        xs.append(x.double())
        ys.append(normalize_target(y.double(), stats, window_mean.double()))
    return torch.cat(xs), torch.cat(ys)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", BASELINE_KEYS)
def test_importing_dmf_models_populates_the_registry(key: str) -> None:
    """An empty registry from import order is a classic bug; it presents as a config typo."""
    import dmf.models  # noqa: F401

    assert key in MODEL_REGISTRY


def test_registry_refuses_a_duplicate_key() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @register_model("persistence")
        class _Clash(BaseForecaster):
            pass


def test_build_model_raises_on_an_unregistered_name() -> None:
    cfg = ModelConfig(name="not_a_model", head="point", quantiles=(), params={}, label="x")
    with pytest.raises(KeyError, match="unknown model"):
        build_model(cfg, PRODUCTION_SPEC, N_IN, N_OUT)


def test_build_model_raises_on_an_unknown_constructor_argument() -> None:
    cfg = ModelConfig(
        name="persistence", head="point", quantiles=(), params={"n_layers": 3}, label="x"
    )
    with pytest.raises(TypeError, match="does not accept"):
        build_model(cfg, PRODUCTION_SPEC, N_IN, N_OUT)


@pytest.mark.parametrize("stem", ["persistence", "damped_persistence", "ar_p20", "dlinear"])
def test_every_shipped_model_config_builds(stem: str) -> None:
    cfg = load_model(CONFIG_ROOT / "model" / f"{stem}.yaml")
    model = build_model(cfg, PRODUCTION_SPEC, N_IN, N_OUT)
    assert model.output_shape(4) == (4, PRODUCTION_SPEC.max_horizon, N_OUT)


def test_ar_configs_have_distinct_labels_but_share_a_registry_name() -> None:
    """F7: three AR configs legitimately share ``name: ar``; the CSV is keyed on ``label``."""
    configs = [load_model(CONFIG_ROOT / "model" / f"ar_p{p}.yaml") for p in (10, 20, 40)]
    assert {c.name for c in configs} == {"ar"}
    assert sorted(c.label for c in configs) == ["ar10", "ar20", "ar40"]


@pytest.mark.parametrize("stem", ["e01_baselines", "e01_baselines_imu"])
def test_the_baselines_experiment_configs_load_with_six_models_and_three_seeds(
    stem: str,
) -> None:
    """Both observation modes must ship the same model set, or the P6.3 ablation is not one."""
    cfg = load_experiment(CONFIG_ROOT / "experiment" / f"{stem}.yaml")
    assert [m.label for m in cfg.models] == [
        "persistence",
        "damped_persistence",
        "ar10",
        "ar20",
        "ar40",
        "ar_attitude_only",
        "dlinear",
    ]
    assert len(cfg.seeds) >= 3
    assert set(cfg.regimes) == set(REGIMES)


@pytest.mark.parametrize("stem", ["e01_baselines", "e01_baselines_imu"])
def test_every_model_in_the_baselines_configs_builds(stem: str) -> None:
    """A config that loads but cannot be instantiated fails four regimes into a sweep."""
    cfg = load_experiment(CONFIG_ROOT / "experiment" / f"{stem}.yaml")
    spec = window_spec_from_config(cfg.data)
    n_in = len(cfg.data.input_channels)
    n_out = len(cfg.data.target_dofs)
    for model_cfg in cfg.models:
        model = build_model(model_cfg, spec, n_in, n_out)
        assert model.output_shape(4) == (4, spec.max_horizon, n_out)


def test_load_experiment_refuses_fewer_than_three_seeds(tmp_path: Path) -> None:
    source = (CONFIG_ROOT / "experiment" / "e01_baselines.yaml").read_text()
    target = tmp_path / "configs" / "experiment"
    target.mkdir(parents=True)
    (target / "bad.yaml").write_text(source.replace("seeds: [0, 1, 2]", "seeds: [0]"))
    for sub in ("data", "model"):
        dest = tmp_path / "configs" / sub
        dest.mkdir(parents=True, exist_ok=True)
        for path in (CONFIG_ROOT / sub).glob("*.yaml"):
            (dest / path.name).write_text(path.read_text())
    with pytest.raises(ValueError, match="at least three training seeds"):
        load_experiment(target / "bad.yaml")


def test_config_and_splits_agree_on_the_regime_names() -> None:
    from dmf.config import _REGIME_NAMES

    assert _REGIME_NAMES == REGIMES


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_in", "n_out"), [(N_IN, N_OUT), (SYNTH_C_IN, SYNTH_C_OUT)], ids=["task", "proper_prefix"]
)
def test_persistence_matches_the_inline_control_expression_bitwise(
    rng: np.random.Generator, n_in: int, n_out: int
) -> None:
    """The Gate 2 criterion-5 control validated *that* expression, not this class."""
    x = torch.from_numpy(rng.normal(size=(32, PRODUCTION_SPEC.lookback, n_in)).astype(np.float32))
    model = Persistence(PRODUCTION_SPEC.lookback, PRODUCTION_SPEC.max_horizon, n_in, n_out)
    inline = x[:, -1:, :n_out].expand(-1, PRODUCTION_SPEC.max_horizon, -1)
    assert torch.equal(model.forward(x), inline)


def test_persistence_is_exact_on_a_constant_signal() -> None:
    x = torch.full((4, 50, SYNTH_C_IN), 2.5)
    model = Persistence(50, 10, SYNTH_C_IN, SYNTH_C_OUT)
    assert torch.equal(model.forward(x), torch.full((4, 10, SYNTH_C_OUT), 2.5))


def test_persistence_has_no_fitted_parameters() -> None:
    model = Persistence(PRODUCTION_SPEC.lookback, PRODUCTION_SPEC.max_horizon, N_IN, N_OUT)
    assert model.n_fitted_parameters == 0
    assert model.FIT_KIND == "none"


# ---------------------------------------------------------------------------
# Damped persistence
# ---------------------------------------------------------------------------


def test_decay_horizon_indexing_is_one_based() -> None:
    """The 1-step-ahead factor is ``exp(-1/tau)``, never ``1.0``.

    A zero-based index makes the first horizon step a plain persistence forecast and shifts
    every later one, which barely moves the SSE while making ``tau`` mean nothing.
    """
    tau = np.array([4.0])
    factors = decay_factors(tau, 3)
    assert factors.shape == (3, 1)
    np.testing.assert_allclose(
        factors[:, 0], [np.exp(-1 / 4), np.exp(-2 / 4), np.exp(-3 / 4)], rtol=1e-12
    )
    assert factors[0, 0] != pytest.approx(1.0)


def test_infinite_tau_reduces_to_persistence(rng: np.random.Generator) -> None:
    x = torch.from_numpy(rng.normal(size=(8, 50, SYNTH_C_IN)).astype(np.float32))
    damped = DampedPersistence(
        50, 10, SYNTH_C_IN, SYNTH_C_OUT, tau_samples=np.full(SYNTH_C_OUT, 1e12)
    )
    plain = Persistence(50, 10, SYNTH_C_IN, SYNTH_C_OUT)
    torch.testing.assert_close(damped.forward(x), plain.forward(x), rtol=1e-6, atol=1e-6)


def test_zero_tau_predicts_the_window_mean(rng: np.random.Generator) -> None:
    """``tau -> 0+`` is exactly the window-mean forecast, the shuffle control's null."""
    x = torch.from_numpy(rng.normal(size=(8, 50, SYNTH_C_IN)).astype(np.float32))
    damped = DampedPersistence(
        50, 10, SYNTH_C_IN, SYNTH_C_OUT, tau_samples=np.full(SYNTH_C_OUT, TAU_WINDOW_MEAN)
    )
    assert torch.equal(damped.forward(x), torch.zeros(8, 10, SYNTH_C_OUT))


def test_damped_persistence_reports_one_parameter_per_channel() -> None:
    model = DampedPersistence(
        50, 10, SYNTH_C_IN, SYNTH_C_OUT, tau_samples=np.full(SYNTH_C_OUT, 5.0)
    )
    assert model.n_fitted_parameters == SYNTH_C_OUT
    assert model.FIT_KIND == "closed_form"


def test_damped_persistence_raises_before_fitting() -> None:
    model = DampedPersistence(50, 10, SYNTH_C_IN, SYNTH_C_OUT)
    with pytest.raises(RuntimeError, match="no decay constants"):
        model.forward(torch.zeros(2, 50, SYNTH_C_IN))


def test_decay_sse_matches_a_direct_sum(rng: np.random.Generator) -> None:
    """The closed form is the point of the streaming fit; check it against brute force."""
    x = rng.normal(size=(64, 20, SYNTH_C_OUT))
    y = rng.normal(size=(64, 6, SYNTH_C_OUT))
    tau = np.array([3.0, 7.0, 11.0])
    last = x[:, -1, :]
    closed = decay_sse(
        tau,
        np.square(last).sum(axis=0),
        np.einsum("nc,nhc->hc", last, y),
        np.square(y).sum(axis=(0, 1)),
    )
    factors = decay_factors(tau, 6)
    direct = np.square(last[:, None, :] * factors[None, :, :] - y).sum(axis=(0, 1))
    np.testing.assert_allclose(closed, direct, rtol=1e-10)


def test_fitted_tau_beats_both_limits_on_a_damped_oscillation(
    rng: np.random.Generator,
) -> None:
    """Positive control: the grid search must find something better than either endpoint."""
    t = np.arange(30)[None, :, None]
    phase = rng.uniform(0, 2 * np.pi, size=(256, 1, SYNTH_C_OUT))
    signal = np.cos(0.3 * t + phase) * np.exp(-t / 12.0)
    x = np.concatenate([np.zeros((256, 19, SYNTH_C_OUT)), signal[:, :1, :]], axis=1)
    y = signal[:, 1:11, :]
    tau = fit_decay_constant(x, y, fs_hz=10.0)
    assert np.all(tau > 0)
    last = x[:, -1, :]
    args = (
        np.square(last).sum(axis=0),
        np.einsum("nc,nhc->hc", last, y),
        np.square(y).sum(axis=(0, 1)),
    )
    assert np.all(decay_sse(tau, *args) < decay_sse(np.full(SYNTH_C_OUT, 1e12), *args))
    assert np.all(decay_sse(tau, *args) < decay_sse(np.full(SYNTH_C_OUT, TAU_WINDOW_MEAN), *args))


def test_solve_decay_tau_flags_an_optimum_on_the_grid_boundary(
    rng: np.random.Generator,
) -> None:
    """Negative control: a boundary optimum is reported, not silently returned as a fit."""
    y = rng.normal(size=(64, 6, SYNTH_C_OUT))
    last = rng.normal(size=(64, SYNTH_C_OUT))
    fitted = solve_decay_tau(
        np.square(last).sum(axis=0),
        np.einsum("nc,nhc->hc", last, y),
        np.square(y).sum(axis=(0, 1)),
        fs_hz=10.0,
        grid=(1.0, 2.0, 5),
    )
    assert fitted.at_grid_boundary.any()
    assert "GRID BOUNDARY" in fitted.summary


def test_decay_fit_reports_tau_in_both_samples_and_seconds() -> None:
    """``fs_hz`` must genuinely reach a report string, not merely satisfy the signature."""
    y = np.ones((16, 4, SYNTH_C_OUT)) * 0.5
    last = np.ones((16, SYNTH_C_OUT))
    fitted = solve_decay_tau(
        np.square(last).sum(axis=0),
        np.einsum("nc,nhc->hc", last, y),
        np.square(y).sum(axis=(0, 1)),
        fs_hz=10.0,
        channels=("roll", "pitch", "heave"),
    )
    np.testing.assert_allclose(fitted.tau_seconds, fitted.tau_samples / 10.0, rtol=1e-12)
    assert "roll" in fitted.summary and "10 Hz" in fitted.summary


def test_fit_decay_constant_rejects_mismatched_shapes(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="must agree on the window count"):
        fit_decay_constant(rng.normal(size=(4, 10, 3)), rng.normal(size=(5, 6, 3)), 10.0)


# ---------------------------------------------------------------------------
# AR: feature map, normal equations, determinism
# ---------------------------------------------------------------------------


def test_lag_features_are_prefix_nested(rng: np.random.Generator) -> None:
    """One accumulation at p=40 must yield exact normal equations for p=10 and p=20."""
    x = torch.from_numpy(rng.normal(size=(16, 60, SYNTH_C_IN)))
    full = lag_features(x, 40)
    for order in (1, 10, 20, 39):
        assert torch.equal(lag_features(x, order), full[:, : order * SYNTH_C_IN])


def test_lag_features_are_most_recent_first(rng: np.random.Generator) -> None:
    x = torch.from_numpy(rng.normal(size=(16, 60, SYNTH_C_IN)))
    feats = lag_features(x, 5)
    for lag in range(5):
        assert torch.equal(feats[:, lag * SYNTH_C_IN : (lag + 1) * SYNTH_C_IN], x[:, -1 - lag, :])


@pytest.mark.parametrize("order", [0, -1, 61])
def test_lag_features_reject_an_order_outside_the_lookback(order: int) -> None:
    with pytest.raises(ValueError, match="order must be in"):
        lag_features(torch.zeros(2, 60, SYNTH_C_IN), order)


def test_ar_rejects_an_order_longer_than_the_lookback() -> None:
    with pytest.raises(ValueError, match="order must be in"):
        ARForecaster(20, 10, SYNTH_C_IN, SYNTH_C_OUT, order=21)


def test_ar_raises_before_fitting() -> None:
    model = ARForecaster(50, 10, SYNTH_C_IN, SYNTH_C_OUT, order=5)
    with pytest.raises(RuntimeError, match="no coefficients"):
        model.forward(torch.zeros(2, 50, SYNTH_C_IN))


def test_ar_reports_its_buffer_held_coefficients_as_parameters(
    rng: np.random.Generator,
) -> None:
    """F5: ``sum(p.numel() for p in parameters())`` is 0 here; the report must not be.

    Asserted as the formula ``order * C_in * H * C_out + H * C_out`` at the shipped
    geometry, not as a pinned integer. For the record, that formula evaluates to 216,900
    for AR(40) at the P3 geometry (order 40, C_in 6, H 150, C_out 6), up from 36,150 at the
    pre-P3 one; the number belongs in the results table, where it is measured, rather than
    mirrored here.
    """
    model = ARForecaster(PRODUCTION_SPEC.lookback, PRODUCTION_SPEC.max_horizon, N_IN, N_OUT, 40)
    model.set_coefficients(
        rng.normal(size=(40 * N_IN, PRODUCTION_SPEC.max_horizon * N_OUT)),
        rng.normal(size=PRODUCTION_SPEC.max_horizon * N_OUT),
    )
    assert sum(p.numel() for p in model.parameters()) == 0
    outputs = PRODUCTION_SPEC.max_horizon * N_OUT
    assert model.n_fitted_parameters == 40 * N_IN * outputs + outputs


def test_ar_recovers_a_planted_var_process(rng: np.random.Generator) -> None:
    """A process that *is* linear in its lags must be recovered to float32 storage."""
    n, lookback, order, horizon = 4000, 30, 5, 4
    x = rng.normal(size=(n, lookback, SYNTH_C_IN))
    weight = rng.normal(size=(order * SYNTH_C_IN, horizon * SYNTH_C_OUT)) / np.sqrt(
        order * SYNTH_C_IN
    )
    bias = rng.normal(size=horizon * SYNTH_C_OUT)
    feats = lag_features(torch.from_numpy(x), order).numpy()
    y = (feats @ weight + bias).reshape(n, horizon, SYNTH_C_OUT)

    model = ARForecaster(lookback, horizon, SYNTH_C_IN, SYNTH_C_OUT, order=order, ridge=0.0)
    model.fit(x, y)
    np.testing.assert_allclose(model.weight.numpy(), weight, atol=1e-5)
    np.testing.assert_allclose(model.bias.numpy(), bias, atol=1e-5)
    pred = model.forward(torch.from_numpy(x).float()).numpy()
    np.testing.assert_allclose(pred, y, atol=1e-3)


def test_ar_refuses_fewer_windows_than_free_parameters(rng: np.random.Generator) -> None:
    model = ARForecaster(30, 4, SYNTH_C_IN, SYNTH_C_OUT, order=5)
    with pytest.raises(ValueError, match="fewer than the"):
        model.fit(rng.normal(size=(10, 30, SYNTH_C_IN)), rng.normal(size=(10, 4, SYNTH_C_OUT)))


def test_ar_fit_is_bitwise_deterministic_across_seeds(rng: np.random.Generator) -> None:
    """Evidence for the seed exemption: the seed *cannot* enter, not merely did not."""
    x = rng.normal(size=(2000, 20, SYNTH_C_IN))
    y = rng.normal(size=(2000, 4, SYNTH_C_OUT))
    coefficients = []
    for seed in (0, 1, 2):
        set_seed(seed)
        model = ARForecaster(20, 4, SYNTH_C_IN, SYNTH_C_OUT, order=3, ridge=1e-6)
        model.fit(x, y)
        coefficients.append((model.weight.clone(), model.bias.clone()))
    for weight, bias in coefficients[1:]:
        assert torch.equal(weight, coefficients[0][0])
        assert torch.equal(bias, coefficients[0][1])


def test_ridge_shrinks_the_coefficients(rng: np.random.Generator) -> None:
    """Negative control on the regulariser: it must actually do something monotone."""
    x = rng.normal(size=(500, 20, SYNTH_C_IN))
    y = rng.normal(size=(500, 4, SYNTH_C_OUT))
    norms = []
    for ridge in (0.0, 1e-3, 1.0):
        model = ARForecaster(20, 4, SYNTH_C_IN, SYNTH_C_OUT, order=6, ridge=ridge)
        model.fit(x, y)
        norms.append(float(torch.linalg.norm(model.weight)))
    assert norms[0] > norms[1] > norms[2]


# ---------------------------------------------------------------------------
# AR: the information-set ablation (`ar_attitude_only`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_used", [1, 3, SYNTH_C_IN])
def test_lag_features_are_channel_subset_nested(rng: np.random.Generator, n_used: int) -> None:
    """The claim the whole free ablation rests on: a channel subset is a *column* subset.

    ``lag_features`` is lag-major with the channels ordered inside each lag block, so the
    design matrix of the leading ``m`` channels is the strided column set
    ``subset_columns`` enumerates -- not a prefix. If this fails, the sliced Gram is the
    Gram of some other regression and ``ar_attitude_only`` is silently wrong rather than
    loudly broken.
    """
    x = torch.from_numpy(rng.normal(size=(16, 60, SYNTH_C_IN)))
    order = 7
    cols = subset_columns(order=order, n_input_channels=SYNTH_C_IN, n_input_used=n_used)
    assert cols.shape == (order * n_used,)
    assert torch.equal(
        lag_features(x[:, :, :n_used], order),
        lag_features(x, order)[:, torch.from_numpy(cols)],
    )


def test_subset_columns_rejects_a_subset_wider_than_the_inputs() -> None:
    with pytest.raises(ValueError, match="n_input_used must be in"):
        subset_columns(order=4, n_input_channels=SYNTH_C_IN, n_input_used=SYNTH_C_IN + 1)


def test_ar_rejects_an_input_subset_wider_than_the_inputs() -> None:
    with pytest.raises(ValueError, match="n_input_used must be in"):
        ARForecaster(50, 10, SYNTH_C_IN, SYNTH_C_OUT, order=5, n_input_used=SYNTH_C_IN + 1)


def test_sliced_moments_fit_matches_a_directly_built_three_channel_fit(
    rng: np.random.Generator,
) -> None:
    """Slicing the six-channel moments must equal fitting a three-channel design matrix.

    The subject solves from the ``C_in = 6`` Gram with ``n_input_used = 3``, i.e. the
    production path where ``ar_attitude_only`` rides on the moments ``ar20`` already paid
    for. The reference is an ordinary ``C_in = 3`` AR whose design matrix was built on
    three channels from the start. Tolerance rather than bitwise: the two solves reach the
    same normal equations through different float64 summation orders, and the coefficients
    are stored in float32.
    """
    n, lookback, order, horizon, n_used = 3000, 30, 5, 4, 3
    x = rng.normal(size=(n, lookback, SYNTH_C_IN))
    y = rng.normal(size=(n, horizon, SYNTH_C_OUT))

    subject = ARForecaster(
        lookback, horizon, SYNTH_C_IN, SYNTH_C_OUT, order=order, ridge=1e-6, n_input_used=n_used
    )
    subject.fit(x, y)
    reference = ARForecaster(lookback, horizon, n_used, SYNTH_C_OUT, order=order, ridge=1e-6)
    reference.fit(x[:, :, :n_used], y)

    np.testing.assert_allclose(
        subject.weight.numpy(), reference.weight.numpy(), rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(subject.bias.numpy(), reference.bias.numpy(), rtol=1e-4, atol=1e-6)
    # And the forecasts agree, which is the property that actually reaches the metrics: the
    # subject slices the full six-channel window internally, the reference is handed three.
    with torch.no_grad():
        torch.testing.assert_close(
            subject.forward(torch.from_numpy(x).float()),
            reference.forward(torch.from_numpy(x[:, :, :n_used]).float()),
            rtol=1e-4,
            atol=1e-5,
        )


def test_streamed_moments_fit_the_ablation_without_a_second_pass(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    """The production path: one streamed moments object, two information sets.

    Checks the streamed six-channel moments sliced to three channels against the same
    windows stacked and fitted directly on three channels, so the equality is asserted on
    corpus data through ``fit_ar`` rather than only on synthetic arrays through ``fit``.
    """
    train = _small_dataset(small_corpus, small_data_cfg, "id", "train")
    spec = window_spec_from_config(small_data_cfg)
    order, n_used = 5, 3
    n_in = len(train.input_columns)
    n_out = len(train.target_columns)
    moments = accumulate_training_moments(train, max_order=order, num_workers=0)

    subject, report = fit_ar(
        moments,
        order=order,
        ridge=1e-6,
        lookback=spec.lookback,
        n_input_channels=n_in,
        n_target_channels=n_out,
        n_input_used=n_used,
    )
    assert report.n_fitted_parameters == order * n_used * spec.max_horizon * n_out + (
        spec.max_horizon * n_out
    )

    # The reference is built from a three-channel design matrix, not from a three-channel
    # model: ``BaseForecaster`` refuses ``C_out > C_in`` (the P2-D4 prefix rule), and this
    # task forecasts all six channels from three. Going through ``LagMoments`` directly
    # keeps the reference free of the slicing under test.
    x, y = _stack(train)
    feats = lag_features(x[:, :, :n_used], order).numpy()
    targets = y.reshape(y.shape[0], -1).numpy()
    direct = LagMoments(
        n=int(x.shape[0]),
        sx=feats.sum(axis=0),
        sy=targets.sum(axis=0),
        syy=np.square(targets).sum(axis=0),
        gram=feats.T @ feats,
        cross=feats.T @ targets,
        max_order=order,
        n_input_channels=n_used,
        max_horizon=spec.max_horizon,
        n_target_channels=n_out,
    )
    weight, bias, _ = solve_ar_coefficients(direct, order=order, ridge=1e-6)
    np.testing.assert_allclose(subject.weight.numpy(), weight, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(subject.bias.numpy(), bias, rtol=1e-3, atol=1e-5)
    with torch.no_grad():
        torch.testing.assert_close(
            subject.forward(x.float()),
            torch.from_numpy((feats @ weight + bias).astype(np.float32)).view(
                x.shape[0], spec.max_horizon, n_out
            ),
            rtol=1e-3,
            atol=1e-4,
        )


def test_ar_reads_only_the_channels_it_is_given(rng: np.random.Generator) -> None:
    """Matched pair, both halves on one architecture.

    The positive half of the DLinear pair asserts an *invariance*, and an invariance is
    the assertion that passes when the model is broken in the direction of ignoring its
    inputs entirely. Perturbing the same slice on the full-input model is the negative
    half: ``ar_attitude_only`` must ignore the rate channels and ``ar20`` must not.
    """
    lookback, order, horizon = 30, 5, 4
    x = rng.normal(size=(2000, lookback, SYNTH_C_IN))
    y = rng.normal(size=(2000, horizon, SYNTH_C_OUT))
    xt = torch.from_numpy(x).float()
    perturbed = xt.clone()
    perturbed[:, :, SYNTH_C_OUT:] += 100.0

    attitude_only = ARForecaster(
        lookback, horizon, SYNTH_C_IN, SYNTH_C_OUT, order=order, ridge=1e-6, n_input_used=3
    )
    attitude_only.fit(x, y)
    all_channels = ARForecaster(lookback, horizon, SYNTH_C_IN, SYNTH_C_OUT, order=order, ridge=1e-6)
    all_channels.fit(x, y)
    with torch.no_grad():
        assert torch.equal(attitude_only.forward(xt), attitude_only.forward(perturbed))
        assert not torch.equal(all_channels.forward(xt), all_channels.forward(perturbed))
    assert attitude_only.n_fitted_parameters < all_channels.n_fitted_parameters


# ---------------------------------------------------------------------------
# Streaming moments, against the explicit design matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", ["id", "unseen_seastate"])
def test_streaming_moments_match_the_stacked_design(
    small_corpus: Path, small_data_cfg: DataConfig, regime: Regime
) -> None:
    """Positive control for the accumulation path itself, to float64 round-off.

    This is the tight half of the claim "one streaming pass gives the exact normal
    equations". The production partitions cannot be stacked -- ``unseen_vessel/train`` is
    1.88 M windows, i.e. 18 GB as a float64 design matrix -- so the equality is checked on
    the fixture corpus and the streaming code path is then the same one production uses.
    """
    train = _small_dataset(small_corpus, small_data_cfg, regime, "train")
    order = 4
    moments = accumulate_training_moments(train, max_order=order, batch_size=256, num_workers=0)

    x, y = _stack(train)
    assert moments.n_windows == x.shape[0]
    feats = lag_features(x, order).numpy()
    flat = y.reshape(y.shape[0], -1).numpy()

    np.testing.assert_allclose(moments.lag.gram, feats.T @ feats, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(moments.lag.cross, feats.T @ flat, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(moments.lag.sx, feats.sum(axis=0), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(moments.lag.sy, flat.sum(axis=0), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(moments.lag.syy, np.square(flat).sum(axis=0), rtol=1e-9, atol=1e-9)
    last = x[:, -1, : len(train.target_columns)].numpy()
    np.testing.assert_allclose(moments.decay.sxx, np.square(last).sum(axis=0), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("regime", ["id", "unseen_seastate"])
def test_streaming_normal_equations_match_lstsq(
    small_corpus: Path, small_data_cfg: DataConfig, regime: Regime
) -> None:
    """The solved model must fit as well as ``np.linalg.lstsq`` on the explicit design.

    **The comparison is on the residual, not on the coefficients, and that is a finding
    rather than a weakened test.** At 10 Hz the six-channel lag design of this corpus is
    numerically rank-deficient: the motion band tops out near 0.4 Hz, so adjacent lags are
    almost collinear and the measured ``cond(R)`` runs past 1e16 -- asserted below, so the
    reason for the tolerance is itself under test. The least-squares *solution* is therefore
    not unique, and this module's whitened minimum-norm solve and ``lstsq``'s raw-coordinate
    one land on different points of the same optimal set with different rank cutoffs.
    Asserting the coefficients agree would be asserting something false. Exact coefficient
    recovery is covered separately, on a well-conditioned synthetic design, by
    ``test_ar_recovers_a_planted_var_process``.
    """
    train = _small_dataset(small_corpus, small_data_cfg, regime, "train")
    order = 4
    moments = accumulate_training_moments(train, max_order=order, batch_size=256, num_workers=0)
    model, report = fit_ar(
        moments,
        order=order,
        ridge=0.0,
        lookback=small_data_cfg.lookback,
        n_input_channels=len(train.input_columns),
        n_target_channels=len(train.target_columns),
    )
    assert report.cond_r > 1e10, (
        f"cond(R) = {report.cond_r:.3e}: the design is better conditioned than this test "
        f"assumes, so the coefficients could and should be compared directly instead"
    )

    x, y = _stack(train)
    feats = lag_features(x, order).numpy()
    flat = y.reshape(y.shape[0], -1).numpy()
    design = np.concatenate([feats, np.ones((feats.shape[0], 1))], axis=1)
    reference = design @ np.linalg.lstsq(design, flat, rcond=None)[0]
    reference_mse = float(np.mean(np.square(reference - flat)))

    assert report.residual_train_mse == pytest.approx(reference_mse, rel=0.5)
    assert report.residual_train_mse < 1e-3 * float(np.mean(np.square(flat)))
    assert report.n_fitted_parameters == model.weight.numel() + model.bias.numel()


def test_moments_from_a_single_batch_equal_moments_from_many(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    """Sums, not means: the accumulation must not depend on the batch size."""
    train = _small_dataset(small_corpus, small_data_cfg, "id", "train")
    a = accumulate_training_moments(train, max_order=3, batch_size=64, num_workers=0)
    b = accumulate_training_moments(train, max_order=3, batch_size=4096, num_workers=0)
    np.testing.assert_allclose(a.lag.gram, b.lag.gram, rtol=1e-9)
    np.testing.assert_allclose(a.lag.cross, b.lag.cross, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(a.decay.sxy, b.decay.sxy, rtol=1e-9, atol=1e-12)


def test_prefix_nesting_makes_one_pass_fit_every_order(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    """AR(2) from a p=6 accumulation must equal AR(2) from its own p=2 accumulation."""
    train = _small_dataset(small_corpus, small_data_cfg, "id", "train")
    kw = {
        "lookback": small_data_cfg.lookback,
        "n_input_channels": len(train.input_columns),
        "n_target_channels": len(train.target_columns),
    }
    wide, _ = fit_ar(
        accumulate_training_moments(train, max_order=6, batch_size=256, num_workers=0),
        order=2,
        ridge=1e-6,
        **kw,
    )
    narrow, _ = fit_ar(
        accumulate_training_moments(train, max_order=2, batch_size=256, num_workers=0),
        order=2,
        ridge=1e-6,
        **kw,
    )
    assert torch.equal(wide.weight, narrow.weight)
    assert torch.equal(wide.bias, narrow.bias)


@pytest.mark.parametrize("partition", ["val", "test"])
def test_moments_refuse_a_non_training_partition(
    small_corpus: Path, small_data_cfg: DataConfig, partition: str
) -> None:
    """F9: ``norm_stats.fitted_on`` proves the *stats* are train-only, not the *windows*."""
    dataset = _small_dataset(small_corpus, small_data_cfg, "id", partition)
    assert dataset.norm_stats.fitted_on.endswith("/train")
    assert dataset.partition == partition
    with pytest.raises(ValueError, match="training partition"):
        accumulate_training_moments(dataset, max_order=3, num_workers=0)


def test_dataset_exposes_its_partition_and_regime(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    dataset = _small_dataset(small_corpus, small_data_cfg, "unseen_heading", "test")
    assert (dataset.partition, dataset.regime) == ("test", "unseen_heading")


def test_shuffling_targets_destroys_the_fit(small_corpus: Path, small_data_cfg: DataConfig) -> None:
    """The shuffle control's premise: with the correspondence gone, so is the signal."""
    train = _small_dataset(small_corpus, small_data_cfg, "id", "train")
    kw = {
        "lookback": small_data_cfg.lookback,
        "n_input_channels": len(train.input_columns),
        "n_target_channels": len(train.target_columns),
    }
    _, honest = fit_ar(
        accumulate_training_moments(train, max_order=4, batch_size=512, num_workers=0),
        order=4,
        ridge=1e-6,
        **kw,
    )
    _, shuffled = fit_ar(
        accumulate_training_moments(
            train, max_order=4, batch_size=512, num_workers=0, shuffle_targets=True
        ),
        order=4,
        ridge=1e-6,
        **kw,
    )
    assert shuffled.residual_train_mse > 10.0 * honest.residual_train_mse


def test_damped_persistence_fits_from_the_same_moments_pass(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    train = _small_dataset(small_corpus, small_data_cfg, "id", "train")
    moments = accumulate_training_moments(train, max_order=4, batch_size=512, num_workers=0)
    model, report = fit_damped_persistence(
        moments,
        fs_hz=small_data_cfg.fs_hz,
        lookback=small_data_cfg.lookback,
        n_input_channels=len(train.input_columns),
        n_target_channels=len(train.target_columns),
        channels=train.target_columns,
    )
    assert report.n_fitted_parameters == len(train.target_columns)
    assert np.all(report.fit.tau_samples > 0)
    np.testing.assert_allclose(
        report.fit.tau_seconds, report.fit.tau_samples / small_data_cfg.fs_hz, rtol=1e-12
    )
    # No assertion that the optimum is interior: on this 60 s fixture the horizon is only
    # 2 s, over which roll barely decays, so the upper grid edge (i.e. plain persistence)
    # genuinely is optimal for some channels. The flag exists to make that visible; on the
    # production corpus every channel lands interior (tau ~ 1.9 s for roll).
    assert report.fit.at_grid_boundary.shape == (len(train.target_columns),)
    assert model.forward(
        torch.zeros(2, small_data_cfg.lookback, len(train.input_columns))
    ).shape == (
        2,
        window_spec_from_config(small_data_cfg).max_horizon,
        len(train.target_columns),
    )


# ---------------------------------------------------------------------------
# Normalisation round trip
# ---------------------------------------------------------------------------


def test_normalize_target_inverts_invert_norm(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    dataset = _small_dataset(small_corpus, small_data_cfg, "id", "train")
    stats = dataset.norm_stats.subset(dataset.target_columns)
    x, y, window_mean = dataset[7]
    y = y[None].double()
    window_mean = window_mean[None].double()
    normalised = normalize_target(y, stats, window_mean)
    torch.testing.assert_close(invert_norm(normalised, stats, window_mean), y)
    assert x.shape == (small_data_cfg.lookback, len(dataset.input_columns))


def test_normalize_target_rejects_a_channel_count_mismatch(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    """Statistics covering a different channel set than ``y`` must be refused, not sliced.

    The mismatch is constructed by dropping a target channel from the statistics rather
    than by passing the full input-channel statistics, which is what this test used to do.
    Since the P3 task revision ``target_dofs == input_channels``, so "full ``C_in`` stats
    where the ``C_out`` subset was wanted" is no longer a mismatch at all and the test
    stopped testing anything -- it raised nothing and the ``pytest.raises`` block failed.
    A proper subset is a mismatch under any task definition.
    """
    dataset = _small_dataset(small_corpus, small_data_cfg, "id", "train")
    assert len(dataset.target_columns) >= 2
    stats = dataset.norm_stats.subset(dataset.target_columns[:-1])
    _, y, window_mean = dataset[0]
    with pytest.raises(ValueError, match="channels but stats cover"):
        normalize_target(y[None].double(), stats, window_mean[None].double())


# ---------------------------------------------------------------------------
# DLinear
# ---------------------------------------------------------------------------


def test_moving_average_replicates_edges() -> None:
    """Replicate, not zero: zero padding depresses the trend at the last lookback sample."""
    x = torch.ones(1, 9, 1) * 3.0
    torch.testing.assert_close(moving_average(x, 5), x)


def test_moving_average_of_a_ramp_is_the_ramp_in_the_interior() -> None:
    ramp = torch.arange(11, dtype=torch.float32).view(1, 11, 1)
    smoothed = moving_average(ramp, 5)
    torch.testing.assert_close(smoothed[:, 2:-2, :], ramp[:, 2:-2, :])


def test_moving_average_preserves_length(rng: np.random.Generator) -> None:
    x = torch.from_numpy(rng.normal(size=(3, 40, SYNTH_C_IN)).astype(np.float32))
    for kernel in (1, 2, 5, 25, 40):
        assert moving_average(x, kernel).shape == x.shape


@pytest.mark.parametrize("kernel", [0, -1, 41])
def test_moving_average_rejects_an_impossible_kernel(kernel: int) -> None:
    with pytest.raises(ValueError, match="kernel_size must be in"):
        moving_average(torch.zeros(2, 40, SYNTH_C_IN), kernel)


def test_series_decompose_sums_to_the_input(rng: np.random.Generator) -> None:
    x = torch.from_numpy(rng.normal(size=(4, 40, SYNTH_C_IN)).astype(np.float32))
    trend, remainder = series_decompose(x, 9)
    torch.testing.assert_close(trend + remainder, x)


@pytest.mark.parametrize("individual", [False, True])
def test_dlinear_output_shape_and_parameter_count(individual: bool) -> None:
    model = DLinear(
        PRODUCTION_SPEC.lookback, PRODUCTION_SPEC.max_horizon, N_IN, N_OUT, individual=individual
    )
    x = torch.zeros(4, PRODUCTION_SPEC.lookback, N_IN)
    assert model(x).shape == model.output_shape(4)
    maps = N_OUT if individual else 1
    # Two linear maps (trend and remainder) of L -> H with bias, per channel map.
    per_map = PRODUCTION_SPEC.lookback * PRODUCTION_SPEC.max_horizon + PRODUCTION_SPEC.max_horizon
    assert model.n_fitted_parameters == 2 * maps * per_map


def test_dlinear_is_channel_independent_and_ignores_the_rate_channels(
    rng: np.random.Generator,
) -> None:
    """The canonical variant must not see roll_rate.

    An invariance assertion passes for free if the perturbed slice is empty, which is
    exactly what happens at the shipped geometry (``target_dofs == input_channels`` since
    P3) -- hence the SYNTH proper-prefix geometry and the explicit non-emptiness check. The
    matched *negative* half of this pair used to be ``dlinear_mc``; it is now
    ``test_ar_reads_only_the_channels_it_is_given``, which perturbs the same slice on a
    model that is supposed to react to it.
    """
    assert SYNTH_C_OUT < SYNTH_C_IN, "the perturbation below is vacuous without spare channels"
    model = DLinear(50, 10, SYNTH_C_IN, SYNTH_C_OUT).eval()
    x = torch.from_numpy(rng.normal(size=(4, 50, SYNTH_C_IN)).astype(np.float32))
    perturbed = x.clone()
    perturbed[:, :, SYNTH_C_OUT:] += 100.0
    assert not torch.equal(x, perturbed)
    with torch.no_grad():
        assert torch.equal(model(x), model(perturbed))


def test_untrained_dlinear_is_worse_than_persistence(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    """Integrity control 2, in unit-test form: a random init must lose to persistence."""
    dataset = _small_dataset(small_corpus, small_data_cfg, "id", "test")
    spec = window_spec_from_config(small_data_cfg)
    set_seed(0)
    untrained = DLinear(spec.lookback, spec.max_horizon, N_IN, N_OUT).eval()
    persistence = Persistence(spec.lookback, spec.max_horizon, N_IN, N_OUT).eval()
    stats = dataset.norm_stats.subset(dataset.target_columns)
    loader = make_dataloader(dataset, batch_size=512, shuffle=False, num_workers=0, seed=0)
    sse = {"untrained": 0.0, "persistence": 0.0}
    with torch.no_grad():
        for x, y, window_mean in loader:
            for name, model in (("untrained", untrained), ("persistence", persistence)):
                pred = invert_norm(model(x).double(), stats, window_mean.double())
                sse[name] += float(torch.square(pred - y.double()).sum())
    assert sse["untrained"] > sse["persistence"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def test_mse_loss_matches_torch(rng: np.random.Generator) -> None:
    a = torch.from_numpy(rng.normal(size=(4, 6, 3)))
    b = torch.from_numpy(rng.normal(size=(4, 6, 3)))
    torch.testing.assert_close(mse_loss(a, b), torch.nn.functional.mse_loss(a, b))


def test_mse_loss_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        mse_loss(torch.zeros(2, 3, 4), torch.zeros(2, 3, 5))


def test_early_stopper_waits_exactly_patience_epochs() -> None:
    stopper = EarlyStopper(patience=2)
    assert not stopper.update(1.0)
    assert not stopper.update(2.0)
    assert not stopper.update(2.0)
    assert stopper.update(2.0)
    assert stopper.best_loss == 1.0
    assert stopper.best_epoch == 0


def test_early_stopper_min_delta_rejects_a_negligible_improvement() -> None:
    stopper = EarlyStopper(patience=0, min_delta=0.1)
    assert not stopper.update(1.0)
    assert stopper.update(0.95)


@pytest.mark.parametrize("bad", [-1])
def test_early_stopper_rejects_a_negative_patience(bad: int) -> None:
    with pytest.raises(ValueError, match="patience must be non-negative"):
        EarlyStopper(bad)


def test_set_seed_makes_initialisation_reproducible() -> None:
    weights = []
    for _ in range(2):
        set_seed(7)
        weights.append(DLinear(50, 10, SYNTH_C_IN, SYNTH_C_OUT).trend[0].weight.detach().clone())
    assert torch.equal(weights[0], weights[1])


def test_different_seeds_give_different_initialisations() -> None:
    set_seed(1)
    a = DLinear(50, 10, SYNTH_C_IN, SYNTH_C_OUT).trend[0].weight.detach().clone()
    set_seed(2)
    b = DLinear(50, 10, SYNTH_C_IN, SYNTH_C_OUT).trend[0].weight.detach().clone()
    assert not torch.equal(a, b)


@pytest.mark.slow
def test_fit_lowers_the_validation_loss(
    small_corpus: Path, small_data_cfg: DataConfig, tmp_path: Path
) -> None:
    train = _small_dataset(small_corpus, small_data_cfg, "id", "train")
    val = _small_dataset(small_corpus, small_data_cfg, "id", "val")
    spec = window_spec_from_config(small_data_cfg)
    cfg = _train_cfg(epochs=8)
    set_seed(0)
    model = DLinear(spec.lookback, spec.max_horizon, N_IN, N_OUT)
    train_loader = make_dataloader(train, batch_size=256, shuffle=True, num_workers=0, seed=0)
    val_loader = make_dataloader(val, batch_size=512, shuffle=False, num_workers=0, seed=0)
    before = validate(model, val_loader, cfg)
    result = fit(model, train_loader, val_loader, cfg, seed=0, checkpoint_dir=tmp_path)
    assert result.best_val_loss < before
    assert result.checkpoint_path.exists()
    assert len(result.val_losses) == result.epochs_run


@pytest.mark.slow
def test_dlinear_sgd_reaches_the_closed_form_optimum(tmp_path: Path) -> None:
    """The oracle: ``DLinear(individual=False)`` is linear, so its MSE optimum is exact.

    This is what turns "the training loop worked" into an assertion, before Phase 4's deep
    models inherit a loop that has never been checked against a known answer. The two biases
    are only ever used summed, so the identifiable parameter set is
    ``[trend_weights, remainder_weights, combined_bias]`` and ordinary least squares over
    that is the global optimum.

    **The oracle runs on a well-conditioned synthetic task, not on the corpus, and that is
    deliberate.** On the real corpus the same comparison is not a test of the training loop:
    the lag design there is numerically rank-deficient (``cond`` past 1e16, asserted in
    ``test_streaming_normal_equations_match_lstsq``), so the exact linear optimum sits at
    the bottom of a quadratic whose condition number is astronomical. First-order methods
    converge at a rate set by that condition number, so SGD falls short of the closed-form
    optimum by orders of magnitude *while working correctly*, and the test would report a
    property of the data as a bug in ``loop.py``. Measured for the record: on ``id`` at the
    fixture geometry the exact val optimum is ~1e-6 and 120 epochs of Adam reach ~5e-2, with
    the loss still falling at the last epoch.
    """
    lookback, horizon, kernel = 24, 6, 5
    rng = np.random.default_rng(11)
    n_train, n_val = 6000, 1500
    weight = rng.normal(size=(2 * lookback, horizon)) / np.sqrt(2 * lookback)
    bias = rng.normal(size=horizon) * 0.1

    def _make(count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a well-conditioned linear task: iid inputs, exactly linear targets."""
        x = torch.from_numpy(rng.normal(size=(count, lookback, SYNTH_C_OUT)).astype(np.float32))
        trend, remainder = series_decompose(x, kernel)
        design = torch.cat(
            [
                trend.transpose(1, 2).reshape(count * SYNTH_C_OUT, lookback),
                remainder.transpose(1, 2).reshape(count * SYNTH_C_OUT, lookback),
            ],
            dim=1,
        ).double()
        flat = design.numpy() @ weight + bias
        y = torch.from_numpy(flat).view(count, SYNTH_C_OUT, horizon).permute(0, 2, 1).float()
        y = y + torch.from_numpy(rng.normal(scale=0.05, size=tuple(y.shape)).astype(np.float32))
        return x, y, torch.zeros(count, 1, SYNTH_C_OUT)

    train_parts = _make(n_train)
    val_parts = _make(n_val)

    def _optimum(parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        """Exact least-squares val MSE over the identifiable parameterisation."""
        x, y, _ = parts
        count = x.shape[0]
        trend, remainder = series_decompose(x, kernel)
        design = (
            torch.cat(
                [
                    trend.transpose(1, 2).reshape(count * SYNTH_C_OUT, lookback),
                    remainder.transpose(1, 2).reshape(count * SYNTH_C_OUT, lookback),
                    torch.ones(count * SYNTH_C_OUT, 1),
                ],
                dim=1,
            )
            .double()
            .numpy()
        )
        targets = y.permute(0, 2, 1).reshape(count * SYNTH_C_OUT, horizon).double().numpy()
        return design, targets  # type: ignore[return-value]

    train_design, train_targets = _optimum(train_parts)  # type: ignore[misc]
    val_design, val_targets = _optimum(val_parts)  # type: ignore[misc]
    solution = np.linalg.lstsq(train_design, train_targets, rcond=None)[0]
    optimum = float(np.mean(np.square(val_design @ solution - val_targets)))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train_parts), batch_size=256, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*val_parts), batch_size=512, shuffle=False
    )
    set_seed(0)
    model = DLinear(lookback, horizon, SYNTH_C_OUT, SYNTH_C_OUT, kernel_size=kernel)
    result = fit(model, train_loader, val_loader, _train_cfg(epochs=200, lr=3e-2), 0, tmp_path)
    assert result.best_val_loss <= optimum * 1.05 + 1e-6, (
        f"SGD reached val MSE {result.best_val_loss:.6f} against the exact linear optimum "
        f"{optimum:.6f} on a well-conditioned task; the training loop, not the data, is "
        f"the suspect"
    )


def _train_cfg(*, epochs: int, lr: float = 1e-2):  # type: ignore[no-untyped-def]
    """Return a small-corpus training config: CPU, no autocast, short patience."""
    from dmf.config import TrainConfig

    return TrainConfig(
        epochs=epochs,
        batch_size=256,
        lr=lr,
        weight_decay=0.0,
        warmup_frac=0.05,
        grad_clip=1.0,
        patience=max(epochs // 4, 3),
        amp_dtype="off",
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# The experiment driver
# ---------------------------------------------------------------------------


def _smoke_experiment(data_cfg: DataConfig):  # type: ignore[no-untyped-def]
    """Return a cut-down `e01_baselines` covering all three fit kinds.

    One model of each ``FIT_KIND`` -- ``none`` (persistence), ``closed_form`` (damped
    persistence and AR) and ``sgd`` (DLinear) -- so that every branch of ``_fit_one`` runs,
    on the small-corpus window geometry and two epochs rather than sixty.

    Args:
        data_cfg: The small-corpus task configuration (50-sample lookback).

    Returns:
        A :class:`dmf.config.ExperimentConfig` sized for a few seconds on CPU.
    """
    from dmf.config import ExperimentConfig

    base = load_experiment(CONFIG_ROOT / "experiment" / "e01_baselines.yaml")
    wanted = ("persistence", "damped_persistence", "ar10", "dlinear")
    models = tuple(m for m in base.models if m.label in wanted)
    assert len(models) == len(wanted), [m.label for m in base.models]
    return ExperimentConfig(
        name="baselines_smoke",
        data=data_cfg,
        models=models,
        train=_train_cfg(epochs=2),
        seeds=base.seeds,
        regimes=("id", "unseen_seastate"),
    )


def test_run_experiment_writes_a_joinable_gate3_artifact(
    small_corpus: Path, small_data_cfg: DataConfig, tmp_path: Path
) -> None:
    """End-to-end cover for the one function that assembles the Gate 3 artifact.

    ``run_experiment`` is where the data pipeline, the three fit kinds, the single-pass
    scorer and the report writer meet, and every one of its parts is unit-tested apart from
    the assembly itself. This exercises the assembly on the fixture corpus: two regimes,
    two epochs, controls off (they are slow, and ``tests/test_metrics.py`` owns them).

    Three seeds rather than one, even though one would be faster. ``run_experiment`` always
    aggregates before it writes, and :func:`dmf.eval.report.aggregate_over_seeds` refuses a
    stochastic group with fewer than three (CLAUDE.md non-negotiable 5), so a single-seed
    run can only be smoke-tested with the SGD models removed -- which would leave the fit
    path every Phase 4 model uses uncovered. Two epochs on the fixture corpus keeps the
    whole test near three seconds.

    The load-bearing assertion is the join. ``baselines.csv`` and ``baselines_by_cell.csv``
    both carry a ``model`` column, and if the per-cell table keeps the internal
    ``"<label>@<seed>"`` run key while the aggregate carries the bare label, the two use
    different key spaces under the same name: the per-cell breakdown silently stops being
    a decomposition of the headline table, and every per-heading row in ``baselines.md``
    reads as a separate model.
    """
    import pandas as pd

    from dmf.eval.report import BASELINES_COLUMNS, BASELINES_GROUP_COLS
    from dmf.train.experiment import run_experiment

    results = tmp_path / "results"
    regimes = ("id", "unseen_seastate")
    per_run = run_experiment(
        _smoke_experiment(small_data_cfg),
        small_corpus,
        results_dir=results,
        seeds=(0, 1, 2),
        regimes=regimes,
        device="cpu",
        run_controls=False,
        checkpoint_root=tmp_path / "checkpoints",
    )
    assert not per_run.empty
    assert set(per_run["regime"]) == set(regimes)

    for name in ("baselines_by_seed.csv", "baselines_by_cell.csv", "baselines.csv"):
        assert (results / name).exists(), name
    assert (results / "baselines.md").exists()
    # The fourth CSV is written only when the controls run, which this test turns off to
    # stay fast; its absence is the evidence that `run_controls=False` is honoured.
    assert not (results / "baselines_controls.csv").exists()

    baselines = pd.read_csv(results / "baselines.csv")
    assert tuple(baselines.columns) == BASELINES_COLUMNS

    # Bitwise, not approximate: skill is formed from sums with the reference's own sums in
    # the denominator, so the reference's row is exactly 1 - 1. Anything else means the
    # denominator drifted away from the model that produced it.
    for frame, column in ((per_run, "skill"), (baselines, "skill_mean")):
        reference = frame[frame["model"] == "persistence"]
        assert len(reference) == (
            len(regimes) * len(small_data_cfg.target_dofs) * len(small_data_cfg.horizons)
        )
        assert (reference[column] == 0.0).all(), reference[column].unique()

    cells = pd.read_csv(results / "baselines_by_cell.csv")
    assert "seed" in cells.columns
    assert not cells["model"].astype(str).str.contains("@").any()
    keys = list(BASELINES_GROUP_COLS)
    joined = cells.merge(baselines[keys].drop_duplicates(), on=keys, how="left", indicator=True)
    unmatched = joined[joined["_merge"] != "both"]
    assert unmatched.empty, unmatched[keys].drop_duplicates().to_dict("records")


def test_the_ablation_adds_no_pass_over_the_training_split(
    small_corpus: Path,
    small_data_cfg: DataConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ar_attitude_only` must ride on `ar20`'s moments, not accumulate its own.

    The pass over the training split *is* the cost of a closed-form fit -- the solve is a
    240x240 Cholesky and the pass is millions of windows through a dataloader -- so a
    second accumulation would make an ablation that is supposed to be free the most
    expensive model in the table. Asserted by counting calls rather than by timing, which
    would be flaky on a fixture corpus this small.
    """
    from dmf.config import ExperimentConfig
    from dmf.train import experiment as experiment_module
    from dmf.train.experiment import run_experiment

    base = load_experiment(CONFIG_ROOT / "experiment" / "e01_baselines.yaml")
    wanted = ("persistence", "ar20", "ar_attitude_only")
    models = tuple(m for m in base.models if m.label in wanted)
    assert len(models) == len(wanted), [m.label for m in base.models]
    cfg = ExperimentConfig(
        name="ablation_smoke",
        data=small_data_cfg,
        models=models,
        train=_train_cfg(epochs=1),
        seeds=base.seeds,
        regimes=("id",),
    )

    calls = []
    original = experiment_module.accumulate_training_moments

    def counting(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs.get("max_order"))
        return original(*args, **kwargs)

    monkeypatch.setattr(experiment_module, "accumulate_training_moments", counting)
    per_run = run_experiment(
        cfg,
        small_corpus,
        results_dir=tmp_path / "results",
        seeds=(0, 1, 2),
        regimes=("id",),
        device="cpu",
        run_controls=False,
        checkpoint_root=tmp_path / "checkpoints",
    )
    assert calls == [20], calls

    params = per_run.groupby("model")["n_params"].nunique()
    assert (params == 1).all()
    counts = per_run.groupby("model")["n_params"].first()
    # Same order, same horizon, same targets: the only difference is the input width, so
    # the ratio of the coefficient blocks is exactly 6/3.
    n_out = len(small_data_cfg.target_dofs) * max(small_data_cfg.horizons)
    assert counts["ar20"] - n_out == 2 * (counts["ar_attitude_only"] - n_out)


# ---------------------------------------------------------------------------
# Real corpus (slow)
# ---------------------------------------------------------------------------


#: Persistence RMSE on ``id``/test in ``ideal`` mode, per (horizon, DOF), at the P3 task
#: geometry: 434,304 windows over 384 realizations, 1131 windows each. Columns follow
#: ``configs/data/default.yaml``'s ``target_dofs`` order. Units: degrees for roll and pitch,
#: metres for heave, degrees per second and metres per second for the rates.
#:
#: **Re-measured, not adjusted, after the P3 revision.** The pre-P3 table was measured at
#: ``max_horizon = 50``, which fits 1151 windows into a 6000-sample realization; at
#: ``max_horizon = 150`` only 1131 fit, so the last 20 window starts of every realization
#: drop out and every denominator moves in the fifth significant figure (roll at 10 samples:
#: 1.946806 before, 1.944663 now). Verified against an independent pandas/numpy pass over
#: the raw Parquet that shares no code with the dataset pipeline; the two agree to 5e-7,
#: i.e. to the rounding of the literals below.
#:
#: These rows are the denominator of every skill score reported on ``id``. If they move, no
#: skill number in the project means what it says.
PERSISTENCE_RMSE_ID_TEST: dict[int, tuple[float, ...]] = {
    10: (1.944663, 0.896319, 0.441743, 1.067514, 0.752844, 0.285223),
    20: (3.744455, 1.639687, 0.838214, 2.048398, 1.341412, 0.535198),
    30: (5.267361, 2.122358, 1.151318, 2.865996, 1.664056, 0.721625),
    50: (7.087561, 2.234325, 1.442156, 3.798346, 1.549273, 0.852486),
    100: (3.793711, 1.559155, 0.833518, 1.958739, 1.273090, 0.478612),
    150: (5.251935, 1.785132, 1.106453, 2.898094, 1.307998, 0.681423),
}


@pytest.mark.slow
def test_real_corpus_persistence_reproduces_the_p2d9_table(
    real_corpus: Path,
) -> None:
    """The P2-D9 denominators, re-measured at the P3 geometry.

    Every cell of :data:`PERSISTENCE_RMSE_ID_TEST` is pinned, not just roll: the P3 task
    forecasts all six DOFs, so all six denominators are load-bearing.
    """
    from dmf.data.splits import load_manifest
    from dmf.eval.controls import persistence_pipeline_sanity

    cfg = PRODUCTION_CFG
    spec = PRODUCTION_SPEC
    split = build_split(load_manifest(real_corpus), "id")
    train = DeckMotionDataset(real_corpus, split, "train", cfg, spec)
    test = DeckMotionDataset(real_corpus, split, "test", cfg, spec, stats=train.norm_stats)
    model = Persistence(spec.lookback, spec.max_horizon, N_IN, N_OUT).eval()
    result = persistence_pipeline_sanity(test, real_corpus, model=model, batch_size=4096)
    assert result.model_matches_inline is True
    assert result.max_rel_diff <= 1e-6
    assert result.n_windows == 434_304
    assert sorted(PERSISTENCE_RMSE_ID_TEST) == sorted(cfg.horizons)

    dofs = tuple(cfg.target_dofs)
    for horizon, row in PERSISTENCE_RMSE_ID_TEST.items():
        assert len(row) == len(dofs)
        for channel, (dof, value) in enumerate(zip(dofs, row, strict=True)):
            measured = result.rmse_pipeline[horizon - 1, channel]
            assert measured == pytest.approx(value, abs=5e-6), (horizon, dof, measured)

    # Non-monotone in horizon, and that is the physics rather than a bug: persistence error
    # tracks the signal's autocorrelation, and 100 samples (10 s) is close to one roll
    # period on this hull, so roll persistence is *better* at 10 s than at 5 s. This is why
    # skill vs persistence is not comparable across horizons on this signal and normalised
    # RMSE is reported alongside it (``configs/data/default.yaml``, ``docs/protocol.md`` P3).
    roll = result.rmse_pipeline[:, dofs.index("roll")]
    assert roll[99] < roll[49]


#: Every (regime, DOF, horizon) cell on the real corpus where AR(20) scores **negative**
#: skill against persistence, measured at the P3 geometry over all four regimes, six DOFs
#: and six horizons -- 144 cells, of which these 12 are losses. Recorded rather than
#: exempted (CLAUDE.md non-negotiable 6); the test below asserts this set exactly, so a
#: cell that starts or stops losing fails the suite instead of passing quietly.
#:
#: ``unseen_heading`` / ``pitch`` and ``pitch_rate``, horizons 20-150. The P1-D2 residual
#: floor: the ``unseen_heading`` test set *is* beam seas, where the pitch heading factor is
#: clamped at ``eps = 0.05``, about 26 dB down, so the test-set pitch signal is the
#: engineering stand-in for hull asymmetry rather than the pitch physics the model was
#: trained on at 180/135/45 deg. AR extrapolates onto it catastrophically (skill -0.5 at
#: 2 s, past -39 at 5 s) while persistence, which needs no training distribution, does
#: fine. ``pitch_rate`` is new at P3 and fails for the same reason and harder -- it is the
#: derivative of that same clamped signal, so it inherits the floor with the noise
#: differentiated up (-6.9 already at 2 s, -57 at 10 s). Only the 1 s horizon survives, and
#: only just for the rate channel (+0.15).
#:
#: ``unseen_vessel`` / ``roll`` and ``roll_rate`` at 150 samples. **New at P3 and invisible
#: before it**, because the pre-P3 task stopped at 50 samples. This is not AR degrading:
#: its own roll RMSE grows smoothly with lead time (0.007, 0.054, 0.19, 0.76, 1.85,
#: 2.03 deg). The *denominator* collapses. 150 samples is 15 s, and the held-out S-175
#: rolls at ``tn_s = 14.5``, so at that lead the hull has come back around almost exactly
#: one roll period and persistence is nearly free: its roll RMSE falls from 3.91 deg at 5 s
#: to 1.68 deg at 15 s. AR, whose coefficients encode the *frigate*'s 12 s roll mode, has no
#: way to know that and predicts 15 s ahead at the wrong period. Bootstrap CI over
#: realizations is [-0.64, -0.26] for roll and [-0.78, -0.33] for roll_rate, so it is a
#: transfer failure, not resampling noise. It is the same recurrence effect
#: ``configs/data/default.yaml`` warns about, at a period the training hull does not have.
AR20_NEGATIVE_SKILL_CELLS: frozenset[tuple[str, str, int]] = frozenset(
    [("unseen_heading", dof, h) for dof in ("pitch", "pitch_rate") for h in (20, 30, 50, 100, 150)]
    + [("unseen_vessel", dof, 150) for dof in ("roll", "roll_rate")]
)


@pytest.mark.slow
@pytest.mark.parametrize("regime", REGIMES)
def test_real_corpus_ar_beats_persistence_at_every_horizon(
    real_corpus: Path, regime: Regime
) -> None:
    """AR(20) is the Gate 3 subject; a regime where it loses to persistence is a finding.

    The finding is encoded, not worked around: :data:`AR20_NEGATIVE_SKILL_CELLS` lists the
    losing cells and this test asserts the sign of **every** cell against it, so neither a
    new loss nor a silent repair can slip through.
    """
    from dmf.data.splits import load_manifest
    from dmf.eval.runner import evaluate_models

    cfg = PRODUCTION_CFG
    spec = PRODUCTION_SPEC
    split = build_split(load_manifest(real_corpus), regime)
    train = DeckMotionDataset(real_corpus, split, "train", cfg, spec)
    test = DeckMotionDataset(real_corpus, split, "test", cfg, spec, stats=train.norm_stats)
    moments = accumulate_training_moments(train, max_order=20, num_workers=4)
    model, _ = fit_ar(
        moments,
        order=20,
        ridge=1e-6,
        lookback=spec.lookback,
        n_input_channels=N_IN,
        n_target_channels=N_OUT,
    )
    table, _ = evaluate_models(
        {
            "persistence": Persistence(spec.lookback, spec.max_horizon, N_IN, N_OUT).eval(),
            "ar20": model,
        },
        test,
        persistence_key="persistence",
        horizons=cfg.horizons,
        fs_hz=cfg.fs_hz,
        num_workers=4,
    )
    persistence_rows = table[table["model"] == "persistence"]
    assert (persistence_rows["skill"] == 0.0).all(), "the reference must score exactly 0.0"

    ar_rows = table[table["model"] == "ar20"]
    assert len(ar_rows) == len(cfg.target_dofs) * len(cfg.horizons)
    expected_losses = {
        (dof, horizon) for r, dof, horizon in AR20_NEGATIVE_SKILL_CELLS if r == regime
    }
    measured_losses = {
        (str(row.dof), int(row.horizon_samples)) for row in ar_rows.itertuples() if row.skill <= 0.0
    }
    assert measured_losses == expected_losses, ar_rows.loc[
        ar_rows["skill"] <= 0.0, ["dof", "horizon_samples", "skill"]
    ].to_dict("records")


def test_unused_import_guard() -> None:
    """Keep ``replace`` and ``ModelConfig`` referenced; both are used by the config tests."""
    cfg = ModelConfig(name="persistence", head="point", quantiles=(), params={}, label="p")
    assert replace(cfg, label="q").label == "q"
    assert SMALL_LOOKBACK > 0
