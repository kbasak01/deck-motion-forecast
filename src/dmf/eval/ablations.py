"""The ablation arm registry, and the matching rules a contrast is only valid under.

Five ablations, six new arms, one reference. ``docs/protocol.md`` P6-D4 fixes what is
matched and what cannot be; this module is that entry as code, so that a contrast which
violates it raises instead of rendering.

**1. Lookback is matched on forecast origin, never on window start.** For lookback ``L`` a
window starting at sample ``s`` observes ``s .. s+L-1`` and forecasts ``s+L+h``. Matching on
``s`` would place an L=100 forecast of absolute time ``s+100+h`` beside an L=400 forecast of
``s+400+h`` -- a comparison of forecasts of *different times*, which is the easiest thing
here to get silently wrong because both tables look perfectly well formed. The quantity to
match is the origin ``t = s + L - 1``. :func:`matched_origins` computes the intersection,
:func:`origin_window_indices` maps it back to window indices per arm, and
:func:`start_matching_compares_different_times` exists so a test can assert the wrong rule
is wrong rather than assert the right rule is right.

**2. Parameter matching is a property of the row, not only of the arm.** The arm-level
``parameter_matched`` column is False for both lookback arms, which is the conservative
arm-wide reading; P6-D4 item 2 corrected the blanket claim that *nothing* there is matched.
Measured: DLinear is channel-shared, so ``dlinear_ols`` is ``2*L*H + 2*H`` parameters and is
**exactly matched on the channels arm** (60 300 either way) while scaling 30 300 / 60 300 /
120 300 with lookback; a dilated convolution's weights do not scale with input length, so
``tcn`` is **exactly matched between L=100 and L=200** (196 804 both) and unmatched only at
L=400. The per-row reading is therefore carried by ``param_delta_frac`` and
``capacity_confounded``, both measured from the two rows' own ``n_params``. Hedging a
comparison that is in fact clean is not conservatism; it discards a result.

**3. The channels arm is read on the three shared DOFs only.** It forecasts roll, pitch and
heave; the rate rows do not exist in it and are not imputed.

**4. The ``imu`` arm is never compared to ``ideal`` on raw RMSE.** Both input and target
are ``imu`` (P2-D8), so the two modes forecast **different targets** with different
``signal_std``, and RMSE, MAE and every other raw-scale column are numbers in different
scales. **This is not a claim about the persistence denominator.** P6-D23 corrects the
version of this entry that said ``heave_imu``'s ~1.3 s lead "roughly halves the 1 s
persistence denominator": that 0.218/0.437 pair is P1-D6's *cross-mode* pairing, which P2-D8
makes structurally impossible and which this arm does not use. Measured same-mode on ``id``,
``imu`` persistence over ``ideal`` persistence is **0.4361 / 0.4417 = 0.9872** at heave / 1 s
and 0.9706-1.0008 (mean 0.9963) over every DOF and horizon -- the denominators agree to about
1%, so skill does cross the boundary meaningfully and only the raw scales do not.
:data:`CROSS_MODE_COLUMNS` names the two columns that may cross that boundary -- ``skill``
and ``nrmse``, both dimensionless and both normalised by a quantity measured inside the same
observation mode -- and :func:`assert_columns_comparable` raises on any other. The arm's rows
also carry different ``dof`` *labels* (``roll_imu``, not ``roll``), so a naive join on ``dof``
returns an **empty** frame rather than a wrong one: silent, and an empty ablation table
renders as "no rows". Joins therefore go through :func:`with_logical_dof`, which uses
:func:`dmf.data.channels.logical_channel` rather than a suffix test, and
:func:`contrast_table` asserts the joined row count.

**5. Sea-state conditioning consumes privileged information, and is not an upper bound.**
At deployment the sea state is estimated online from the same motion record the forecaster
consumes, so a ground-truth one-hot is information the deployed system does not have: that
is what ``privileged_information`` records, and it is why no row of this arm is a deployable
result. The plan called the arm an *upper bound*; **P6-D18 measured that it is not one**, and
this module does not render the retracted claim. Conditioning buys +0.0014 mean skill on
``id`` (+0.0049 at ``tcn``/pitch/``horizon_samples = 100``, paired CI [+0.0040, +0.0058],
P6-D20) and costs -0.0445 mean on ``unseen_seastate``, where 24 of 36 cells exclude zero and
all 24 are negative. An upper bound that lies below the unconditioned baseline on the regime
that matters is not an upper bound on anything. Two further defects (P6-D8) are carried as
columns rather than as prose:

- ``vehicle_blind_to_arm`` -- ``dlinear``, ``dlinear_ols`` and the three trivial baselines
  slice the leading ``C_out`` input channels and are **bitwise** blind to the indicator, so
  their contrast on this arm is exactly zero as a property of the architecture. Dropping
  those rows would breach ``CLAUDE.md`` non-negotiable 6; printing them unflagged would
  invite "no effect".
- ``capacity_confounded`` -- the arm changes this model's parameter count, so a difference
  is not attributable to the information change alone. **Measured from the two rows' own
  ``n_params``**, not asserted from a model name: on ``ss_conditioned`` the one-hot costs
  ``ar20`` +66.12% (108 900 -> 180 900) where ``tcn`` pays +0.52% and the channel-shared
  ``dlinear_ols`` pays nothing (P6-D8 defect 4). The measured rule generalises the entry
  rather than transcribing it, and it fires in the other direction too -- see
  :func:`capacity_is_confounded`.
- ``not_fittable`` -- ``ar20`` cannot be fitted on this arm in ``unseen_seastate`` at all:
  the SS6 indicator is constant zero in that regime's training split and the normal
  equations are singular. The row ships with the reason instead of being absent, because an
  absent row and a row with nothing to report are indistinguishable to a reader.

**6. The observation mode is a column, never inferred at read time.** A suffix test is one
renamed channel away from being wrong, and the arm registry already knows the answer.

**7. The join is on the *logical* vehicle, not on the model label.** The L=400 arm's deep
vehicle is labelled ``tcn_l400`` and its reference's is ``tcn``, so a join keyed on ``model``
dropped it silently: it was fitted, scored and committed, and reached no rendered table and
no ``not_fittable`` reason -- the exact silent-drop failure ``with_logical_dof`` exists to
prevent, one column over. :func:`with_logical_model` is the same device for the vehicle, and
the architectural difference the two labels record is carried as ``model_reference``,
``architecture_differs`` and ``param_delta_frac`` (221 636 vs 196 804 parameters, +12.6%,
from the extra dilation stage the 40 s receptive field needs) rather than by dropping the
row, which ``CLAUDE.md`` non-negotiable 6 forbids.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dmf.config import ObservationMode
from dmf.data.channels import logical_channel
from dmf.data.splits import RealizationKey
from dmf.data.windows import WindowSpec, n_windows
from dmf.data.windows import window_origins as _origins_of
from dmf.eval.controls import RESIDUAL_FLOOR_MARGIN, floored_dofs

# Private on purpose, and imported on purpose. The whole point of a *paired* bootstrap is
# that both sides are weighted by the SAME draw, so the two functions must not each own a
# generator: `_bootstrap_counts` is a pure function of `(n_units, n_boot, seed)` and reusing
# it is what makes the pairing exact rather than approximate. `_percentile_interval` is
# imported for the same reason -- one definition of "central 95%", not two.
from dmf.eval.runner import _bootstrap_counts, _percentile_interval
from dmf.typedefs import FloatArray, IntArray

__all__ = [
    "ABLATIONS",
    "ABLATION_BOOTSTRAP_SEED",
    "ABLATION_CI_LEVEL",
    "ABLATION_N_BOOT",
    "ARMS",
    "CAPACITY_CONFOUND_TOL",
    "CELL_KEYS",
    "CI_MIN_UNITS",
    "CI_POINT_TOL",
    "CI_RESAMPLE_UNIT",
    "CONTRAST_CI_COLUMNS",
    "CONTRAST_COLUMNS",
    "CROSS_MODE_COLUMNS",
    "JOIN_KEYS",
    "MATCHED_ORIGIN_COLUMNS",
    "RAW_SCALE_COLUMNS",
    "REFERENCE_ARM",
    "REFERENCE_RESULTS_DIR",
    "AblationArm",
    "architecture_differs",
    "arms_for_experiment",
    "assert_columns_comparable",
    "capacity_is_confounded",
    "cell_floored_dofs",
    "contrast_table",
    "contrast_uncertainty",
    "logical_model",
    "matched_origin_provenance",
    "matched_origins",
    "matched_window_counts",
    "not_fittable_reason",
    "not_fittable_rows",
    "origin_window_indices",
    "parameter_delta_fraction",
    "parameter_is_matched",
    "start_matching_compares_different_times",
    "unmatched_origin_reason",
    "vehicle_is_blind",
    "window_origins",
    "with_logical_dof",
    "with_logical_model",
]

#: Committed results directory the reference arm is *read* from. ``results/e02/`` fitted both
#: vehicles under a byte-identical train block and ``configs/data/default.yaml``; it is a
#: Gate 4 audit trail and is never regenerated.
REFERENCE_RESULTS_DIR: Path = Path("results/e02")

#: Arm key of the unablated reference.
REFERENCE_ARM: str = "reference"

#: The columns that may be compared across the ``ideal``/``imu`` boundary, and the only
#: ones (P6-D4 item 4). Both are dimensionless and both are normalised by a quantity
#: measured *inside* the same observation mode, so neither carries that mode's scale.
CROSS_MODE_COLUMNS: tuple[str, ...] = ("skill", "nrmse")

#: Columns carrying the observation mode's own scale. Comparing any of these across the
#: mode boundary is refused, not warned about.
RAW_SCALE_COLUMNS: tuple[str, ...] = (
    "rmse",
    "mae",
    "rmse_persistence",
    "signal_std",
    "sse",
    "sae",
    "mean_width",
    "pinball",
    "crps",
    "winkler",
)

#: Columns a contrast is joined on. Two of the four are *logical* names rather than the
#: labels the tables carry, and for the same reason in both cases -- an arm may spell a thing
#: differently from its reference, and an inner join on the spelling drops the row without
#: saying so.
#:
#: - ``logical_dof`` rather than ``dof``: the ``imu`` arm's rows are keyed on the resolved
#:   corpus columns (``roll_imu``), so a join on ``dof`` yields an empty frame.
#: - ``logical_model`` rather than ``model``: the L=400 arm's deep vehicle is ``tcn_l400``
#:   and its reference's is ``tcn``. Joining on ``model`` silently dropped 216 committed,
#:   fitted, scored rows from every rendered table -- no empty frame to notice, no
#:   ``not_fittable`` reason, just an arm that looked as though it had only closed-form
#:   vehicles. See :func:`with_logical_model`.
JOIN_KEYS: tuple[str, ...] = ("logical_model", "regime", "logical_dof", "horizon_samples")


@dataclass(frozen=True)
class AblationArm:
    """One arm of one ablation, and everything a fair contrast against it needs.

    Attributes:
        ablation: Which ablation this arm belongs to, e.g. ``"lookback"``.
        arm: Arm key, unique across :data:`ARMS`.
        experiments: Experiment config stems that produce this arm's tables. Two per arm,
            because ``ExperimentConfig.regimes`` applies to every model in the list and the
            deep and closed-form vehicles have different budgets; the ``_ood`` file carries
            the closed-form vehicles on the two out-of-distribution regimes.
        data_config: The ``configs/data/*.yaml`` stem that defines the arm.
        reference: Arm key this arm is contrasted against.
        observation_mode: ``"ideal"`` or ``"imu"``. A column, never inferred from a channel
            suffix at read time.
        lookback: Input window length, samples.
        comparable_dofs: Logical channels the contrast is read on. All six except on the
            channels arm, which forecasts three.
        parameter_matched: What the arm was *built* to hold fixed. An **arm-level design
            intent**, and a coarse one: it is True for ``attitude_only`` because the arm's
            closed-form vehicle ``dlinear_ols`` is exactly matched there (60 300 either way,
            P6-D4 item 2), while ``ar20`` on the same arm drops from 108 900 to 27 450 and
            ``tcn`` from 196 804 to 166 786. It therefore **disagrees with the measurement
            on real rows**, and it is rendered under its own name,
            ``arm_parameter_matched``, for exactly that reason: the row-level
            ``parameter_matched`` is derived from ``param_delta_frac`` by
            :func:`parameter_is_matched` and is the negation of ``capacity_confounded``.
            Publishing this field as ``parameter_matched`` put ``yes`` beside
            ``param_delta_frac = -0.1525`` and ``capacity_confounded = yes`` on one row and
            ``no`` beside ``param_delta_frac = 0.0000`` on another -- both backwards.
        privileged_information: Whether the arm feeds the model something the deployed
            system cannot observe. True only for sea-state conditioning, whose one-hot is
            ground truth where deployment has an online estimate (P6-D4 item 5).
            **This column was called ``upper_bound`` and the rename is the point**: the plan
            and P6-D4 both called this arm an upper bound on deployable performance, and
            P6-D18 measured that it is not one -- conditioning costs -0.0445 mean skill on
            ``unseen_seastate``, so the "bound" lies below the thing it was supposed to
            bound. What survives the retraction is that the information is not available at
            deployment, which is what this flag now says and all it says.
        note: One-line reading instruction carried into the table.
    """

    ablation: str
    arm: str
    experiments: tuple[str, ...]
    data_config: str
    reference: str
    observation_mode: ObservationMode
    lookback: int
    comparable_dofs: tuple[str, ...]
    parameter_matched: bool
    privileged_information: bool
    note: str


#: The six motion channels, logical spelling.
_ALL_DOFS: tuple[str, ...] = ("roll", "pitch", "heave", "roll_rate", "pitch_rate", "heave_rate")

#: The three the channels arm forecasts.
_ATTITUDE_DOFS: tuple[str, ...] = ("roll", "pitch", "heave")

#: The arm registry. ``reference`` and ``lookback_20s`` are the same task definition read
#: two different ways: ``reference`` is ``results/e02/`` as committed, while
#: ``lookback_20s`` is that geometry **re-scored on the matched origin set** from e02's
#: checkpoints (a forward pass, not a refit), because the lookback contrast is only paired
#: on the intersection of the three origin sets.
ARMS: dict[str, AblationArm] = {
    REFERENCE_ARM: AblationArm(
        ablation="reference",
        arm=REFERENCE_ARM,
        experiments=("e02_deep",),
        data_config="default",
        reference=REFERENCE_ARM,
        observation_mode="ideal",
        lookback=200,
        comparable_dofs=_ALL_DOFS,
        parameter_matched=True,
        privileged_information=False,
        note="unablated reference, read from results/e02/ and never regenerated",
    ),
    "imu": AblationArm(
        ablation="observation_mode",
        arm="imu",
        experiments=("e04a_obs_mode", "e04a_obs_mode_ood"),
        data_config="imu",
        reference=REFERENCE_ARM,
        observation_mode="imu",
        lookback=200,
        comparable_dofs=_ALL_DOFS,
        parameter_matched=True,
        privileged_information=False,
        note=(
            "input and target are both imu; skill and nrmse cross the mode boundary, raw "
            "RMSE never does (P6-D4 item 4)"
        ),
    ),
    "attitude_only": AblationArm(
        ablation="channels",
        arm="attitude_only",
        experiments=("e04b_channels", "e04b_channels_ood"),
        data_config="attitude_only",
        reference=REFERENCE_ARM,
        observation_mode="ideal",
        lookback=200,
        comparable_dofs=_ATTITUDE_DOFS,
        parameter_matched=True,
        privileged_information=False,
        note=(
            "forecasts roll/pitch/heave only; the rate rows do not exist here and are not "
            "imputed, and the quiescence metric cannot be evaluated on this arm"
        ),
    ),
    "ss_conditioned": AblationArm(
        ablation="sea_state_conditioning",
        arm="ss_conditioned",
        experiments=(
            "e04c_ss_conditioned",
            "e04c_ss_conditioned_ood",
            "e04c_ss_conditioned_ar_id",
        ),
        data_config="ss_conditioned",
        reference=REFERENCE_ARM,
        observation_mode="ideal",
        lookback=200,
        comparable_dofs=_ALL_DOFS,
        parameter_matched=False,
        privileged_information=True,
        note=(
            "PRIVILEGED INFORMATION, and NOT an upper bound (P6-D18 supersedes P6-D4 item "
            "5): the one-hot is ground-truth sea state, which deployment estimates online "
            "rather than knows, and conditioning on it costs -0.0445 mean skill on "
            "unseen_seastate while buying +0.0014 on id. DLinear and the three trivial "
            "baselines are bitwise blind to the indicator (P6-D8 defect 1)"
        ),
    ),
    "lookback_10s": AblationArm(
        ablation="lookback",
        arm="lookback_10s",
        experiments=("e04d_lookback_10s", "e04d_lookback_10s_ood"),
        data_config="lookback_10s",
        reference="lookback_20s",
        observation_mode="ideal",
        lookback=100,
        comparable_dofs=_ALL_DOFS,
        parameter_matched=False,
        privileged_information=False,
        note="scored on the L=400 origin intersection; drops its leading 60 windows",
    ),
    "lookback_20s": AblationArm(
        ablation="lookback",
        arm="lookback_20s",
        experiments=("e02_deep",),
        data_config="default",
        reference="lookback_20s",
        observation_mode="ideal",
        lookback=200,
        comparable_dofs=_ALL_DOFS,
        parameter_matched=False,
        privileged_information=False,
        note=(
            "the lookback reference, RE-SCORED from e02's checkpoints on the matched origin "
            "set; drops its leading 40 windows. Not the committed results/e02/ rows"
        ),
    ),
    "lookback_40s": AblationArm(
        ablation="lookback",
        arm="lookback_40s",
        experiments=("e04e_lookback_40s", "e04e_lookback_40s_ood"),
        data_config="lookback_40s",
        reference="lookback_20s",
        observation_mode="ideal",
        lookback=400,
        comparable_dofs=_ALL_DOFS,
        parameter_matched=False,
        privileged_information=False,
        note="the binding arm: its 1091 origins are the intersection the others match",
    ),
    "revin": AblationArm(
        ablation="normalization",
        arm="revin",
        experiments=("e04f_revin", "e04f_revin_ood"),
        data_config="revin",
        reference=REFERENCE_ARM,
        observation_mode="ideal",
        lookback=200,
        comparable_dofs=_ALL_DOFS,
        parameter_matched=False,
        privileged_information=False,
        note="affine RevIN adds 2*C_in parameters; the counts are printed beside the rows",
    ),
}

#: Ablation -> its arms, reference first. Derived from :data:`ARMS` so the two cannot drift.
ABLATIONS: dict[str, tuple[str, ...]] = {
    ablation: tuple(sorted(key for key, arm in ARMS.items() if arm.ablation == ablation))
    for ablation in sorted({arm.ablation for arm in ARMS.values()})
}

#: Models that cannot read the arm's feature at all, by arm. Measured, not inferred:
#: ``src/dmf/models/dlinear.py`` slices ``x[:, :, :C_out]``, and two forward passes
#: differing only in the one-hot are bitwise identical (P6-D8 defect 1).
#:
#: P6-D8 named ``dlinear`` and ``dlinear_ols`` because DLinear was the *designated* vehicle
#: for this arm. The three trivial baselines take the same slice --
#: ``persistence.py:87`` is ``x[:, -1:, : self.n_target_channels]`` and
#: ``persistence.py:344`` (``DampedPersistence``, which ``WindowMean`` subclasses) is
#: ``x[:, -1:, : self.n_target_channels]`` -- so they are blind by exactly the same
#: mechanism and their contrast on this arm is exactly zero for exactly the same reason.
#: Confirmed on the committed rows: over the 72 joined ``unseen_heading``/``unseen_vessel``
#: cells of ``results/e04/e04c_ss_conditioned_ood/`` against ``results/e02/``, the skill
#: difference is 0.0 to the last bit for ``persistence``, ``window_mean``,
#: ``damped_persistence`` and ``dlinear_ols``, and non-zero only for ``ar20``. Leaving the
#: three unflagged would have published four more "no effect" rows than P6-D8 anticipated.
_VEHICLE_BLIND: dict[str, frozenset[str]] = {
    "ss_conditioned": frozenset(
        {"dlinear", "dlinear_ols", "persistence", "window_mean", "damped_persistence"}
    ),
}

#: The vehicle each arm-specific model label stands in for, when the label differs from the
#: reference arm's. **The one entry on this corpus is why B1 of the second Phase 6 audit
#: exists**: ``configs/model/tcn_l400.yaml`` is registered as ``tcn_l400`` because its
#: dilation stack differs, the L=400 arm therefore committed 216 rows labelled ``tcn_l400``,
#: and an inner join on ``model`` against a reference labelled ``tcn`` matched none of them.
#: The arm rendered as though it had no deep vehicle at all -- fitted, scored, committed, and
#: absent from every table, with no ``not_fittable`` reason to notice.
#:
#: **Not derived from the model registry**, which would be the obvious mechanisation and is
#: wrong: ``ar10``, ``ar20`` and ``ar40`` share a class and are three different models, so a
#: class-keyed rule would collapse a genuine model comparison. The mapping is a declaration
#: about which label an *arm* uses for the reference's vehicle, and
#: ``tests/test_ablations.py`` checks it against the committed tables rather than trusting
#: it: every fitted (arm, model) must reach a contrast row or carry a reason.
_LOGICAL_MODEL: dict[str, str] = {"tcn_l400": "tcn"}

#: Relative parameter-count change above which an arm is recorded as changing this model's
#: capacity as well as its information set. **Not a tuning knob**, for the reason
#: :data:`dmf.eval.controls.RESIDUAL_FLOOR_MARGIN` is not: the measured values on this
#: corpus are +0.52% (``tcn``, ``ss_conditioned``), 0.00% (``dlinear_ols``, channel-shared)
#: and +66.12% (``ar20``, ``ss_conditioned``), so every threshold in (0.006, 0.66) selects
#: the same rows -- two orders of magnitude of slack.
CAPACITY_CONFOUND_TOL: float = 0.05

#: ``arm -> models`` whose capacity the arm changes, used **only** where the two rows carry
#: no ``n_params`` to measure from -- i.e. on the :func:`not_fittable_rows` placeholders,
#: which have no fit and therefore no parameter count. The values are P6-D8 defect 4 as
#: measured there (``ar20``: 108 900 -> 180 900, +66.12%), and
#: ``tests/test_ablations.py`` asserts this fallback agrees with the measured flag on the
#: committed rows, so the declaration is checked against data rather than trusted.
_CAPACITY_CONFOUNDED: dict[str, frozenset[str]] = {
    "ss_conditioned": frozenset({"ar20"}),
}

#: ``(arm, regime, model)`` -> why no fit exists. P6-D8 defect 2, reproduced on the real
#: corpus rather than predicted from the source.
_NOT_FITTABLE: dict[tuple[str, str, str], str] = {
    ("ss_conditioned", "unseen_seastate", "ar20"): (
        "not fittable: unseen_seastate/train is SS3/SS4/SS5, so the SS6 indicator is "
        "constant zero and the normal equations are singular. Sea-state conditioning is "
        "structurally undefined for the regime that would most need it (P6-D8 defect 2)"
    ),
}

#: Number of bootstrap resamples every ablation interval is drawn at. The value
#: :func:`dmf.eval.runner.paired_skill_difference_ci` defaults to, restated here so the two
#: paths cannot silently diverge.
ABLATION_N_BOOT: int = 1000

#: Central confidence level of every ablation interval.
ABLATION_CI_LEVEL: float = 0.95

#: Seed of the resampling generator. Fixed, so the interval is a function of the committed
#: CSVs and nothing else.
ABLATION_BOOTSTRAP_SEED: int = 0

#: What one resample draw is. **Not the realization**, which is what
#: :func:`dmf.eval.runner.bootstrap_skill_ci` resamples inside a training run: no committed
#: table carries per-realization SSE, and ``baselines_by_cell.csv`` is the finest committed
#: granularity there is. One unit is therefore a corpus grid cell -- 8 realizations on
#: ``id``/``unseen_vessel``, 40 on the two out-of-distribution regimes.
#:
#: **This makes the interval wider than a realization bootstrap of the same contrast, and by
#: a measured factor**, because a cluster resample of a fixed factorial design charges the
#: statistic for between-cell heterogeneity that is design, not sampling. The direction is
#: therefore structural: an interval that excludes zero here would also exclude zero under a
#: realization bootstrap, and an interval that contains zero may be doing so only because of
#: the coarser unit.
#:
#: **The magnitude is a measurement without a committed producer, and this docstring used to
#: point at the wrong place for it.** It said "the ratio is recorded in
#: ``docs/protocol.md``"; no entry there carries it. The four numbers
#: (median 2.8 on ``id`` and ``unseen_vessel``, 6.3-7.1 on the two out-of-distribution
#: regimes, narrower in 1 of 5616) live in ``dmf.eval.report`` as prose and nowhere else.
#: They are **reconstructible read-only** from two committed files -- resample the grid cells
#: of ``results/e02/baselines_by_cell.csv`` under the same
#: ``(n_units, n_boot, seed)`` draw and compare each width against
#: ``results/e02/paired_contrasts.csv`` -- and the rendered document says both that they have
#: no producer and how to check them. Until a producer exists, they are the one substantive
#: quantity in ``results.md`` outside Gate 6's traceability contract.
CI_RESAMPLE_UNIT: str = "grid_cell"

#: Largest discrepancy tolerated between the skill difference recomputed from the per-cell
#: SSE and the one the contrast row carries. They are the same quantity -- pooled skill is
#: ``1 - sum(sse) / sum(sse_persistence)`` over exactly these cells -- so a mismatch means
#: the cell table and the metric row describe different window populations, and an interval
#: drawn on the first would be attached to the second.
CI_POINT_TOL: float = 1e-6

#: Fewest resampling units an interval will be drawn from. Two cells give a bootstrap that
#: is a coin flip between two numbers; printing a percentile interval of it would be
#: decoration.
CI_MIN_UNITS: int = 4

#: The uncertainty columns every contrast row carries, and the provenance that says what
#: they were drawn under. ``skill_diff_ci_reason`` is non-empty exactly when the two bounds
#: are NaN, and it names which precondition failed -- P5-D12 records that an unpaired delta
#: reported as paired is a mistake this project has already made, so an unpairable arm emits
#: no interval at all rather than an unpaired one.
CONTRAST_CI_COLUMNS: tuple[str, ...] = (
    "skill_diff_ci_lo",
    "skill_diff_ci_hi",
    "skill_diff_ci_point",
    "skill_diff_ci_reason",
    "ci_resample_unit",
    "ci_n_units",
    "ci_n_realizations",
    "n_boot",
    "ci_level",
    "bootstrap_seed",
)

#: Column order of the contrast table.
CONTRAST_COLUMNS: tuple[str, ...] = (
    "ablation",
    "arm",
    "reference_arm",
    "model",
    # The vehicle the row is joined on, and the reference arm's own label for it. They
    # differ on exactly one pair on this corpus -- `tcn_l400` against `tcn` -- and printing
    # both is what stops the lookback contrast reading as a comparison of one network with
    # itself at a different input length. `architecture_differs` is the flag; the two labels
    # are the evidence for it.
    "logical_model",
    "model_reference",
    "architecture_differs",
    "regime",
    "observation_mode",
    "reference_observation_mode",
    "logical_dof",
    "dof",
    "dof_reference",
    "horizon_samples",
    # Carried from the arm's rows when they have it, NaN otherwise. Not decoration: the
    # section 6.3 table is displayed on `horizon_s`, and without it six rows of one cell
    # render as six identical-looking rows with no column saying which lead time each is.
    "horizon_s",
    "lookback",
    "reference_lookback",
    # Measured per row from `param_delta_frac`, exactly as `capacity_confounded` is, and
    # exactly the negation of it (P6-D4 item 2, and B2 of the second Phase 6 audit). The
    # arm-level design intent is `arm_parameter_matched` beside it, and the two disagree on
    # real rows: the channels arm is *built* to hold parameters fixed and does so for its
    # closed-form vehicle only, so `attitude_only`/`tcn` is arm-matched and row-unmatched
    # at -15.25%. Rendering the arm's intent under the row's name is how a row came to read
    # `parameter_matched = yes, param_delta_frac = -0.1525, capacity_confounded = yes`.
    "parameter_matched",
    "arm_parameter_matched",
    "privileged_information",
    "vehicle_blind_to_arm",
    "capacity_confounded",
    "n_params",
    "n_params_reference",
    "param_delta_frac",
    "not_fittable",
    "raw_rmse_comparable",
    # The P1-D2 residual floor, derived by `dmf.eval.controls.floored_dofs` from the cells
    # the arm was actually scored on -- never re-derived here, and never asserted from a
    # regime name. Sixth flag of the five ABLATION_FLAG_COLUMNS already carried, and it is
    # the one that stops `unseen_heading`/`ar20`/`pitch_rate`/10 s reading as a finding.
    # Five arms score that cell and their skill_diff runs -0.000003 (`revin`), +0.000585
    # (`ss_conditioned`), +3.29 (`lookback_40s`), +30.30 (`lookback_10s`), +41.67 (`imu`)
    # against a signal_std of 0.0658 deg/s -- read the whole set rather than one member of
    # it (P6-D17). The unfloored rows of the same file have median |skill_diff| 0.0002.
    "on_residual_floor",
    "note",
    *CONTRAST_CI_COLUMNS,
)


def arms_for_experiment(name: str) -> tuple[AblationArm, ...]:
    """Return every arm an experiment config feeds.

    Plural, and deliberately so: ``e02_deep`` feeds **two** arms. It is the unablated
    ``reference`` when read as committed from ``results/e02/``, and it is ``lookback_20s``
    when its checkpoints are re-scored on the matched origin set. Those are two different
    row sets from one training run, and a function that returned "the" arm would have to
    pick one silently.

    Args:
        name: Experiment config stem, e.g. ``"e04d_lookback_10s_ood"``.

    Returns:
        The claiming arms, in registry order.

    Raises:
        ValueError: If no arm claims that experiment.
    """
    claimants = tuple(arm for arm in ARMS.values() if name in arm.experiments)
    if not claimants:
        known = sorted({experiment for arm in ARMS.values() for experiment in arm.experiments})
        raise ValueError(
            f"no ablation arm claims experiment {name!r}; known experiments are {known}"
        )
    return claimants


def logical_model(model: str) -> str:
    """Return the vehicle a model label stands in for across arms.

    The model twin of :func:`dmf.data.channels.logical_channel`, and it exists for the same
    failure: an arm that spells a thing differently from its reference joins to nothing, and
    an inner join that matches nothing is silent. ``tcn_l400`` is the L=400 arm's label for
    the vehicle the reference arm calls ``tcn``; they are **not the same network** (221 636
    parameters against 196 804, from the extra dilation stage a 40 s receptive field needs),
    which is carried on the row as ``architecture_differs`` and ``param_delta_frac`` rather
    than resolved away here. Mapping them to one logical vehicle is what makes the contrast
    render at all; the flags are what stop it being read as one network at two lookbacks.

    Args:
        model: Model label as a committed table spells it.

    Returns:
        The logical vehicle, or ``model`` itself when the label needs no mapping.
    """
    return _LOGICAL_MODEL.get(model, model)


def architecture_differs(model: str, reference_model: str) -> bool:
    """Report whether two labels joined as one logical vehicle are two networks.

    Args:
        model: The arm's model label.
        reference_model: The reference arm's label for the same logical vehicle.

    Returns:
        True when the labels differ, i.e. the contrast pairs two architectures rather than
        one architecture under two conditions. Never inferred from the parameter counts: two
        networks can coincide in parameter count, and one network's count can change with
        the ablation (the ``ss_conditioned`` one-hot costs ``ar20`` 66%).
    """
    return model != reference_model


def parameter_is_matched(
    param_delta_frac: float = float("nan"),
    *,
    tol: float = CAPACITY_CONFOUND_TOL,
) -> bool | None:
    """Report whether the two rows of a contrast hold the parameter count fixed.

    Exactly the negation of :func:`capacity_is_confounded` where both are measurable, and
    computed from the same number by the same threshold so that the two cannot contradict
    each other on one row. They did: the arm-level flag rendered
    ``parameter_matched = yes`` beside ``param_delta_frac = -0.1525`` and
    ``capacity_confounded = yes`` on ``channels``/``attitude_only``/``tcn``, and
    ``parameter_matched = no`` beside ``param_delta_frac = 0.0000`` on
    ``lookback``/``lookback_10s``/``tcn`` -- both backwards, because the arm's *design
    intent* was being printed under the row's name (P6-D4 item 2 says the reading is per
    row). The intent is still carried, as ``arm_parameter_matched``.

    Args:
        param_delta_frac: Relative parameter change from
            :func:`parameter_delta_fraction`, or NaN if the counts were not available.
        tol: Threshold on ``|param_delta_frac|``; see :data:`CAPACITY_CONFOUND_TOL`.

    Returns:
        True if ``|param_delta_frac| <= tol``, False if not, and **None if the counts were
        not available**. None rather than False: an unmeasured match is not a measured
        mismatch, and the ``not_fittable`` placeholders have no fit to count parameters of.
    """
    if not np.isfinite(param_delta_frac):
        return None
    return bool(abs(param_delta_frac) <= tol)


def vehicle_is_blind(arm: str, model: str) -> bool:
    """Report whether a model is structurally blind to what an arm changes.

    Args:
        arm: Arm key.
        model: Model label, e.g. ``"dlinear_ols"``. Resolved through
            :func:`logical_model` as well as matched literally, so an arm-specific label
            cannot escape a declaration made about the vehicle it stands in for.

    Returns:
        True if the arm's feature cannot reach this model's forward pass, so that a zero
        contrast is an architectural artefact rather than a finding about the feature.
    """
    blind = _VEHICLE_BLIND.get(arm, frozenset())
    return model in blind or logical_model(model) in blind


def parameter_delta_fraction(n_params: float, reference_n_params: float) -> float:
    """Return the arm's relative parameter-count change against its reference.

    Args:
        n_params: The arm row's fitted parameter count.
        reference_n_params: The reference row's fitted parameter count.

    Returns:
        ``(n - ref) / ref``, signed, so a negative value means the arm is *smaller*. NaN if
        either count is missing -- an unmeasured change is not a measured zero. ``0.0`` when
        both are zero (the trivial baselines fit nothing, so nothing changed), and ``inf``
        when the reference fits nothing and the arm fits something, where no relative change
        is defined but the capacity plainly did change.
    """
    if not np.isfinite(n_params) or not np.isfinite(reference_n_params):
        return float("nan")
    if reference_n_params == 0.0:
        return 0.0 if n_params == 0.0 else float("inf")
    return float((n_params - reference_n_params) / reference_n_params)


def capacity_is_confounded(
    arm: str,
    model: str,
    *,
    param_delta_frac: float = float("nan"),
    tol: float = CAPACITY_CONFOUND_TOL,
) -> bool:
    """Report whether the arm changes this model's capacity as well as its information set.

    P3-D13 deleted ``dlinear_mc`` for exactly this defect: a pair introduced to isolate an
    information set that differs in capacity instead. P6-D8 defect 4 found it again on the
    sea-state arm, where the one-hot's four extra channels cost ``ar20`` 66.12% more
    parameters (108 900 -> 180 900) against ``tcn``'s 0.52%. If the conditioned AR row beats
    the unconditioned one, the increase is not attributable to sea-state information without
    further work, and the flag is what stops that row being read as if it were.

    **Measured where it can be, declared only where it cannot.** Given a finite
    ``param_delta_frac`` the answer is ``|delta| > tol`` and nothing else -- no model name
    enters it, so the rule follows the corpus if an architecture changes. The registry
    :data:`_CAPACITY_CONFOUNDED` is consulted only for a row with no parameter count at all,
    which on this corpus means the :func:`not_fittable_rows` placeholders.

    **The flag is symmetric in direction, and that is deliberate.** An arm that *removes*
    capacity confounds a loss exactly as an arm that adds it confounds a win: on the
    ``attitude_only`` arm ``ar20`` drops from 108 900 to 27 450 parameters because it has
    three channels instead of six, and its lower skill there is not attributable to the
    narrower information set alone. Where the parameter change is a *consequence* of the
    information change the two cannot be separated at all, which is the point of printing
    the flag rather than silently correcting for it.

    Args:
        arm: Arm key.
        model: Model label.
        param_delta_frac: Relative parameter change from
            :func:`parameter_delta_fraction`, or NaN if the counts were not available.
        tol: Threshold on ``|param_delta_frac|``; see :data:`CAPACITY_CONFOUND_TOL`.

    Returns:
        True if a difference on this row mixes a capacity change with the arm's effect.
    """
    if np.isfinite(param_delta_frac):
        return bool(abs(param_delta_frac) > tol)
    if param_delta_frac == float("inf") or param_delta_frac == float("-inf"):
        return True
    declared = _CAPACITY_CONFOUNDED.get(arm, frozenset())
    return model in declared or logical_model(model) in declared


def not_fittable_reason(arm: str, regime: str, model: str) -> str:
    """Return why a (arm, regime, model) row has no fit, or the empty string if it has one.

    Args:
        arm: Arm key.
        regime: Evaluation regime.
        model: Model label.

    Returns:
        The reason, or ``""``.
    """
    reason = _NOT_FITTABLE.get((arm, regime, model), "")
    return reason or _NOT_FITTABLE.get((arm, regime, logical_model(model)), "")


def not_fittable_rows(arm: str) -> pd.DataFrame:
    """Return the placeholder rows for every fit this arm cannot produce.

    Emitted so the contrast table carries the absence explicitly. An absent row and a row
    with nothing in it look the same to a reader only if the second is never written.

    Args:
        arm: Arm key.

    Returns:
        One row per unfittable (regime, model), with :data:`CONTRAST_COLUMNS` filled where
        they are known and ``not_fittable`` carrying the reason. Empty if the arm can fit
        everything.
    """
    spec = ARMS[arm]
    rows = [
        {
            "ablation": spec.ablation,
            "arm": arm,
            "reference_arm": spec.reference,
            "model": model,
            "logical_model": logical_model(model),
            # No fit and therefore no reference row to name a label from. Blank rather
            # than the arm's own label: this row's whole content is the reason column.
            "model_reference": "",
            "architecture_differs": None,
            "regime": regime,
            "observation_mode": spec.observation_mode,
            "reference_observation_mode": ARMS[spec.reference].observation_mode,
            "logical_dof": "",
            "dof": "",
            "dof_reference": "",
            "horizon_samples": -1,
            "horizon_s": float("nan"),
            "lookback": spec.lookback,
            "reference_lookback": ARMS[spec.reference].lookback,
            # None, not the arm's declaration: with no fit there is no parameter count to
            # measure a match from, and `parameter_matched` is a per-row measurement.
            "parameter_matched": None,
            "arm_parameter_matched": spec.parameter_matched,
            "privileged_information": spec.privileged_information,
            "vehicle_blind_to_arm": vehicle_is_blind(arm, model),
            # No fit, so no parameter count to measure from: this is the one place the
            # declared fallback of `capacity_is_confounded` is consulted, and the NaN
            # `param_delta_frac` beside it is what says the flag was declared rather than
            # measured.
            "capacity_confounded": capacity_is_confounded(arm, model),
            "n_params": float("nan"),
            "n_params_reference": float("nan"),
            "param_delta_frac": float("nan"),
            "not_fittable": reason,
            "raw_rmse_comparable": spec.observation_mode == ARMS[spec.reference].observation_mode,
            # No fit, so no per-cell SSE and no channel to place on a floor. Both are left
            # unknown rather than defaulted: this row's whole content is the reason column.
            "on_residual_floor": None,
            "note": spec.note,
            "skill_diff_ci_lo": float("nan"),
            "skill_diff_ci_hi": float("nan"),
            "skill_diff_ci_point": float("nan"),
            "skill_diff_ci_reason": (
                "no paired interval: this cell has no fit at all, so there is no skill "
                "difference for an interval to be drawn around"
            ),
            "ci_resample_unit": CI_RESAMPLE_UNIT,
            "ci_n_units": float("nan"),
            "ci_n_realizations": float("nan"),
            "n_boot": float("nan"),
            "ci_level": float("nan"),
            "bootstrap_seed": float("nan"),
        }
        for (arm_key, regime, model), reason in sorted(_NOT_FITTABLE.items())
        if arm_key == arm
    ]
    return pd.DataFrame(rows, columns=list(CONTRAST_COLUMNS))


# ---------------------------------------------------------------------------
# Window matching, on forecast origin
# ---------------------------------------------------------------------------


def window_origins(spec: WindowSpec, n_samples: int) -> IntArray:
    """Return the forecast origin of every window of one realization.

    The origin is the **last observed sample**, ``s + L - 1``: the moment the decision is
    taken and the sample the forecast is anchored to. It, not the window start, is what two
    arms with different lookbacks must share for their rows to describe the same forecast.

    The arithmetic itself lives in :func:`dmf.data.windows.window_origins` and is delegated
    to here rather than repeated: :class:`dmf.data.dataset.DeckMotionDataset` restricts
    itself to a chosen origin set and must agree, to the sample, with the intersection this
    module computes. Two copies of ``s + L - 1`` in two packages is exactly how a matched
    table stops being matched without anything failing.

    Args:
        spec: Window geometry.
        n_samples: Realization length, samples.

    Returns:
        Origins, shape ``(n_windows,)``, ascending.
    """
    return _origins_of(n_samples, spec)


def matched_origins(specs: Sequence[WindowSpec], n_samples: int) -> IntArray:
    """Return the origins every one of ``specs`` can score.

    Args:
        specs: The geometries to match, e.g. the L=100, L=200 and L=400 lookback arms.
        n_samples: Realization length, samples.

    Returns:
        The intersection of the arms' origin sets, ascending. On the production geometry
        this is the L=400 set: 1091 origins, 399 to 5849 in steps of 5.

    Raises:
        ValueError: If ``specs`` is empty, or if the intersection is empty -- two arms with
            nothing in common cannot be contrasted, and an empty paired table would render
            as a comparison that was made.
    """
    if not specs:
        raise ValueError("specs is empty; there is nothing to match")
    common = window_origins(specs[0], n_samples)
    for spec in specs[1:]:
        common = np.intersect1d(common, window_origins(spec, n_samples), assume_unique=True)
    if common.size == 0:
        lookbacks = [spec.lookback for spec in specs]
        raise ValueError(
            f"lookbacks {lookbacks} share no forecast origin on a {n_samples}-sample "
            f"realization, so no paired contrast between them exists"
        )
    return common.astype(np.int64)


def origin_window_indices(spec: WindowSpec, n_samples: int, origins: IntArray) -> IntArray:
    """Map a matched origin set back to this arm's window indices.

    Args:
        spec: The arm's window geometry.
        n_samples: Realization length, samples.
        origins: Origins to keep, from :func:`matched_origins`.

    Returns:
        Indices into this arm's per-realization window list, ascending.

    Raises:
        ValueError: If any requested origin is not one this arm scores, which would mean
            the caller matched against the wrong intersection.
    """
    available = window_origins(spec, n_samples)
    positions = np.searchsorted(available, origins)
    if positions.size and (
        positions.max() >= available.size or not np.array_equal(available[positions], origins)
    ):
        missing = sorted(set(origins.tolist()) - set(available.tolist()))[:5]
        raise ValueError(
            f"origins {missing} are not scored at lookback {spec.lookback}; the matched set "
            f"must be the intersection of every arm's origins"
        )
    return positions.astype(np.int64)


def start_matching_compares_different_times(
    spec_a: WindowSpec, spec_b: WindowSpec
) -> tuple[bool, int]:
    """Report whether matching two arms on window *start* would misalign them, and by how much.

    Exists so the wrong rule can be asserted wrong. Two arms matched on ``s`` forecast
    absolute times ``s + L_a + h`` and ``s + L_b + h``; those differ by ``L_a - L_b``, which
    at the production geometry is up to 300 samples, i.e. 30 s -- twice the longest horizon
    reported. The tables would both look well formed.

    Args:
        spec_a: One arm's geometry.
        spec_b: The other's.

    Returns:
        Tuple ``(misaligned, offset_samples)``: whether a start-matched comparison would
        compare different absolute times, and the signed offset ``L_a - L_b``.
    """
    offset = spec_a.lookback - spec_b.lookback
    return offset != 0, offset


#: Provenance a matched-origin table carries on **every row**, so that a reader -- and
#: :mod:`dmf.eval.assemble` -- can re-derive the matching from the file alone rather than
#: trusting the directory it was found in.
#:
#: This is the P6-D4 item 1 claim written down as data. Every value is recorded by the
#: writer and re-computed by :func:`unmatched_origin_reason` from ``matched_n_samples``,
#: ``matched_stride``, ``matched_max_horizon`` and ``matched_lookbacks``, which are the only
#: four facts the intersection depends on. A table lacking these columns is an **unmatched**
#: table -- which is what an arm's own committed ``baselines_by_seed.csv`` is -- and is
#: refused rather than joined.
#:
#: Every name is prefixed ``matched_`` so that none of them collides with a column
#: :func:`contrast_table` sets from the arm registry (``lookback`` is the one that would).
MATCHED_ORIGIN_COLUMNS: tuple[str, ...] = (
    "matched_n_samples",
    "matched_lookback",
    "matched_stride",
    "matched_max_horizon",
    "matched_lookbacks",
    "matched_n_origins",
    "matched_origin_first",
    "matched_origin_last",
    "matched_origin_stride",
    "matched_windows_before",
)

#: Separator between the lookbacks in the ``matched_lookbacks`` field. A string field rather
#: than a list, because the value round-trips through CSV.
MATCHED_LOOKBACK_SEP: str = "|"


def matched_origin_provenance(
    spec: WindowSpec, specs: Sequence[WindowSpec], n_samples: int
) -> dict[str, int | str]:
    """Describe the matched-origin scoring of one arm, for the rows it writes.

    Args:
        spec: The geometry of the arm being scored.
        specs: Every geometry the matching is over, this one included.
        n_samples: Realization length, samples.

    Returns:
        A mapping on :data:`MATCHED_ORIGIN_COLUMNS`, to be written on every row of the
        arm's table.

    Raises:
        ValueError: If ``spec`` is not among ``specs``, or if the matched origins are not
            evenly spaced -- which cannot happen for the congruent grids P6-D4 measured,
            and which would mean ``matched_origin_stride`` describes nothing.
    """
    lookbacks = sorted(other.lookback for other in specs)
    if spec.lookback not in lookbacks:
        raise ValueError(
            f"the arm being scored has lookback {spec.lookback}, which is not among the "
            f"matched geometries {lookbacks}; an arm must be matched against itself too"
        )
    origins = matched_origins(specs, n_samples)
    steps = np.unique(np.diff(origins)) if origins.size > 1 else np.array([spec.stride])
    if steps.size != 1:
        raise ValueError(
            f"the matched origins are not evenly spaced (steps {steps.tolist()}), so no "
            f"single origin stride describes them"
        )
    return {
        "matched_n_samples": int(n_samples),
        "matched_lookback": int(spec.lookback),
        "matched_stride": int(spec.stride),
        "matched_max_horizon": int(spec.max_horizon),
        "matched_lookbacks": MATCHED_LOOKBACK_SEP.join(str(value) for value in lookbacks),
        "matched_n_origins": int(origins.size),
        "matched_origin_first": int(origins[0]),
        "matched_origin_last": int(origins[-1]),
        "matched_origin_stride": int(steps[0]),
        "matched_windows_before": int(n_windows(n_samples, spec)),
    }


def _as_int(value: object) -> int:
    """Coerce a value read back from a CSV cell to an integer.

    Args:
        value: The cell.

    Returns:
        Its integer value.

    Raises:
        TypeError: If the cell holds something no integer can be read from, which the
            caller reports as an unusable provenance rather than raising.
        ValueError: Likewise, for a string that is not a number.
    """
    if isinstance(value, bool) or not isinstance(value, int | float | str | np.integer):
        raise TypeError(f"expected a number, got {type(value).__name__} {value!r}")
    return int(value)


def unmatched_origin_reason(record: Mapping[str, object], arm: str) -> str:
    """Say why a table claiming to be matched is not, or return the empty string.

    The verifier half of :func:`matched_origin_provenance`, and the reason
    :mod:`dmf.eval.assemble` can accept a matched table without a corpus: the four facts the
    intersection depends on are on the row, so the intersection is **re-derived** here and
    compared against the origin set the file says it scored. A file that carries the columns
    but the wrong numbers is refused exactly as one that carries no columns at all.

    Args:
        record: One row's worth of :data:`MATCHED_ORIGIN_COLUMNS`, plus optionally
            ``n_windows`` and ``n_realizations`` from the metrics schema, which are
            cross-checked when both are present.
        arm: The arm the file claims to be.

    Returns:
        Empty if the record describes a correct matching of ``arm``; otherwise one sentence
        saying which part does not hold.
    """
    if arm not in ARMS:
        return f"unknown arm {arm!r}; known arms are {sorted(ARMS)}"
    missing = [name for name in MATCHED_ORIGIN_COLUMNS if name not in record]
    if missing:
        return (
            f"the table carries no {missing} column(s), so it does not record which origin "
            f"set it was scored on. An arm's own committed baselines_by_seed.csv looks "
            f"exactly like this: it scored that arm's full window set, and differencing it "
            f"against another lookback would compare forecasts of different absolute times "
            f"(docs/protocol.md P6-D4 item 1)"
        )
    try:
        values = {
            name: _as_int(record[name])
            for name in MATCHED_ORIGIN_COLUMNS
            if name != "matched_lookbacks"
        }
        lookbacks = [
            int(part) for part in str(record["matched_lookbacks"]).split(MATCHED_LOOKBACK_SEP)
        ]
    except (TypeError, ValueError) as error:
        return f"the matched-origin provenance is not numeric: {error}"
    declared = ARMS[arm].lookback
    if values["matched_lookback"] != declared:
        return (
            f"the table was scored at lookback {values['matched_lookback']} but arm {arm!r} "
            f"is declared at {declared}"
        )
    if declared not in lookbacks:
        return (
            f"arm {arm!r} has lookback {declared}, which is not among the matched set "
            f"{lookbacks}; an arm must be matched against itself too"
        )
    expected = sorted(ARMS[key].lookback for key in ABLATIONS[ARMS[arm].ablation])
    if lookbacks != expected:
        return (
            f"the table declares the matching over lookbacks {lookbacks}, but the "
            f"{ARMS[arm].ablation!r} ablation has arms at {expected}. A contrast is paired "
            f"only on the intersection of EVERY arm's origins"
        )
    horizons = (values["matched_max_horizon"],)
    specs = [
        WindowSpec(lookback=value, horizons=horizons, stride=values["matched_stride"])
        for value in lookbacks
    ]
    own = WindowSpec(lookback=declared, horizons=horizons, stride=values["matched_stride"])
    origins = matched_origins(specs, values["matched_n_samples"])
    recomputed = {
        "matched_n_origins": int(origins.size),
        "matched_origin_first": int(origins[0]),
        "matched_origin_last": int(origins[-1]),
        "matched_windows_before": int(n_windows(values["matched_n_samples"], own)),
    }
    wrong = {
        name: (values[name], value) for name, value in recomputed.items() if values[name] != value
    }
    if wrong:
        return (
            f"the table states { ({k: v[0] for k, v in wrong.items()}) } but the intersection "
            f"of lookbacks {lookbacks} on a {values['matched_n_samples']}-sample "
            f"realization at stride {values['matched_stride']} gives "
            f"{ ({k: v[1] for k, v in wrong.items()}) }. The recorded matching does not "
            f"reproduce, so the rows were not scored on the set they claim"
        )
    n_windows_row = record.get("n_windows")
    n_keys = record.get("n_realizations")
    if n_windows_row is not None and n_keys is not None:
        scored = _as_int(n_windows_row)
        keys = _as_int(n_keys)
        if keys > 0 and scored != keys * values["matched_n_origins"]:
            return (
                f"a row reports {scored} scored windows over {keys} realizations, which is "
                f"{scored / keys:.6g} per realization against the "
                f"{values['matched_n_origins']} matched origins the table declares"
            )
    return ""


def matched_window_counts(
    specs: Sequence[WindowSpec], n_samples: int
) -> dict[int, tuple[int, int]]:
    """Return, per lookback, the window count before and after origin matching.

    The number a lookback table must print beside its rows: at the production geometry the
    L=100 arm goes from 1151 windows to 1091 and the L=200 arm from 1131 to 1091, and a
    reader who is not told that will read the three arms as having been scored on the same
    data because they were scored on the same *realizations*.

    Args:
        specs: The geometries to match.
        n_samples: Realization length, samples.

    Returns:
        Mapping ``lookback -> (windows before matching, windows after)``.
    """
    origins = matched_origins(specs, n_samples)
    return {spec.lookback: (n_windows(n_samples, spec), int(origins.size)) for spec in specs}


# ---------------------------------------------------------------------------
# Contrasts
# ---------------------------------------------------------------------------


def with_logical_dof(frame: pd.DataFrame, column: str = "dof") -> pd.DataFrame:
    """Add the logical channel name beside a table's observation-mode-resolved one.

    The ``imu`` arm's rows are keyed on ``roll_imu``; every other arm's on ``roll``. Joining
    on ``dof`` across that boundary returns an **empty** frame -- a safe failure, but a
    silent one, and an empty ablation table renders as "no rows". The mapping comes from
    :func:`dmf.data.channels.logical_channel`, which is a lookup and not a suffix test.

    Args:
        frame: Any table with a channel column. Not mutated.
        column: Name of that column.

    Returns:
        A copy carrying an additional ``logical_dof`` column.

    Raises:
        ValueError: If ``column`` is absent, or holds a name that is not a corpus channel.
    """
    if column not in frame.columns:
        raise ValueError(f"frame has no {column!r} column; columns are {list(frame.columns)}")
    out = frame.copy()
    # Mapped over the distinct channel names rather than over the rows: a per-cell table
    # runs to tens of thousands of rows and holds six distinct channels, and this function
    # is called once per arm against the same reference table.
    lookup = {name: logical_channel(str(name)) for name in frame[column].unique()}
    out["logical_dof"] = frame[column].map(lookup)
    return out


def with_logical_model(frame: pd.DataFrame, column: str = "model") -> pd.DataFrame:
    """Add the logical vehicle name beside a table's own model label.

    The twin of :func:`with_logical_dof`, one column over. The L=400 arm labels its deep
    vehicle ``tcn_l400`` and its reference labels the same vehicle ``tcn``, so an inner join
    on ``model`` drops all 216 of that arm's deep rows -- and unlike the ``dof`` case it does
    **not** produce an empty frame, because the arm's five closed-form vehicles still match.
    The table renders, looks complete, and is missing the one fitted model the arm exists to
    measure. That is what happened (B1, second Phase 6 audit), and it is why the join key is
    derived here rather than assumed to be the label.

    Args:
        frame: Any table with a model column. Not mutated.
        column: Name of that column.

    Returns:
        A copy carrying an additional ``logical_model`` column.

    Raises:
        ValueError: If ``column`` is absent.
    """
    if column not in frame.columns:
        raise ValueError(f"frame has no {column!r} column; columns are {list(frame.columns)}")
    out = frame.copy()
    lookup = {name: logical_model(str(name)) for name in frame[column].unique()}
    out["logical_model"] = frame[column].map(lookup)
    return out


def assert_columns_comparable(arm: str, reference: str, columns: Sequence[str]) -> None:
    """Refuse a contrast that would compare a raw-scale column across observation modes.

    P6-D4 item 4 as corrected by P6-D23, enforced rather than documented. The two modes
    forecast **different targets** -- both input and target are ``imu`` on that arm (P2-D8)
    -- so an ``imu`` RMSE and an ``ideal`` RMSE are numbers in different scales and their
    difference is not an ablation effect. The *persistence denominators*, by contrast, agree
    to about 1% (0.9706-1.0008 across ``id``, mean 0.9963), so skill and nrmse do cross the
    boundary; the earlier claim that the ``imu`` denominator is roughly halved was P1-D6's
    cross-mode pairing quoted into a configuration that does not use it.

    Args:
        arm: Arm key.
        reference: Reference arm key.
        columns: Metric columns the caller intends to difference.

    Raises:
        ValueError: If the two arms differ in observation mode and any requested column is
            not in :data:`CROSS_MODE_COLUMNS`.
    """
    left, right = ARMS[arm], ARMS[reference]
    if left.observation_mode == right.observation_mode:
        return
    offending = [name for name in columns if name not in CROSS_MODE_COLUMNS]
    if offending:
        raise ValueError(
            f"arm {arm!r} is {left.observation_mode!r} and its reference {reference!r} is "
            f"{right.observation_mode!r}; columns {offending} carry the observation mode's "
            f"own scale and must not cross that boundary. Only {list(CROSS_MODE_COLUMNS)} "
            f"may (docs/protocol.md P6-D4 item 4, corrected by P6-D23): both input and "
            f"target are imu on that arm, so the two arms forecast different targets with "
            f"different signal_std and their raw errors are not in the same units of "
            f"difficulty. The persistence denominators agree to ~1% (0.9963 mean ratio on "
            f"id), so skill and nrmse do cross; the raw scales do not."
        )


# ---------------------------------------------------------------------------
# Uncertainty on a contrast
# ---------------------------------------------------------------------------

#: The grid axes one resampling unit is identified by, in ``baselines_by_cell.csv``.
CELL_KEYS: tuple[str, ...] = ("vessel", "ss", "heading_deg", "speed_kn")

#: Columns a per-cell table must carry for an interval to be drawn from it.
_CELL_REQUIRED: tuple[str, ...] = (
    *CELL_KEYS,
    "model",
    "regime",
    "dof",
    "horizon_samples",
    "sse",
    "sse_persistence",
    "n_realizations",
)


def cell_floored_dofs(
    cells: pd.DataFrame,
    dof_names: Sequence[str],
    *,
    margin: float = RESIDUAL_FLOOR_MARGIN,
) -> frozenset[str]:
    """Name the channels on their P1-D2 residual floor over the cells a table was scored on.

    A thin adapter, and deliberately nothing more: the derivation is
    :func:`dmf.eval.controls.floored_dofs`, which already takes the floor from
    ``configs/vessel/*.yaml`` and the heading factor from :func:`dmf.sim.response.heading_factor`
    and requires the channel to be floored at **every** cell of the partition. Re-deriving it
    here from a regime name would be a second definition of the same fact, and the two would
    drift the first time the corpus grid changed.

    ``floored_dofs`` wants realization keys; a per-cell table has grid cells. It reads only
    the heading and the vessel out of a key, so a cell becomes one synthetic key with an
    arbitrary seed. That is a real restriction rather than a shortcut: this function cannot
    see a heading the arm was scored on but that no cell row mentions, so a partial cell
    table would understate the floor. Callers pass the table the metrics were pooled from.

    Args:
        cells: A ``baselines_by_cell.csv``-shaped frame, already restricted to one regime.
        dof_names: Target channel names in either observation-mode spelling.
        margin: Multiple of the floor the directional factor may reach and still count as
            floored; see :data:`dmf.eval.controls.RESIDUAL_FLOOR_MARGIN`.

    Returns:
        The subset of ``dof_names`` that is on its floor across every cell in ``cells``. On
        the production corpus this is ``{pitch, pitch_rate}`` for ``unseen_heading`` and
        empty for the other three regimes.

    Raises:
        ValueError: If ``cells`` lacks a grid axis, or holds no rows -- an empty partition
            would floor every channel vacuously.
    """
    missing = [name for name in CELL_KEYS if name not in cells.columns]
    if missing:
        raise ValueError(f"the per-cell table is missing grid axes {missing}")
    if cells.empty:
        raise ValueError("cannot derive the residual-floor set from an empty per-cell table")
    grid = cells[list(CELL_KEYS)].drop_duplicates()
    keys: list[RealizationKey] = [
        (str(ss), float(heading), float(speed), str(vessel), 0)
        for vessel, ss, heading, speed in zip(
            grid["vessel"], grid["ss"], grid["heading_deg"], grid["speed_kn"], strict=True
        )
    ]
    return floored_dofs(dof_names, keys, margin=margin)


def _cell_table_reason(frame: pd.DataFrame | None, side: str) -> str:
    """Say why a per-cell table cannot support an interval, or return the empty string.

    Args:
        frame: The table, or None when the side committed none.
        side: ``"arm"`` or ``"reference"``, for the message.

    Returns:
        Empty if the table can be used; otherwise one sentence naming what is absent.
    """
    if frame is None:
        return (
            f"no paired interval: the {side} committed no baselines_by_cell.csv, which is "
            f"the only committed table carrying per-cell sse and sse_persistence. The "
            f"matched-origin re-scoring writes pooled rows only, so a lookback arm has no "
            f"resampling unit to draw on until it does"
        )
    if frame.empty:
        return f"no paired interval: the {side}'s per-cell table holds no rows"
    absent = [name for name in _CELL_REQUIRED if name not in frame.columns]
    if absent:
        return f"no paired interval: the {side}'s per-cell table is missing {absent}"
    return ""


def _aligned_cells(
    left: pd.DataFrame, right: pd.DataFrame
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, int, str]:
    """Align two cell groups on the grid and return their SSE vectors.

    Args:
        left: The arm's rows for one contrast cell, one row per grid cell.
        right: The reference's rows for the same contrast cell.

    Returns:
        ``(sse_arm, persistence_arm, sse_reference, persistence_reference, n_realizations,
        reason)``. On a non-empty reason the arrays are empty and no interval may be drawn.
    """
    empty = np.zeros(0, dtype=np.float64)
    keys = list(CELL_KEYS)
    for side, frame in (("arm", left), ("reference", right)):
        repeated = frame.duplicated(subset=keys).sum()
        if repeated:
            return (
                empty,
                empty,
                empty,
                empty,
                0,
                f"no paired interval: the {side}'s per-cell rows hold {repeated} repeated "
                f"grid cell(s) for this contrast cell, so there is no single SSE per "
                f"resampling unit. De-duplicating would pick one silently",
            )
    a = left.set_index(keys).sort_index()
    b = right.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        only_a = sorted(set(a.index) - set(b.index))[:3]
        only_b = sorted(set(b.index) - set(a.index))[:3]
        return (
            empty,
            empty,
            empty,
            empty,
            0,
            f"no paired interval: the arm and its reference were scored on different grid "
            f"cells ({len(a)} and {len(b)}); cells only in the arm {only_a}, only in the "
            f"reference {only_b}. A bootstrap paired on cells requires one cell set",
        )
    if len(a) < CI_MIN_UNITS:
        return (
            empty,
            empty,
            empty,
            empty,
            0,
            f"no paired interval: {len(a)} resampling unit(s) is under the minimum "
            f"{CI_MIN_UNITS}; a percentile interval over that few draws is decoration",
        )
    n_realizations = int(a["n_realizations"].sum()) if "n_realizations" in a.columns else 0
    return (
        np.asarray(a["sse"].to_numpy(), dtype=np.float64),
        np.asarray(a["sse_persistence"].to_numpy(), dtype=np.float64),
        np.asarray(b["sse"].to_numpy(), dtype=np.float64),
        np.asarray(b["sse_persistence"].to_numpy(), dtype=np.float64),
        n_realizations,
        "",
    )


def contrast_uncertainty(
    arm_cells: pd.DataFrame | None,
    reference_cells: pd.DataFrame | None,
    *,
    keys: Sequence[str],
    n_boot: int = ABLATION_N_BOOT,
    ci_level: float = ABLATION_CI_LEVEL,
    bootstrap_seed: int = ABLATION_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Bootstrap a **paired** interval for ``skill(arm) - skill(reference)``, per contrast cell.

    The ablation twin of :func:`dmf.eval.runner.paired_skill_difference_ci`, which does not
    fit here for two reasons and follows its conventions in every other respect
    (``n_boot = 1000``, ``ci_level = 0.95``, ``bootstrap_seed = 0``, the same
    ``_bootstrap_counts`` draw, the same sign convention -- positive means the **arm** has
    the higher skill):

    1. **Its inputs are per-realization SSE tensors from a live scoring pass.** Nothing
       committed carries those. ``baselines_by_cell.csv`` is the finest committed
       granularity, so one draw resamples a **grid cell** and not a realization. That is
       recorded on every row in ``ci_resample_unit``, and it is a conservative substitution:
       see :data:`CI_RESAMPLE_UNIT`.
    2. **It assumes one shared persistence denominator**, which is exactly right within a
       training run and wrong across arms: the ``imu`` arm's persistence is a different
       number from ``ideal``'s (P6-D4 item 4), and a lookback arm's is taken over different
       windows. So the statistic here is the difference of two ratios,
       ``skill_arm - skill_ref = sum(w*sse_ref)/sum(w*p_ref) - sum(w*sse_arm)/sum(w*p_arm)``,
       under one shared weight vector ``w``. Where the two denominators coincide it reduces
       algebraically to the runner's ``(b - a) / p``, and ``tests/test_ablations.py``
       asserts that reduction numerically rather than by inspection.

    Args:
        arm_cells: The arm's ``baselines_by_cell.csv`` rows, or None if it committed none.
        reference_cells: The reference arm's, likewise.
        keys: Contrast-cell keys to group on, i.e. :data:`JOIN_KEYS` plus ``"seed"`` when
            both sides carry seeds -- the same keys :func:`contrast_table` merges on, so
            that the interval lands on the row whose difference it describes.
        n_boot: Bootstrap resamples.
        ci_level: Central confidence level.
        bootstrap_seed: Seed of the resampling generator.

    Returns:
        One row per contrast cell present in **both** tables, carrying ``keys`` and
        :data:`CONTRAST_CI_COLUMNS`. Cells the arm has and the reference does not are absent
        rather than NaN-filled: :func:`contrast_table` left-joins this onto its own rows and
        fills the gap with the reason it computed for the pair as a whole.

    Raises:
        ValueError: If ``n_boot`` is not positive or ``ci_level`` is outside ``(0, 1)``.
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    reason = _cell_table_reason(arm_cells, "arm") or _cell_table_reason(
        reference_cells, "reference arm"
    )
    if reason or arm_cells is None or reference_cells is None:
        return pd.DataFrame(columns=[*keys, *CONTRAST_CI_COLUMNS])

    group = list(keys)
    # Both logical keys, and for the same reason `contrast_table` needs them: a per-cell
    # table labelled `tcn_l400` groups to nothing against one labelled `tcn`, and a missing
    # group is a missing interval rather than an error.
    left = with_logical_model(with_logical_dof(arm_cells))
    right = with_logical_model(with_logical_dof(reference_cells))
    right_groups = dict(tuple(right.groupby(group, sort=False)))
    # One counts matrix per unit count, not one per contrast cell: `_bootstrap_counts` is a
    # pure function of (n_units, n_boot, seed), so every cell of one regime is resampled by
    # the identical draw. That is what makes two rows of one table comparable to each other
    # as well as each to zero.
    counts_cache: dict[int, FloatArray] = {}
    rows: list[dict[str, object]] = []
    for name, arm_group in left.groupby(group, sort=False):
        label = name if isinstance(name, tuple) else (name,)
        reference_group = right_groups.get(label)
        if reference_group is None:
            continue
        sse_a, p_a, sse_b, p_b, n_realizations, why = _aligned_cells(arm_group, reference_group)
        record: dict[str, object] = dict(zip(group, label, strict=True))
        record.update(
            {
                "skill_diff_ci_lo": float("nan"),
                "skill_diff_ci_hi": float("nan"),
                "skill_diff_ci_point": float("nan"),
                "skill_diff_ci_reason": why,
                "ci_resample_unit": CI_RESAMPLE_UNIT,
                "ci_n_units": len(arm_group),
                "ci_n_realizations": n_realizations,
                "n_boot": n_boot,
                "ci_level": ci_level,
                "bootstrap_seed": bootstrap_seed,
            }
        )
        if why:
            rows.append(record)
            continue
        n_units = int(sse_a.size)
        if n_units not in counts_cache:
            counts_cache[n_units] = _bootstrap_counts(n_units, n_boot, bootstrap_seed)
        counts = counts_cache[n_units]
        denominator_a = counts @ p_a
        denominator_b = counts @ p_b
        if float(p_a.sum()) == 0.0 or float(p_b.sum()) == 0.0:
            record["skill_diff_ci_reason"] = (
                "no paired interval: the pooled persistence SSE is zero on one side, so the "
                "skill difference is undefined"
            )
            rows.append(record)
            continue
        if np.any(denominator_a == 0.0) or np.any(denominator_b == 0.0):
            record["skill_diff_ci_reason"] = (
                "no paired interval: a resample produced zero persistence SSE, so the skill "
                "score is undefined on that draw"
            )
            rows.append(record)
            continue
        draws = (counts @ sse_b) / denominator_b - (counts @ sse_a) / denominator_a
        lo, hi = _percentile_interval(draws.reshape(n_boot, 1), ci_level=ci_level, shape=(1, 1))
        record["skill_diff_ci_lo"] = float(lo.reshape(-1)[0])
        record["skill_diff_ci_hi"] = float(hi.reshape(-1)[0])
        record["skill_diff_ci_point"] = float(sse_b.sum() / p_b.sum() - sse_a.sum() / p_a.sum())
        record["ci_n_units"] = n_units
        rows.append(record)
    if not rows:
        return pd.DataFrame(columns=[*keys, *CONTRAST_CI_COLUMNS])
    return pd.DataFrame(rows)[[*group, *CONTRAST_CI_COLUMNS]]


def _with_residual_floor(merged: pd.DataFrame, cells: pd.DataFrame | None) -> pd.DataFrame:
    """Flag every contrast row whose channel is nothing but its P1-D2 residual floor.

    The last artefact flag. The others exist so that an architectural or bookkeeping zero
    cannot read as a finding; this one stops the opposite reading, an enormous number that
    is not one. Five arms score ``unseen_heading`` / ``ar20`` / ``pitch_rate`` / 10 s, and
    their ``skill_diff`` runs -0.000003 (``revin``), +0.000585 (``ss_conditioned``), +3.29
    (``lookback_40s``), +30.30 (``lookback_10s``) and **+41.67** (``imu``) against
    ``signal_std = 0.0658 deg/s``; the largest on any floored cell of the committed table is
    that +41.67, and ``attitude_only`` / ``ar20`` / ``pitch`` / 10 s carries +27.69. The
    unfloored rows of the same file have median ``|skill_diff| = 0.0002`` and reach 2.67 at
    most. On a 26 dB-suppressed channel the skill denominator is itself floor, so the ratio
    is arithmetic about the floor and not a measurement of the ablation.

    The ``+30.27`` this docstring used to quote came from the audit rather than from a row
    and is retracted (P6-D20); the whole set is printed instead of one member of it (P6-D17).

    Derived, never listed: the set comes from :func:`cell_floored_dofs`, which delegates to
    :func:`dmf.eval.controls.floored_dofs` -- the same function the shuffle control's
    narrowing uses (P6-D12) -- so the two cannot disagree about which channel is floored.

    Args:
        merged: The joined contrast rows.
        cells: A per-cell table naming the grid cells the rows were scored on, or None. With
            None the flag cannot be derived and is left as pandas NA rather than False: a
            missing derivation must not read as "checked, and not floored".

    Returns:
        ``merged`` with a nullable-boolean ``on_residual_floor`` column.
    """
    out = merged.copy()
    if cells is None or cells.empty or "regime" not in cells.columns:
        out["on_residual_floor"] = pd.array([None] * len(out), dtype="boolean")
        return out
    names = sorted({str(value) for value in out["dof"]})
    floored_by_regime: dict[str, frozenset[str] | None] = {}
    for regime in {str(value) for value in out["regime"]}:
        scoped = cells.loc[cells["regime"].astype(str) == regime]
        # None, not an empty set: a regime the per-cell table does not cover has an
        # underived flag, and an empty set would say "derived, and nothing is floored".
        floored_by_regime[regime] = None if scoped.empty else cell_floored_dofs(scoped, names)
    flags: list[bool | None] = []
    for regime, dof in zip(out["regime"], out["dof"], strict=True):
        floored = floored_by_regime[str(regime)]
        flags.append(None if floored is None else str(dof) in floored)
    out["on_residual_floor"] = pd.array(flags, dtype="boolean")
    return out


def _with_uncertainty(
    merged: pd.DataFrame,
    keys: Sequence[str],
    arm: str,
    arm_cells: pd.DataFrame | None,
    reference_cells: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach the paired interval to every contrast row, or the reason it has none.

    Args:
        merged: The joined contrast rows, already carrying ``skill_diff``.
        keys: The columns ``merged`` was joined on.
        arm: Arm key, for the message.
        arm_cells: The arm's per-cell rows, or None.
        reference_cells: The reference's, or None.

    Returns:
        ``merged`` with every column of :data:`CONTRAST_CI_COLUMNS`. A row whose interval
        could not be drawn carries NaN bounds and a non-empty ``skill_diff_ci_reason``;
        the two are never both filled and never both empty.
    """
    out = merged.copy()
    absent = _cell_table_reason(arm_cells, "arm") or _cell_table_reason(
        reference_cells, "reference arm"
    )
    # An arm fed by two experiment configs commits two per-cell tables, and both carry
    # `persistence` on `id` because every config must -- so the concatenated frame holds two
    # rows per grid cell for that label. The contrast rows are per experiment (see
    # `dmf.eval.assemble.read_arm_frames`), so the cells are narrowed to the same experiment
    # rather than de-duplicated: picking one of two silently is how an interval ends up
    # describing a run its row did not come from.
    scoped = arm_cells
    if scoped is not None and "experiment" in scoped.columns and "experiment" in merged.columns:
        wanted = set(merged["experiment"].astype(str))
        scoped = scoped.loc[scoped["experiment"].astype(str).isin(wanted)]
    intervals = contrast_uncertainty(scoped, reference_cells, keys=keys)
    if intervals.empty:
        for name in CONTRAST_CI_COLUMNS:
            out[name] = float("nan")
        out["ci_resample_unit"] = CI_RESAMPLE_UNIT
        out["skill_diff_ci_reason"] = absent or (
            f"no paired interval: no contrast cell of arm {arm!r} appears in both per-cell "
            f"tables, so no cell set is shared to resample"
        )
        return out
    join = [name for name in keys if name in intervals.columns]
    out = out.merge(intervals, on=join, how="left", validate="one_to_one")
    out["ci_resample_unit"] = out["ci_resample_unit"].fillna(CI_RESAMPLE_UNIT)
    missing = out["skill_diff_ci_reason"].isna()
    out.loc[missing, "skill_diff_ci_reason"] = (
        f"no paired interval: this contrast cell of arm {arm!r} has no counterpart in the "
        f"per-cell tables, so the pooled row it was read from cannot be resampled"
    )
    # The interval and the row must describe the same quantity. Pooled skill is
    # `1 - sum(sse) / sum(sse_persistence)` over exactly these cells, so the two agree to
    # floating point unless the cell table and the metric row were scored on different
    # window populations -- in which case the interval belongs to a different number and is
    # withdrawn rather than printed beside it.
    if "skill_diff" in out.columns:
        drawn = out["skill_diff_ci_lo"].notna()
        mismatch = drawn & ((out["skill_diff_ci_point"] - out["skill_diff"]).abs() > CI_POINT_TOL)
        if bool(mismatch.any()):
            for name in ("skill_diff_ci_lo", "skill_diff_ci_hi"):
                out.loc[mismatch, name] = float("nan")
            out.loc[mismatch, "skill_diff_ci_reason"] = [
                f"no paired interval: the per-cell tables give a skill difference of "
                f"{point:.9g} where the pooled row carries {value:.9g}. The two describe "
                f"different window populations, and an interval drawn on the first would "
                f"be attached to the second"
                for point, value in zip(
                    out.loc[mismatch, "skill_diff_ci_point"],
                    out.loc[mismatch, "skill_diff"],
                    strict=True,
                )
            ]
    return out


def contrast_table(
    arm_rows: pd.DataFrame,
    reference_rows: pd.DataFrame,
    arm: str,
    *,
    columns: Sequence[str] = CROSS_MODE_COLUMNS,
    arm_cells: pd.DataFrame | None = None,
    reference_cells: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pair an arm's rows against its reference's, under this module's matching rules.

    Args:
        arm_rows: The arm's metric rows, carrying ``model``, ``regime``, ``dof``,
            ``horizon_samples`` and every column in ``columns``. An ``experiment`` column,
            if present, is carried through as the leading provenance column.
        reference_rows: The reference arm's rows, same schema.
        arm: Arm key, used for the registry's matching rules and flags.
        columns: Metric columns to difference. Defaults to the two that may cross an
            observation-mode boundary; pass raw-scale columns only for a within-mode
            contrast, which :func:`assert_columns_comparable` enforces.
        arm_cells: The arm's ``baselines_by_cell.csv`` rows. Supplying both this and
            ``reference_cells`` is what puts a **paired** interval on ``skill_diff``
            (:func:`contrast_uncertainty`) and what lets ``on_residual_floor`` be derived
            from the cells the arm was scored on rather than asserted from a regime name.
            Omitting either leaves both bounds NaN and fills ``skill_diff_ci_reason``;
            ``skill_diff_std`` over training seeds is **not** a substitute, and P4-D13
            records why: it measures initialisation and data order, not the realization
            sampling that decides whether a skill difference is real.
        reference_cells: The reference arm's per-cell rows, likewise.

    Returns:
        One row per matched (model, regime, DOF, horizon), with :data:`CONTRAST_COLUMNS`,
        each requested metric as ``<name>`` and ``reference_<name>``, and their difference
        as ``<name>_diff`` (**arm minus reference**, so positive means the ablated arm is
        higher on that metric). ``n_params``, ``n_params_reference`` and
        ``param_delta_frac`` are carried when both frames have a parameter count, and
        ``capacity_confounded`` is measured from them; all four are NaN/declared otherwise
        (:func:`capacity_is_confounded`).

    Raises:
        ValueError: If ``arm`` is unknown, if a required column is missing, if the contrast
            would cross the observation-mode boundary on a raw-scale column, or if the join
            matched no rows at all.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; known arms are {sorted(ARMS)}")
    spec = ARMS[arm]
    reference = ARMS[spec.reference]
    assert_columns_comparable(arm, spec.reference, columns)

    needed = {"model", "regime", "dof", "horizon_samples", *columns}
    for name, frame in (("arm_rows", arm_rows), ("reference_rows", reference_rows)):
        missing = sorted(needed - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns {missing}")

    left = with_logical_model(with_logical_dof(arm_rows))
    right = with_logical_model(with_logical_dof(reference_rows))
    left = left.loc[left["logical_dof"].isin(spec.comparable_dofs)]
    right = right.loc[right["logical_dof"].isin(spec.comparable_dofs)]

    # A per-run table carries three seeds of the SGD vehicle, so the join is seed-wise when
    # both sides have a seed: seed 0 of the arm against seed 0 of the reference, which is
    # the paired quantity. Joining without it would be a many-to-many merge of nine pairs
    # per cell, and `validate` would (correctly) refuse. Aggregating the three differences
    # is the renderer's job, under the >= 3 seed rule.
    keys = (
        [*JOIN_KEYS, "seed"] if {"seed"} <= set(left.columns) & set(right.columns) else [*JOIN_KEYS]
    )
    # `n_params` is carried from BOTH sides when both have it, because the capacity flag is
    # measured from the pair rather than asserted from a model name. Carried and not merely
    # used: CLAUDE.md requires the parameter count in every model comparison table, and a
    # flag whose input is not printed beside it cannot be checked by a reader.
    both_have_params = {"n_params"} <= set(left.columns) & set(right.columns)
    # `model` is carried from the right as well, so the reference's own label lands on the
    # row as `model_reference`. It is the evidence for `architecture_differs`, and on the
    # L=400 arm it is the difference between "tcn at 40 s" and "a deeper TCN at 40 s".
    carried = ["model", *(["n_params"] if both_have_params else [])]
    if "horizon_s" not in left.columns:
        left = left.assign(horizon_s=float("nan"))
    merged = left.merge(
        right[[*keys, "dof", *columns, *carried]],
        on=keys,
        how="inner",
        suffixes=("", "_reference"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(
            f"the {arm!r} contrast matched no rows against {spec.reference!r}. The arms "
            f"carry {sorted(set(arm_rows['dof']))} and {sorted(set(reference_rows['dof']))} "
            f"as dof labels and {sorted(set(arm_rows['model']))} and "
            f"{sorted(set(reference_rows['model']))} as model labels, and are joined on the "
            f"logical channel and the logical vehicle, so an empty result means the regimes "
            f"or the horizons do not overlap -- not that the effect is zero."
        )

    merged["ablation"] = spec.ablation
    merged["arm"] = arm
    merged["reference_arm"] = spec.reference
    merged["observation_mode"] = spec.observation_mode
    merged["reference_observation_mode"] = reference.observation_mode
    merged["lookback"] = spec.lookback
    merged["reference_lookback"] = reference.lookback
    merged["arm_parameter_matched"] = spec.parameter_matched
    merged["privileged_information"] = spec.privileged_information
    merged["architecture_differs"] = [
        architecture_differs(str(model), str(reference_model))
        for model, reference_model in zip(merged["model"], merged["model_reference"], strict=True)
    ]
    merged["vehicle_blind_to_arm"] = [vehicle_is_blind(arm, str(m)) for m in merged["model"]]
    if not both_have_params:
        merged["n_params"] = float("nan")
        merged["n_params_reference"] = float("nan")
    merged["param_delta_frac"] = [
        parameter_delta_fraction(float(n), float(ref))
        for n, ref in zip(merged["n_params"], merged["n_params_reference"], strict=True)
    ]
    merged["capacity_confounded"] = [
        capacity_is_confounded(arm, str(model), param_delta_frac=float(delta))
        for model, delta in zip(merged["model"], merged["param_delta_frac"], strict=True)
    ]
    # Measured from the same number by the same threshold as the flag above, and therefore
    # its exact negation on every row that has one. pandas' nullable boolean, because a row
    # whose parameter counts are missing is unmeasured rather than unmatched.
    merged["parameter_matched"] = pd.array(
        [parameter_is_matched(float(delta)) for delta in merged["param_delta_frac"]],
        dtype="boolean",
    )
    merged["not_fittable"] = [
        not_fittable_reason(arm, str(regime), str(model))
        for regime, model in zip(merged["regime"], merged["model"], strict=True)
    ]
    merged["raw_rmse_comparable"] = spec.observation_mode == reference.observation_mode
    merged["note"] = spec.note
    for name in columns:
        merged[f"{name}_diff"] = merged[name] - merged[f"{name}_reference"]
    merged = _with_residual_floor(merged, arm_cells if arm_cells is not None else reference_cells)
    merged = _with_uncertainty(merged, keys, arm, arm_cells, reference_cells)

    ordered = [
        # `experiment` leads when the arm's rows carry it, because an arm can be fed by more
        # than one experiment config (the deep and closed-form halves cover different
        # regimes) and `dmf.eval.report._group_cols` groups on it when it is present. A row
        # that does not say which run produced it cannot be told from a duplicate of another.
        *(["experiment"] if "experiment" in left.columns else []),
        *CONTRAST_COLUMNS[:4],
        *(["seed"] if "seed" in keys else []),
        *[name for name in CONTRAST_COLUMNS[4:] if name not in CONTRAST_CI_COLUMNS],
        *[part for name in columns for part in (name, f"{name}_reference", f"{name}_diff")],
        *CONTRAST_CI_COLUMNS,
    ]
    return merged[ordered].sort_values(
        ["arm", "regime", "model", "logical_dof", "horizon_samples"], ignore_index=True
    )
