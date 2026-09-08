"""The ablation matching rules, asserted rather than described.

Four things are pinned here, and each of them is a way an ablation table can look perfectly
well formed while comparing the wrong quantities:

1. **Origin matching.** The production numbers of ``docs/protocol.md`` P6-D4 item 1 --
   1151/1131/1091 windows at L=100/200/400, origins 399..5849, nested sets -- are asserted
   against :mod:`dmf.data.windows` rather than trusted from the entry. The *wrong* rule is
   asserted wrong too: matching on window start puts an L=100 forecast of one absolute time
   beside an L=400 forecast of a time 300 samples later, and both tables look fine.
2. **The observation-mode boundary.** A raw-RMSE contrast between the ``imu`` arm and the
   ``ideal`` reference raises; a skill contrast does not. The join is on the logical channel,
   because the two arms carry different ``dof`` labels and a join on ``dof`` returns an empty
   frame -- which renders as "no rows", not as an error.
3. **The channels arm** is read on roll/pitch/heave and the rate rows are not imputed.
4. **The sea-state arm's two defects** travel as columns: ``vehicle_blind_to_arm`` on the
   DLinear rows, and a ``not_fittable`` row for AR(20) on ``unseen_seastate``.

Units: samples for lookbacks, horizons, origins and window counts.
"""

import numpy as np
import pandas as pd
import pytest

from dmf.data.windows import WindowSpec, n_windows
from dmf.eval.ablations import (
    ABLATION_BOOTSTRAP_SEED,
    ABLATION_CI_LEVEL,
    ABLATION_N_BOOT,
    ABLATIONS,
    ARMS,
    CAPACITY_CONFOUND_TOL,
    CI_RESAMPLE_UNIT,
    CONTRAST_CI_COLUMNS,
    CONTRAST_COLUMNS,
    CROSS_MODE_COLUMNS,
    REFERENCE_ARM,
    arms_for_experiment,
    assert_columns_comparable,
    capacity_is_confounded,
    cell_floored_dofs,
    contrast_table,
    contrast_uncertainty,
    matched_origins,
    matched_window_counts,
    not_fittable_reason,
    not_fittable_rows,
    origin_window_indices,
    start_matching_compares_different_times,
    vehicle_is_blind,
    window_origins,
    with_logical_dof,
)

#: The production corpus record length, samples: 600 s at 10 Hz.
N_SAMPLES = 6000

#: The production horizon set and stride.
HORIZONS: tuple[int, ...] = (10, 20, 30, 50, 100, 150)
STRIDE = 5


def _spec(lookback: int) -> WindowSpec:
    """Return the production window geometry at one lookback."""
    return WindowSpec(lookback=lookback, horizons=HORIZONS, stride=STRIDE)


# ---------------------------------------------------------------------------
# Origin matching
# ---------------------------------------------------------------------------


def test_the_production_origin_sets_are_the_ones_the_protocol_records() -> None:
    expected = {100: (1151, 99), 200: (1131, 199), 400: (1091, 399)}
    for lookback, (count, first) in expected.items():
        spec = _spec(lookback)
        origins = window_origins(spec, N_SAMPLES)
        assert n_windows(N_SAMPLES, spec) == count
        assert int(origins[0]) == first
        assert int(origins[-1]) == 5849
        assert np.array_equal(np.diff(origins), np.full(origins.size - 1, STRIDE))


def test_the_three_lookback_origin_sets_are_nested_and_the_intersection_is_the_longest() -> None:
    specs = [_spec(100), _spec(200), _spec(400)]
    common = matched_origins(specs, N_SAMPLES)
    assert common.size == 1091
    assert int(common[0]) == 399
    assert int(common[-1]) == 5849
    # Nested, not merely overlapping: L=400's set is a subset of L=200's, which is a subset
    # of L=100's. That is what makes the matched contrast paired rather than approximate.
    assert np.array_equal(common, window_origins(_spec(400), N_SAMPLES))
    for lookback in (100, 200):
        assert set(common.tolist()) <= set(window_origins(_spec(lookback), N_SAMPLES).tolist())


def test_matching_drops_the_leading_windows_the_protocol_names() -> None:
    counts = matched_window_counts([_spec(100), _spec(200), _spec(400)], N_SAMPLES)
    assert counts == {100: (1151, 1091), 200: (1131, 1091), 400: (1091, 1091)}
    # 60 leading windows at L=100 and 40 at L=200, which is what "the arms were scored on
    # the same realizations" hides if the counts are not printed beside the rows.
    assert counts[100][0] - counts[100][1] == 60
    assert counts[200][0] - counts[200][1] == 40


def test_origin_window_indices_select_the_matched_windows_of_each_arm() -> None:
    common = matched_origins([_spec(100), _spec(200), _spec(400)], N_SAMPLES)
    for lookback, dropped in ((100, 60), (200, 40), (400, 0)):
        spec = _spec(lookback)
        index = origin_window_indices(spec, N_SAMPLES, common)
        assert index.size == common.size
        assert int(index[0]) == dropped
        # The selected windows really do carry the matched origins, in order.
        assert np.array_equal(window_origins(spec, N_SAMPLES)[index], common)


def test_origin_window_indices_refuse_an_origin_the_arm_does_not_score() -> None:
    with pytest.raises(ValueError, match="not scored at lookback"):
        origin_window_indices(_spec(400), N_SAMPLES, np.asarray([199, 399], dtype=np.int64))


def test_matching_on_window_start_would_compare_different_absolute_times() -> None:
    # The wrong rule, asserted wrong. A window starting at s forecasts s + L + h, so two
    # arms matched on s are compared at times 300 samples -- 30 s, twice the longest horizon
    # reported -- apart, and neither table shows it.
    misaligned, offset = start_matching_compares_different_times(_spec(100), _spec(400))
    assert misaligned
    assert offset == -300
    same, zero = start_matching_compares_different_times(_spec(200), _spec(200))
    assert not same
    assert zero == 0
    # Concretely: window 0 of each arm shares a start of 0 and forecasts different samples.
    starts_equal = 0
    assert starts_equal + 100 + HORIZONS[0] != starts_equal + 400 + HORIZONS[0]


def test_matched_origins_refuses_geometries_with_nothing_in_common() -> None:
    # Two arms whose origin sets are disjoint by parity: at stride 2 from origin 99 every
    # origin is odd, and at stride 2 from origin 100 every origin is even. An empty paired
    # table would render as a comparison that was made and found nothing.
    odd = WindowSpec(lookback=100, horizons=HORIZONS, stride=2)
    even = WindowSpec(lookback=101, horizons=HORIZONS, stride=2)
    assert int(window_origins(odd, N_SAMPLES)[0]) % 2 == 1
    assert int(window_origins(even, N_SAMPLES)[0]) % 2 == 0
    with pytest.raises(ValueError, match="share no forecast origin"):
        matched_origins([odd, even], N_SAMPLES)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_the_registry_covers_five_ablations_and_six_new_arms() -> None:
    # Six new arms, plus the re-scored L=200 reference the lookback contrast is paired
    # against, plus the unablated reference itself.
    assert set(ARMS) == {
        REFERENCE_ARM,
        "imu",
        "attitude_only",
        "ss_conditioned",
        "lookback_10s",
        "lookback_20s",
        "lookback_40s",
        "revin",
    }
    assert set(ABLATIONS) == {
        "reference",
        "observation_mode",
        "channels",
        "sea_state_conditioning",
        "lookback",
        "normalization",
    }
    # Every arm's reference is itself a registered arm, so a contrast can always be formed.
    for arm in ARMS.values():
        assert arm.reference in ARMS


def test_the_lookback_arms_reference_the_rescored_geometry_not_the_committed_rows() -> None:
    for arm in ("lookback_10s", "lookback_40s"):
        assert ARMS[arm].reference == "lookback_20s"
    # And the plain reference arm is the committed e02 one, which the other four arms use.
    for arm in ("imu", "attitude_only", "ss_conditioned", "revin"):
        assert ARMS[arm].reference == REFERENCE_ARM


def test_one_experiment_can_feed_two_arms_and_the_registry_says_so() -> None:
    claimed = {arm.arm for arm in arms_for_experiment("e02_deep")}
    # e02_deep is the unablated reference when read as committed and the lookback reference
    # when re-scored on the matched origin set. Two row sets, one training run.
    assert claimed == {REFERENCE_ARM, "lookback_20s"}
    assert {arm.arm for arm in arms_for_experiment("e04a_obs_mode_ood")} == {"imu"}
    with pytest.raises(ValueError, match="no ablation arm claims"):
        arms_for_experiment("e99_nonexistent")


def test_every_ablation_experiment_config_is_claimed_by_an_arm() -> None:
    from pathlib import Path

    from dmf.eval.assemble import RESIDUAL_FLOOR_EXPERIMENT

    root = Path(__file__).resolve().parents[1] / "configs" / "experiment"
    for path in sorted(root.glob("e04*.yaml")):
        if path.stem == RESIDUAL_FLOOR_EXPERIMENT:
            # The one Phase 6 config that is not an ablation arm: it fits the unconditional
            # probabilistic floor (P6-D6), which nothing is contrasted against. The
            # exception is named from `dmf.eval.assemble` rather than spelled here, so a
            # second unclaimed config still fails this test instead of being waved through
            # by a pattern.
            continue
        assert arms_for_experiment(path.stem), path.name


#: Measured parameter counts, ``(reference at L=200, arm)``, from ``docs/protocol.md``
#: P6-D4 item 2. DLinear is channel-shared, so ``dlinear_ols`` is ``2*L*H + 2*H`` and scales
#: with lookback; a dilated convolution's weights do not scale with input length, so ``tcn``
#: is unchanged between L=100 and L=200 and changes at L=400 only because the receptive
#: field needs one more dilation stage (which is also why that arm's vehicle is a different
#: network, labelled ``tcn_l400``).
LOOKBACK_PARAMS: dict[str, dict[str, tuple[int, int]]] = {
    "lookback_10s": {"dlinear_ols": (60_300, 30_300), "tcn": (196_804, 196_804)},
    "lookback_40s": {"dlinear_ols": (60_300, 120_300), "tcn_l400": (196_804, 221_636)},
}


def test_the_lookback_arm_holds_parameters_fixed_for_the_tcn_and_not_for_dlinear() -> None:
    """P6-D4 item 2 corrected the blanket claim, and the flag is now per row.

    The arm-level field records what the arm was *built* to hold fixed and is False here;
    the row-level ``parameter_matched`` is measured from the two rows' own counts and says
    ``True`` for ``tcn`` at L=100, because 196 804 parameters is 196 804 parameters however
    long the input is. A blanket arm-level claim printed under the row's name is what put
    the wrong flag in the published table, so what is asserted here is the **split**, model
    by model, and never a statement about the arm as a whole.
    """
    # The design intent, unchanged: the arm as a whole does not hold capacity fixed.
    for arm in ("lookback_10s", "lookback_20s", "lookback_40s"):
        assert not ARMS[arm].parameter_matched

    dofs = ("roll", "pitch")
    for arm, counts in LOOKBACK_PARAMS.items():
        table = contrast_table(
            _params_rows({m: c for m, (_, c) in counts.items()}, dofs, 0.70),
            _params_rows({m: c for m, (c, _) in counts.items()}, dofs, 0.65),
            arm,
        )
        matched = table.groupby("model")["parameter_matched"].all().to_dict()
        confounded = table.groupby("model")["capacity_confounded"].all().to_dict()
        deltas = table.groupby("model")["param_delta_frac"].first().to_dict()
        # The arm's intent travels on every row too, under a name of its own, so the two
        # claims are both visible and cannot be read as one.
        assert not bool(table["arm_parameter_matched"].any())
        for model in counts:
            # The row-level flag is exactly the negation of the capacity flag: both are a
            # threshold on the same measured number, so they cannot contradict each other.
            assert bool(matched[model]) is not bool(confounded[model])

        if arm == "lookback_10s":
            assert deltas["tcn"] == 0.0
            assert bool(matched["tcn"]), "196 804 parameters at both lookbacks"
            assert deltas["dlinear_ols"] == pytest.approx(-0.4975, abs=1e-4)
            assert not bool(matched["dlinear_ols"]), "30 300 against 60 300 is confounded"
        else:
            assert deltas["tcn_l400"] == pytest.approx(0.1262, abs=1e-4)
            assert not bool(matched["tcn_l400"])
            assert deltas["dlinear_ols"] == pytest.approx(0.9950, abs=1e-4)
            assert not bool(matched["dlinear_ols"])


def test_the_arm_intent_and_the_row_measurement_are_allowed_to_disagree() -> None:
    """And where they do, the row is the measurement.

    ``attitude_only`` is *built* to hold parameters fixed, and does -- for its closed-form
    vehicle. ``dlinear_ols`` is channel-shared and is 60 300 either way; ``tcn``'s input
    stem narrows with the channel count and drops to 166 786, a 15.25% loss. Publishing the
    arm's intent under the row's name printed ``parameter_matched = yes`` beside
    ``param_delta_frac = -0.1525`` and ``capacity_confounded = yes`` on that row.
    """
    dofs = ("roll", "pitch")
    counts = {"dlinear_ols": (60_300, 60_300), "tcn": (196_804, 166_786)}
    table = contrast_table(
        _params_rows({m: c for m, (_, c) in counts.items()}, dofs, 0.70),
        _params_rows({m: c for m, (c, _) in counts.items()}, dofs, 0.65),
        "attitude_only",
    )
    assert ARMS["attitude_only"].parameter_matched
    assert bool(table["arm_parameter_matched"].all())
    matched = table.groupby("model")["parameter_matched"].all().to_dict()
    deltas = table.groupby("model")["param_delta_frac"].first().to_dict()
    assert bool(matched["dlinear_ols"]) and deltas["dlinear_ols"] == 0.0
    assert not bool(matched["tcn"])
    assert deltas["tcn"] == pytest.approx(-0.1525, abs=1e-4)
    # The disagreement is between the two columns, not inside either of them.
    assert bool(table.loc[table["model"] == "tcn", "capacity_confounded"].all())


def test_the_sea_state_arm_is_flagged_as_privileged_information_not_as_an_upper_bound() -> None:
    """P6-D18 measured that the arm is not an upper bound; the flag says what survives.

    What survives the retraction is that the one-hot is ground truth where deployment has an
    online estimate, so no row of the arm is deployable. What does not survive is the word
    *bound*: conditioning costs -0.0445 mean skill on ``unseen_seastate``, so the "bound"
    lies below the baseline it was supposed to bound.
    """
    assert ARMS["ss_conditioned"].privileged_information
    assert not ARMS["imu"].privileged_information
    assert not hasattr(ARMS["imu"], "upper_bound")
    note = ARMS["ss_conditioned"].note
    assert "NOT an upper bound" in note
    assert "PRIVILEGED INFORMATION" in note


def test_the_channels_arm_is_read_on_the_three_shared_dofs() -> None:
    assert ARMS["attitude_only"].comparable_dofs == ("roll", "pitch", "heave")
    assert "heave_rate" in ARMS[REFERENCE_ARM].comparable_dofs


# ---------------------------------------------------------------------------
# The observation-mode boundary
# ---------------------------------------------------------------------------


def _rows(models: tuple[str, ...], dofs: tuple[str, ...], value: float) -> pd.DataFrame:
    """Return a minimal metric frame."""
    return pd.DataFrame(
        [
            {
                "model": model,
                "regime": "id",
                "dof": dof,
                "horizon_samples": horizon,
                "skill": value,
                "nrmse": value,
                "rmse": value,
            }
            for model in models
            for dof in dofs
            for horizon in (10, 30)
        ]
    )


def test_a_raw_rmse_contrast_across_the_observation_mode_boundary_raises() -> None:
    with pytest.raises(ValueError, match="must not cross that boundary"):
        assert_columns_comparable("imu", REFERENCE_ARM, ("skill", "rmse"))
    # And the two dimensionless columns are permitted, which is what makes the refusal a
    # rule rather than a blanket ban on comparing the arms at all.
    assert_columns_comparable("imu", REFERENCE_ARM, CROSS_MODE_COLUMNS)
    # Within one mode, everything is comparable.
    assert_columns_comparable("revin", REFERENCE_ARM, ("rmse", "mae", "signal_std"))


def test_the_imu_contrast_joins_on_the_logical_channel_and_refuses_raw_rmse() -> None:
    imu_dofs = (
        "roll_imu",
        "pitch_imu",
        "heave_imu",
        "roll_rate_imu",
        "pitch_rate_imu",
        "heave_rate_imu",
    )
    ideal_dofs = ("roll", "pitch", "heave", "roll_rate", "pitch_rate", "heave_rate")
    arm_rows = _rows(("ar20", "dlinear_ols"), imu_dofs, 0.9)
    reference_rows = _rows(("ar20", "dlinear_ols"), ideal_dofs, 0.8)

    # A naive join on `dof` finds nothing, which is the silent failure this guards.
    naive = arm_rows.merge(reference_rows, on=["model", "regime", "dof", "horizon_samples"])
    assert naive.empty

    table = contrast_table(arm_rows, reference_rows, "imu")
    assert len(table) == len(arm_rows)
    assert set(table["logical_dof"]) == set(ideal_dofs)
    assert (table["dof"].str.endswith("_imu")).all()
    assert (table["dof_reference"] == table["logical_dof"]).all()
    assert table["skill_diff"].to_numpy() == pytest.approx(0.9 - 0.8)
    assert not table["raw_rmse_comparable"].any()
    assert (table["observation_mode"] == "imu").all()
    assert (table["reference_observation_mode"] == "ideal").all()
    with pytest.raises(ValueError, match="must not cross that boundary"):
        contrast_table(arm_rows, reference_rows, "imu", columns=("rmse",))


def test_a_contrast_that_matches_nothing_raises_instead_of_rendering_empty() -> None:
    arm_rows = _rows(("ar20",), ("roll",), 0.9)
    reference_rows = _rows(("tcn",), ("roll",), 0.8)
    with pytest.raises(ValueError, match="matched no rows"):
        contrast_table(arm_rows, reference_rows, "revin")


def test_the_channels_contrast_drops_the_rate_rows_rather_than_imputing_them() -> None:
    attitude = ("roll", "pitch", "heave")
    arm_rows = _rows(("dlinear_ols",), attitude, 0.7)
    reference_rows = _rows(
        ("dlinear_ols",), (*attitude, "roll_rate", "pitch_rate", "heave_rate"), 0.6
    )
    table = contrast_table(arm_rows, reference_rows, "attitude_only")
    assert set(table["logical_dof"]) == set(attitude)
    assert len(table) == len(arm_rows)
    assert table["nrmse_diff"].to_numpy() == pytest.approx(0.7 - 0.6)


def test_with_logical_dof_refuses_a_name_that_is_not_a_corpus_channel() -> None:
    with pytest.raises(ValueError, match="unknown channel"):
        with_logical_dof(pd.DataFrame({"dof": ["yaw"]}))
    with pytest.raises(ValueError, match="has no 'dof' column"):
        with_logical_dof(pd.DataFrame({"channel": ["roll"]}))


# ---------------------------------------------------------------------------
# The sea-state arm's two defects
# ---------------------------------------------------------------------------


def test_the_dlinear_rows_of_the_sea_state_arm_are_flagged_blind_not_dropped() -> None:
    dofs = ("roll", "pitch")
    arm_rows = _rows(("dlinear_ols", "tcn"), dofs, 0.75)
    reference_rows = _rows(("dlinear_ols", "tcn"), dofs, 0.75)
    table = contrast_table(arm_rows, reference_rows, "ss_conditioned")

    blind = table.loc[table["model"] == "dlinear_ols"]
    seeing = table.loc[table["model"] == "tcn"]
    assert not blind.empty, "the blind rows ship (CLAUDE.md non-negotiable 6), flagged"
    assert blind["vehicle_blind_to_arm"].all()
    assert not seeing["vehicle_blind_to_arm"].any()
    # The contrast is exactly zero for the blind vehicle, and the flag is what stops that
    # being read as "sea-state conditioning has no effect".
    assert blind["skill_diff"].to_numpy() == pytest.approx(0.0)
    assert table["privileged_information"].all()
    assert vehicle_is_blind("ss_conditioned", "dlinear")
    assert not vehicle_is_blind("revin", "dlinear_ols")


def test_the_unfittable_ar_row_ships_with_its_reason() -> None:
    rows = not_fittable_rows("ss_conditioned")
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["model"] == "ar20"
    assert row["regime"] == "unseen_seastate"
    assert "singular" in row["not_fittable"]
    assert tuple(rows.columns) == CONTRAST_COLUMNS
    # Every other arm can fit everything it configures.
    for arm in ARMS:
        if arm != "ss_conditioned":
            assert not_fittable_rows(arm).empty
    assert not_fittable_reason("ss_conditioned", "id", "ar20") == ""
    assert not_fittable_reason("ss_conditioned", "unseen_seastate", "tcn") == ""


def test_each_arm_matches_the_data_config_it_names() -> None:
    from pathlib import Path

    from dmf.config import load_data

    root = Path(__file__).resolve().parents[1] / "configs" / "data"
    for key, arm in ARMS.items():
        cfg = load_data(root / f"{arm.data_config}.yaml")
        # The registry is a second statement of what each arm is, and a second statement
        # that can disagree with the first is how an ablation table ends up labelled with
        # the wrong lookback. Checked against the YAML rather than trusted.
        assert cfg.lookback == arm.lookback, key
        assert cfg.observation_mode == arm.observation_mode, key
        assert tuple(cfg.target_dofs) == arm.comparable_dofs or key != "attitude_only"
        assert cfg.condition_on_sea_state is (key == "ss_conditioned"), key
        assert cfg.revin is (key == "revin"), key


def test_a_multi_seed_contrast_is_paired_seed_by_seed() -> None:
    dofs = ("roll", "pitch")
    arm_rows = pd.concat(
        [_rows(("tcn",), dofs, 0.7 + 0.01 * seed).assign(seed=seed) for seed in (0, 1, 2)],
        ignore_index=True,
    )
    reference_rows = pd.concat(
        [_rows(("tcn",), dofs, 0.6).assign(seed=seed) for seed in (0, 1, 2)], ignore_index=True
    )
    table = contrast_table(arm_rows, reference_rows, "revin")
    # Three seeds of one model are three paired differences, not nine cross-seed ones.
    assert len(table) == len(arm_rows)
    assert set(table["seed"]) == {0, 1, 2}
    for seed in (0, 1, 2):
        rows = table.loc[table["seed"] == seed]
        assert rows["skill_diff"].to_numpy() == pytest.approx(0.1 + 0.01 * seed)


def test_the_contrast_runs_on_the_committed_imu_arm_if_it_has_landed() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    arm_path = root / "results" / "e04" / "e04a_obs_mode_ood" / "baselines_by_seed.csv"
    reference_path = root / "results" / "e02" / "baselines_by_seed.csv"
    if not arm_path.exists():
        pytest.skip("the imu ablation arm has not been scored yet")
    arm_rows = pd.read_csv(arm_path)
    reference_rows = pd.read_csv(reference_path)
    shared = sorted(set(arm_rows["model"]) & set(reference_rows["model"]))
    regimes = sorted(set(arm_rows["regime"]))
    arm_rows = arm_rows.loc[arm_rows["model"].isin(shared)]
    reference_rows = reference_rows.loc[
        reference_rows["model"].isin(shared) & reference_rows["regime"].isin(regimes)
    ]
    table = contrast_table(arm_rows, reference_rows, "imu")
    # The committed imu rows are keyed on roll_imu / pitch_imu / ...; a join on `dof` would
    # be empty. The row count is asserted rather than the merge trusted.
    assert len(table) == len(arm_rows)
    assert not table["raw_rmse_comparable"].any()
    assert set(table["logical_dof"]) == {
        "roll",
        "pitch",
        "heave",
        "roll_rate",
        "pitch_rate",
        "heave_rate",
    }


# ---------------------------------------------------------------------------
# The logical vehicle: the L=400 arm's deep rows must reach the table
# ---------------------------------------------------------------------------


def test_the_lookback_40s_contrast_pairs_tcn_l400_against_tcn() -> None:
    """B1 of the second Phase 6 audit: the arm's only deep vehicle reached no table.

    ``tcn_l400`` and ``tcn`` are the same logical vehicle at two input lengths and are
    **not** the same network. The join has to be on the logical name or the rows vanish; the
    two labels and ``architecture_differs`` are what stop the pairing being read as one
    network at two lookbacks.
    """
    dofs = ("roll", "pitch")
    arm_rows = _rows(("tcn_l400", "dlinear_ols"), dofs, 0.70)
    reference_rows = _rows(("tcn", "dlinear_ols"), dofs, 0.65)

    table = contrast_table(arm_rows, reference_rows, "lookback_40s")
    assert len(table) == len(arm_rows), "every arm row is paired, none dropped"
    deep = table.loc[table["model"] == "tcn_l400"]
    assert not deep.empty
    assert set(deep["logical_model"]) == {"tcn"}
    assert set(deep["model_reference"]) == {"tcn"}
    assert bool(deep["architecture_differs"].all())
    assert deep["skill_diff"].to_numpy() == pytest.approx(0.05)
    # The closed-form vehicle is the same network under both arms, so the flag is False
    # there: it is measured from the labels, not asserted from the arm.
    shared = table.loc[table["model"] == "dlinear_ols"]
    assert not bool(shared["architecture_differs"].any())


def test_a_join_on_the_model_label_drops_the_deep_rows_without_emptying_the_table() -> None:
    """Why the silent drop was silent, asserted rather than described.

    The ``imu`` arm's label mismatch is on ``dof`` and produces an **empty** frame, which
    :func:`contrast_table` raises on. This one is on ``model`` and is *partial*: the five
    closed-form vehicles still match, so the table renders, looks complete, and is missing
    the only model the arm exists to measure. Nothing raises and nothing is empty.
    """
    dofs = ("roll", "pitch")
    arm_rows = _rows(("tcn_l400", "dlinear_ols"), dofs, 0.70)
    reference_rows = _rows(("tcn", "dlinear_ols"), dofs, 0.65)
    naive = arm_rows.merge(reference_rows, on=["model", "regime", "dof", "horizon_samples"])
    assert not naive.empty, "the failure is partial, which is what made it survive review"
    assert set(naive["model"]) == {"dlinear_ols"}
    assert len(naive) == len(arm_rows) // 2
    # And the fixed path keeps them.
    assert set(contrast_table(arm_rows, reference_rows, "lookback_40s")["model"]) == {
        "tcn_l400",
        "dlinear_ols",
    }


def test_every_fitted_model_of_every_committed_arm_reaches_a_contrast_row() -> None:
    """The guard against B1 recurring on an arm nobody has looked at yet.

    A model that was fitted, scored and committed must reach a contrast row or carry a
    ``not_fittable`` reason. Reaching neither is the failure that hid 216 ``tcn_l400`` rows:
    ``CLAUDE.md`` non-negotiable 6 forbids dropping a model, and a silent inner join drops
    one without anybody choosing to.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    table_path = root / "results" / "e04" / "ablations.csv"
    if not table_path.exists():
        pytest.skip("the ablation table has not been assembled yet")
    table = pd.read_csv(table_path)
    matched_root = root / "results" / "e04" / "matched_origins"
    for arm in sorted(set(table["arm"].astype(str))):
        directory = matched_root / arm
        source = directory / "baselines_by_seed.csv"
        if not source.exists():
            continue
        fitted = set(pd.read_csv(source)["model"].astype(str))
        rendered = set(table.loc[table["arm"] == arm, "model"].astype(str))
        regimes = sorted(set(table.loc[table["arm"] == arm, "regime"].astype(str)))
        excused = {
            model
            for model in fitted - rendered
            if any(not_fittable_reason(arm, regime, model) for regime in regimes)
        }
        assert not (fitted - rendered - excused), (
            f"arm {arm!r} committed fitted rows for {sorted(fitted - rendered - excused)} "
            f"that reach no contrast row and carry no reason"
        )


# ---------------------------------------------------------------------------
# The capacity confound (P6-D8 defect 4), measured rather than asserted
# ---------------------------------------------------------------------------

#: Parameter counts P6-D8 defect 4 measured for the sea-state arm, and P6-D4 item 2 for the
#: reference. Written here as the two-column table the protocol prints, so a test that reads
#: like the entry is checking the entry.
SS_CONDITIONED_PARAMS: dict[str, tuple[int, int]] = {
    # model: (unconditioned C_in=6, conditioned C_in=10)
    "tcn": (196_804, 197_828),
    "ar20": (108_900, 180_900),
    "dlinear_ols": (60_300, 60_300),
}


def _params_rows(counts: dict[str, int], dofs: tuple[str, ...], value: float) -> pd.DataFrame:
    """Return a metric frame carrying one parameter count per model."""
    frame = _rows(tuple(counts), dofs, value)
    frame["n_params"] = [counts[str(model)] for model in frame["model"]]
    return frame


def test_the_capacity_flag_is_measured_from_the_parameter_counts_not_the_model_name() -> None:
    dofs = ("roll", "pitch")
    arm = _params_rows({m: c for m, (_, c) in SS_CONDITIONED_PARAMS.items()}, dofs, 0.70)
    reference = _params_rows({m: c for m, (c, _) in SS_CONDITIONED_PARAMS.items()}, dofs, 0.65)
    table = contrast_table(arm, reference, "ss_conditioned")
    flagged = table.groupby("model")["capacity_confounded"].all().to_dict()
    # AR(20) pays 66.12% more parameters for the one-hot's four channels; TCN pays 0.52%,
    # and channel-shared DLinear pays nothing. The rule is a threshold on the measured
    # change, so the answer follows the counts if an architecture changes.
    assert flagged == {"ar20": True, "dlinear_ols": False, "tcn": False}
    deltas = table.groupby("model")["param_delta_frac"].first()
    assert deltas["ar20"] == pytest.approx(0.6612, abs=1e-4)
    assert deltas["tcn"] == pytest.approx(0.0052, abs=1e-4)
    assert deltas["dlinear_ols"] == 0.0
    # The inputs to the flag are printed beside it: a flag whose evidence is not in the
    # table cannot be checked by a reader of the table.
    assert set(table["n_params"]) == {c for _, c in SS_CONDITIONED_PARAMS.values()}
    assert set(table["n_params_reference"]) == {c for c, _ in SS_CONDITIONED_PARAMS.values()}


def test_the_capacity_flag_fires_in_both_directions() -> None:
    """An arm that removes capacity confounds a loss as an arm that adds it confounds a win.

    The ``attitude_only`` arm forecasts three channels instead of six, so AR(20) drops from
    108 900 parameters to 27 450 -- a *narrower* model on a *narrower* information set, and
    the two cannot be separated. A one-sided rule would call that comparison clean.
    """
    dofs = ("roll", "pitch")
    arm = _params_rows({"ar20": 27_450}, dofs, 0.50)
    reference = _params_rows({"ar20": 108_900}, dofs, 0.65)
    table = contrast_table(arm, reference, "attitude_only")
    assert bool(table["capacity_confounded"].all())
    assert table["param_delta_frac"].iloc[0] == pytest.approx(-0.7479, abs=1e-4)


def test_a_row_with_no_parameter_count_falls_back_to_the_declaration_and_says_so() -> None:
    rows = not_fittable_rows("ss_conditioned")
    row = rows.iloc[0]
    # No fit, so nothing to measure: the flag is declared, and the NaN beside it is what
    # says so. A 0.0 there would claim a measured absence of change.
    assert bool(row["capacity_confounded"])
    assert np.isnan(float(row["param_delta_frac"]))
    assert np.isnan(float(row["n_params"]))
    assert capacity_is_confounded("ss_conditioned", "ar20")
    assert not capacity_is_confounded("ss_conditioned", "tcn")
    # A measured delta always wins over the declaration, in both directions.
    assert not capacity_is_confounded("ss_conditioned", "ar20", param_delta_frac=0.0)
    assert capacity_is_confounded("ss_conditioned", "tcn", param_delta_frac=0.20)


def test_the_capacity_threshold_is_not_load_bearing() -> None:
    """Every threshold between the two measured extremes selects the same rows.

    0.52% for TCN and 66.12% for AR(20) leaves two orders of magnitude of slack, which is
    what makes :data:`CAPACITY_CONFOUND_TOL` a documented constant rather than a knob.
    """
    for tol in (0.01, 0.05, 0.2, 0.5):
        assert capacity_is_confounded("ss_conditioned", "ar20", param_delta_frac=0.6612, tol=tol)
        assert not capacity_is_confounded("ss_conditioned", "tcn", param_delta_frac=0.0052, tol=tol)
    assert 0.0052 < CAPACITY_CONFOUND_TOL < 0.6612


def test_a_contrast_without_parameter_counts_still_carries_the_columns() -> None:
    dofs = ("roll",)
    table = contrast_table(_rows(("ar20",), dofs, 0.7), _rows(("ar20",), dofs, 0.6), "revin")
    assert set(CONTRAST_COLUMNS) <= set(table.columns)
    assert table["param_delta_frac"].isna().all()
    # `revin` declares nothing, so an unmeasurable row is not flagged rather than guessed.
    assert not bool(table["capacity_confounded"].any())


def test_every_model_blind_to_the_sea_state_indicator_is_flagged_as_blind() -> None:
    """The three trivial baselines take the same ``[:C_out]`` slice DLinear does.

    P6-D8 named DLinear because it was the designated vehicle. ``Persistence.forward`` is
    ``x[:, -1:, : self.n_target_channels]`` and ``DampedPersistence`` -- which ``WindowMean``
    subclasses -- takes the same slice, so all three are blind by the same mechanism and
    their contrast on this arm is exactly zero for the same reason. Unflagged, that would be
    four more "no effect" rows than the entry anticipated.
    """
    for model in ("persistence", "window_mean", "damped_persistence", "dlinear", "dlinear_ols"):
        assert vehicle_is_blind("ss_conditioned", model), model
    # AR(20) is the one closed-form vehicle that does read the indicator -- which is why it
    # is also the one that pays for it.
    assert not vehicle_is_blind("ss_conditioned", "ar20")


# ---------------------------------------------------------------------------
# Uncertainty on a contrast, and the residual-floor flag
# ---------------------------------------------------------------------------


def _cells(
    models: tuple[str, ...],
    dofs: tuple[str, ...],
    *,
    regime: str = "id",
    headings: tuple[float, ...] = (45.0, 90.0, 135.0, 180.0),
    speeds: tuple[float, ...] = (0.0, 6.0, 12.0),
    vessel: str = "frigate",
    sse: float = 1.0,
    persistence: float = 10.0,
    jitter: float = 0.0,
) -> pd.DataFrame:
    """Return a minimal ``baselines_by_cell.csv``-shaped frame.

    ``jitter`` varies the SSE across cells, which is what gives the bootstrap something to
    resample; with ``jitter = 0`` every draw is identical and the interval collapses to a
    point, which is itself an assertion worth making.
    """
    rows = []
    for model in models:
        for dof in dofs:
            for horizon in (10, 30):
                for index, heading in enumerate(headings):
                    for offset, speed in enumerate(speeds):
                        step = index * len(speeds) + offset
                        rows.append(
                            {
                                "model": model,
                                "regime": regime,
                                "vessel": vessel,
                                "ss": "SS4",
                                "heading_deg": heading,
                                "speed_kn": speed,
                                "dof": dof,
                                "horizon_samples": horizon,
                                "sse": sse * (1.0 + jitter * step),
                                "sse_persistence": persistence,
                                "n_realizations": 8,
                            }
                        )
    return pd.DataFrame(rows)


def test_the_paired_cell_bootstrap_reduces_to_the_runners_paired_interval() -> None:
    """The two estimators are the same statistic where the runner's assumption holds.

    ``dmf.eval.runner.paired_skill_difference_ci`` divides one shared persistence SSE into
    ``sse_b - sse_a``; this module divides each side by its own, because the ``imu`` arm's
    denominator is a different number from ``ideal``'s (P6-D4 item 4). Where the two
    denominators coincide -- which is every within-mode arm -- the algebra must agree, and
    it is asserted numerically rather than by reading the two expressions.
    """
    import torch

    from dmf.eval.runner import paired_skill_difference_ci

    rng = np.random.default_rng(11)
    n_units = 12
    sse_a = rng.uniform(1.0, 4.0, size=n_units)
    sse_b = rng.uniform(1.0, 4.0, size=n_units)
    persistence = rng.uniform(8.0, 12.0, size=n_units)

    arm = _cells(("tcn",), ("roll",), headings=(45.0, 90.0, 135.0, 180.0), speeds=(0.0, 6.0, 12.0))
    reference = arm.copy()
    one = (arm["dof"] == "roll") & (arm["horizon_samples"] == 10)
    arm.loc[one, "sse"] = sse_a
    arm.loc[one, "sse_persistence"] = persistence
    reference.loc[one, "sse"] = sse_b
    reference.loc[one, "sse_persistence"] = persistence

    table = contrast_uncertainty(
        arm, reference, keys=("model", "regime", "logical_dof", "horizon_samples")
    )
    row = table.loc[table["horizon_samples"] == 10].iloc[0]

    # The runner wants (n_keys, H, C_out) tensors; one horizon, one channel, n_units "keys".
    def _tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float64).reshape(n_units, 1, 1)

    point, lo, hi = paired_skill_difference_ci(
        _tensor(sse_a), _tensor(sse_b), _tensor(persistence), horizons=(1,)
    )
    assert float(row["skill_diff_ci_point"]) == pytest.approx(float(point.reshape(-1)[0]))
    assert float(row["skill_diff_ci_lo"]) == pytest.approx(float(lo.reshape(-1)[0]))
    assert float(row["skill_diff_ci_hi"]) == pytest.approx(float(hi.reshape(-1)[0]))
    # And the sign convention is the runner's: positive means the ARM has the higher skill.
    assert float(row["skill_diff_ci_point"]) > 0.0 if sse_a.sum() < sse_b.sum() else True


def test_an_arm_with_no_per_cell_table_gets_a_reason_and_not_an_unpaired_interval() -> None:
    """P5-D12: an unpaired delta reported as paired is a mistake already made here."""
    dofs = ("roll", "pitch")
    table = contrast_table(_rows(("tcn",), dofs, 0.7), _rows(("tcn",), dofs, 0.6), "revin")
    assert set(CONTRAST_CI_COLUMNS) <= set(table.columns)
    assert table["skill_diff_ci_lo"].isna().all()
    assert table["skill_diff_ci_hi"].isna().all()
    assert table["skill_diff_ci_reason"].str.contains("no paired interval").all()
    # The contrast itself still ships. An ablation is not deleted to protest a missing bar.
    assert table["skill_diff"].to_numpy() == pytest.approx(0.1)


def test_two_arms_scored_on_different_grid_cells_get_a_reason_and_no_interval() -> None:
    dofs = ("roll",)
    arm_cells = _cells(("tcn",), dofs, jitter=0.1)
    # The reference is missing one heading, so the two were not scored on one cell set.
    reference_cells = arm_cells.loc[arm_cells["heading_deg"] != 45.0].copy()
    table = contrast_table(
        _rows(("tcn",), dofs, 0.7),
        _rows(("tcn",), dofs, 0.6),
        "revin",
        arm_cells=arm_cells,
        reference_cells=reference_cells,
    )
    assert table["skill_diff_ci_lo"].isna().all()
    assert table["skill_diff_ci_reason"].str.contains("different grid cells").all()


def test_an_interval_is_withdrawn_when_the_cells_do_not_reproduce_the_rows_difference() -> None:
    """The interval must belong to the number printed beside it, or it must not be printed."""
    dofs = ("roll",)
    arm_cells = _cells(("tcn",), dofs, sse=1.0, jitter=0.1)
    reference_cells = _cells(("tcn",), dofs, sse=2.0, jitter=0.1)
    # The pooled rows claim a difference the cells do not give: a window-population mismatch.
    table = contrast_table(
        _rows(("tcn",), dofs, 0.7),
        _rows(("tcn",), dofs, 0.6),
        "revin",
        arm_cells=arm_cells,
        reference_cells=reference_cells,
    )
    assert table["skill_diff"].to_numpy() == pytest.approx(0.1)
    assert table["skill_diff_ci_lo"].isna().all()
    assert table["skill_diff_ci_reason"].str.contains("different window populations").all()


def test_the_interval_is_a_pure_function_of_the_committed_numbers() -> None:
    dofs = ("roll",)
    arm_cells = _cells(("tcn",), dofs, sse=1.0, jitter=0.2)
    reference_cells = _cells(("tcn",), dofs, sse=1.3, jitter=0.05)
    keys = ("model", "regime", "logical_dof", "horizon_samples")
    first = contrast_uncertainty(arm_cells, reference_cells, keys=keys)
    second = contrast_uncertainty(arm_cells.iloc[::-1], reference_cells, keys=keys)
    pd.testing.assert_frame_equal(
        first.sort_values(list(keys), ignore_index=True),
        second.sort_values(list(keys), ignore_index=True),
    )
    assert (first["n_boot"] == ABLATION_N_BOOT).all()
    assert (first["ci_level"] == ABLATION_CI_LEVEL).all()
    assert (first["bootstrap_seed"] == ABLATION_BOOTSTRAP_SEED).all()
    assert (first["ci_resample_unit"] == CI_RESAMPLE_UNIT).all()
    assert (first["ci_n_units"] == 12).all()
    assert (first["ci_n_realizations"] == 96).all()
    assert (first["skill_diff_ci_lo"] <= first["skill_diff_ci_point"]).all()
    assert (first["skill_diff_ci_point"] <= first["skill_diff_ci_hi"]).all()


def test_too_few_resampling_units_gets_a_reason_rather_than_a_two_point_interval() -> None:
    dofs = ("roll",)
    arm_cells = _cells(("tcn",), dofs, headings=(45.0,), speeds=(0.0, 6.0), jitter=0.1)
    reference_cells = _cells(("tcn",), dofs, headings=(45.0,), speeds=(0.0, 6.0), jitter=0.2)
    keys = ("model", "regime", "logical_dof", "horizon_samples")
    table = contrast_uncertainty(arm_cells, reference_cells, keys=keys)
    assert table["skill_diff_ci_lo"].isna().all()
    assert table["skill_diff_ci_reason"].str.contains("under the minimum").all()


def test_the_residual_floor_flag_is_derived_from_the_control_and_not_from_the_regime_name() -> None:
    """The sixth artefact flag, and it must agree with the shuffle control's own derivation.

    ``unseen_heading``'s test partition is beam-on only, where P1-D2 clamps pitch at its
    26 dB residual; ``id`` spans every heading, so nothing is floored there. The set comes
    from :func:`dmf.eval.controls.floored_dofs` either way -- this asserts the two agree
    rather than that the ablation table has its own opinion.
    """
    from dmf.eval.controls import floored_dofs

    dofs = ("pitch", "pitch_rate", "roll", "heave")
    beam = _cells(("ar20",), dofs, regime="unseen_heading", headings=(90.0,), speeds=(0.0, 12.0))
    assert cell_floored_dofs(beam, dofs) == floored_dofs(
        dofs, [("SS4", 90.0, 0.0, "frigate", 0), ("SS4", 90.0, 12.0, "frigate", 0)]
    )
    assert cell_floored_dofs(beam, dofs) == frozenset({"pitch", "pitch_rate"})
    everywhere = _cells(("ar20",), dofs, regime="id")
    assert cell_floored_dofs(everywhere, dofs) == frozenset()


def test_the_floored_channel_is_flagged_on_the_contrast_row_that_carries_its_huge_delta() -> None:
    dofs = ("pitch_rate", "roll")
    arm_cells = _cells(("ar20",), dofs, regime="unseen_heading", headings=(90.0,), jitter=0.1)
    reference_cells = _cells(("ar20",), dofs, regime="unseen_heading", headings=(90.0,), jitter=0.1)
    arm_rows = _rows(("ar20",), dofs, 0.7)
    reference_rows = _rows(("ar20",), dofs, 0.6)
    for frame in (arm_rows, reference_rows):
        frame["regime"] = "unseen_heading"
    table = contrast_table(
        arm_rows, reference_rows, "revin", arm_cells=arm_cells, reference_cells=reference_cells
    )
    flagged = table.loc[table["logical_dof"] == "pitch_rate", "on_residual_floor"]
    unflagged = table.loc[table["logical_dof"] == "roll", "on_residual_floor"]
    assert flagged.astype("boolean").all()
    assert not unflagged.astype("boolean").any()


def test_the_floor_flag_is_unknown_rather_than_false_when_it_could_not_be_derived() -> None:
    """A missing derivation must not read as "checked, and not floored"."""
    dofs = ("pitch",)
    table = contrast_table(_rows(("ar20",), dofs, 0.7), _rows(("ar20",), dofs, 0.6), "revin")
    assert table["on_residual_floor"].isna().all()
