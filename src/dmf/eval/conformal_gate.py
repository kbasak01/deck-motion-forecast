"""Gate 10 — split-conformal calibration, checked as a process rather than as a number.

Phase 10's honest outcome is not known in advance in three of its four regimes, and P10-D1
predicts that calibration will **fail** to restore coverage under sea-state shift. A numeric
bar on that would be a bar set by whoever already knew where the answer landed, which is the
reason Gate 8 is a process gate too. **A failure to calibrate under shift is a PASS with a
negative finding** (``CLAUDE.md`` non-negotiable 6).

The seven predicates:

1. the committed uncalibrated tables carry no uncommitted modification (it prints their
   digests but compares them to nothing, so a *committed* rewrite would still pass -- the
   stronger byte-identity claim against `main` is checked outside the gate);
2. the calibrated labels are disjoint from the committed ones and the two tables share a
   schema and a window population, so they join rather than collide;
3. every calibration was fitted on its regime's ``val`` partition with train-fitted
   normalisation statistics;
4. ``id`` coverage lands inside Gate 5's band -- the one regime where split conformal has a
   guarantee, so this is a correctness check on the implementation, not a result;
5. every calibrated row carries at least three seeds;
6. the pre-registration was committed before the first calibrated artifact, from ``git log``;
7. the degradation table exists for every out-of-distribution regime **and carries no
   threshold** -- the non-predicate is stated as a predicate so that a later contributor
   cannot add a bar without deleting a line that says not to.

Predicate 6 is checked from ``git log`` because it is the one predicate no artifact can
testify to -- the same reasoning :mod:`dmf.eval.gate` gives for Gate 6's predicate 1.

The logic lives here rather than in ``scripts/gate10.py`` because ``CLAUDE.md`` puts every
importable and testable thing under ``src/dmf/``; the script is an argparse wrapper, and
``tests/test_gate10.py`` imports this module.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pandas as pd

from dmf.eval.conformal_runner import CONFORMAL_ARTIFACTS
from dmf.eval.gate import GATE5_DOF, GATE5_HORIZON_SAMPLES, GATE5_PICP_BAND, GATE5_REGIME
from dmf.eval.report import MIN_SEEDS

#: The committed Phase 5 artifacts predicate 1 protects. Their content is the uncalibrated
#: reading Gate 5 was decided on; if this arm can change them it can rewrite a closed gate.
PROTECTED: tuple[str, ...] = (
    "results/e03/probabilistic.csv",
    "results/e03/probabilistic_by_seed.csv",
    "results/e03/gate5.csv",
    "results/e03/gate5_degradation.csv",
)

#: Out-of-distribution regimes predicate 7 requires a degradation row for.
OOD_REGIMES: tuple[str, ...] = ("unseen_seastate", "unseen_heading", "unseen_vessel")

#: Verdict strings, shared with the renderer so the gate CSV and the markdown agree.
PASS = "PASS"
FAIL = "FAIL"

__all__ = [
    "FAIL",
    "OOD_REGIMES",
    "PASS",
    "PROTECTED",
    "read_gate10",
]


def _digest(path: Path) -> str:
    """Return the SHA-256 of a file, or ``"absent"``.

    Args:
        path: File to digest.

    Returns:
        Hex digest, or the literal ``"absent"``.
    """
    if not path.is_file():
        return "absent"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _predicate_1() -> tuple[bool, str]:
    """The committed uncalibrated tables are unchanged.

    Checked against the git index rather than against a recorded constant, so that the
    predicate keeps working after a legitimate future change to those files: what it forbids
    is *this arm* changing them, i.e. an uncommitted or unstaged modification.

    Returns:
        ``(passed, note)``.
    """
    dirty: list[str] = []
    for name in PROTECTED:
        proc = subprocess.run(
            ["git", "diff", "HEAD", "--quiet", "--", name], capture_output=True, check=False
        )
        if proc.returncode != 0:
            dirty.append(name)
    if dirty:
        return False, f"modified since HEAD: {dirty}. The uncalibrated reading is not additive."
    digests = {Path(n).name: _digest(Path(n))[:12] for n in PROTECTED}
    return True, f"all {len(PROTECTED)} clean at HEAD; digests {digests}"


def _predicate_2(conformal: pd.DataFrame, heads: pd.DataFrame) -> tuple[bool, str]:
    """The two tables join: disjoint labels, shared schema, same window population.

    Args:
        conformal: The calibrated table.
        heads: The committed uncalibrated table.

    Returns:
        ``(passed, note)``.
    """
    overlap = set(conformal["model"]) & set(heads["model"])
    if overlap:
        return False, f"label collision with the committed heads: {sorted(overlap)}"
    if tuple(conformal.columns) != tuple(heads.columns):
        missing = set(heads.columns) ^ set(conformal.columns)
        return False, f"schema differs from the committed table; symmetric difference {missing}"
    keys = ["regime", "dof", "horizon_samples"]
    left = conformal.groupby(keys)["n_windows"].first()
    right = heads.groupby(keys)["n_windows"].first()
    shared = left.index.intersection(right.index)
    bad = [k for k in shared if int(left[k]) != int(right[k])]
    if bad:
        return False, f"{len(bad)} cells scored a different window count, e.g. {bad[0]}"
    return True, (
        f"{conformal['model'].nunique()} calibrated labels, disjoint; schema identical; "
        f"window counts agree on all {len(shared)} shared cells"
    )


def _predicate_3(calibration: pd.DataFrame) -> tuple[bool, str]:
    """Calibration came from each regime's validation split, under train-fitted statistics.

    Args:
        calibration: The calibration table.

    Returns:
        ``(passed, note)``.
    """
    bad_partition = calibration[calibration["fitted_on"] != calibration["regime"] + "/val"]
    if not bad_partition.empty:
        seen = sorted(set(bad_partition["fitted_on"]))
        return False, f"calibration not fitted on <regime>/val: {seen}"
    bad_stats = calibration[~calibration["norm_stats_fitted_on"].str.endswith("/train")]
    if not bad_stats.empty:
        seen = sorted(set(bad_stats["norm_stats_fitted_on"]))
        return False, f"normalisation statistics not train-fitted: {seen}"
    used = sorted(set(calibration["n_windows_used"]))
    return True, (
        f"{calibration['regime'].nunique()} regimes, all fitted on <regime>/val under "
        f"<regime>/train statistics; calibration windows {used}"
    )


def _predicate_4(conformal: pd.DataFrame) -> tuple[bool, str]:
    """In-distribution coverage lands in Gate 5's band at the gate cell.

    The one regime where the calibration set is exchangeable with the test set, so this is a
    correctness check on the construction rather than a finding. A failure here means the
    implementation is wrong, not that the data is interesting.

    Args:
        conformal: The calibrated table.

    Returns:
        ``(passed, note)``.
    """
    lo, hi = GATE5_PICP_BAND
    cell = conformal[
        (conformal["regime"] == GATE5_REGIME)
        & (conformal["dof"] == GATE5_DOF)
        & (conformal["horizon_samples"] == GATE5_HORIZON_SAMPLES)
    ]
    if cell.empty:
        return (
            False,
            f"no {GATE5_REGIME}/{GATE5_DOF}/{GATE5_HORIZON_SAMPLES} cell; absent is not a pass",
        )
    out = cell[(cell["picp_mean"] < lo) | (cell["picp_mean"] > hi)]
    worst = float((cell["picp_mean"] - 0.90).abs().max())
    if not out.empty:
        rows = ", ".join(f"{r.model} {r.picp_mean:.4f}" for r in out.itertuples())
        return False, f"{len(out)} of {len(cell)} outside [{lo}, {hi}]: {rows}"
    return True, (
        f"{len(cell)} of {len(cell)} inside [{lo}, {hi}]; worst deviation from 0.90 is {worst:.4f}"
    )


def _predicate_5(conformal: pd.DataFrame) -> tuple[bool, str]:
    """Every calibrated row rests on at least three seeds.

    Args:
        conformal: The calibrated table.

    Returns:
        ``(passed, note)``.
    """
    thin = conformal[(~conformal["deterministic"]) & (conformal["n_seeds"] < MIN_SEEDS)]
    if not thin.empty:
        return False, f"{len(thin)} stochastic rows carry fewer than {MIN_SEEDS} seeds"
    return True, f"all {len(conformal)} rows carry >= {MIN_SEEDS} seeds or are deterministic"


def _predicate_6(results_dir: Path) -> tuple[bool, str]:
    """The pre-registration was committed before the first calibrated artifact.

    Args:
        results_dir: Where the calibrated tables live.

    Returns:
        ``(passed, note)``.
    """
    table = results_dir / CONFORMAL_ARTIFACTS["conformal"]

    def first_commit(path: str, needle: str | None = None) -> str:
        cmd = ["git", "log", "--diff-filter=A", "--format=%H %cI", "--reverse", "--", path]
        if needle is not None:
            cmd = ["git", "log", "-S", needle, "--format=%H %cI", "--reverse", "--", path]
        out = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
        return out.splitlines()[0] if out else ""

    prereg = first_commit("docs/protocol.md", "P10-D1")
    artifact = first_commit(str(table))
    if not prereg:
        return False, "no commit introduces P10-D1 in docs/protocol.md"
    if not artifact:
        return False, f"{table} is not committed yet, so the ordering cannot be checked"
    if prereg.split()[1] >= artifact.split()[1]:
        return False, f"P10-D1 committed at {prereg.split()[1]}, not before {artifact.split()[1]}"
    return (
        True,
        f"P10-D1 at {prereg.split()[1]} precedes the first {table.name} at {artifact.split()[1]}",
    )


def _predicate_7(results_dir: Path, conformal: pd.DataFrame) -> tuple[bool, str]:
    """The degradation table exists for every OOD regime, and carries no threshold.

    **This predicate imposes no bar on the degradation and must never acquire one.** P5-D2
    and P10-D1 both require the out-of-distribution behaviour to be reported rather than
    fixed; a gate that failed on a large degradation would be a gate that rewards hiding it.

    Args:
        results_dir: Where the calibrated tables live.
        conformal: The calibrated table.

    Returns:
        ``(passed, note)``.
    """
    path = results_dir / CONFORMAL_ARTIFACTS["degradation"]
    if not path.is_file():
        return False, f"{path} is absent; the degradation must be reported, not omitted"
    frame = pd.read_csv(path)
    present = set(frame["regime"]) & set(OOD_REGIMES)
    missing = [r for r in OOD_REGIMES if r not in present and r in set(conformal["regime"])]
    if missing:
        return False, f"no degradation rows for {missing}"
    spans = ", ".join(
        f"{r}: picp_delta {frame[frame.regime == r].picp_delta.mean():+.4f} mean"
        for r in sorted(present)
    )
    return True, f"{len(frame)} rows, no threshold applied by design. {spans}"


def read_gate10(
    results_dir: Path, heads_path: Path = Path("results/e03/probabilistic.csv")
) -> pd.DataFrame:
    """Evaluate all seven predicates and return them as a frame.

    Args:
        results_dir: Directory holding the calibrated tables.
        heads_path: The committed uncalibrated table.

    Returns:
        One row per predicate, with ``criterion``, ``description``, ``verdict`` and ``note``.

    Raises:
        FileNotFoundError: If the calibrated table is absent -- there is nothing to read, and
            reporting that as a failed predicate would conflate "not run" with "ran and
            failed".
    """
    conformal_path = results_dir / CONFORMAL_ARTIFACTS["conformal"]
    if not conformal_path.is_file():
        raise FileNotFoundError(f"{conformal_path} is absent; run `make conformal` first")
    conformal = pd.read_csv(conformal_path)
    calibration = pd.read_csv(results_dir / CONFORMAL_ARTIFACTS["calibration"])
    heads = pd.read_csv(heads_path)
    checks = [
        ("1", "committed uncalibrated tables unchanged", _predicate_1()),
        ("2", "calibrated table joins the committed one", _predicate_2(conformal, heads)),
        ("3", "calibrated on <regime>/val under train statistics", _predicate_3(calibration)),
        ("4", "id coverage in band at the gate cell", _predicate_4(conformal)),
        ("5", f">= {MIN_SEEDS} seeds on every stochastic row", _predicate_5(conformal)),
        ("6", "pre-registration precedes the run", _predicate_6(results_dir)),
        ("7", "degradation reported, no threshold", _predicate_7(results_dir, conformal)),
    ]
    return pd.DataFrame(
        [
            {
                "criterion": key,
                "description": description,
                "verdict": PASS if ok else FAIL,
                "note": note,
            }
            for key, description, (ok, note) in checks
        ]
    )


def build_gate10_markdown(frame: pd.DataFrame) -> str:
    """Render the gate read-out.

    Args:
        frame: The frame :func:`read_gate10` returned.

    Returns:
        The markdown document.
    """
    passed = int((frame["verdict"] == PASS).sum())
    lines = [
        "# Gate 10 — split-conformal calibration",
        "",
        f"**{passed} of {len(frame)} predicates pass.**",
        "",
        "A failure to calibrate under distribution shift is a PASS with a negative finding: "
        "predicate 7 requires the degradation to be reported and deliberately sets no bar on "
        "it (P5-D2, P10-D1).",
        "",
        "| # | predicate | verdict | note |",
        "|---|---|---|---|",
    ]
    for row in frame.itertuples():
        lines.append(f"| {row.criterion} | {row.description} | {row.verdict} | {row.note} |")
    return "\n".join(lines) + "\n"
