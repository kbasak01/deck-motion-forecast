#!/usr/bin/env python3
"""Gate 8 — MSS cross-validation, re-derived independently of the run that produced it.

Phase 8 is the only phase the implementation plan gives no numeric gate line for, and that
absence is load-bearing rather than an oversight. Carry-forward delta 1 shows that both "the
deep model held up on MSS data" and "the deep model collapsed" are purchasable from the same
run by choosing how tightly the spectrum is matched, so a numeric bar here would be doing no
work: it would be set by whoever already knew roughly where the answer lands. Gate 8 is
therefore a **process** gate, in the shape of Gate 6 (P6-D1), with five predicates:

1. the realized spectrum matches SS5 within 5%, read on the mean over realizations;
2. units and signs are asserted at the CSV boundary, and the sign-flip ablation was run;
3. `persistence`, `window_mean` and `dlinear_ols` were recomputed on the MSS trajectories;
4. every model-vs-model statement rests on at least three seeds;
5. the pre-registration was committed before the first evaluation run.

**A collapse in skill is a PASS with a negative finding.** The gate asks whether the protocol
was followed and the result reported whichever way it fell, not whether the number was
flattering (CLAUDE.md non-negotiable 6).

Predicate 5 is checked from `git log`, because it is the one predicate no artifact can
testify to -- the same reasoning `scripts/gate6.py` gives for its predicate 1.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

import dmf.models  # noqa: F401  -- import populates MODEL_REGISTRY
from dmf.config import load_experiment
from dmf.train.registry import MODEL_REGISTRY

REQUIRED_BASELINES: tuple[str, ...] = ("persistence", "window_mean", "dlinear_ols")
MIN_SEEDS: int = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/mss/s175_ss5.yaml"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/mss"))
    parser.add_argument("--mss-dir", type=Path, default=Path("artifacts/mss"))
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiment/e02_deep.yaml"),
        help="Experiment whose model configs say which rows are closed-form and therefore "
        "exempt from the three-seed rule.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Exit 0 even when a predicate is unverified. The predicate still records as "
        "a failure in the artifact; this only changes the exit code.",
    )
    return parser


def _predicate_1(cfg: dict, mss_dir: Path, results_dir: Path) -> tuple[bool, str]:
    """Spectrum match, read from the COMMITTED summary.

    `artifacts/` is gitignored, so `artifacts/mss/manifest.csv` cannot testify to anything
    on a fresh clone -- a gate that reads it passes only on the machine that happens to
    still hold the run. The committed `results/mss/spectrum_match.csv` is the evidence; the
    manifest is used only as a fallback when the summary has not been written yet.
    """
    tol = cfg["match_tolerance"]
    summary = results_dir / "spectrum_match.csv"
    if summary.exists():
        m = pd.read_csv(summary)
        hs = float(m["hs_rel_err_mean"].mean())
        hs_sd = float(m["hs_rel_err_std"].mean())
        tz = float(m["tz_rel_err_mean"].mean())
        tz_sd = float(m["tz_rel_err_std"].mean())
        n = int(m["hs_rel_err_count"].sum())
        source = summary
    else:
        path = mss_dir / "manifest.csv"
        if not path.exists():
            return False, (
                f"neither {summary} (committed) nor {path} (local) exists; the spectrum "
                "match is unverified"
            )
        raw = pd.read_csv(path)
        hs, hs_sd = float(raw["hs_rel_err"].mean()), float(raw["hs_rel_err"].std())
        tz, tz_sd = float(raw["tz_rel_err"].mean()), float(raw["tz_rel_err"].std())
        n = len(raw)
        source = path
    ok = abs(hs) <= float(tol["hs_rel"]) and abs(tz) <= float(tol["tz_rel"])
    return ok, (
        f"Hs {hs:+.4f} +/- {hs_sd:.4f}, Tz {tz:+.4f} +/- {tz_sd:.4f} "
        f"(tol {tol['hs_rel']}/{tol['tz_rel']}, mean over {n} records, from {source})"
    )


def _predicate_2(results_dir: Path) -> tuple[bool, str]:
    hits = sorted(results_dir.glob("skill_mss_*.csv"))
    if not hits:
        return False, "no skill table; the evaluation did not run"
    frames = [pd.read_csv(p) for p in hits]
    table = pd.concat(frames, ignore_index=True)
    if "sign_convention" not in table.columns:
        return False, "skill table has no sign_convention column"
    conventions = sorted(table["sign_convention"].unique())
    ok = "nominal" in conventions and "sign_flipped" in conventions
    return ok, f"sign conventions scored: {conventions}"


def _predicate_3(results_dir: Path) -> tuple[bool, str]:
    hits = sorted(results_dir.glob("skill_mss_*.csv"))
    if not hits:
        return False, "no skill table"
    table = pd.concat([pd.read_csv(p) for p in hits], ignore_index=True)
    models = {str(m).split("|")[0] for m in table["model"].unique()}
    missing = [b for b in REQUIRED_BASELINES if b not in models]
    if missing:
        return False, f"baselines absent from the MSS table: {missing}"
    ref = table[table["model"].astype(str).str.startswith("persistence")]
    worst = float(ref["skill"].abs().max()) if "skill" in ref.columns else float("nan")
    ok = worst == 0.0
    return ok, (
        f"all of {list(REQUIRED_BASELINES)} recomputed on MSS data; "
        f"persistence self-skill max |.| = {worst:.3e} (must be exactly 0)"
    )


def _predicate_4(results_dir: Path, experiment: Path) -> tuple[bool, str]:
    """Non-negotiable 5: every model-vs-model comparison rests on at least three seeds.

    Which rows the rule applies to is **derived from each model's `FIT_KIND`**, not from a
    hardcoded list of baseline names. A closed-form or untrained model is seed-independent
    by construction -- `dmf.models.base.FitKind` is the property that says so -- and carries
    one row legitimately; only `sgd` rows can differ between seeds and therefore only they
    owe a spread.

    An earlier version tested membership of a hardcoded three-name baseline tuple, which
    passed until the AR family and `damped_persistence` were added and then failed four
    closed-form models for having exactly the one seed they can have. That is the same
    defect as the first version of predicate 5: reading a proxy for the property instead of
    the property.
    """
    hits = sorted(results_dir.glob("skill_mss_*.csv"))
    if not hits:
        return False, "no skill table"
    table = pd.concat([pd.read_csv(p) for p in hits], ignore_index=True)
    table["label"] = table["model"].astype(str).str.split("|").str[0]

    cfg = load_experiment(experiment)
    deterministic: set[str] = set()
    for model_cfg in cfg.models:
        cls = MODEL_REGISTRY.get(model_cfg.name)
        if cls is not None and getattr(cls, "FIT_KIND", "sgd") != "sgd":
            deterministic.add(model_cfg.label)

    if not MODEL_REGISTRY:
        return False, (
            "MODEL_REGISTRY is empty -- `dmf.models` was not imported, so every model "
            "would be treated as SGD and the exemption would silently vanish"
        )

    sgd = table[~table["label"].isin(deterministic)]
    if sgd.empty:
        return False, "no SGD model rows present"
    counts = sgd.groupby("label")["model"].nunique()
    short = counts[counts < MIN_SEEDS]
    ok = bool(short.empty)
    detail = (
        f"seeds per SGD model: {counts.to_dict()}; "
        f"{len(deterministic)} closed-form models exempt by FIT_KIND ({sorted(deterministic)})"
    )
    if not ok:
        detail += f"; below {MIN_SEEDS}: {short.to_dict()}"
    return ok, detail


def _predicate_5() -> tuple[bool, str]:
    """Was the pre-registration committed before the first evaluation run?

    Checked from git, because no artifact can testify to its own ordering.

    The pre-registration is located by the commit that *introduced* the P8-D1 heading, found
    with `git log -S`, not by the newest commit touching `docs/protocol.md` -- the results
    entries are appended to that same file later, so "newest commit touching the protocol"
    would compare the results commit against itself and fail a phase that did the right
    thing. That is the first thing this predicate got wrong, and it is recorded rather than
    quietly corrected.
    """
    marker = "P8-D1 — PRE-REGISTRATION"
    prereg_log = subprocess.run(
        ["git", "log", "--format=%H", "-S", marker, "--", "docs/protocol.md"],
        capture_output=True,
        text=True,
    )
    prereg_commits = [line for line in prereg_log.stdout.split("\n") if line]
    if not prereg_commits:
        return False, f"no commit introduces {marker!r} in docs/protocol.md"
    prereg = prereg_commits[-1]  # oldest commit touching that string = the one that added it

    skill_log = subprocess.run(
        ["git", "log", "--format=%H", "--", "results/mss/skill_mss_mss.csv"],
        capture_output=True,
        text=True,
    )
    skill_commits = [line for line in skill_log.stdout.split("\n") if line]
    if not skill_commits:
        return False, (
            "the skill table is not committed yet, so the ordering cannot be verified. "
            "Commit the results and re-run this gate."
        )
    first_skill = skill_commits[-1]

    if prereg == first_skill:
        return False, (
            f"the pre-registration and the first skill table are the same commit "
            f"({prereg[:8]}); the pre-registration must land first"
        )
    ordered = subprocess.run(
        ["git", "merge-base", "--is-ancestor", prereg, first_skill], capture_output=True
    )
    ok = ordered.returncode == 0
    return ok, (
        f"pre-registration {prereg[:8]} precedes first skill table {first_skill[:8]}: "
        + ("yes" if ok else "NO -- the evaluation was committed first")
    )


def _predicate_6(results_dir: Path) -> tuple[bool, str]:
    """Delta 7: the operational metric was run, and every F1 carries its base rate.

    `CLAUDE.md` §Known traps requires the base rate beside every F1, and delta 7 names
    quiescence the part of this phase most exposed to a units or scale error. The first
    Phase 8 pass skipped the metric entirely, so the gate now checks for it.
    """
    path = results_dir / "quiescence_mss.csv"
    if not path.exists():
        return False, f"{path} absent; the operational metric was not run (delta 7)"
    table = pd.read_csv(path)
    for column in ("f1", "base_rate"):
        if column not in table.columns:
            return False, f"quiescence table has no {column!r} column"
    if table["base_rate"].isna().any():
        return False, "some rows report an F1 with no base rate beside it"
    rates = table.groupby("threshold_set")["base_rate"].mean().round(4).to_dict()
    return True, f"{len(table)} rows, every F1 carries a base rate; mean base rate {rates}"


def _predicate_7(results_dir: Path) -> tuple[bool, str]:
    """The wave-grid attribution control the config declares was actually run.

    `configs/mss/s175_ss5.yaml` calls the corpus wave grid "the CONTROL", and the first
    Phase 8 pass never invoked it while conceding the confound it removes. A control that
    exists in the config and not in the results is not a control.
    """
    mss = results_dir / "skill_mss_mss.csv"
    corpus = results_dir / "skill_mss_corpus.csv"
    if not corpus.exists():
        return False, f"{corpus} absent; --grid-kind corpus was never run"
    if not mss.exists():
        return False, f"{mss} absent"
    return True, "both wave-grid conventions scored; attribution is separable"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    out_dir = args.out_dir or args.results_dir

    checks = [
        ("1. spectrum matched to SS5 within 5%", _predicate_1(cfg, args.mss_dir, args.results_dir)),
        ("2. units and signs asserted; sign ablation run", _predicate_2(args.results_dir)),
        ("3. baselines re-scored on MSS trajectories", _predicate_3(args.results_dir)),
        (
            "4. at least three seeds behind model comparisons",
            _predicate_4(args.results_dir, args.experiment),
        ),
        ("5. pre-registration committed before evaluation", _predicate_5()),
        ("6. quiescence run with base rate beside every F1", _predicate_6(args.results_dir)),
        ("7. declared wave-grid control actually run", _predicate_7(args.results_dir)),
    ]

    rows = []
    for name, (ok, detail) in checks:
        rows.append({"predicate": name, "passed": bool(ok), "evidence": detail})
        print(f"[gate8] {'PASS' if ok else 'FAIL'}  {name}", file=sys.stderr)
        print(f"[gate8]        {detail}", file=sys.stderr)

    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gate8.csv"
    table.to_csv(path, index=False)
    print(f"[gate8] wrote {path}", file=sys.stderr)

    n_fail = int((~table["passed"]).sum())
    print(
        f"[gate8] {len(table) - n_fail}/{len(table)} predicates verified. "
        "A collapse in skill is a PASS with a negative finding; only an unfollowed "
        "protocol is a failure.",
        file=sys.stderr,
    )
    if n_fail and not args.allow_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
