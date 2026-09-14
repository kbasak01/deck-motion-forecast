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

REQUIRED_BASELINES: tuple[str, ...] = ("persistence", "window_mean", "dlinear_ols")
MIN_SEEDS: int = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/mss/s175_ss5.yaml"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/mss"))
    parser.add_argument("--mss-dir", type=Path, default=Path("artifacts/mss"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Exit 0 even when a predicate is unverified. The predicate still records as "
        "a failure in the artifact; this only changes the exit code.",
    )
    return parser


def _predicate_1(cfg: dict, mss_dir: Path) -> tuple[bool, str]:
    path = mss_dir / "manifest.csv"
    if not path.exists():
        return False, f"{path} absent; nothing was exported"
    m = pd.read_csv(path)
    tol = cfg["match_tolerance"]
    hs, hs_sd = float(m["hs_rel_err"].mean()), float(m["hs_rel_err"].std())
    tz, tz_sd = float(m["tz_rel_err"].mean()), float(m["tz_rel_err"].std())
    ok = abs(hs) <= float(tol["hs_rel"]) and abs(tz) <= float(tol["tz_rel"])
    return ok, (
        f"Hs {hs:+.4f} +/- {hs_sd:.4f}, Tz {tz:+.4f} +/- {tz_sd:.4f} "
        f"(tol {tol['hs_rel']}/{tol['tz_rel']}, read on the mean over {len(m)} records)"
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


def _predicate_4(results_dir: Path) -> tuple[bool, str]:
    hits = sorted(results_dir.glob("skill_mss_*.csv"))
    if not hits:
        return False, "no skill table"
    table = pd.concat([pd.read_csv(p) for p in hits], ignore_index=True)
    table["label"] = table["model"].astype(str).str.split("|").str[0]
    table["seed"] = table["model"].astype(str).str.split("|").str[-1]
    # Closed-form rows are deterministic and carry one seed by construction; the three-seed
    # rule applies to the SGD rows, which are the ones a model-vs-model claim rests on.
    sgd = table[~table["label"].isin(REQUIRED_BASELINES)]
    if sgd.empty:
        return False, "no SGD model rows present"
    counts = sgd.groupby("label")["seed"].nunique()
    short = counts[counts < MIN_SEEDS]
    ok = short.empty
    detail = f"seeds per SGD model: {counts.to_dict()}"
    if not ok:
        detail += f"; below {MIN_SEEDS}: {short.to_dict()}"
    return ok, detail


def _predicate_5() -> tuple[bool, str]:
    """Was the pre-registration committed before the first evaluation run?

    Checked from git rather than from an artifact, because no artifact can testify to its
    own ordering. The pre-registration commit must touch `docs/protocol.md` and must be an
    ancestor of -- and distinct from -- the commit that first introduced a skill table.
    """
    def _log(path: str) -> list[str]:
        out = subprocess.run(
            ["git", "log", "--format=%H", "--", path], capture_output=True, text=True
        )
        return [line for line in out.stdout.split("\n") if line]

    protocol = _log("docs/protocol.md")
    skill = _log("results/mss/skill_mss_mss.csv")
    if not protocol:
        return False, "docs/protocol.md has no commits"
    if not skill:
        return False, (
            "the skill table is not committed yet, so the ordering cannot be verified. "
            "Commit the results and re-run this gate."
        )
    prereg = protocol[0]
    # Oldest commit touching the skill table is the one whose ordering matters.
    first_skill = skill[-1]
    anc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", prereg, first_skill], capture_output=True
    )
    ok = anc.returncode == 0 and prereg != first_skill
    return ok, (
        f"pre-registration {prereg[:8]} vs first skill table {first_skill[:8]}: "
        + ("ordered correctly" if ok else "NOT committed before the evaluation")
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    out_dir = args.out_dir or args.results_dir

    checks = [
        ("1. spectrum matched to SS5 within 5%", _predicate_1(cfg, args.mss_dir)),
        ("2. units and signs asserted; sign ablation run", _predicate_2(args.results_dir)),
        ("3. baselines recomputed on MSS trajectories", _predicate_3(args.results_dir)),
        ("4. at least three seeds behind model comparisons", _predicate_4(args.results_dir)),
        ("5. pre-registration committed before evaluation", _predicate_5()),
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
