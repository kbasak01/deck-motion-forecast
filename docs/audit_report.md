# Pre-release audit — `deck-motion-forecast`

**These are simulated results.** Nothing audited here is evidence about real deck motion; this
document checks that the project's numbers came from the artifacts beside them and that its claims
match what those artifacts say.

Run date 2026-09-14, branch `phase-8-mss-crossvalidation`, HEAD `9d95573`, working tree dirty.
Protocol: `docs/IMPLEMENTATION_PLAN.md` §5, all six subsections, checked against committed
artifacts rather than against code comments or the README's prose. The four integrity controls of
the audit brief were re-run here rather than read off a CSV; the adversarial pass was delegated to
the `results-skeptic` agent and its findings are merged in below, each one re-verified against the
artifact it names before being written down.

Machine: NVIDIA RTX A4000, torch 2.13.0+cu130, onnxruntime 1.29.0. Corpus and checkpoints present
locally (`artifacts/corpus` 1.1 GB, `artifacts/checkpoints` 335 MB, 180 checkpoints).

---

## Integrity controls, measured

### 1. Shuffle control — PASS, and run on the best model for the first time

The committed shuffle control refits **AR(20)** on shuffled targets
(`src/dmf/train/experiment.py:1308-1348`); `results/e04/controls.csv` carries no deep-model
subject. §5.2 asks for *the best model*, which at the gate cell is `lstm`. I retrained `lstm`
(`configs/model/lstm.yaml`, e02_deep budget, seed 0, regime `id`) on training targets permuted by a
single global permutation over all 1 465 776 training windows — so a re-paired target comes from a
different realization, sea state, heading and seed — and scored it on all 434 304 `id/test`
windows. Early stopping used the **honest** validation targets, the conservative direction, since
it hands the shuffled model the best epoch it could possibly have.

| `id` / pitch / 10 s | skill vs persistence |
|---|---:|
| `window_mean` (the null) | 0.3656 |
| shuffled-target `lstm` | **0.3658** |
| untrained `lstm` | 0.3659 |
| trained `lstm` (committed) | **0.8680** |

Worst excess of the shuffled model over the window-mean null across all 36 (DOF, horizon) cells:
**0.0038**, against the project's 0.02 tolerance. **0 of 36** cells exceed it, and the largest
skill gap from the window mean anywhere is 0.012. The fit early-stopped at epoch 19 with its best
epoch at 2 and a validation MSE of 1.049 against ~0.13 for an honest run — it learned nothing and
stopped.

The protocol's literal null ("skill must collapse to ~0") is **wrong for this task and would fire a
false alarm**: the shuffled model reaches +0.71 skill at 5 s. That is not leakage, it is the window
mean, which beats persistence on a narrowband signal from about 2 s out. The project's restated
null — the window-mean forecast — is the correct one, is derived at
`src/dmf/eval/controls.py:36-52`, and my independent run reproduces both the restatement and the
pass. **No leakage.**

### 2. Untrained control — fails its literal criterion, for the documented reason

A randomly initialised `lstm`, same geometry, scored on the same 434 304 windows. Worse than
persistence at 1 s on all six DOF (skill −0.68 to −3.42), but it **beats persistence in 28 of 36
cells**, peaking at +0.711 at 5 s. The literal §5.2 criterion therefore fails.

It fails because it is measuring the window mean: untrained skill sits within 0.053 of
`window_mean` skill in every cell and within 0.0003 at the gate cell. A small-weight random
initialisation emits ≈0, and `invert_norm` maps that to the window mean. This reproduces the
project's own finding (P3-D9; `results/e04/controls.csv` `control=untrained`, `enforced=False`,
76 of 144 rows failing on `e02_deep`) on a different architecture. Reported, not enforced, is the
right disposition. See **N1** for the one thing the README understates about it.

### 3. Pipeline sanity — PASS

Persistence through the full `DeckMotionDataset` → `Persistence` → `invert_norm` path against
persistence recomputed with plain NumPy directly off the Parquet blocks, in **all four regimes**,
1 954 368 windows total, 288 (regime, DOF, horizon) cells:

| regime | windows | max relative difference |
|---|---:|---:|
| `id` | 434 304 | 1.65e-10 |
| `unseen_seastate` | 542 880 | 8.23e-11 |
| `unseen_heading` | 542 880 | 1.90e-10 |
| `unseen_vessel` | 434 304 | 2.83e-10 |

Against a 1e-6 tolerance. My raw-side figures also agree with the committed
`results/e04/pipeline_sanity.csv` to 2.9e-8 across all 288 rows, and the window counts match
exactly. The skill denominator was confirmed a second way: `rmse_persistence` = 1.55915455 at
`id`/pitch/10 s in `results/e04/metrics_full.csv` against 1.5591545467 computed independently
(8.99e-11 relative).

### 4. Benchmark stability — p50 PASS, p99 does not, and one p50 row is marginal

> **Completed after the audit agent terminated.** The agent hit a rate limit while running this
> control and never wrote the section that two places above and in §5.4 forward-reference. Those
> references pointed at nothing — a PASS asserted on evidence that did not exist, which is the
> defect shape P8-D15 and P7-D14 both record. Verified here against the committed two-run
> comparison rather than by re-sweeping, per the disposition agreed for this pass.

From `results/latency_stability.csv`, 52 configurations measured twice:

| statistic | within 10 % | worst drift |
|---|---:|---|
| **p50** | **52 of 52** | −9.91 % — `tcn` / torch-eager / cpu / batch 32 |
| **p99** | **43 of 52** | −43.59 % — `tcn_quantile` / torch-eager / cuda / batch 32 |

The plan's §5.4 checkbox asks only about p50, and on that criterion this passes outright. Two things
belong beside the tick.

**The worst p50 row sits at 9.91 % against a 10 % bar.** It passes by 0.09 of a percentage point on
one row of 52. That is a pass, not a comfortable one, and a re-run could plausibly move it across.

**The p99 failures are not scattered.** All nine land on `torch-compile` (4), `torch-eager` (3) and
TensorRT engine-build (2) — the three backends with a compile or build step — and none on the ORT
CPU or ORT CUDA providers that carry the README's main claims. One of the nine,
`tcn_quantile`/ort-trt at batch 1, is a row supporting the README's claim that TensorRT wins the
convolutional median; that claim is read on the median, which is stable in both runs, and the
README says so. `docs/protocol.md` P7-D13 records the p99 outcome as a **partial failure of the
plan's own checkbox** rather than rounding it into a pass, and the README's limitations repeat it.

**Environment stamp:** every row carries `git_commit = 7bd8d86-dirty` — see **N4**; the `-dirty`
suffix means the stamp cannot identify the source tree exactly.


---

## BLOCKING

### B1 — The MSS headline table tells the reader to compare two columns that are not comparable, and the prose deltas come from a third column that is never printed

`README.md:52-54`:

> "read it against the `unseen_vessel` column beside it, **because only the generator changes
> between the two**"

That is false. `unseen_vessel` is the full held-out-vessel grid (4 sea states × 4 headings ×
3 speeds); the MSS column is the matched subset only (SS5, headings 135/180, 3 speeds). The correct
comparator is the *matched-corpus* column, which `docs/mss_crossvalidation.md:96-105` prints and the
README does not:

| model | README `unseen_vessel` | matched corpus (the real comparator) | MSS |
|---|---:|---:|---:|
| `tcn` | 0.8298 | **0.8604 ± 0.0040** | −0.9033 |
| `transformer` | 0.7250 | **0.7654 ± 0.0323** | −0.1773 |
| `lstm` | 0.8007 | **0.8032 ± 0.0169** | −1.5564 |
| `ar40` | 0.5263 | **0.5412** | −0.7243 |
| `ar20` | 0.4786 | **0.5258** | 0.1560 |
| `window_mean` | 0.1724 | **0.0733** | −0.0586 |

Every number in the paragraph at `README.md:109-116` is computed against the matched column:
"`tcn` goes from **0.860** to −0.903" (`README.md:111` — 0.860 appears nowhere in the README's own
table), and "`ar40` loses **1.27** where `ar20` loses **0.37**" (`README.md:115`; matched
0.5412−(−0.7243)=1.2655 and 0.5258−0.1560=0.3698, whereas the README's printed table gives 1.25 and
0.32). A reader who does the subtraction the README instructs them to do gets different numbers from
the ones it states. This is the column P8-D12 (`docs/protocol.md:5024`) recomputed after the AR
baselines were restored.

**Smallest fix:** replace the MSS table's `unseen_vessel` column with the matched-corpus column
already computed in `docs/mss_crossvalidation.md`, and delete the clause "because only the generator
changes between the two".

### B2 — The result that the deep models lose to the closed-form linear model in the operational band was deleted from the README

Paired bootstrap against `dlinear_ols` from `results/e02/paired_contrasts.csv`, excluding
`unseen_heading`, cells where the 95 % CI excludes zero (reproduced independently for this audit):

| band | `tcn` | `transformer` | `lstm` |
|---|---|---|---|
| **1–5 s** — the operational band | 34 win – 35 lose | **20 – 49** | **25 – 43** |
| 10–15 s — where Gate 4 is read | 27 – 8 | 26 – 9 | 27 – 8 |

The previously committed README carried this table (as 34–34 / 19–47 / 23–42 and 26–8 / 25–8 /
26–7, within one or two cells of my recount — the direction and the ratio are identical) and the
sentence "*the gate is read where the deep models look best, and that belongs next to the word
'passes'*" (`git show HEAD:README.md`, lines 118-127).
The rewritten working-tree README deletes both. What remains is a main table read at pitch/10 s,
where the deep models win by ~0.3 skill, and a heave/3 s table where everything is 0.93–0.999 and
the difference is invisible. A reader of the README alone concludes the deep models win at the
operational horizon; the committed artifact says `transformer` and `lstm` lose there roughly 2:1.

This is CLAUDE.md non-negotiable 6. The content survives only in `docs/findings.md:120-130`, which
is untracked — see B3.

**Smallest fix:** restore the two-row band table and its one-sentence caveat to the README body.

### B3 — The release candidate depends on nine files that are not in the repository, and the README asserts they are

`README.md:374`: "**`results/` is committed in full, including every figure** and the 6 MB
`results.md`." `README.md:40`: "regenerated by `make figures` from a **12 kB committed trace**."
`git ls-files` disagrees:

```
THIRD_PARTY_NOTICES.md                    UNTRACKED   (the P9-D6 licensing fix)
docs/findings.md                          UNTRACKED   (linked 4x from the README; holds B2's table)
results/headline_forecast_intervals.png   UNTRACKED   (rendered at README.md:23)
results/quiescence_lead_time_hist.png     UNTRACKED   (rendered at README.md:165)
results/headline_trace.npz                UNTRACKED   (the "committed trace")
scripts/make_figures.py                   UNTRACKED
src/dmf/viz/traces.py                     UNTRACKED
tests/test_traces.py                      UNTRACKED
tests/test_forecast_plots.py              UNTRACKED
Makefile                                  MODIFIED, uncommitted
```

Three consequences, each independently disqualifying for §5.5:

1. **`make all` on a fresh clone is the wrong recipe.** `git show HEAD:Makefile:59` is
   `all: data train eval bench report gate7` — it trains only `e01` and never runs gates 4, 5, 6
   or 8. The corrected recipe (`all: data sweeps gate4 gate5 gate6-full bench report gate7 mss
   figures-extract`) exists only in the uncommitted working tree. P9-D2 records the fix; the fix is
   not in the repository.
2. **The licensing fix is not in the repository.** P9-D6 records that
   `mss/waveMotionRAO_seeded.m` redistributed an MIT-licensed upstream file with neither the
   copyright notice nor the permission notice, and that this was fixed in three places. The notice
   is now in the `.m` file (line 57-72, uncommitted) and `THIRD_PARTY_NOTICES.md` exists and is
   correct — but both are untracked. A clone made today still carries the defect.
3. **The wall-clock table's seven logged rows are not checkable by a cloner.** `README.md:351`
   sources them to "timestamps in `artifacts/logs/`", which `.gitignore:4` excludes. This is a
   minor third item — the table is otherwise exemplary, marking per row which figures are measured
   and which are not, and saying so for the two unlogged stages and the empty `e03` log — but it is
   the same class of defect: a stated provenance that the repository does not carry.

**Smallest fix:** `git add` the nine files listed above plus the modified `Makefile`,
`mss/waveMotionRAO_seeded.m`, `mss/PATCHES.md` and `docs/protocol.md` before the README is
committed. For item 3, one clause at `README.md:351` saying the logs are not redistributed.

---

## SHOULD FIX

**S1 — Neither committed shuffle control nor untrained control runs on the best model.**
`src/dmf/train/experiment.py:1319` pins the shuffle subject to AR(20); `:1353` picks the untrained
subject as `sgd_cfgs[0]`, which `configs/experiment/e02_deep.yaml` deliberately orders as
`dlinear`. §5.2 asks for the best model, and `README.md:89` leans on the shuffle control to defend
absolute skill numbers that include the deep models. I ran both on `lstm` for this audit and both
behave correctly (Controls 1 and 2 above), so this is a reporting gap, not a defect.
*Fix:* add the two `lstm` rows to `controls.csv`, or state at `README.md:89` that the control's
subject is AR(20).

**S2 — `README.md:89` understates the shuffle control's own bound.** It quotes "excess 0.001
against a 2 % tolerance". From `results/e04/controls.csv` (`arm=reference`, `control=shuffle`,
144 rows) the maximum excess is **0.00946** and **46 of 144 rows exceed 0.001**. The conclusion
survives; the number is not the file's bound. The README also omits that the residual-floor
exemption (P6-D11) was introduced *after* the control failed at 5.52 % against the same tolerance.
*Fix:* quote 0.0095 and add a clause on the exemption.

**S3 — The README contradicts itself on `unseen_seastate` interval coverage.** `README.md:206`:
"**Every family scores zero cells in band on `unseen_seastate`**". The table three lines above
(`README.md:196`) says 5 of 216, and `results/e03/probabilistic.csv` gives `tcn_quantile` 3 of 36
and `tcn_gaussian` 2 of 36 inside [0.85, 0.95] — verified.
*Fix:* "Only the TCN family scores any cell in band (5 of 72); DLinear and LSTM score zero, as does
the residual-interval floor."

**S4 — "our generator runs about 2x hot" is quoted bare, against the source document's explicit
instruction.** `README.md:174`. `results/mss/summary_marginal_mss.csv` gives corpus/MSS RMS ratios
of 1.01x, 1.28x, 1.32x at 135° and 2.27x, 2.34x at 180°, with per-speed spread inside the roll row
alone running 0.61x to 1.54x. `docs/mss_crossvalidation.md:83` says outright: *"'our generator is
1.6x hot' is a summary that should not be quoted without its spread."* The README picks a number
above the source's own and drops the spread.
*Fix:* "runs 1.0–2.3x hot depending on channel and heading".

**S5 — The MSS column carries no seed spread, and one row's std is 38 % of its mean.**
`README.md:56-69` prints ± for every SGD model in the four corpus columns and bare numbers in the
MSS column. From `results/mss/skill_mss_mss.csv`: `lstm` **−1.5564 ± 0.5995** (per-seed −1.008,
−2.196, −1.465), `tcn` −0.9033 ± 0.1384, `transformer` −0.1773 ± 0.0295. CLAUDE.md
non-negotiable 5.
*Fix:* print the ± that `docs/mss_crossvalidation.md:104` already carries.

**S6 — The heave / 3 s table drops the worst learned model and the `nrmse` column.**
`README.md:73-81` lists 7 of 12 point models. Omitted, with `id` skill from
`results/e04/metrics_full.csv`: `dlinear` **0.9502** — the lowest of any learned model at this cell
— plus `ar10` 0.9956, `ar_attitude_only` 0.9947, `damped_persistence` 0.4663, `persistence` 0.0000.
`dlinear` appears in the pitch table four lines above, so its absence here is not a family
convention. The table also drops the `nrmse` column that the README's own caveat 3 says must sit
beside every skill score.
*Fix:* add the five rows and the `nrmse` column.

**S7 — Parameter counts are reported nowhere.** The previously committed README had a `params`
column; the rewrite dropped it. From `results/e04/metrics_full.csv`: `transformer` **2 712 708**,
`lstm` 317 828, `tcn` 196 804, `dlinear`/`dlinear_ols` 60 300, `ar40` 216 900 — a 45x capacity
spread between models being compared head to head.
*Fix:* restore the column to the pitch/10 s table.

**S8 — The quiescence table reports no seed spread.** `README.md:131-140` compares models with a
single F1 each. From `results/e04/quiescence.csv` the three-seed stds run 0.001–0.049; e.g.
`lstm_quantile` on `id`/strict/interval is 0.3712 **± 0.0481** against the winning `lstm_gaussian`
0.4207 ± 0.0064. The winners still separate. CLAUDE.md non-negotiable 5.
*Fix:* add ± to the two "best" columns.

**S9 — `results/physics_validation.md` is stale against P8-D6 in the one paragraph that matters.**
Its closing "Supporting checks" section reports roll–pitch cross-correlation 0.693 from the shared
phase set and concludes: *"That structure is what a multivariate forecaster is supposed to exploit,
and it exists in the corpus by construction."* P8-D6 (`docs/protocol.md:4779`) establishes that this
cross-DOF phase structure is **wrong by 90°** — `src/dmf/sim/response.py:229` applies a real
wave-slope excitation where strip theory requires a factor of `i` — and is unfixed. P8-D6 correctly
notes that no Gate 1 *invariant* is affected, and the README discloses the defect at `:397-403`, but
a reader of the Gate 1 document alone is told the cross-DOF structure is sound.
*Fix:* one forward-referencing sentence in that paragraph pointing at P8-D6.

**S10 — `tests/test_response.py:20` claims a directional test for `gamma` that does not exist.**
The module docstring says "changing `gamma`, `Hs`, `Tp`, `beta` and `U` each move the output in the
physically correct direction". `Hs` (`:444`), `Tp` (`:481`), `beta` (`:253`) and `U` (`:413`) each
have one; `gamma` is held at 3.3 in every construction in the file. The physics is right — I
measured `S(wp)` rising 1.51 → 2.41 → 3.27 → 4.05 → 4.65 across gamma 1.0 → 7.0 at fixed `Hs`, with
`4*sqrt(m0)` flat at 3.30 — so this is a missing assertion, not a bug. §5.1 checkbox 4 is otherwise
satisfied.
*Fix:* one test asserting `S(wp)` is strictly increasing in `gamma` at fixed `Hs`, `Tp`.

---

## NOTE

**N1 — The README quotes the untrained control's failure rate for one arm only.**
`README.md:423` says "fails its literal criterion in 76 of 144 rows" — correct for `e02_deep`. From
`results/e04/controls.csv` the `e03_probabilistic` arm fails **107 of 144** and the six ablation
arms 23–53 of 36–72. Worth one clause.

**N2 — ONNX parity does not meet the plan's literal `< 1e-4`, and this is disclosed.** 5 of 16
(model, provider) rows fail the unscaled bar (`lstm`/torch:cuda 2.79e-4; `lstm_quantile` on CPU
2.04e-4, CUDA 3.46e-4, TensorRT 5.85e-4, torch:cuda 5.30e-3). P7-D3 records the change to a
scale-relative criterion, `results/gate7.md` criterion 1 names every failing row, `parity.csv`
keeps `passed_absolute` as a separate column, and the one configuration that fails even the scaled
bar (`lstm_quantile`/torch:cuda) is **refused and not benchmarked**. This is a threshold change
handled exactly as CLAUDE.md requires. Recorded here so the §5.4 checkbox is not read as a literal
pass.

**N3 — p99 latency is not stable; p50 is.** `results/latency_stability.csv`: p50 within 10 % in
52 of 52 configurations, but p99 fails in **9 of 52**, worst −43.6 % (`tcn_quantile`/torch-eager/
cuda/b32). The plan's checkbox asks only about p50, and the README reports the p99 outcome. Any
future claim resting on a p99 figure should carry this.

**N4 — Benchmark artifacts are stamped `7bd8d86-dirty`.** `results/latency.csv`,
`parity.csv` and `latency_stability.csv` all carry that `git_commit`. It is an ancestor of HEAD, so
the provenance is real, but the `-dirty` suffix means the tree was modified when they were
measured and the stamp cannot identify the source exactly.

**N5 — Smaller wording issues, each verified.** `README.md:33` calls 0.294° / 3.42° a "**median**
90 % interval width"; the source column in `results/e03/probabilistic.csv` is
`mean_interval_width_mean`, and the "0.961 and 0.984" on the same line are the roll / 3 s / `id`
cell rather than the whole partition. `README.md:169-170`'s "**median** lead is 0.49–3.02 s /
0.53–7.70 s" are means of per-cell medians (`lead_p50` averaged over cells — I reproduce
0.48–2.97 s and 0.53–7.82 s under my own best-F1 grouping), not medians of the pooled lead-time
distribution in `results/e04/quiescence_lead_times.csv.gz`. `README.md:408`'s
"−0.0445 mean skill on `unseen_seastate`" is the `tcn`-only mean over 108 rows of
`results/e04/ablations.csv`; pooled over the arm it is −0.0191.

**N6 — `results/headline_trace.npz` cannot testify to its own caption.** It stores `labels`,
`regimes` and training `seeds` but no realization key, so "SS5, beam seas, 12 kn" is checkable only
from `src/dmf/viz/traces.py:56,198` — code provenance, not artifact provenance. That module also
correctly draws from the `test` partition with train-fitted statistics.

---

## Protocol §5, checkbox by checkbox

### 5.1 Physics — clean

| checkbox | verdict | evidence |
|---|---|---|
| Eight Gate 1 invariants pass from a clean seed | PASS | `results/physics_validation.md`; `make test` 1129 passed, exit 0 |
| Autocorrelation shows no synthesis periodicity | PASS | `r(T_period)` = −0.051 / +0.005 / −0.014 jittered against **+0.9923** un-jittered; the test asserts **both** directions, so the spike is demonstrably detectable and jitter demonstrably removes it |
| Roll/pitch/heave magnitudes plausible, numbers written down | PASS | SS5 beam-seas roll RMS 3.554° (≈7.1° significant single amplitude) for a 124 m unstabilised frigate, `gain = 1.0`, no calibration constant; per-sea-state table in `docs/corpus_card.md:115` |
| `gamma`, `Hs`, `Tp`, `beta`, `U` each move the output correctly | PASS with a gap | `Hs`, `Tp`, `beta`, `U` each have a test; `gamma` does not — **S10**. Measured correct here |

Carried forward, not a §5.1 failure: the P8-D6 cross-DOF quadrature defect (**S9**), which the
README discloses and the Gate 1 document does not.

### 5.2 Leakage — clean

| checkbox | verdict | evidence |
|---|---|---|
| Seed-disjointness asserted for all four regimes | PASS | `tests/test_splits.py` parametrizes `assert_seed_disjoint` over all four; independently re-derived here — key overlap **0** in every regime, and the held-out axis value disjoint in all three OOD regimes |
| Normalization statistics from train split only | PASS | `src/dmf/data/dataset.py:190` **refuses** to construct a val or test partition without train-fitted stats; `normalize.py:124,169` raise unless `fitted_on` ends `/train` |
| No global shuffle before splitting | PASS | `splits.py:190-263` partitions a file list by seed ordinal; no RNG, no shuffle |
| Shuffle-label control | PASS | Control 1 above: worst excess 0.0038 vs 0.02 tolerance, 0/36 cells |
| Untrained-model control | FAILS LITERALLY, correctly reported | Control 2 above |

The one non-obvious hazard — whether `unseen_vessel`'s S175 test records reuse a frigate training
record's wave field, since seed ordinals do repeat across cells (40, 40 and 8 shared ordinals) — does
not occur: `src/dmf/sim/generate.py:128-155` keys the generator entropy on a blake2b digest of the
full `<vessel>|<ss>|<heading>|<speed>|<seed>` tuple, so no two grid cells share a realization.

### 5.3 Metrics — clean

| checkbox | verdict | evidence |
|---|---|---|
| Skill score reproduced by hand | PASS | `id`/pitch/10 s, `lstm`, three seeds: `1 − (rmse/rmse_persistence)²` reproduces the `skill` column to **0.00e+00**; `nrmse = rmse/signal_std` likewise |
| Persistence RMSE computed two independent ways | PASS | 1.5591545467 (NumPy on raw Parquet) vs 1.55915455 (`metrics_full.csv`), 8.99e-11 relative; and 288 cells at ≤2.9e-8 |
| Quiescence ground truth spot-checked on 10 random windows | PASS | 10 random `id/test` realizations × 2 threshold sets = 20 masks, each recomputed from the written definition with a naive Python loop: **0 disagreements**. Geometry sane — SS3 head seas permissive base rate 1.000, SS6 strict 0.000 |
| Base rates reported alongside F1 | PASS | `results/gate6.csv` criterion 6, 20 of 20 tables |
| Every table reports mean ± std over 3 seeds | PARTIAL | true of `results/`; three README tables print bare numbers — **S5**, **S8** |

### 5.4 Deployment

| checkbox | verdict | evidence |
|---|---|---|
| ONNX parity `< 1e-4` on 1000 windows | PASS as restated, fails literally | **N2** |
| Warmup present; CUDA synchronization correct | PASS | 200 warmup / 2000 timed on every row; `src/dmf/deploy/harness.py:160-175` hard-codes `synchronize=True` at the only construction site and `bench.py:371` refuses a CUDA run without it |
| Latency stable across two runs (< 10 % p50 drift) | PASS, marginally | Control 4 above — p50 52 of 52, but the worst row is at 9.91 % against the 10 % bar; p99 fails 9 of 52 (**N3**) |
| Environment stamp in every benchmark artifact | PASS | `gpu`, `torch`, `onnxruntime`, `git_commit`, `tf32`, `intra_op_threads`, `providers_realized` on every row; see **N4** |

### 5.5 Reproducibility

| checkbox | verdict | evidence |
|---|---|---|
| Fresh clone + `make all` reproduces every committed number | **FAIL** | **B3** — the committed `make all` is the superseded recipe and nine referenced files are untracked. Not otherwise testable here: the full pipeline is ~160 h |
| Seeds fixed and recorded; `use_deterministic_algorithms` where feasible | PASS | `src/dmf/train/loop.py:76-100` seeds `random`, NumPy and torch, sets `CUBLAS_WORKSPACE_CONFIG`, enables `use_deterministic_algorithms(warn_only=True)` and the cuDNN deterministic path |
| No absolute paths, no machine-specific config | PASS | `git grep` over `src`, `scripts`, `configs`, `tests`, `Makefile`, `pyproject.toml` finds no absolute path; the only `A4000` mention is a measured-cost comment |
| `results/` regenerable and consistent with the README | PASS for the documents, see B1/B2/B3 for the README | re-rendered here from the committed CSVs alone: `results/results.md` and `results/latency.md` both reproduce **byte for byte**. `gate6.py` and `gate7.py` re-run to the same verdicts (the two criteria that flip in a scratch directory are the two that depend on files I did not copy) |

### 5.6 Honesty pass

| checkbox | verdict |
|---|---|
| Every README claim traceable to a committed artifact | **FAIL** — B1, B3 |
| No model omitted because it underperformed | **FAIL** — S6 (and B2, which removes the comparison rather than the model) |
| Negative results in the README body, not a footnote | **FAIL** — B2. Otherwise exemplary: the MSS collapse, the trivial interval baseline beating every learned head, `persistence` `nrmse` > 1.0, the Gate 3 failure, the `unseen_heading` collapse and the vacuous interval control are all in the body, several in bold |
| Simulation-only caveat in the first paragraph | PASS — `README.md:5-7`, bolded, repeated at `:322`, `:403`, `:460`. No sentence implies real-deck validation |

### Known traps (CLAUDE.md) — all six covered

Wave-synthesis periodicity, with a converse test (`tests/test_spectra.py:297,323`); TCN receptive
field asserted arithmetically, not in a comment (`tests/test_models.py:3776-3807`); CUDA
synchronization forced rather than configurable (`harness.py:172`, `bench.py:371`); following-seas
non-monotonic encounter cells kept, labelled and tested rather than silently excluded
(`tests/test_response.py:198`, `tests/test_generate.py:384`); quantile crossing measured on the raw
fan then sorted in the constructor (`prob_runner.py:323-329`, `heads.py:285`) — the nonzero
`crossing_rate_mean` in the CSVs is the pre-sort diagnostic, not a defect; base rates beside every
F1 (`gate6.csv` criterion 6).

### Retracted and corrected claims — clean

Every superseded string from P6-D17/D19/D20/D22/D23, P7-D9/D12/D13 and P8-D11/D12 was grepped for
across `README.md`, `docs/findings.md` and `results/*.md`: "keeps positive skill through 5 s", the
18 %/82 % decomposition, the 1.13 / +3.10 roll row, "denominator roughly halved", the "≤ 0.006" sign
bound and "the CPU is faster". **None survives outside its own retraction note.** P8-D12's restored
AR baselines are present in both README tables.

### Toolchain

`make test` — 1129 passed, exit 0, 623 s. `make lint` — `ruff check`, `ruff format --check` and
`mypy` on 75 source files, all clean, exit 0.

---

## Verdict

**NOT RELEASE READY — 3 blocking findings** (B1, B2, B3).

None of the three is a defect in the science. The integrity controls pass on the best model, the
splits are sound, the metrics reproduce by hand, the pipeline is exact to 1e-10, the rendered
documents are byte-for-byte functions of their CSVs, and the project's own record of what it got
wrong is unusually complete. All three blocking findings are in the **Phase 9 rewrite of the
README and the state of the git index**: a comparison column swapped for a non-comparable one, an
unfavourable table dropped in the rewrite, and nine files — including the licensing fix and the
document that holds the dropped table — never added to the repository. B3 is one `git add` away;
B1 and B2 are each a table edit. Re-run this audit's §5.6 and §5.5 sections after those and the
verdict flips.

---

# Resolution — 2026-09-14

Appended after the audit, by the session that made the changes. **The audit's findings and its
verdict above are left exactly as written**; this section records what was done about them. Every
number quoted below was recomputed from the committed CSVs before being published, including the
ones that arrived from the audit itself.

## The audit did not finish on its own

**The agent terminated early on a usage limit, mid-Control-4**, with the body of the report written
and one section missing. Two places — the Control 4 stub and the §5.4 stability checkbox — carried
forward references to a section that did not exist, and the checkbox recorded a **PASS citing it**.
A verdict resting on a pointer to nothing is the defect P8-D15 recorded in Gate 8 predicate 1 and
P7-D14 recorded in the Phase 7 documentation checker; it is not acceptable in the document that
clears this repository for release.

Control 4 has been completed and written up, against the committed two-run comparison. It changes
the checkbox: p50 passes **52 of 52**, but its worst row sits at **9.91 % against a 10 % bar**, so
the entry now reads "PASS, marginally" rather than a bare PASS. The p99 result (9 of 52 failing,
worst −43.6 %) was already disclosed in N3, the README and P7-D13, and is unchanged. Every other
section of the report, N6 included, was written in full before the agent stopped — checked by
resolving every B/S/N cross-reference in the document.

## Blocking — all three fixed

**B1 — fixed.** The MSS column has been removed from the four-regime accuracy table and the transfer
result moved into its own section, scored against the **matched-cell corpus column** that
`docs/mss_crossvalidation.md` already carried, with the seed spreads it already carried (which also
resolves S5). The false clause "because only the generator changes between the two" is gone, and
every delta in the surrounding prose now reproduces by subtracting the two printed columns — 1.266
and 0.370, where the old table would have given 1.25 and 0.32.

**B2 — fixed.** The 1–5 s versus 10–15 s band table is back in the README body with its caveat
sentence. **The restored numbers are the previously committed ones, not the audit's.** The audit
reported 1–5 s as `tcn` 34-35, `transformer` 20-49, `lstm` 25-43; recomputing from
`results/e02/paired_contrasts.csv` under this project's documented aggregation — a multi-seed
interval is the **envelope** of the per-run intervals (P3-D22) — gives **34-34, 19-47 and 23-42**,
matching `git show HEAD:README.md`. The finding was correct; its evidence was not.

**B3 — fixed, with one part that went further than the finding.** All new files are staged, and the
runtime table has been rebuilt. The audit said the table was unverifiable from a clone; it was also
**wrong**, because "the seven logged stages sum to 163 hours" attributed to logs a figure of which
only **66.5 h** is logged. `scripts/collect_runtimes.py` now derives
`results/runtime_stages.csv` from the stage logs, and the README's table carries a per-row `source`
column marking each figure as logged, protocol prose, or derived — with the four traceable rows in
bold and `make mss` marked "not recorded" rather than estimated.

## Should fix — all ten actioned

| | action |
|---|---|
| S1 | The README now states at the point it leans on the control that the subject is **AR(20)**, not the best model. |
| S2 | Bound corrected from "excess 0.001" to **0.0095 over 144 rows**, with the residual-floor exemption's post-hoc origin (after a 5.52 % failure) stated. |
| S3 | Self-contradiction removed: **only the TCN family scores any cell in band on `unseen_seastate`, 5 of 72**; DLinear, LSTM and the floor score zero. |
| S4 | "about 2x hot" → **1.0–2.3x depending on channel and heading**, with the 0.61–1.54x per-speed spread the source document requires. |
| S5 | MSS column now carries ±, including `lstm` **−1.5564 ± 0.5995**. |
| S6 | Heave table restored to all **12** models and the `nrmse` column, including `dlinear` at 0.9502 — the lowest learned model at that cell. |
| S7 | `params` column restored; the 45x spread between `transformer` and `dlinear` is now visible. |
| S8 | Quiescence table carries ± across training seeds, closed-form rows marked `det.` |
| S9 | `results/physics_validation.md` now forward-references P8-D6 in the cross-DOF paragraph. |
| S10 | `test_peak_enhancement_concentrates_energy_without_changing_hs` added: `S(wp)` rises 1.51 → 4.65 across gamma 1.0 → 7.0 while realized `Hs` stays bounded near nominal. |

## Notes — N1, N5 and N6 actioned; N2, N3, N4 already disclosed

- **N1** — the untrained control's failure rate is now given for the reference arm **and** the range
  across the other seven (23 to 107 rows), rather than for one arm as if it were the whole.
- **N5** — "median interval width" corrected to **mean** (the column is
  `mean_interval_width_mean`); the PICP pair is now attributed to its cell rather than "the whole
  test partition"; the lead-time ranges are labelled as **means of per-cell medians**; and
  "−0.0445 mean skill" is now given as **−0.0191 pooled over the arm, −0.0445 on `tcn` alone**.
- **N6** — `results/headline_trace.npz` now stores the grid cell, channel and lead time, and
  `scripts/make_figures.py` builds the caption from the artifact instead of from module constants.
  A trace re-extracted at a different cell retitles the figure rather than inheriting a caption that
  is no longer true. The rendered PNG is byte-identical, so this is provenance, not a new figure.
- **N2, N3, N4** — no action. Each is already disclosed in the README, `results/gate7.md` or the
  protocol, which the audit states.

## One correction to this session's own work, not the audit's

S8's spread was first recomputed here as 0.06–0.18 by taking a standard deviation across cells *and*
seeds together. The right quantity for a "mean F1 over scorable cells" row is the spread of that
mean across the three training seeds — average over cells within a seed, then take the standard
deviation over seeds — which gives **0.0008–0.0255** at the published cells and matches the audit's
0.001–0.049 across all rows. The same error was made and caught again on the MSS ± values before
publication. Recorded in `docs/protocol.md` P9-D7; it is the third instance in this project of a
statistic computed over the wrong grouping.

**Re-verification after the fixes:** `make lint` clean on 75 source files; `make test`
**1135 passed, 0 failed** (1129 before this phase, plus six new: five on the figure and trace code,
one the `gamma` directional test of S10); `make gate7` PASS,
including its mechanical check of the README's latency section; Gates 4 and 5 PASS; Gate 6 unchanged
at its committed 7/7. The three blocking findings are resolved. **Publication remains the
repository owner's decision, and the commit itself has deliberately been left to them.**

---

# Re-verification — 2026-09-15, read against commit `6aa98e5`

Appended by the session that closed Gate 9. **The audit's body, its checkbox tables and its verdict
are left exactly as written**, per the Resolution section's own rule; this section re-runs the two
checkbox tables that the audit's exit condition at the end of its verdict asked to be re-run
("Re-run this audit's §5.6 and §5.5 sections after those and the verdict flips"). That re-run had
never happened, so §5.5, §5.6 and the verdict above still described the repository as it stood
before the fixes.

## §5.5 Reproducibility — re-run

| checkbox | audit verdict | now | evidence, measured 2026-09-15 |
|---|---|---|---|
| Fresh clone + `make all` reproduces every committed number | **FAIL** (B3) | **PASS as far as it is checkable without a 160 h run** | `Makefile` carries `all: data sweeps gate4 gate5 gate6-full bench report gate7 mss figures-extract`, the corrected recipe, and it is committed. Every file the README asserts is in the repository is tracked, and `git ls-files --others --exclude-standard results/` is empty. **The full pipeline was not re-run for this verification**; what changed is that a clone now contains the recipe and the files, which is what B3 said it did not. |
| Seeds fixed and recorded | PASS | PASS | unchanged |
| No absolute paths, no machine-specific config | PASS | PASS | unchanged |
| `results/` regenerable and consistent with the README | PASS for the documents, see B1/B2/B3 for the README | **PASS** | B1 and B2 are fixed (below). `make figures` and `make report` were re-run here and both reproduce their outputs **byte-identically** from committed artifacts with no corpus, no checkpoints and no GPU. |

## §5.6 Honesty pass — re-run

| checkbox | audit verdict | now | evidence |
|---|---|---|---|
| Every README claim traceable to a committed artifact | **FAIL** (B1, B3) | **PASS** | B1 fixed: the MSS result is in its own section against the matched-cell column, and the false "only the generator changes" clause is gone. B3 fixed: committed in `6aa98e5`. An independent adversarial pass on 2026-09-15 re-derived every numeric claim in the README from the committed CSVs; **five sentences and row sets did not reproduce and all five are corrected** — recorded as P9-D9 in `docs/protocol.md`. |
| No model omitted because it underperformed | **FAIL** (S6, B2) | **PASS** | S6 fixed (all 12 models and the `nrmse` column restored). Gate 9 found the same defect once more in the MSS transfer table — `ar10`, `damped_persistence`, `window_mean` and `persistence` were missing — and all four are now in the README, `docs/findings.md` and `docs/mss_crossvalidation.md`. Adding `ar10` **falsified a published conclusion**; see P9-D9 item 3. |
| Negative results in the README body, not a footnote | **FAIL** (B2) | **PASS** | B2 fixed: the 1–5 s versus 10–15 s band table is back in the body with its caveat sentence, carrying the committed envelope numbers rather than the audit's. |
| Simulation-only caveat in the first paragraph | PASS | PASS | See the citation note below. |

**Citation note, and it is the audit's own defect class.** §5.6's last row cites the simulation-only
caveat at `README.md:322`, `:403` and `:460`. The README has been restructured since, and those
lines now hold a latency sentence, a runtime-provenance sentence and a blank line. The caveat
currently appears at **`:6-7`, `:172`, `:385`, `:481` and `:543`**. The original citation is left in
place above rather than silently repaired, because it is an instance of exactly what P9-D8 recorded —
a reference that was correct when written and stopped being correct when the thing it pointed at
moved — and the third such instance this project has found. Line numbers are the weakest form of
citation available and this document now says so.

## Verdict — superseding the one above

**RELEASE READY**, read against commit `6aa98e5` on 2026-09-15.

All three blocking findings are fixed, all ten SHOULD FIX items are actioned, and the two integrity
controls the audit flagged (the untrained control's literal failure, the p99 latency drift) remain
**disclosed rather than repaired**, which is the disposition the audit itself recommended.

**What this verdict does not say.** It does not say every committed number was re-derived by a fresh
`make all` — that is ~160 h and was not run. It says that what the repository publishes is traceable
to what the repository commits, which is the question this audit asked.

**Superseded the same day, and left visible rather than silently rewritten.** As first written, the
paragraph above continued: *"It does not say the project's deliverables are complete: the README
states in three places that no conformal calibration is run anywhere, so the 'calibrated prediction
intervals' deliverable in `CLAUDE.md` is **not met**."* That was true when it was committed at
**17:18** on 2026-09-15 and false by **21:04**, when the calibration run finished. The deliverable
is now **met in distribution and not met under shift**: split conformal puts 216 of 216 `id` cells
inside Gate 5's band at a median PICP of 0.9001, and makes coverage *worse* in 78–87 percent of
cells in all three shifted regimes, because the calibration split is in-distribution by construction
in every regime (P10-D2). The correction is
recorded here rather than applied invisibly, because a stale sentence inside a release-readiness
verdict is the exact defect P9-D10 recorded two paragraphs of this document ago, and the fifth
instance of it in this project.
