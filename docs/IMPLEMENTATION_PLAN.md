# `deck-motion-forecast` — Implementation Plan

**Short-Horizon 6-DOF Deck-Motion Forecasting from JONSWAP-Driven Vessel Simulation**

Source: Project 4 of *Novel, Single-GPU Engineering Projects for Vision-Based GNC / Autonomous Ship-Deck Landing*.
Target hardware: 1× RTX A4000 16 GB. Target duration: 6 working days core, 9 with the probabilistic + publication extensions.

---

## 0. Three design decisions that differ from the source document

Read this section before anything else — it changes the shape of the build.

### 0.1 MSS becomes a *validation* dependency, not the critical path

The source doc puts Fossen's MSS (MATLAB/Octave) on the critical path for data generation. That is a schedule risk: MSS is MATLAB-first, Octave compatibility across its toolbox is uneven, and debugging Octave from inside a Python project on Day 1 is the worst possible place to lose a day.

**Instead:** implement the wave-and-response model natively in Python (NumPy) as the primary generator, and use MSS as an *independent cross-check* in Phase 8. The physics involved — JONSWAP spectrum, random-phase wave synthesis, encounter frequency, second-order DOF response — is ~250 lines of NumPy and is fully verifiable against closed-form spectral invariants. This gives you:

- No Octave dependency for anyone reproducing the repo (a real portability win for a public GitHub project).
- Analytic ground truth for unit tests (spectral moments, Rayleigh peak statistics).
- MSS agreement in Phase 8 as *corroborating evidence* rather than a blocker. If MSS installs cleanly, you get a stronger writeup. If it doesn't, you lose nothing.

### 0.2 The headline metric is not RMSE

Multi-step RMSE on a narrowband quasi-periodic signal is easy to make look good and easy to make meaningless. A persistence baseline on 0.5 s horizons will already look excellent. The metrics that carry the project:

1. **Skill score vs. persistence** — `1 − MSE_model / MSE_persistence`, per horizon, per DOF. If this is not clearly positive at 2–3 s, the model is not doing anything.
2. **Quiescent-window detection** — the actual GNC question. Can the model predict, 2–3 s ahead, a window in which `|roll|`, `|pitch|`, and `|heave rate|` all stay below landing limits for ≥ 2 s? Scored as precision / recall / F1 / lead time / false-alarm rate.
3. **Calibrated intervals** — pinball loss, CRPS, PICP@90, mean interval width.

Quiescent-window detection is the differentiator. It is what a landing controller actually consumes, it is almost absent from the deck-motion-forecasting literature (which is dominated by RMSE tables), and it converts "I trained a TCN" into "I built the module that decides when to commit to touchdown."

### 0.3 Include a linear model you might lose to

Ship `DLinear` (a single linear layer on a decomposed, de-meaned window) alongside the TCN and Transformer. On long-horizon time-series forecasting, linear models are notoriously competitive with transformers. If DLinear wins, that is a **result**, not a failure, and it is the honest, senior-engineer version of this project. Reporting it costs you nothing and buys you enormous credibility in an interview. Reporting only the TCN because it happened to win is how portfolios lose trust.

---

## 1. Deliverables and definition of done

The project is done when all of the following exist in a public repo:

| # | Deliverable | Acceptance criterion |
|---|---|---|
| D1 | Reproducible data generator | `make data` produces an identical corpus from a seed on a clean machine; physics tests pass |
| D2 | Corpus card | Documented sea states, headings, speeds, vessel params, sample rate, seeds, split policy |
| D3 | 6 models | persistence, AR(p), DLinear, LSTM, TCN, Transformer — one shared interface |
| D4 | Evaluation report | Per-DOF × per-horizon RMSE/MAE + skill score vs persistence, across 4 test regimes |
| D5 | Quiescence report | P/R/F1, lead-time distribution, false-alarm rate at 2 operational thresholds |
| D6 | Probabilistic report | Pinball / CRPS / PICP@90 / MIW for the quantile heads |
| D7 | ONNX artifacts + parity | `max_abs_err < 1e-4` FP32 vs PyTorch on 1000 random windows |
| D8 | Latency benchmark | p50/p90/p99 + throughput + memory, CPU EP vs CUDA EP, batch 1 and 32, with warmup |
| D9 | README | Reproduces every number in the tables; states simulation-only caveat prominently |
| D10 | MSS cross-check (optional) | Spectral agreement plot, or a documented note on why it was skipped |

**Explicit non-goals.** No real deck data collection. No closed-loop control (that is Project 8). No sim-to-real claims — this is a simulation study and the README must say so in the first paragraph.

---

## 2. Repository layout

```
deck-motion-forecast/
├── CLAUDE.md                    # Claude Code project context (see kit)
├── README.md
├── pyproject.toml
├── Makefile                     # data / train / eval / bench / all
├── .claude/                     # agents + skills (see kit)
├── configs/
│   ├── sim/
│   │   ├── sea_states.yaml       # SS3-SS6: Hs, Tp, gamma
│   │   ├── headings.yaml         # 0/45/90/135/180 deg
│   │   └── vessels/
│   │       ├── frigate.yaml      # L=120m, roll Tn=12s, zeta=0.08 ...
│   │       └── s175.yaml
│   ├── data/default.yaml         # fs, lookback, horizon, dofs, splits
│   ├── model/{persistence,ar,dlinear,lstm,tcn,transformer}.yaml
│   └── experiment/
│       ├── e01_baselines.yaml
│       ├── e02_deep.yaml
│       ├── e03_quantile.yaml
│       └── e04_observability.yaml
├── src/dmf/
│   ├── sim/
│   │   ├── spectra.py            # JONSWAP, spectral moments, Hs/Tz recovery
│   │   ├── response.py           # RAO / 2nd-order DOF transfer functions
│   │   ├── encounter.py          # encounter frequency, heading transforms
│   │   ├── vessel.py             # vessel parameter dataclass
│   │   ├── imu.py                # IMU observability model (noise, bias, HP filter)
│   │   └── generate.py           # corpus driver -> parquet
│   ├── data/
│   │   ├── windows.py            # lookback/horizon windowing
│   │   ├── splits.py             # realization-level split logic (LEAKAGE GUARD)
│   │   ├── normalize.py          # per-window de-mean / RevIN
│   │   └── dataset.py            # torch Dataset + DataLoader factory
│   ├── models/
│   │   ├── base.py               # ForecastModel protocol: (B,L,C) -> (B,H,C[,Q])
│   │   ├── persistence.py
│   │   ├── ar.py                 # least-squares AR(p), direct multi-horizon
│   │   ├── dlinear.py
│   │   ├── lstm.py
│   │   ├── tcn.py                # dilated causal conv, WaveNet-style
│   │   ├── transformer.py        # encoder-only + patch embedding
│   │   └── heads.py              # point head, quantile head, gaussian head
│   ├── train/
│   │   ├── loop.py               # AMP, cosine schedule, early stop, seed control
│   │   ├── losses.py             # MSE, MAE, pinball, gaussian NLL
│   │   └── registry.py
│   ├── eval/
│   │   ├── metrics.py            # RMSE/MAE per (dof, horizon), skill score
│   │   ├── quiescence.py         # window detection + P/R/F1 + lead time
│   │   ├── probabilistic.py      # pinball, CRPS, PICP, MIW, Winkler
│   │   ├── phase.py              # peak-timing / cross-correlation lag
│   │   └── report.py             # -> results/*.csv + markdown tables
│   ├── deploy/
│   │   ├── export_onnx.py
│   │   ├── parity.py
│   │   └── bench.py              # warmup + percentile harness
│   └── viz/
│       ├── forecast_plots.py
│       ├── spectra_plots.py
│       └── pareto.py
├── scripts/                      # thin CLI entry points only
├── tests/
│   ├── test_spectra.py           # PHYSICS INVARIANTS
│   ├── test_response.py
│   ├── test_splits.py            # LEAKAGE
│   ├── test_windows.py
│   ├── test_metrics.py
│   └── test_onnx_parity.py
├── mss/                          # Phase 8 Octave scripts (optional)
├── artifacts/                    # gitignored: data, checkpoints, onnx
├── results/                      # committed: csv tables + figures
└── docs/
    ├── corpus_card.md
    ├── protocol.md               # split + metric definitions
    └── results.md
```

**Rule:** everything in `src/dmf/` is importable and testable; `scripts/` contains only argparse wrappers. This matters because Claude Code will otherwise drift toward monolithic scripts that cannot be unit-tested.

---

## 3. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pandas pyarrow pyyaml matplotlib \
            onnx onnxruntime-gpu netron \
            pytest pytest-cov ruff mypy rich tqdm
# optional
pip install tensorboard properscoring
```

Pin everything in `pyproject.toml`. Record `torch.__version__`, `onnxruntime.__version__`, driver and CUDA version in every benchmark JSON — benchmark numbers without an environment stamp are not reproducible and a reviewer will notice.

---

## 4. Phase-by-phase build

Each phase ends with a **gate**. Do not start the next phase until the gate passes. This is the single most important structural property of the plan when working with an agentic coder: it prevents silent compounding of a bad assumption.

---

### Phase 0 — Bootstrap (0.5 day)

**Tasks**
1. `git init`, MIT or Apache-2.0 license, `.gitignore` (`artifacts/`, `*.parquet`, `*.onnx`, `.venv/`).
2. `pyproject.toml` with `[project]` + `[tool.ruff]` + `[tool.pytest.ini_options]`.
3. Package skeleton with every module stubbed and typed signatures only — no bodies.
4. `Makefile` targets: `data`, `train`, `eval`, `bench`, `test`, `lint`, `all`.
5. Drop in `CLAUDE.md`, `.claude/agents/`, `.claude/skills/` from the kit.
6. First commit.

**Gate 0:** `make lint && make test` runs green on an empty test suite; `import dmf` succeeds.

---

### Phase 1 — Wave and vessel-response simulation (1.5 days)

This is the phase where correctness matters most and where an agentic coder is most likely to produce plausible-looking wrong physics. Write the tests first.

#### 1.1 JONSWAP spectrum (`sim/spectra.py`)

Use the Hs-scaled DNV form:

```
S(w) = (5/16) * Hs^2 * wp^4 * w^-5 * exp(-1.25 * (wp/w)^4) * (1 - 0.287*ln(gamma)) * gamma^r

r     = exp( -(w - wp)^2 / (2 * sigma^2 * wp^2) )
sigma = 0.07 for w <= wp,  0.09 for w > wp
wp    = 2*pi/Tp
```

Spectral moments: `m_n = integral(w^n * S(w) dw)`.

#### 1.2 Wave elevation synthesis

```
eta(t) = sum_i  sqrt(2 * S(w_i) * dw_i) * cos(w_i * t + phi_i),   phi_i ~ U(0, 2pi)
```

Use **200–400 components** over `w in [0.2, 2.5] rad/s`, and **jitter each `w_i` within its bin** (`w_i = w_lo + u_i * dw`, `u_i ~ U(0,1)`) rather than using a uniform grid. Uniform-grid synthesis produces a signal that repeats with period `2*pi/dw` — if that period is shorter than your record you have manufactured periodicity, the forecaster will memorize it, and every result is invalid. This is the single most likely silent bug in the whole project.

#### 1.3 Encounter frequency (`sim/encounter.py`)

```
w_e = w - (w^2 * U / g) * cos(beta)
```
with `beta` the encounter angle (180° = head seas, 90° = beam, 0° = following). Guard the following-seas case where `dw_e/dw` changes sign — either restrict `U` and `beta` so it does not occur in the corpus, or document it explicitly. Do not let it pass silently.

#### 1.4 DOF response (`sim/response.py`)

Model each DOF as a linear second-order system driven by wave elevation (heave) or wave slope (roll, pitch):

```
H_dof(w_e) = K_dof * F_exc(w) / (1 - (w_e/wn)^2 + 2*j*zeta*(w_e/wn))
```

Excitation and coupling:

| DOF | Driven by | Heading dependence | Typical `Tn` | Typical `zeta` |
|---|---|---|---|---|
| Heave | wave elevation × Smith factor `exp(-k*T_draft)` | weak | 7–10 s | 0.25–0.45 |
| Pitch | wave slope `k*a` | `|cos(beta)|` | 6–9 s | 0.30–0.50 |
| Roll | wave slope `k*a` | `|sin(beta)|` | 10–16 s | **0.05–0.12** |

The lightly damped roll mode is the whole reason this problem is interesting: it is narrowband, resonant, and near-predictable over a few periods, whereas heave is broadband and much harder. Expect and report per-DOF skill scores that differ sharply.

Also apply a wavelength-vs-ship-length attenuation (`exp(-(k*L/(4*pi))^2)` or similar rolloff) so that short waves do not drive full-amplitude response. Document whatever form you use.

Synthesize each DOF by superposition in the frequency domain with the *same phase set* `phi_i` as the elevation — this preserves the physical phase relationships between roll, pitch, and heave, which is exactly what a multivariate forecaster should be able to exploit.

#### 1.5 Corpus generation (`sim/generate.py`)

Grid:

| Axis | Values |
|---|---|
| Sea state | SS3 (Hs 1.0 m, Tp 7.5 s), SS4 (1.9, 8.8), SS5 (3.3, 9.7), SS6 (5.0, 12.4) |
| Heading | 180, 135, 90, 45 deg |
| Speed | 0, 6, 12 kn |
| Vessel | frigate (primary), S175 (held out) |
| Seeds | 40 realizations per (SS, heading, speed) cell |

Each realization: 600 s at `fs = 10 Hz` after discarding a 120 s spin-up transient. Store as Parquet with columns `t, roll, pitch, heave, roll_rate, pitch_rate, heave_rate, heave_acc` plus metadata columns `seed, ss, heading, speed, vessel`.

Corpus size sanity check: 4 × 4 × 3 × 40 = 1920 realizations × 6000 samples ≈ 11.5 M rows. As float32 Parquet that is roughly 500 MB — comfortable. Generation is embarrassingly parallel; use `multiprocessing`.

#### 1.6 IMU observability model (`sim/imu.py`)

Real deck sensors give angular rates and linear accelerations, not absolute heave. Provide two observation modes:

- `ideal` — clean `roll, pitch, heave`.
- `imu` — attitude with white noise (0.02 deg) + slow bias random walk; heave reconstructed by double-integrating vertical acceleration through a 2nd-order high-pass at 0.03 Hz (this is what real heave-compensation systems do, and it distorts the low-frequency content).

Phase 6 runs both as an ablation. This one addition is what separates a toy study from something a GNC engineer takes seriously.

**Gate 1 — physics validation (must all pass in `tests/test_spectra.py`, `tests/test_response.py`):**

1. `4*sqrt(m0)` recovers the requested `Hs` to within 2%.
2. Zero-crossing period `Tz = 2*pi*sqrt(m0/m2)` falls in the expected band for the requested `Tp` (`Tz/Tp ≈ 0.71–0.78` for `gamma=3.3`).
3. Welch PSD of a synthesized 3600 s elevation record matches the analytic `S(w)` within 15% across `[0.4, 1.5] rad/s`.
4. Peak amplitudes of the elevation record follow a Rayleigh distribution (KS test, `p > 0.01`).
5. Autocorrelation of a 1 hr record shows **no** spurious spike at `2*pi/dw` — the anti-periodicity check.
6. In beam seas, the roll response spectrum peaks within 5% of `wn_roll`; in head seas, roll RMS drops by at least an order of magnitude.
7. Roll RMS at SS5 beam seas lands in a physically sensible range (single-digit degrees). Print it and eyeball it against published seakeeping figures.
8. Increasing forward speed in head seas shifts the response spectrum peak to higher `w_e`.

Do not proceed until every one of these passes. A checked-in `results/physics_validation.md` with the plots is a genuine portfolio asset in its own right.

---

### Phase 2 — Windowing, splits, normalization (0.5 day)

#### 2.1 Task definition (`configs/data/default.yaml`)

```yaml
fs: 10
lookback: 200        # 20 s
horizons: [10, 20, 30, 50]   # 1, 2, 3, 5 s
target_dofs: [roll, pitch, heave]
input_channels: [roll, pitch, heave, roll_rate, pitch_rate, heave_rate]
stride: 5            # window stride in samples
observation_mode: ideal   # or imu
```

**Direct multi-horizon output**, shape `(B, H_max, C)` — not autoregressive rollout. Direct avoids error compounding, trains faster, and gives a single fixed-shape ONNX graph. Slice shorter horizons out of the same output tensor.

#### 2.2 Split policy (`data/splits.py`) — the leakage guard

Splits are **by realization seed**, never by time window. Two windows from the same realization that overlap in time share information; putting one in train and one in test inflates every metric and is the classic failure mode of time-series portfolio projects.

Four evaluation regimes, all built here:

| Regime | Train | Test | Question answered |
|---|---|---|---|
| `id` | seeds 0–31 of every cell | seeds 32–39 of every cell | In-distribution accuracy |
| `unseen_seastate` | SS3, SS4, SS5 | SS6 | Extrapolation to rougher seas |
| `unseen_heading` | 180, 135, 45 | 90 (beam) | Generalization to the worst-case roll condition |
| `unseen_vessel` | frigate | S175 | Transfer across hull dynamics |

Additionally, insert a **guard band** of `lookback + max_horizon` samples between any two segments that come from the same realization if you ever split within one — but the primary design avoids this entirely by splitting at realization granularity.

#### 2.3 Normalization (`data/normalize.py`)

Per-window de-meaning (subtract the lookback-window mean, add it back to predictions) plus a **global per-channel scale computed on the training split only**. Do not compute scale statistics over the whole corpus — that is a subtle leak. Optionally implement RevIN as a config flag and ablate it.

**Gate 2 (`tests/test_splits.py`, `tests/test_windows.py`):**
1. Zero seed overlap between any train and test split, asserted programmatically for all four regimes.
2. No time index appears in both a train window and a test window for the same realization.
3. A reconstructed window from the dataset exactly matches the raw Parquet slice.
4. Normalization statistics computed on train only; a test-only fixture raises if train stats are absent.
5. Persistence baseline evaluated through the full dataset pipeline reproduces the same RMSE as computed directly on raw arrays (end-to-end pipeline sanity).

---

### Phase 3 — Baselines (0.5 day)

Implement, in this order, against the shared `ForecastModel` interface:

1. **Persistence** — repeat the last observed value across the horizon. Zero parameters.
2. **Damped persistence** — decay toward the window mean with a fitted time constant. Surprisingly strong on oscillatory signals; include it so nobody can accuse you of a straw-man baseline.
3. **AR(p)** — least-squares multivariate AR, `p ∈ {10, 20, 40}`, fit per training split, direct multi-horizon (one coefficient matrix per horizon step). Fit on CPU with NumPy; no training loop needed.
4. **DLinear** — series decomposition (moving-average trend + remainder) followed by one linear layer per component, mapping `(L,) → (H,)` per channel.

**Gate 3:** a `results/baselines.csv` with skill-score-vs-persistence for every baseline, every horizon, every DOF, every regime. Read it. If AR(p) already achieves 0.8 skill at 3 s on roll, your task is too easy — lengthen the horizon or add the `imu` observation mode before spending a day on deep models.

This gate is where you decide whether the rest of the project is worth building as specified. Do not skip it.

---

### Phase 4 — Deep models (1 day)

> **Before you start — carried forward from Gate 3. Read `docs/protocol.md` §Phase 3 first.**
>
> Phase 3 measured things that change this section. The text below is the original plan and has
> **not** been rewritten; these are the deltas.
>
> **1. Do quiescence detection before the deep models.** Implement `src/dmf/eval/quiescence.py`
> (Phase 6.2, seven stubbed functions) and validate it against the AR baseline *first*. AR(20)
> already exceeds 0.8 skill in 28 of 36 `id` cells and 89 of 96 across the 1–5 s band, so selecting
> architectures on `id` RMSE tunes against a metric that cannot separate them. The discrimination is
> in the OOD regimes and in the landing decision at SS5/SS6 (base rates 0.27 and 0.058).
>
> **2. Gate 4 as written is a weak bar in a saturated cell.** It asks that deep models beat
> `damped_persistence` at 3 s on `id`. That cell is saturated (AR(20) 0.9987), and
> `damped_persistence` loses to the zero-parameter `window_mean` in 54 of 144 cells (P3-D20). If
> Gate 4 is kept as-is it will pass on models that have learned nothing useful. Restate it at the
> decision horizon against the strongest trivial baseline, and record the change per CLAUDE.md
> §Gates.
>
> **3. The stated training defaults are wrong for this corpus.** "batch 256 … expect minutes per
> run" — measured 22 min/seed for a 60 300-parameter linear map. Committed configs use batch 1024,
> lr 2e-3, `num_workers` 16 (P3-D17). More important: the 60-epoch cap bound in **22 of 24** DLinear
> runs, i.e. early stopping almost never fired and no run converged (P3-D19, P3-D23). Check
> `epochs_run` on every deep model; do not inherit the budget, and do not compare a converged deep
> model against the under-converged `dlinear` row.
>
> **4. Two prerequisites before any model-vs-model claim.** Neither exists yet, and the absence of
> the first has already produced a wrong published conclusion (P3-D13):
> - wire `dmf.eval.runner.paired_skill_difference_ci` — implemented, unwired. Unpaired marginal CIs
>   are far too loose for effects of this size on models scored over identical realizations.
> - add the normalised-RMSE column P3-D5 mandates twice and no artifact carries. Skill vs
>   persistence is not comparable across horizons here; the denominator oscillates with the roll
>   period.
>
> **5. Geometry changed (P3-D4).** Six target channels, horizons `[10, 20, 30, 50, 100, 150]`, so
> `max_horizon` is 150, not 50. The TCN receptive-field arithmetic below still holds (lookback is
> unchanged at 200), but output shapes and parameter counts do not match anything quoted here.

Shared interface: `forward(x: Tensor[B, L, C_in]) -> Tensor[B, H, C_out]` (point) or `[B, H, C_out, Q]` (quantile).

**TCN** — dilated causal 1-D convolutions, dilations `[1,2,4,8,16,32]`, kernel 3, 64 channels, residual blocks with weight norm and dropout. Receptive field must cover the full lookback: `1 + 2*(k-1)*sum(dilations) = 1 + 2*2*63 = 253 ≥ 200`. State this calculation in a docstring; getting it wrong is a common and invisible bug.

**Transformer** — encoder-only. Patch the lookback into non-overlapping patches of 10 samples (1 s) → 20 tokens, linear embed to `d_model=128`, 3 layers, 4 heads, learned positional encoding, flatten-and-project head. Patching matters: token-per-sample attention over 200 steps is wasteful and trains worse.

**LSTM/GRU** — 2 layers, hidden 128, last hidden state → linear head to `(H, C)`.

Training defaults: AdamW, lr 1e-3, cosine schedule with 5% warmup, batch 256, AMP (`bf16`), gradient clip 1.0, early stopping on validation loss with patience 15, 3 seeds per configuration. Models are small — expect minutes per run on an A4000, so **always run 3 seeds and report mean ± std**. Single-seed comparisons between models this small are noise.

**Gate 4:** all deep models beat damped persistence at 3 s horizon on the `id` regime by a margin exceeding the seed-to-seed standard deviation. Any model that does not, gets debugged or gets reported as not beating the baseline — both are acceptable, silently dropping it is not.

---

### Phase 5 — Probabilistic heads (0.5 day)

Add two heads behind a config flag:

- **Quantile head** — outputs `Q = 9` quantiles (0.05 … 0.95), trained with pinball loss. Apply quantile sorting post-hoc to prevent crossing.
- **Gaussian head** — mean and log-variance, trained with Gaussian NLL. Cheaper, but assumes symmetry.

Attach to the best point model from Phase 4 (probably TCN) and to DLinear (cheap, and gives a probabilistic baseline).

This phase is also the clean bridge into Projects 6 and 7 from the source doc — the quantile head is exactly what split conformal calibration wraps. Structure `heads.py` so a `ConformalWrapper` can be added later without touching the models.

**Gate 5:** PICP@90 within `[0.85, 0.95]` on the `id` regime. Report — do not fix — the degradation on `unseen_seastate`; that degradation *is* the finding, and it is the same story the source doc flags for conformal prediction under domain shift.

---

### Phase 6 — Evaluation, quiescence, ablations (1 day)

> **Before you start — carried forward from Gate 5. Read `docs/protocol.md` §Phase 5 first.**
>
> Phases 3–5 measured things that change this section. The text below is the original plan and has
> **not** been rewritten; these are the deltas.
>
> **1. Quiescence detection is still not built, and it is now three phases overdue.** The Gate 3
> carry-forward said to implement `src/dmf/eval/quiescence.py` *before* the deep models, because
> `id` RMSE cannot separate architectures. It was not done, and Phase 4 and Phase 5 both confirmed
> the prediction: Gate 4 passed on a cell where all three deep models clear the bar by two orders
> of magnitude, and Gate 5 passes on six rows of 216. All seven functions in `quiescence.py` are
> still `raise NotImplementedError`. **Do this first, before any ablation.** The operational
> metric is the only one left that discriminates, and P3-D4 already made `heave_rate` a forecast
> target specifically so this metric could supply its own decision variable.
>
> **2. There is no probabilistic baseline anywhere in Phase 5, and it is that phase's largest
> gap.** All five non-head rows in `e03` are point models, so `probabilistic.csv` holds six learned
> heads and nothing else. CLAUDE.md non-negotiable 4 is satisfied for the point column and has **no
> analogue for the coverage column**: nobody can say whether 102 of 216 cells in band is good,
> because nothing trivial was measured on that axis. Build it before any further interval claim --
> an empirical-residual interval around `persistence` or `dlinear_ols`, with the residual quantiles
> taken from the **validation** split, is closed-form, costs one pass, and would be near-perfectly
> calibrated on `id` by construction. That is exactly what makes it the right floor: a head that
> cannot beat it has learned nothing conditional about its own uncertainty. It needs a new
> closed-form branch in `dmf.train.experiment._fit_one`, which dispatches on `issubclass`
> (P5-D17 item 7).
>
> **3. The probabilistic heads change what quiescence detection can be.** A landing decision is a
> decision under uncertainty, and there are now calibrated intervals to make it with. Detecting a
> quiescent window from the 0.05 quantile of each channel — "will the deck *stay* inside limits
> with 95% confidence" — is a different and more defensible operational rule than thresholding a
> point forecast, and the plan's §6.2 predates the heads existing. Report both; the point-forecast
> rule is the baseline the interval rule has to beat.
>
> **4. Coverage is band-dependent, so an ablation that pools horizons will mislead.** P5-D15: the
> deep heads are calibrated 12 of 12 at 10–15 s on `id` and over-cover in 18–23 of 24 cells at
> 1–5 s. Any Phase 6 table that reports one coverage number per model is averaging two opposite
> behaviours.
>
> **5. The observation-mode ablation (§6.3) is the one Phase 5 could not do.** `e03` is `ideal`
> only, for the same reason `e02` was. P1-D6 forbids mixing an `imu` input with an `ideal` target,
> so `imu` is a separate task definition and a separate sweep — and at Phase 5 rates that is
> another ~65 h. Budget it explicitly or cut the model list; do not assume it is cheap.
>
> **6. Two integrity gaps are inherited, both recorded and neither fixed** (P5-D10): there is no
> shuffle control on any *head* — the one that runs refits AR, a point model — and no untrained
> control on an interval. If Phase 6 makes a claim that turns on interval quality, it needs a
> control that turns on interval quality.
>
> **7. `results/e03/` is the Phase 5 record and must not be regenerated.** `results/`, `results/imu/`
> and `results/e02/` are likewise the Gate 3 and Gate 4 records. Phase 6 writes its own directory.

#### 6.1 Core metric table (`eval/metrics.py`)

For every (model, regime, DOF, horizon): RMSE, MAE, skill score vs persistence, phase lag at max cross-correlation.

#### 6.2 Quiescence detection (`eval/quiescence.py`)

Define an operational landing window at two threshold sets:

| Threshold set | `|roll|` | `|pitch|` | `|heave rate|` | Sustained |
|---|---|---|---|---|
| `permissive` | ≤ 3.0° | ≤ 2.0° | ≤ 0.8 m/s | ≥ 2.0 s |
| `strict` | ≤ 1.5° | ≤ 1.0° | ≤ 0.4 m/s | ≥ 3.0 s |

Ground truth: apply the thresholds to the true future trajectory. Prediction: apply them to the forecast. Score:

- Precision, recall, F1 on window onsets (with a ±0.5 s tolerance on onset timing).
- **Lead time distribution** — how far ahead of onset the model first flags the window.
- **False-alarm rate per minute** — the number a controller engineer actually asks about.
- Base rate of quiescent windows per sea state (report it; a model can score high F1 by exploiting a high base rate).

#### 6.3 Ablations

| Ablation | Compares |
|---|---|
| Observation mode | `ideal` vs `imu` observability |
| Channels | attitude only vs attitude + rates |
| Sea-state conditioning | unconditioned vs one-hot SS as input feature |
| Lookback | 10 s / 20 s / 40 s |
| Normalization | de-mean vs RevIN |

The sea-state-conditioning ablation is worth highlighting: at deployment you would estimate sea state online, so "known SS" is an upper bound, not a deployable result. Say that.

**Gate 6:** `results/results.md` regenerated end-to-end by `make eval`, containing every table above. Every number traceable to a CSV in `results/`.

---

### Phase 7 — ONNX export and latency benchmark (0.5 day)

1. `torch.onnx.export`, opset 18, dynamic batch axis only (fixed sequence length — dynamic sequence length prevents useful kernel selection and you gain nothing here).
2. **Parity check** (`deploy/parity.py`): 1000 random windows, assert `max_abs_err < 1e-4` in FP32. Run this before benchmarking; a fast wrong graph is worthless.
3. **Benchmark harness** (`deploy/bench.py`):
   - 200 warmup iterations, then 2000 timed iterations.
   - Report p50 / p90 / p99 / mean / std, throughput, peak host and device memory.
   - Use `torch.cuda.synchronize()` / ORT's own timing correctly — un-synchronized CUDA timing is the most common benchmarking error and produces impossibly fast numbers.
   - Configurations: PyTorch eager, PyTorch `torch.compile`, ORT CPU EP, ORT CUDA EP, (optionally TensorRT EP). Batch 1 (the real-time case) and batch 32.
   - Stamp every run with hardware, driver, CUDA, torch, ORT versions.

**Expect and report the honest result:** for a model of this size, **ORT CPU EP will very likely beat CUDA EP at batch 1**, because kernel launch overhead dominates a few hundred microseconds of compute. That is the correct and interesting finding — a deck-motion predictor should run on the flight controller's CPU, not contend for the GPU that perception is using. Do not bury it because it is less flashy than a GPU speedup number. It is a better interview answer than a speedup multiple.

**Gate 7:** parity passes; `results/latency.csv` and a Pareto plot exist; the README states measured numbers with methodology, and cites no generic speedup multipliers.

---

### Phase 8 — MSS cross-validation (0.5 day, optional)

Timebox this hard. Install GNU Octave, clone `github.com/cybergalactic/MSS`, run an S175 or supply-vessel case under a JONSWAP spectrum matched to your SS5 head-seas cell, export the 6-DOF time series to CSV.

Compare, not point-by-point (the phase realizations differ), but **statistically**:
- Response spectra overlay (Welch PSD of roll/pitch/heave, yours vs MSS).
- RMS and significant single-amplitude per DOF.
- Zero-crossing periods.

Then the honest test: **train on your Python corpus, evaluate on MSS-generated trajectories**. If skill scores hold up, you have evidence the model learned wave-response structure rather than generator artifacts. That single plot is worth more than another model architecture.

If Octave fights you for more than half a day, stop and write `docs/mss_crossvalidation.md` explaining what was attempted and what the limitation means for the results. A documented, bounded limitation reads as maturity. A silently missing validation reads as an oversight.

---

### Phase 9 — Documentation and packaging (0.5 day)

README structure:

1. One-sentence what-and-why, then immediately: **"All results are from simulated vessel motion; no real deck data is used."**
2. Headline figure: forecast-vs-truth with 90% intervals, roll at SS5 beam seas, 3 s horizon.
3. Skill-score-vs-persistence table across the four regimes.
4. Quiescence detection table with lead-time histogram.
5. Latency table with methodology note.
6. `make all` reproduction instructions and expected runtime.
7. Corpus card link, protocol link, limitations section.
8. Citations: Fossen (MSS), DNV-RP-C205 (JONSWAP form), the source-doc references.

**Limitations section — write it, and write it honestly:**
- Linear seakeeping model; no nonlinear roll damping, no green water, no parametric resonance.
- Unidirectional seas; no spreading, no swell/wind-sea bimodality.
- Fixed heading and speed within a realization; no manoeuvring.
- Simulation only; no real-deck validation.
- Sea state assumed known in the conditioning ablation.

---

## 5. Validation protocol (run before you call it finished)

The user's stated workflow is plan → implement with Claude Code → validate thoroughly. This is the validation checklist. Run it as a distinct pass, ideally in a fresh session so you are reading the code rather than remembering writing it.

### 5.1 Physics
- [ ] All eight Gate 1 invariants pass from a clean seed.
- [ ] Autocorrelation shows no synthesis periodicity.
- [ ] Roll/pitch/heave magnitudes plausible at each sea state; write the numbers down and sanity-check them.
- [ ] Changing `gamma`, `Hs`, `Tp`, `beta`, `U` each move the output in the physically correct direction.

### 5.2 Leakage
- [ ] Seed-disjointness asserted in tests for all four regimes.
- [ ] Normalization statistics derived from train split only.
- [ ] No global shuffle anywhere before splitting.
- [ ] Shuffle-label control: retrain the best model on time-shuffled targets. Skill score must collapse to ~0. If it does not, there is leakage.
- [ ] Untrained-model control: a randomly initialized model must score worse than persistence.

### 5.3 Metrics
- [ ] Skill score reproduced by hand on one worked example.
- [ ] Persistence RMSE computed two independent ways and matched.
- [ ] Quiescence ground truth spot-checked visually on 10 random windows.
- [ ] Base rates reported alongside F1.
- [ ] Every table reports mean ± std over 3 seeds.

### 5.4 Deployment
- [ ] ONNX parity `< 1e-4` on 1000 windows.
- [ ] Warmup present; CUDA synchronization correct.
- [ ] Latency numbers stable across two separate runs (< 10% p50 drift).
- [ ] Environment stamp in every benchmark artifact.

### 5.5 Reproducibility
- [ ] Fresh clone + `make all` on a clean venv reproduces every committed number.
- [ ] All seeds fixed and recorded; `torch.use_deterministic_algorithms` where feasible.
- [ ] No absolute paths, no machine-specific config.
- [ ] `results/` regenerable and consistent with the README.

### 5.6 Honesty pass
- [ ] Every claim in the README traceable to a committed artifact.
- [ ] No model omitted because it underperformed.
- [ ] Negative results (e.g. DLinear beating the Transformer, CPU beating GPU) reported in the README body, not a footnote.
- [ ] Simulation-only caveat in the first paragraph.

---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Synthesis periodicity makes the task artificially easy | **High** | **Critical** | Frequency jitter + autocorrelation test in Gate 1 |
| Task too easy; persistence already near-optimal | Medium | High | Gate 3 decision point; lengthen horizon, add `imu` mode |
| Window-level split leakage | Medium | Critical | Seed-level splits + assertion tests + shuffle control |
| MSS/Octave consumes days | Medium | Medium | Made optional and timeboxed by design (§0.1) |
| TCN receptive field < lookback | Medium | Medium | Explicit calculation asserted in a unit test |
| Un-synchronized CUDA timing | Medium | Medium | Benchmark harness reviewed by the deploy agent |
| Reviewer dismisses it as "just simulation" | High | Medium | Own it up front; MSS cross-check; IMU observability model |
| Scope creep into closed-loop control | Medium | Medium | Explicit non-goal; that is Project 8 |

---

## 7. Extension path toward a publishable contribution

Ranking (c) in the source doc places this second for publication potential. The specific claim that would carry a workshop or RA-L submission:

> *A reproducible, open benchmark for sea-state-generalizing probabilistic deck-motion forecasting, evaluated on operational quiescent-window detection rather than raw RMSE, with calibrated intervals whose coverage degradation under sea-state shift is characterized.*

To get there, add in order of value:
1. Sea-state extrapolation as the headline result (already built: `unseen_seastate`).
2. Split conformal calibration on the quantile head, with coverage measured in-distribution and under shift — this is the merge point with Project 6.
3. Directional wave spreading and bimodal (swell + wind-sea) spectra, so "sea state" is a richer conditioning variable than `(Hs, Tp)`.
4. MSS cross-generator evaluation as external validity evidence.
5. A decision-theoretic evaluation: expected cost of a missed vs false quiescent window, given a landing-attempt cost model.

Items 1, 2 and 4 are already inside the plan. Items 3 and 5 are the incremental research push.
