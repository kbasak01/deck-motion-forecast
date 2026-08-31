"""Evaluation: accuracy metrics, quiescent-window detection, and probabilistic scoring.

Two rules hold across this subpackage:

1. **Persistence is the reference.** Every accuracy number is reported as a skill score
   against persistence alongside its raw RMSE. A number without its persistence baseline
   is not a result.
2. **Metrics are computed in corpus units** -- degrees for roll and pitch, metres for
   heave, metres per second for heave rate -- never in normalised space. An RMSE in units
   of the *training* normalisation scale cannot be compared across channels and cannot be
   checked against a landing limit. The ``nrmse`` column is a different quantity and does
   not breach this: it is a corpus-unit RMSE divided by the corpus-unit standard deviation
   of the held-out targets themselves, reported because the skill denominator is not
   comparable across horizons on a narrowband signal (``docs/protocol.md`` P3-D5).

The headline metric of the project is not RMSE. Multi-step RMSE on a narrowband
quasi-periodic signal is easy to make look good and easy to make meaningless; persistence
alone already looks excellent at half-second horizons. What carries the project is the
skill score at 2-3 s, quiescent-window detection, and interval calibration.

Layout:

- :mod:`dmf.eval.metrics` -- the metric definitions, and the streamed-sums table builder
  that keeps a test partition's predictions from ever being materialised in full. (No
  literal window count here on purpose: the one that stood in this sentence was the
  pre-P3-D6 geometry's and went stale in place, the same way the caveat literal did.)
- :mod:`dmf.eval.runner` -- one pass over a test partition that scores **every** model on
  identical windows, accumulating error per realization key. Scoring the models together
  is what makes the skill denominator structurally correct rather than correct by
  inspection, and per-key accumulation is what makes the per-cell breakdown and the
  realization-level bootstrap free.
- :mod:`dmf.eval.controls` -- the integrity controls, each a whole-pipeline experiment with
  an outcome known in advance.
- :mod:`dmf.eval.report` -- table aggregation and rendering, including the seed policy.
- :mod:`dmf.eval.gate` -- the Gate 4 read-out: the gate criterion computed from the
  committed tables rather than by an ad-hoc script, in both the readings that ship (the
  original cell and the P4-D1 restatement), each corroborated by the paired bootstrap.

**Two definitions that are fixed here and not re-litigated downstream.** "Horizon ``h``" is
the error at lead time *exactly* ``h`` samples, never a cumulative average over 1..h. And
``n_windows`` is a window count, not an independent-sample count: at ``stride = 5`` with a
200-sample lookback, consecutive windows share 195 of 200 input samples, so every
uncertainty statement resamples whole **realizations**.
"""
