"""Metric definitions and evaluation integrity -- Gate 6.

Empty in Phase 0. Implemented in Phase 6.

Covered here:

- Skill score reproduced by hand on one worked example, and equal to 0 exactly when the
  model's predictions equal the persistence baseline's.
- Persistence RMSE computed two independent ways and matched.
- Metrics are computed in corpus units (degrees, metres), not normalised space.
- Quiescence: base rate reported alongside every F1; a constant-"yes" detector scores the
  base rate as precision, which is the check that makes a high F1 interpretable.
- Quiescence sustain logic: a run shorter than ``sustain_s`` is not a window, a run exactly
  ``sustain_s`` long is.
- Onset matching is one-to-one -- one predicted onset cannot claim two true onsets.
- Probabilistic: PICP of a fan sorted by :func:`dmf.models.heads.sort_quantiles` is
  monotone in interval width; ``picp`` raises on a crossed (unsorted) fan.
- ``aggregate_over_seeds`` refuses a group with fewer than three seeds.

Controls, from the validation protocol:

- **Shuffle-label control.** Retraining the best model on time-shuffled targets must
  collapse skill to approximately 0. If it does not, there is leakage.
- **Untrained-model control.** A randomly initialised model must score worse than
  persistence.
"""
