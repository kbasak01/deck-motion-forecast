"""Evaluation: accuracy metrics, quiescent-window detection, and probabilistic scoring.

Two rules hold across this subpackage:

1. **Persistence is the reference.** Every accuracy number is reported as a skill score
   against persistence alongside its raw RMSE. A number without its persistence baseline
   is not a result.
2. **Metrics are computed in corpus units** -- degrees for roll and pitch, metres for
   heave, metres per second for heave rate -- never in normalised space. A normalised RMSE
   cannot be compared across channels and cannot be checked against a landing limit.

The headline metric of the project is not RMSE. Multi-step RMSE on a narrowband
quasi-periodic signal is easy to make look good and easy to make meaningless; persistence
alone already looks excellent at half-second horizons. What carries the project is the
skill score at 2-3 s, quiescent-window detection, and interval calibration.
"""
