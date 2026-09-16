# Gate 10 — split-conformal calibration

**6 of 7 predicates pass.**

A failure to calibrate under distribution shift is a PASS with a negative finding: predicate 7 requires the degradation to be reported and deliberately sets no bar on it (P5-D2, P10-D1).

| # | predicate | verdict | note |
|---|---|---|---|
| 1 | committed uncalibrated tables unchanged | PASS | all 4 clean at HEAD; digests {'probabilistic.csv': '2eca352071ac', 'probabilistic_by_seed.csv': '9789a92d5af6', 'gate5.csv': '8d51894a948c', 'gate5_degradation.csv': '6c7a996cf8fb'} |
| 2 | calibrated table joins the committed one | PASS | 6 calibrated labels, disjoint; schema identical; window counts agree on all 144 shared cells |
| 3 | calibrated on <regime>/val under train statistics | PASS | 4 regimes, all fitted on <regime>/val under <regime>/train statistics; calibration windows [23267, 24430, 24677] |
| 4 | id coverage in band at the gate cell | PASS | 6 of 6 inside [0.85, 0.95]; worst deviation from 0.90 is 0.0025 |
| 5 | >= 3 seeds on every stochastic row | PASS | all 864 rows carry >= 3 seeds or are deterministic |
| 6 | pre-registration precedes the run | FAIL | results/e05/conformal.csv is not committed yet, so the ordering cannot be checked |
| 7 | degradation reported, no threshold | PASS | 648 rows, no threshold applied by design. unseen_heading: picp_delta -0.4044 mean, unseen_seastate: picp_delta -0.4036 mean, unseen_vessel: picp_delta -0.2888 mean |
