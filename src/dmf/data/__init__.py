"""Windowing, splitting, and normalisation of the simulated corpus.

Two rules govern this subpackage and are asserted by ``tests/test_splits.py`` and
``tests/test_windows.py`` rather than merely documented:

1. **Splits are by realization seed, never by time window.** Windows are only ever formed
   *after* the seed-level split has been decided. Any code path that shuffles windows and
   then splits them is a bug, not a style preference.
2. **Normalisation statistics come from the training split only.** Fitting a scale over
   the whole corpus leaks test-set variance into training and is the subtler of the two
   failure modes, because it does not show up as an obviously implausible metric.
"""
