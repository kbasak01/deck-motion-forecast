"""Split integrity and leakage guards -- Gate 2, part one.

Empty in Phase 0. Implemented in Phase 2.

Gate 2 criteria owned by this module:

1. Zero seed overlap between any train and test split, asserted programmatically for all
   four regimes (``id``, ``unseen_seastate``, ``unseen_heading``, ``unseen_vessel``).
2. No time index appears in both a train window and a test window for the same
   realization.
4. Normalisation statistics are computed on train only; a fixture that constructs a
   test-only dataset without training statistics must raise rather than silently refit.

Also covered here:

- The validation partition is drawn from training seeds, never from test seeds.
- ``build_split`` raises rather than returning an empty test set when a regime's condition
  selects nothing (for example ``unseen_vessel`` on a single-vessel corpus).
"""
