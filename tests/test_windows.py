"""Windowing and pipeline-fidelity checks -- Gate 2, part two.

Empty in Phase 0. Implemented in Phase 2, except the receptive-field check, which Phase 4
adds.

Gate 2 criteria owned by this module:

3. A window reconstructed from the dataset exactly matches the raw Parquet slice.
5. A persistence baseline evaluated through the full dataset pipeline reproduces the same
   RMSE as computed directly on the raw arrays -- an end-to-end sanity check that catches
   an off-by-one in windowing, a mis-inverted normalisation, or a channel-order mismatch,
   none of which would be visible in a loss curve.

Also covered here:

- **TCN receptive field.** ``receptive_field(3, (1,2,4,8,16,32)) == 253 >= lookback``,
  asserted arithmetically against :func:`dmf.models.tcn.receptive_field` rather than
  restated in a comment, plus a check that ``TCN.__init__`` raises when the configured
  stack is too shallow for the configured lookback.
- Windows never span a realization boundary.
"""
