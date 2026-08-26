"""Physics invariants for vessel response -- Gate 1, part two.

Empty in Phase 0. Implemented in Phase 1.

Gate 1 criteria owned by this module:

6. In beam seas the roll response spectrum peaks within 5 percent of ``wn_roll``; in head
   seas roll RMS drops by at least an order of magnitude.
7. Roll RMS at SS5 beam seas lands in a physically sensible range (single-digit degrees).
   Print the value and check it against published seakeeping figures rather than only
   asserting a wide band.
8. Increasing forward speed in head seas shifts the response spectrum peak to higher
   ``w_e``.

Also covered here:

- Encounter frequency: ``w_e > w`` in head seas, ``w_e < w`` in following seas, and
  :func:`dmf.sim.encounter.encounter_frequency_is_monotonic` returns False somewhere in
  the band for the following-seas cases that the corpus grid must exclude.
- Directional monotonicity: changing ``gamma``, ``Hs``, ``Tp``, ``beta`` and ``U`` each
  move the output in the physically correct direction.
- Phase coherence: roll, pitch and heave synthesised from a shared phase set are
  cross-correlated, and are not when the phase set is redrawn per DOF.
"""
