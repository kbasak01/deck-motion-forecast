"""Physics invariants for JONSWAP synthesis -- Gate 1, part one.

Empty in Phase 0. Phase 1 must implement these before writing ``src/dmf/sim/spectra.py``;
this is the phase where an agentic coder is most likely to produce plausible-looking wrong
physics, and the tests are the only defence.

Gate 1 criteria owned by this module:

1. ``4*sqrt(m0)`` recovers the requested ``Hs`` to within 2 percent.
2. Zero-crossing period ``Tz = 2*pi*sqrt(m0/m2)`` falls in ``Tz/Tp`` in 0.71-0.78 for
   ``gamma = 3.3``.
3. Welch PSD of a synthesised 3600 s elevation record matches the analytic ``S(w)`` within
   15 percent across [0.4, 1.5] rad/s.
4. Peak amplitudes of the elevation record follow a Rayleigh distribution
   (KS test, ``p > 0.01``).
5. **Anti-periodicity.** Autocorrelation of a 1 hr record shows no spurious spike at
   ``2*pi/dw``. This is the single most important test in the project: a uniform frequency
   grid makes the record repeat, the forecaster memorises the repeat, and every downstream
   result becomes meaningless without anything looking wrong. The test should also assert
   the converse -- that synthesis with ``jitter=False`` *does* produce the spike -- so that
   it is demonstrably capable of failing.
"""
