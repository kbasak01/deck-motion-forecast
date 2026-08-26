"""Wave and vessel-response simulation.

Everything in this subpackage is **pure NumPy**. Importing ``torch`` anywhere under
``dmf.sim`` is an architecture violation and ``tests/test_spectra.py`` enforces it by
AST-parsing every module here.

Unit conventions, restated because mixing them has been the most common bug class in this
domain:

- Frequencies (wave ``w``, encounter ``w_e``, natural ``wn``) are **radians per second**.
  Sampling rates are the sole Hz quantity and always carry an ``_hz`` suffix.
- Headings are **degrees** at every public boundary (180 head seas, 90 beam, 0 following)
  and converted to radians only inside the physics.
- Roll and pitch are **radians** inside the physics and **degrees** on
  :class:`~dmf.sim.response.MotionRecord` and in Parquet; rates follow the same rule
  (rad/s internally, deg/s stored).
- Heave is **metres**, heave rate **metres per second**, heave acceleration
  **metres per second squared**, everywhere.
- Speeds are **metres per second** inside the physics; knots only at the config boundary,
  converted by :func:`~dmf.sim.encounter.knots_to_m_s`.

All randomness flows through an explicit :class:`numpy.random.Generator` passed in by the
caller. There are no module-level ``np.random`` calls.
"""

__all__: list[str] = []
