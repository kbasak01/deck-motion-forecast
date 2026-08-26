"""Deck-motion forecasting from JONSWAP-driven vessel simulation.

All motion in this package is *simulated*. Nothing here is derived from, or validated
against, real ship-deck measurements.

This top-level module deliberately imports nothing heavyweight (no torch, no pandas) so
that ``import dmf`` stays cheap and can serve as a bootstrap smoke check.

Unit conventions used throughout the package, stated once here and repeated in every
docstring that touches them:

- Angles are **degrees** at every public boundary (Parquet columns, YAML configs, public
  function signatures). They are converted to radians only inside physics internals.
- Angular rates are **degrees per second** at public boundaries, **radians per second**
  inside physics internals.
- Wave and response frequencies are **radians per second** everywhere, never Hz. Sampling
  rates are the sole exception and are **Hz**.
- Linear displacements are **metres**, linear rates **metres per second**, linear
  accelerations **metres per second squared**.
- Times and periods are **seconds**; window lengths and horizons are **samples** unless
  the name ends in ``_s``.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
