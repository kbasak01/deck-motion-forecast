"""Ingest of MSS (Marine Systems Simulator) S175 strip-theory data for cross-validation.

This package exists to answer one question that `src/dmf/sim/` cannot answer about itself:
do the forecasting models transfer to deck motion produced by an *independent*
hydrodynamic computation, or did they learn artifacts of our own generator?

The independence here is in the **hydrodynamics**, not in the wave synthesis. MSS ships
ShipX (VERES) strip-theory motion RAOs for the ITTC S-175 containership on a
36-frequency x 36-heading x speed grid; those tables are the thing our own
`configs/sim/vessels/s175.yaml` explicitly says it is not::

    "a reduced-order stand-in for a different ship, not a strip-theory computation of the
     S-175's actual RAOs, so `unseen_vessel` measures transfer across a parameter shift
     rather than across a genuinely different hull form."

Everything here is simulated. MSS trajectories are another simulation, not measurements of
a real ship, and no statement derived from them may imply real-world validation.

Like `dmf.sim`, this package is pure NumPy/SciPy and must not import torch.

Unit and sign conventions are stated in :mod:`dmf.mss.convert`; they were derived by
measurement rather than from MSS's own source comments, two of which are stale (see that
module's docstring).
"""

from dmf.mss.convert import (
    MSS_TO_CORPUS_SIGN,
    mss_motion_to_frame,
)
from dmf.mss.synth import (
    MSSMotion,
    sample_corpus_grid,
    sample_mss_grid,
    synthesize_mss_motion,
)
from dmf.mss.vessel import MSSVessel, load_mss_vessel

__all__ = [
    "MSS_TO_CORPUS_SIGN",
    "MSSMotion",
    "MSSVessel",
    "load_mss_vessel",
    "mss_motion_to_frame",
    "sample_corpus_grid",
    "sample_mss_grid",
    "synthesize_mss_motion",
]
