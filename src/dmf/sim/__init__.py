"""Wave and vessel-response simulation.

This subpackage is **pure NumPy and has no torch dependency**. That is an architectural
rule, not an accident: the simulator must be runnable, testable, and reviewable by someone
who has no deep-learning stack installed, and its physics must be verifiable against
closed-form spectral invariants rather than against a training curve.

The physics implemented here is a linear seakeeping model: JONSWAP spectrum, random-phase
wave synthesis, encounter-frequency transformation, and second-order DOF response. It has
no nonlinear roll damping, no green water, and no parametric resonance. Every trajectory
it produces is simulated; none is measured.
"""
