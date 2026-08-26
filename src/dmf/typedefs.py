"""Shared type aliases for array-typed signatures.

Kept in one place so that every module in the package spells a float array the same way,
which is what makes ``mypy --strict`` useful rather than merely noisy.
"""

import numpy as np
import numpy.typing as npt

__all__ = ["BoolArray", "ComplexArray", "FloatArray", "IntArray"]

#: Real-valued array, float64. The working dtype for all simulation and metric code.
#: Serialised to Parquet as float32; float64 is used in-memory so that spectral moment
#: integrals and long-record accumulations do not lose precision.
type FloatArray = npt.NDArray[np.float64]

#: Complex-valued array, complex128. Used for frequency-domain transfer functions, where
#: the argument carries the phase lag between wave excitation and DOF response.
type ComplexArray = npt.NDArray[np.complex128]

#: Integer array, int64. Used for sample indices, seeds, and window offsets.
type IntArray = npt.NDArray[np.int64]

#: Boolean mask array. Used for per-sample quiescence masks and split membership.
type BoolArray = npt.NDArray[np.bool_]
