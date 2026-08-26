"""ONNX export parity and benchmark-harness integrity -- Gate 7.

Empty in Phase 0. Implemented in Phase 7.

Covered here:

- FP32 parity: ``max_abs_err < 1e-4`` between PyTorch and ONNX Runtime over 1000 random
  windows, for every exported model.
- The exported graph has a dynamic batch axis and a fixed sequence axis.
- Output shape matches ``BaseForecaster.output_shape`` for both point and quantile heads.
- ``benchmark_onnxruntime`` records the provider actually selected, not the one requested,
  so an unavailable CUDA provider cannot silently produce a "CUDA" row holding CPU timings.
- ``benchmark_torch`` raises when asked to time a CUDA run with ``synchronize=False``:
  un-synchronised CUDA timing produces impossibly fast numbers, and the harness must refuse
  to generate them rather than leave it to the reader to notice.
"""
