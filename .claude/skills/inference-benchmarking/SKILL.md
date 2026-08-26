---
name: inference-benchmarking
description: Methodology for ONNX export, numerical parity checking, and honest inference latency measurement across execution providers. Use whenever exporting a model to ONNX, measuring latency or throughput, comparing PyTorch to onnxruntime or TensorRT, or writing a deployment or performance claim. Use proactively before any timing number is recorded.
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# Inference benchmarking methodology

## Export

- `torch.onnx.export`, opset 18.
- Dynamic axes: batch only. Keep sequence length fixed — dynamic sequence length blocks useful
  kernel selection and buys nothing for a fixed-lookback forecaster.
- Export in eval mode with `torch.no_grad()`.

## Parity before timing

1000 random windows, assert `max_abs_err < 1e-4` (FP32) between PyTorch and ORT outputs. If parity
fails, stop and report — a fast graph computing the wrong thing is worthless. For FP16, relax to
`1e-2` and state the tolerance next to the number.

## Timing

- 200 warmup iterations, then 2000 timed iterations.
- Report **p50, p90, p99, mean, std**, throughput, peak host memory, peak device memory. A mean
  without percentiles hides tail latency, which is what a real-time control loop cares about.
- `torch.cuda.synchronize()` around every timed GPU region; use ORT's own timing API correctly.
  **Un-synchronized CUDA timing is the most common benchmarking error** and produces impossibly fast
  numbers. If a measurement looks too good, check synchronization before believing it.
- Configurations: PyTorch eager, `torch.compile`, ORT CPU EP, ORT CUDA EP, TensorRT EP if available.
  Batch 1 (real-time case) and batch 32.
- Pin CPU threads and record the setting; ORT CPU results swing widely with thread count.

## Environment stamp

Every result JSON records: GPU model, driver version, CUDA version, CPU model, thread count,
`torch.__version__`, `onnxruntime.__version__`, and the git commit. Benchmark numbers without an
environment stamp are not reproducible.

## Stability

Re-run the full sweep once. p50 must agree within 10% across runs before reporting. Report the
spread if it does not.

## Claims

Report only what was measured on this machine. Never cite generic speedup multipliers. State the
target platform for any "real-time" claim. Note explicitly when a workstation GPU is standing in for
an embedded target — absolute numbers do not transfer.

For a small forecasting model, expect ORT CPU EP to beat CUDA EP at batch 1 because kernel launch
overhead dominates. Report that plainly; it is the more interesting and more useful finding.
