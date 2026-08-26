"""ONNX export, numerical parity, and latency benchmarking.

Order matters here: **parity before benchmarking**. A fast graph that computes the wrong
thing is worthless, and it is easy to produce one by accident with a bad opset or a
mishandled dynamic axis.

The expected result for a model of this size is that the ONNX Runtime **CPU** execution
provider beats the CUDA one at batch 1, because kernel-launch overhead dominates a few
hundred microseconds of compute. That is the correct and useful finding -- a deck-motion
predictor should run on the flight controller's CPU rather than contend for the GPU that
perception is using -- and it belongs in the README body, not a footnote.
"""
