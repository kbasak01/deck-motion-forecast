# Gate 7 read-out

`docs/IMPLEMENTATION_PLAN.md` §Phase 7: parity passes; `results/latency.csv` and a Pareto plot exist; the README states measured numbers with methodology and cites no generic speedup multipliers. Computed from committed artifacts by `scripts/gate7.py`; nothing here measures anything.

**Gate 7: PASS**

| criterion | description | verdict | detail |
|---|---|---|---|
| 1 | every benchmarked configuration has a passing FP32 parity row | PASS | 16 (model, provider) pairs checked over 5 window draws each; worst margin 0.2x. Rows that fail the criterion and are consequently NOT benchmarked: ['lstm_quantile/torch:cuda']. Unscaled 1e-4 additionally fails for ['lstm/torch:cuda', 'lstm_quantile/CPUExecutionProvider', 'lstm_quantile/CUDAExecutionProvider', 'lstm_quantile/TensorrtExecutionProvider', 'lstm_quantile/torch:cuda'] (P7-D3 records the criterion change). Benchmarked configurations without a passing parity row: none. |
| 2 | results/latency.csv and the Pareto figure exist and are non-empty | PASS | latency.csv carries 52 configurations over 5 backends and 2 batch sizes; the Pareto figure exists. |
| 3 | README states measured numbers with methodology and no generic multipliers | PASS | 6 millisecond figures quoted, 0 not in the committed latency tables; 7 multipliers quoted, 0 not reproducible as a same-model same-batch p50 ratio (pool of 665 ratios, which would accept 65 percent of plausible multipliers -- the ms check is the stricter half); methodology tokens missing: none. |

Clause 3 is checked syntactically rather than read charitably: every `Nx` claim in the README's latency section must be reproducible as a ratio of two p50 values **for the same model at the same batch size**, and every millisecond figure must appear in a committed latency table. It cannot tell a speedup from any other ratio, and a fabricated multiple that coincided with a real one would pass; what it catches is a multiplier that is not a ratio of two numbers this machine produced. Its first version pooled every p50 pair in the file and accepted essentially any value -- see `dmf.deploy.gate`.

