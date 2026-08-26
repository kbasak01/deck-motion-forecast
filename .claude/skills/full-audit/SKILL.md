---
name: full-audit
description: Run the complete pre-release validation pass over the repository - physics, leakage, metrics, deployment, reproducibility, and honesty checks - and produce a written audit report.
disable-model-invocation: true
context: fork
allowed-tools: Read, Grep, Glob, Bash, Write
---

Run the full validation protocol from section 5 of `docs/IMPLEMENTATION_PLAN.md`. Work through all
six subsections and verify each checkbox against committed artifacts rather than against code
comments or the README's own claims.

Run these integrity controls yourself and report the measured outcome:

1. **Shuffle control** — retrain the best model on time-shuffled targets. Skill score must collapse
   to approximately zero.
2. **Untrained control** — a randomly initialized model must score worse than persistence.
3. **Pipeline sanity** — persistence through the full dataset pipeline must match persistence
   computed directly on raw arrays.
4. **Benchmark stability** — re-run the latency sweep and confirm p50 agrees within 10%.

Then delegate to the `results-skeptic` agent for an adversarial read and merge its findings.

Write the result to `docs/audit_report.md` with findings in three tiers — BLOCKING, SHOULD FIX,
NOTE — each with concrete evidence (file, line, or artifact) and the smallest change that would
resolve it.

Finish with a single verdict line: **RELEASE READY** or **NOT RELEASE READY**, plus the count of
blocking findings. If nothing is blocking, say so plainly rather than manufacturing concerns.
