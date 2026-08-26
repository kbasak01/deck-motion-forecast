---
name: phase-gate
description: Run the phase gate check for a given build phase before moving on. Verifies the phase's acceptance criteria against committed artifacts and reports pass/fail per criterion.
argument-hint: <phase-number 0-9>
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash
---

Run the gate check for phase $ARGUMENTS of `docs/IMPLEMENTATION_PLAN.md`.

Procedure:

1. Read the phase section and its gate criteria from `docs/IMPLEMENTATION_PLAN.md`.
2. Run `make test` and `make lint`. Report failures by name only, not full output.
3. Check each gate criterion against **committed artifacts** — files in `results/`, test outcomes,
   generated data — not against what the code claims to do. A criterion is unverified until an
   artifact demonstrates it.
4. For phases 3 and above, delegate a review to the `results-skeptic` agent and fold its findings in.
5. Produce a table: criterion | measured value | PASS/FAIL/UNVERIFIED.

Then state one of:

- **GATE PASSED** — with the one-line justification, and the first task of the next phase.
- **GATE FAILED** — with the specific criteria that failed and the smallest change that would fix
  each. Do not proceed to the next phase, and do not propose relaxing a threshold. If a threshold
  genuinely needs to change, say so explicitly and record the change and its reason in
  `docs/protocol.md`.

Be concise. This is a checkpoint, not a report.
