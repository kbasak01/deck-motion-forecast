# Local modifications to the MSS toolbox

Upstream is pinned at `98970f71a21cfe81e7e29abdcc1bb6741789cddc` (2026-09-07) and is cloned
to `mss/upstream/`, which is gitignored. Nothing in the clone is edited in place.

## `waveMotionRAO_seeded.m`

A copy of `upstream/LIBRARY/environment/waveMotionRAO.m` with **three** functional changes, all to
the source of the random phases and nothing else, plus a comment-only attribution block. Reproduce
with
`diff -u <(sed 's/\r$//' mss/upstream/LIBRARY/environment/waveMotionRAO.m) mss/waveMotionRAO_seeded.m`.

1. Renamed, and `phasesIn` added as a trailing argument.
2. `randomPhases` dropped from the `persistent` list and assigned from `phasesIn`.
3. The `rng(12345,"twister")` draw removed.
4. **Comment only, no behaviour:** a `Revisions:` note recording 1-3, and the upstream MIT
   copyright and permission notice. MSS is MIT-licensed, Copyright (c) 2004 Thor I. Fossen, and
   that licence requires the notice to travel with "copies or substantial portions of the
   Software". This file is a substantial portion of one, so it carries the notice; before Phase 9
   it carried only the upstream `Author:` line, which is attribution but is not the required
   notice. See also `THIRD_PARTY_NOTICES.md` at the repository root.

**Why.** Upstream seeds `rng(12345,"twister")` into a `persistent` variable, so every fresh
MATLAB/Octave process produces the *same* realization. Generating three "seeds" by calling
it three times would silently yield one record three times. The patch is also what lets the
Octave and NumPy backends be driven from an identical phase set, without which a parity
check compares two different random draws and can only ever be a distributional statement.

**The patch is verified behaviour-preserving**, not assumed: `scripts/mss_octave_check.py`
runs stock `waveMotionRAO.m` and the patched copy fed with the phases stock would have
drawn (`rng(12345,'twister'); 2*pi*rand(n,1)`), and requires the two outputs to agree to
machine precision before any parity number is reported.

## Behaviour worth knowing about, not patched

`waveMotionRAO.m` caches its RAO frequency interpolation in a `persistent` variable guarded
by `if isempty(...)`. Calling it with a *different* `Omega` grid inside one session silently
reuses the previous interpolation. `mss/run_case.m` issues `clear -f` before each case.
