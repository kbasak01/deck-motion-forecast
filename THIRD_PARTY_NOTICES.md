# Third-party notices

This project is MIT-licensed (see [LICENSE](LICENSE)). It also redistributes, in modified form, a
file from a third-party project, and depends on others at runtime. Those are listed here.

## Redistributed in this repository

### Marine Systems Simulator (MSS)

- **Upstream:** <https://github.com/cybergalactic/MSS>, pinned at commit
  `98970f71a21cfe81e7e29abdcc1bb6741789cddc` (2026-09-07).
- **License:** MIT, Copyright (c) 2004 Thor I. Fossen.
- **What is redistributed:** [`mss/waveMotionRAO_seeded.m`](mss/waveMotionRAO_seeded.m) is a
  modified copy of `LIBRARY/environment/waveMotionRAO.m`. Three functional changes, all to the
  source of the random phases, documented and justified in [`mss/PATCHES.md`](mss/PATCHES.md) and
  verified behaviour-preserving by `scripts/mss_octave_check.py`. The file carries the upstream
  copyright and permission notice in its header, as the MIT license requires.
- **What is *not* redistributed:** the MSS toolbox itself. `mss/upstream/` is a gitignored clone;
  `make mss` expects it to be present locally. `mss/run_case.m` and `mss/check_patch.m` are this
  project's own drivers, not MSS files.

MSS is used in Phase 8 as an **independent hydrodynamic reference** — its ShipX strip-theory motion
RAOs are the second generator this project's models are cross-validated against. It is not used to
generate the training corpus.

## Runtime dependencies

Python dependencies are pinned in [`pyproject.toml`](pyproject.toml) and carry their own licenses;
none of them is vendored here. The principal ones are PyTorch (BSD-3-Clause), ONNX Runtime (MIT),
NumPy (BSD-3-Clause), pandas (BSD-3-Clause), PyArrow (Apache-2.0), SciPy (BSD-3-Clause) and
Matplotlib (PSF-based, matplotlib license).

## Standards and published methods

The JONSWAP spectral form, the RAO and encounter-frequency formulations, and the forecasting
architectures implemented here are published methods, cited in the README's
[Citations](README.md#citations) section. No text or code from those sources is reproduced in this
repository.
