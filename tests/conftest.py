"""Shared pytest fixtures.

Empty in Phase 0. Phase 1 adds the seeded ``rng`` fixture and the small synthetic corpus
fixture that the physics and split tests build on, so that no test constructs its own
generator and reproducibility is a property of the fixture rather than of each test.
"""
