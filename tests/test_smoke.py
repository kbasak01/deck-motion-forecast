"""Bootstrap smoke test -- Gate 0.

The other test modules in this directory are intentionally empty until their phase; this
one exists so that ``make test`` has something to run, and it asserts the second half of
Gate 0's criterion: ``import dmf`` succeeds.
"""

import dmf


def test_import_dmf() -> None:
    assert dmf.__version__ == "0.1.0"
