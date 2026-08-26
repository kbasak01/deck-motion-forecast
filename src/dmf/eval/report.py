"""Results table assembly and rendering.

Every number that appears in ``results/results.md`` or the README traces back to a CSV
written by this module. ``make eval`` regenerates all of it end to end; nothing in
``results/`` is hand-edited.

Model comparisons are reported as **mean +/- std over at least three seeds**, and no model
is dropped for underperforming. If DLinear beats the transformer, both rows appear.
"""

from pathlib import Path

import pandas as pd

__all__ = ["aggregate_over_seeds", "build_results_report", "to_markdown", "write_table"]


def write_table(df: pd.DataFrame, path: Path) -> Path:
    """Write a results table to CSV.

    Args:
        df: The table to write.
        path: Destination under ``results/``. Parent directories are created if absent.

    Returns:
        The path written.
    """
    raise NotImplementedError


def aggregate_over_seeds(df: pd.DataFrame, group_cols: tuple[str, ...]) -> pd.DataFrame:
    """Collapse per-seed rows into mean and standard deviation.

    Args:
        df: Per-seed results, with a ``seed`` column and one or more metric columns.
        group_cols: Columns identifying a comparison cell, e.g.
            ``("model", "regime", "dof", "horizon_samples")``.

    Returns:
        One row per group, with ``<metric>_mean``, ``<metric>_std`` and ``n_seeds``
        columns. Units are unchanged from the input.

    Raises:
        ValueError: If ``df`` has no ``seed`` column, or if any group contains fewer than
            three seeds -- a comparison over fewer seeds than that measures initialisation
            noise, so it is refused rather than reported with a caveat.
    """
    raise NotImplementedError


def to_markdown(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Render a results table as a GitHub-flavoured Markdown table.

    Args:
        df: The table to render.
        float_fmt: Format string applied to float columns.

    Returns:
        The rendered table, ready to paste into ``results/results.md``.
    """
    raise NotImplementedError


def build_results_report(results_dir: Path, out_path: Path) -> Path:
    """Assemble every committed CSV in ``results/`` into the Markdown report.

    Args:
        results_dir: Directory of committed result CSVs.
        out_path: Destination Markdown file, normally ``results/results.md``.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If an expected CSV is missing, so that a partially regenerated
            report cannot be mistaken for a complete one.
    """
    raise NotImplementedError
