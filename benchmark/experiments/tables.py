# benchmark/experiments/tables.py
"""Write a results table as both CSV and LaTeX (the paper's text/table outputs)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def write_table(df: pd.DataFrame, out_dir: Path, name: str, *,
                float_format: str = "%.4f", caption: str = "") -> List[Path]:
    """Write ``df`` to ``out_dir/name.csv`` and ``out_dir/name.tex``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{name}.csv"
    tex = out_dir / f"{name}.tex"
    df.to_csv(csv, index=False)
    try:
        latex = df.to_latex(index=False, float_format=lambda x: float_format % x,
                            caption=caption or name.replace("_", " "), label=f"tab:{name}",
                            escape=True, longtable=False)
    except TypeError:  # older pandas without caption/label kwargs
        latex = df.to_latex(index=False, float_format=lambda x: float_format % x, escape=True)
    tex.write_text(latex, encoding="utf-8")
    return [csv, tex]


def agg_to_plot_wide(agg: pd.DataFrame) -> pd.DataFrame:
    """Convert long aggregate rows (algo, x_value, metric, mean, std, ...) into a wide
    frame for plotting: one row per (algo, x_value) with a column per metric (= mean)
    and a ``<metric>_err`` column (= std) for the error band."""
    if agg.empty:
        return agg
    mean_w = agg.pivot_table(index=["algo", "x_value"], columns="metric", values="mean").reset_index()
    std_w = agg.pivot_table(index=["algo", "x_value"], columns="metric", values="std").reset_index()
    std_w = std_w.rename(columns={c: f"{c}_err" for c in std_w.columns if c not in ("algo", "x_value")})
    wide = mean_w.merge(std_w, on=["algo", "x_value"], how="left")
    wide.columns.name = None
    return wide
