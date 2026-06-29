# benchmark/paper_viz/aggregation.py
"""Aggregation: turn long-format metric rows into curves / error bars / tables.

Vendored from the HighwayOCD ``performance_module``. Pure statistics — no I/O,
no algorithm execution. Optimization direction comes from ``specs.METRICS``.

  - mean +/- std / 95% CI across seeds  -> seed-dispersion + error bands
  - best baseline per metric + delta + rank  -> "don't average the baselines"
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .specs import METRICS, is_highway


def aggregate_by_x(
    df: pd.DataFrame,
    *,
    x_col: str,
    metric_cols: List[str],
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Group by (group_cols + [algo, x_col]) and compute mean/std/n/ci95 per metric.

    Output columns: [group_cols...] algo x_value metric mean std n ci95 ci95_low ci95_high
    ci95 = 1.96 * std / sqrt(n); n<2 -> std=0, ci95=0.
    """
    group_cols = group_cols or []
    keys = group_cols + ["algo", x_col]

    out_rows = []
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        sub = df[keys + [metric]].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub.dropna(subset=[metric])
        if sub.empty:
            continue
        g = sub.groupby(keys, as_index=False).agg(
            mean=(metric, "mean"),
            std=(metric, lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
            n=(metric, "size"),
        )
        g["metric"] = metric
        g["ci95"] = np.where(g["n"] > 1, 1.96 * g["std"] / np.sqrt(g["n"]), 0.0)
        g["ci95_low"] = g["mean"] - g["ci95"]
        g["ci95_high"] = g["mean"] + g["ci95"]
        g = g.rename(columns={x_col: "x_value"})
        out_rows.append(g)

    cols = group_cols + ["algo", "x_value", "metric", "mean", "std", "n", "ci95", "ci95_low", "ci95_high"]
    if not out_rows:
        return pd.DataFrame(columns=cols)
    return pd.concat(out_rows, ignore_index=True)[cols]


def best_baseline_table(
    df: pd.DataFrame,
    *,
    metric_cols: List[str],
    group_cols: Optional[List[str]] = None,
    highway_key: str = "highway",
) -> pd.DataFrame:
    """Per (group, metric): Highway mean vs the BEST baseline mean, delta, and
    Highway's rank among all algorithms. Direction from METRICS[metric].

    Output: [group_cols...] metric highway_mean best_baseline best_baseline_mean delta_vs_best rank n_algos
    """
    group_cols = group_cols or []
    rows = []

    for metric in metric_cols:
        if metric not in df.columns:
            continue
        hib = METRICS[metric].higher_is_better if metric in METRICS else True
        sub = df.copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub.dropna(subset=[metric])
        if sub.empty:
            continue

        keyset = sub.groupby(group_cols) if group_cols else [((), sub)]
        for gval, gdf in keyset:
            per_algo = gdf.groupby("algo", as_index=False)[metric].mean()
            if per_algo.empty:
                continue

            baselines = per_algo[~per_algo["algo"].map(is_highway)]
            hw = per_algo[per_algo["algo"] == highway_key]
            if baselines.empty or hw.empty:
                continue

            if hib:
                bidx = baselines[metric].idxmax()
                rank = int((per_algo[metric] > hw[metric].iloc[0]).sum()) + 1
            else:
                bidx = baselines[metric].idxmin()
                rank = int((per_algo[metric] < hw[metric].iloc[0]).sum()) + 1

            best_algo = baselines.loc[bidx, "algo"]
            best_mean = float(baselines.loc[bidx, metric])
            hw_mean = float(hw[metric].iloc[0])
            delta = hw_mean - best_mean if hib else best_mean - hw_mean

            rec = dict(zip(group_cols, gval if isinstance(gval, tuple) else (gval,))) if group_cols else {}
            rec.update(
                metric=metric,
                highway_mean=hw_mean,
                best_baseline=best_algo,
                best_baseline_mean=best_mean,
                delta_vs_best=delta,
                rank=rank,
                n_algos=int(per_algo["algo"].nunique()),
            )
            rows.append(rec)

    cols = group_cols + ["metric", "highway_mean", "best_baseline",
                         "best_baseline_mean", "delta_vs_best", "rank", "n_algos"]
    return pd.DataFrame(rows, columns=cols)


def quantile_bins(
    df: pd.DataFrame,
    *,
    x_col: str,
    value_col: str = "runtime",
    num_bins: int = 12,
    agg: str = "mean",
) -> pd.DataFrame:
    """Quantile-bin x_col, aggregate value_col per bin. Output: algo x_value value n."""
    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")
    work = df[["algo", x_col, value_col]].copy()
    work["x_bin"] = pd.qcut(work[x_col], q=num_bins, duplicates="drop")
    grouped = (
        work.groupby(["algo", "x_bin"], as_index=False, observed=True)
        .agg(x_value=(x_col, "median"), value=(value_col, agg), n=(value_col, "size"))
        .sort_values(["algo", "x_value"])
    )
    return grouped[grouped["value"] > 0].copy()
