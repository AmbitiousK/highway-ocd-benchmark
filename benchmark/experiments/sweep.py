# benchmark/experiments/sweep.py
"""Run a benchmark sweep: for every graph in a corpus, run all algorithms and
emit one long-format metric table (the integration seam for plots + tables).

Long-format schema (one row per graph x algo x metric):

    benchmark, graph_id, algo, x_value, seed, num_nodes, num_edges, status, metric, value

Both quality metrics AND runtime (metric == "algo_runtime_sec") are emitted, so a
single sweep feeds the performance figure, the scalability figure, and the tables.
Every registered algorithm — including one you added with register_algorithm —
flows through automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from ..compare import compare_algorithms
from ..paper_viz.specs import PERF_PANEL_ORDER, display_name
from .corpus import iter_corpus

# metric columns produced by compare_algorithms that we melt into long rows
_QUALITY = list(PERF_PANEL_ORDER)


def run_sweep(
    corpus_root: Path,
    *,
    x_param: str,
    benchmark: str,
    algos: Optional[Sequence[str]] = None,
    seed_param: Optional[str] = "seed",
    ground_truth="auto",
    metrics: Optional[Sequence[str]] = None,
    timeout: Optional[float] = 120.0,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every algorithm on every graph under ``corpus_root``.

    Args:
        corpus_root: directory tree of graph.gpickle files.
        x_param: graph parameter used as the sweep x-axis ("muw" | "xi" | "eta").
            Use "num_nodes" to sweep by graph size (scalability).
        benchmark: label stored in the ``benchmark`` column ("lfr" | "abcdo2" | ...).
        algos: algorithms to run (default: all built-ins + registered customs).
        seed_param: graph parameter recorded as ``seed`` (for error bars); None to skip.
        ground_truth: "auto" (read from graph), or None (unsupervised-only).
        metrics: quality metrics to record (default: the full panel).
        timeout: optional per-algorithm hard timeout (seconds).
        limit: cap the number of graphs (smoke runs).
    """
    quality = list(metrics) if metrics is not None else _QUALITY
    rows: List[dict] = []
    n_graphs = 0

    for cg in iter_corpus(corpus_root, limit=limit):
        n_graphs += 1
        x_value = cg.number(x_param) if x_param not in ("num_nodes", "num_edges") \
            else getattr(cg, x_param)
        seed = cg.number(seed_param) if seed_param else None
        if verbose:
            print(f"[sweep:{benchmark}] {cg.graph_id}  {x_param}={x_value}  "
                  f"(n={cg.num_nodes} m={cg.num_edges})")

        df = compare_algorithms(cg.G, algos=algos, ground_truth=ground_truth,
                                metrics=quality, timeout=timeout, verbose=False)

        for _, r in df.iterrows():
            if r["status"] != "ok":
                continue
            base = dict(benchmark=benchmark, graph_id=cg.graph_id, algo=r["algo"],
                        x_value=x_value, seed=seed,
                        num_nodes=cg.num_nodes, num_edges=cg.num_edges, status=r["status"])
            for m in quality:
                rows.append({**base, "metric": m, "value": r.get(m)})
            rows.append({**base, "metric": "algo_runtime_sec", "value": r.get("runtime_sec")})

    out = pd.DataFrame(rows)
    if not out.empty:
        out["algo_display"] = out["algo"].map(display_name)
    if verbose:
        print(f"[sweep:{benchmark}] {n_graphs} graphs, {len(out)} metric rows")
    return out


def to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long rows -> one row per (benchmark, graph_id, algo, x_value, seed) with
    a column per metric. Convenient for per-graph inspection and for plotting."""
    idx = ["benchmark", "graph_id", "algo", "algo_display", "x_value", "seed",
           "num_nodes", "num_edges"]
    idx = [c for c in idx if c in long_df.columns]
    work = long_df.copy()
    # pivot_table drops rows with NaN in any index level; benchmarks without a
    # 'seed' (e.g. LFR) would vanish entirely. Fill NaN index keys with a sentinel.
    for c in idx:
        if work[c].isna().any():
            work[c] = work[c].fillna(-1)
    wide = work.pivot_table(index=idx, columns="metric", values="value",
                            aggfunc="first").reset_index()
    wide.columns.name = None
    return wide
