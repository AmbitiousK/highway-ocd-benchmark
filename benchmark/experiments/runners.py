# benchmark/experiments/runners.py
"""The four benchmark experiments, each producing a paper-style figure + tables
from a LIVE run of all algorithms (Highway + baselines + your registered method):

    run_performance  metric panel vs muw / xi          (1x5 figure + best-baseline & dispersion tables)
    run_scalability  runtime vs graph size (log y)      (figure + runtime summary table)
    run_overlap      metric panel vs overlap eta        (1x4 figure with seed error bands + stability table)
    run_realworld    Q_ov + runtime on real graphs      (grouped-bar figure + main table)

Outputs land under ``<out_dir>/plots`` and ``<out_dir>/tables``.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from ..compare import compare_algorithms
from ..paper_viz import (
    PERF_PANEL_ORDER, OVERLAP_PANEL_ORDER, panel_specs,
    metric_panel_grid, save_figure, aggregate_by_x, best_baseline_table, quantile_bins,
    build_color_map,
)
from ..paper_viz.specs import MetricSpec, METRICS, display_name
from .sweep import run_sweep, to_wide
from .tables import write_table, agg_to_plot_wide


def _algo_order(long_df: pd.DataFrame) -> List[str]:
    """Stable algorithm order: baselines (as encountered) then Highway last/highlighted."""
    algos = list(dict.fromkeys(long_df["algo"].tolist()))
    base = [a for a in algos if a not in ("highway", "highwayfull")]
    hw = [a for a in algos if a in ("highway", "highwayfull")]
    return base + hw


def _display_map(order: Sequence[str]) -> dict:
    return {a: display_name(a) for a in order}


# ---------------------------------------------------------------------------
# 1. performance: metric panel vs muw / xi
# ---------------------------------------------------------------------------
def run_performance(corpus_root: Path, *, benchmark: str, x_param: str, x_label: str,
                    out_dir: Path, algos: Optional[Sequence[str]] = None,
                    timeout: Optional[float] = None, limit: Optional[int] = None) -> pd.DataFrame:
    long = run_sweep(corpus_root, x_param=x_param, benchmark=benchmark, algos=algos,
                     timeout=timeout, limit=limit)
    if long.empty:
        raise SystemExit(f"no metric rows produced from {corpus_root}")
    wide = to_wide(long)
    order = _algo_order(long)
    dmap = _display_map(order)

    agg = aggregate_by_x(wide, x_col="x_value", metric_cols=PERF_PANEL_ORDER)
    plot_wide = agg_to_plot_wide(agg)

    out_dir = Path(out_dir)
    fig, _ = metric_panel_grid(panel_specs(PERF_PANEL_ORDER), plot_wide,
                               x_col="x_value", x_label=x_label, error_suffix="_err",
                               algos=order, display_map=dmap)
    figs = save_figure(fig, out_dir / "plots" / f"performance_{benchmark}_1x5.pdf")
    import matplotlib.pyplot as plt; plt.close(fig)

    wide["benchmark"] = benchmark
    bb = best_baseline_table(wide, metric_cols=PERF_PANEL_ORDER, group_cols=["benchmark"])
    bb["best_baseline"] = bb["best_baseline"].map(display_name)
    disp = agg.copy()
    disp["algo"] = disp["algo"].map(display_name)
    write_table(bb, out_dir / "tables", f"table_performance_{benchmark}_best_baseline",
                caption=f"Highway vs best baseline ({benchmark})")
    write_table(disp, out_dir / "tables", f"table_performance_{benchmark}_dispersion",
                caption=f"Per-setting mean/std/CI ({benchmark})")
    print(f"[performance:{benchmark}] -> {figs[0]}  (+ 2 tables)")
    return long


# ---------------------------------------------------------------------------
# 2. scalability: runtime vs graph size
# ---------------------------------------------------------------------------
def run_scalability(corpus_root: Path, *, benchmark: str, out_dir: Path,
                    algos: Optional[Sequence[str]] = None, num_bins: int = 8,
                    timeout: Optional[float] = None, limit: Optional[int] = None) -> pd.DataFrame:
    # only runtime needed -> skip quality metrics for speed
    long = run_sweep(corpus_root, x_param="num_nodes", benchmark=benchmark, algos=algos,
                     metrics=[], ground_truth=None, timeout=timeout, limit=limit)
    rt = long[long["metric"] == "algo_runtime_sec"].copy()
    rt["runtime"] = pd.to_numeric(rt["value"], errors="coerce")
    rt = rt.dropna(subset=["runtime", "num_nodes"])
    if rt.empty:
        raise SystemExit("no runtime rows produced")
    order = _algo_order(long)
    dmap = _display_map(order)

    # bin by size; guard tiny corpora (qcut needs >= num_bins distinct values)
    nbins = min(num_bins, max(1, rt["num_nodes"].nunique()))
    binned = quantile_bins(rt, x_col="num_nodes", value_col="runtime", num_bins=nbins, agg="median")
    binned = binned.rename(columns={"value": "algo_runtime_sec"})

    out_dir = Path(out_dir)
    fig, _ = metric_panel_grid([METRICS["algo_runtime_sec"]], binned,
                               x_col="x_value", x_label="number of nodes",
                               algos=order, display_map=dmap, log_y=True,
                               figsize_per_panel=10.0)
    figs = save_figure(fig, out_dir / "plots" / f"scalability_{benchmark}_runtime_logy.pdf")
    import matplotlib.pyplot as plt; plt.close(fig)

    summary = binned.pivot_table(index="x_value", columns="algo", values="algo_runtime_sec").reset_index()
    summary.columns = ["num_nodes_bin"] + [display_name(c) for c in summary.columns[1:]]
    write_table(summary, out_dir / "tables", f"table_scalability_{benchmark}_runtime",
                caption=f"Median runtime by graph-size bin ({benchmark})")
    print(f"[scalability:{benchmark}] -> {figs[0]}  (+ 1 table)")
    return long


# ---------------------------------------------------------------------------
# 3. overlap: metric panel vs eta with seed error bands
# ---------------------------------------------------------------------------
def run_overlap(corpus_root: Path, *, benchmark: str, out_dir: Path,
                algos: Optional[Sequence[str]] = None,
                timeout: Optional[float] = None, limit: Optional[int] = None) -> pd.DataFrame:
    long = run_sweep(corpus_root, x_param="eta", benchmark=benchmark, algos=algos,
                     seed_param="seed", metrics=OVERLAP_PANEL_ORDER, timeout=timeout, limit=limit)
    if long.empty:
        raise SystemExit(f"no metric rows produced from {corpus_root}")
    wide = to_wide(long)
    order = _algo_order(long)
    dmap = _display_map(order)

    agg = aggregate_by_x(wide, x_col="x_value", metric_cols=OVERLAP_PANEL_ORDER)
    plot_wide = agg_to_plot_wide(agg)

    out_dir = Path(out_dir)
    fig, _ = metric_panel_grid(panel_specs(OVERLAP_PANEL_ORDER), plot_wide,
                               x_col="x_value", x_label=r"overlap $\eta$  (mean #comm / node)",
                               error_suffix="_err", algos=order, display_map=dmap)
    figs = save_figure(fig, out_dir / "plots" / f"overlap_{benchmark}_1x4.pdf")
    import matplotlib.pyplot as plt; plt.close(fig)

    wide["benchmark"] = benchmark
    bb = best_baseline_table(wide, metric_cols=OVERLAP_PANEL_ORDER, group_cols=["benchmark"])
    bb["best_baseline"] = bb["best_baseline"].map(display_name)
    write_table(bb, out_dir / "tables", f"table_overlap_{benchmark}_stability",
                caption=f"Overlap-sweep stability: Highway vs best baseline ({benchmark})")
    print(f"[overlap:{benchmark}] -> {figs[0]}  (+ 1 table, seeds aggregated for error bands)")
    return long


# ---------------------------------------------------------------------------
# 4. real-world: Q_ov + runtime on real graphs
# ---------------------------------------------------------------------------
def run_realworld(graph_paths: Sequence[Path], *, out_dir: Path,
                  algos: Optional[Sequence[str]] = None, timeout: Optional[float] = 300.0) -> pd.DataFrame:
    rows: List[dict] = []
    for gp in graph_paths:
        gp = Path(gp)
        with gp.open("rb") as f:
            G = pickle.load(f)
        dataset = gp.parent.name or gp.stem
        print(f"[realworld] {dataset}  (n={G.number_of_nodes()} m={G.number_of_edges()})")
        df = compare_algorithms(G, algos=algos, ground_truth=None,
                                metrics=["extended_modularity"], timeout=timeout)
        for _, r in df.iterrows():
            rows.append(dict(dataset=dataset, algo=r["algo"], algo_display=display_name(r["algo"]),
                             num_nodes=G.number_of_nodes(), num_edges=G.number_of_edges(),
                             extended_modularity=r.get("extended_modularity"),
                             runtime_sec=r.get("runtime_sec"), status=r["status"]))
    table = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    write_table(table, out_dir / "tables", "table_realworld_main",
                caption="Real-world overlapping modularity and runtime")

    # grouped-bar figure: Q_ov (left) + runtime log-y (right)
    ok = table[table["status"] == "ok"].copy()
    if not ok.empty:
        figs = _realworld_bar(ok, out_dir / "plots" / "realworld_qov_runtime.pdf")
        print(f"[realworld] -> {figs[0]}  (+ 1 table)")
    else:
        print("[realworld] all runs failed/timed out; table written, figure skipped")
    return table


def _realworld_bar(ok: pd.DataFrame, path: Path) -> List[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from ..paper_viz.style import RC_PARAMS, AXIS_LABEL_FONTSIZE, TICK_LABELSIZE, LEGEND_FONTSIZE

    plt.rcParams.update(RC_PARAMS)
    datasets = sorted(ok["dataset"].unique())
    algos = list(dict.fromkeys(ok["algo"].tolist()))
    base = [a for a in algos if a not in ("highway", "highwayfull")]
    order = base + [a for a in algos if a in ("highway", "highwayfull")]
    cmap = build_color_map(order)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, (col, ylab, logy) in zip(
            axes, [("extended_modularity", "Overlapping modularity", False),
                   ("runtime_sec", "Run time (s)", True)]):
        w = 0.8 / max(1, len(order))
        for i, algo in enumerate(order):
            sub = ok[ok["algo"] == algo].set_index("dataset")
            vals = [float(sub[col].get(d, np.nan)) if d in sub.index else np.nan for d in datasets]
            xs = np.arange(len(datasets)) + i * w
            edge = dict(edgecolor="black", linewidth=1.5) if algo in ("highway", "highwayfull") else {}
            ax.bar(xs, vals, width=w, color=cmap.get(algo, "gray"), label=display_name(algo), **edge)
        ax.set_xticks(np.arange(len(datasets)) + 0.4 - w / 2)
        ax.set_xticklabels(datasets, fontsize=TICK_LABELSIZE)
        ax.set_ylabel(ylab, fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABELSIZE)
        if logy:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=8, frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = save_figure(fig, path)
    plt.close(fig)
    return out
