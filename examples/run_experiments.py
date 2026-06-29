#!/usr/bin/env python
"""Run the full benchmark experiment suite and write paper-style figures + tables.

Each experiment runs ALL algorithms live (Highway + baselines + anything you
registered) over a graph corpus, then renders the figure and the numeric tables.

    # all synthetic experiments on the shipped corpora
    python examples/run_experiments.py

    # pick experiments / algorithms / a real-world corpus
    python examples/run_experiments.py --only performance,overlap
    python examples/run_experiments.py --algos highway,slpa,demon,lfm --timeout 120
    python examples/run_experiments.py --only realworld \
        --realworld /path/to/processed/realworld_nx

Outputs: results/experiments/plots/*.pdf|png  and  results/experiments/tables/*.csv|tex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.experiments import (  # noqa: E402
    run_performance, run_scalability, run_overlap, run_realworld,
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "synthetic"
OUT = REPO / "results" / "experiments"
ALL = ["performance", "scalability", "overlap", "realworld"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(ALL), help=f"subset of {ALL}")
    ap.add_argument("--algos", default=None, help="comma list (default: all built-ins + custom)")
    ap.add_argument("--timeout", type=float, default=120.0, help="per-algorithm seconds")
    ap.add_argument("--realworld", default=None,
                    help="dir of real-world graph.gpickle files (for the realworld experiment)")
    args = ap.parse_args()

    which = [w.strip() for w in args.only.split(",")]
    algos = args.algos.split(",") if args.algos else None

    if "performance" in which:
        print("\n##### PERFORMANCE (metric panel vs mixing muw) #####")
        run_performance(DATA / "performance_lfr", benchmark="lfr", x_param="muw",
                        x_label=r"$\mu_w$", out_dir=OUT, algos=algos, timeout=args.timeout)

    if "scalability" in which:
        print("\n##### SCALABILITY (runtime vs graph size) #####")
        run_scalability(DATA / "scalability_lfr", benchmark="lfr", out_dir=OUT,
                        algos=algos, timeout=args.timeout)

    if "overlap" in which:
        print("\n##### OVERLAPPING SWEEP (metric panel vs overlap eta, seed error bands) #####")
        run_overlap(DATA / "overlap_abcdo2", benchmark="abcdo2", out_dir=OUT,
                    algos=algos, timeout=args.timeout)

    if "realworld" in which:
        print("\n##### REAL-WORLD (Q_ov + runtime) #####")
        rw_root = Path(args.realworld) if args.realworld else None
        if rw_root and rw_root.exists():
            graphs = sorted(rw_root.rglob("graph.gpickle"))
        else:
            # fall back to the shipped small graphs so the command always runs
            graphs = [DATA / "lfr_N200" / "graph.gpickle", DATA / "abcdo2_N200" / "graph.gpickle"]
            print(f"  (no --realworld dir given; demoing on shipped small graphs: {[g.parent.name for g in graphs]})")
        run_realworld(graphs, out_dir=OUT, algos=algos, timeout=args.timeout)

    print(f"\nDone. Figures + tables under {OUT}")


if __name__ == "__main__":
    main()
