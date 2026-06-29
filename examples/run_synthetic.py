#!/usr/bin/env python
"""One-click synthetic benchmark: run every algorithm on the shipped LFR / ABCD+o2
graphs (which carry ground truth) and score the full 5-metric panel.

    python examples/run_synthetic.py
    python examples/run_synthetic.py --algos highway,slpa,lfm
    python examples/run_synthetic.py --data data/synthetic/lfr_N200

Writes one CSV per graph to results/.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# make `benchmark` importable when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import compare_algorithms  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data" / "synthetic"),
                    help="a synthetic dataset dir, or a parent dir of several")
    ap.add_argument("--algos", default=None, help="comma list (default: all built-ins + custom)")
    ap.add_argument("--out", default=str(REPO / "results"))
    args = ap.parse_args()

    algos = args.algos.split(",") if args.algos else None
    data_root = Path(args.data)
    graphs = sorted(data_root.rglob("graph.gpickle"))
    if not graphs:
        raise SystemExit(f"no graph.gpickle found under {data_root}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for gpath in graphs:
        name = gpath.parent.name
        with gpath.open("rb") as f:
            G = pickle.load(f)
        print(f"\n=== {name}  (n={G.number_of_nodes()} m={G.number_of_edges()}) ===")
        df = compare_algorithms(G, algos=algos)
        csv = out_dir / f"synthetic_{name}.csv"
        df.to_csv(csv, index=False)
        print(f"  -> {csv}")


if __name__ == "__main__":
    main()
