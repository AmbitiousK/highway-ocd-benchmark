#!/usr/bin/env python
"""One-click real-world benchmark on a SNAP-style graph.

Real-world graphs are large and not shipped in this repo. Point this at a
pickled networkx graph (``graph.gpickle``). Real-world graphs usually have no
ground-truth cover, so only the unsupervised metric (Q_ov) + runtime are
reported, with a per-algorithm hard timeout.

    python examples/run_realworld.py --graph /path/to/graph.gpickle
    python examples/run_realworld.py --graph g.gpickle --timeout 300 \
        --algos highway,slpa,demon,kclique

See README "Real-world data" for how to obtain the SNAP datasets
(Amazon / DBLP / YouTube) used in the paper.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import compare_algorithms  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# scalable subset: the seq-graph baselines (conga/congo/lais2) are infeasible on
# large graphs even with a timeout.
DEFAULT_RW_ALGOS = ["highway", "slpa", "demon", "kclique", "lfm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="path to a pickled networkx graph")
    ap.add_argument("--algos", default=",".join(DEFAULT_RW_ALGOS))
    ap.add_argument("--timeout", type=float, default=300.0, help="per-algorithm seconds")
    ap.add_argument("--out", default=str(REPO / "results"))
    args = ap.parse_args()

    gpath = Path(args.graph)
    with gpath.open("rb") as f:
        G = pickle.load(f)
    print(f"=== {gpath.stem}  (n={G.number_of_nodes()} m={G.number_of_edges()}) ===")

    df = compare_algorithms(
        G,
        algos=args.algos.split(","),
        ground_truth=None,            # real-world: unsupervised only
        metrics=["extended_modularity"],
        timeout=args.timeout,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"realworld_{gpath.parent.name or gpath.stem}.csv"
    df.to_csv(csv, index=False)
    print(f"\n{df.to_string(index=False)}\n-> {csv}")


if __name__ == "__main__":
    main()
