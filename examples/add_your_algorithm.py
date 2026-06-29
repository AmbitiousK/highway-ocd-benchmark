#!/usr/bin/env python
"""TEMPLATE — benchmark YOUR overlapping community detection algorithm.

Copy this file, drop your method into ``my_algorithm`` (it just needs to take a
networkx graph and return a list of communities), and run it. Your algorithm is
scored side by side with Highway and the baselines on the same metrics.

    python examples/add_your_algorithm.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx  # noqa: E402
from benchmark import compare_algorithms, register_algorithm  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Implement your algorithm.
#    Contract:  my_algorithm(G: nx.Graph) -> List[List[node]]
#    Each inner list is one community; communities may overlap (a node can
#    appear in several). Node labels must be the graph's own node labels.
# ---------------------------------------------------------------------------
def my_algorithm(G: nx.Graph):
    # ---- replace this toy body with your real method ----
    communities = []
    for cc in nx.connected_components(G):
        communities.append(list(cc))
    return communities


# ---------------------------------------------------------------------------
# 2. Register it (do this once).
# ---------------------------------------------------------------------------
register_algorithm("my_algorithm", my_algorithm, note="my new OCD method")


# ---------------------------------------------------------------------------
# 3. Compare against Highway + baselines on a ground-truth graph.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gpath = REPO / "data" / "synthetic" / "lfr_N200" / "graph.gpickle"
    with gpath.open("rb") as f:
        G = pickle.load(f)

    df = compare_algorithms(
        G,
        algos=["highway", "slpa", "demon", "lfm", "my_algorithm"],
    )
    print("\n", df.to_string(index=False), sep="")
