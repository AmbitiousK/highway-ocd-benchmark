# benchmark/__init__.py
"""Highway OCD Benchmark — compare overlapping community detection algorithms.

Quick start:

    import pickle, networkx as nx
    from benchmark import compare_algorithms

    G = pickle.load(open("data/synthetic/lfr_small/graph.gpickle", "rb"))
    df = compare_algorithms(G)        # baselines + Highway, auto ground truth
    print(df)

Add your own algorithm:

    from benchmark import register_algorithm
    register_algorithm("my_ocd", lambda G: [[1, 2, 3], [3, 4, 5]])
    compare_algorithms(G, algos=["highway", "my_ocd"])
"""
from __future__ import annotations

from .algorithms import (
    register_algorithm,
    unregister_algorithm,
    available_algorithms,
    get_algorithm,
    DEFAULT_BASELINES,
)
from .compare import compare_algorithms
from .metrics import evaluate, ground_truth_cover, METRIC_ORDER

__all__ = [
    "compare_algorithms",
    "register_algorithm",
    "unregister_algorithm",
    "available_algorithms",
    "get_algorithm",
    "evaluate",
    "ground_truth_cover",
    "METRIC_ORDER",
    "DEFAULT_BASELINES",
]
