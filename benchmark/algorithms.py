# benchmark/algorithms.py
"""Algorithm registry for the overlapping-community-detection benchmark.

Every algorithm is a callable with a uniform contract:

    fn(G: nx.Graph) -> List[List[node]]     # a list of (possibly overlapping) communities

Built-in algorithms (zero external dependencies):
    - 8 cdlib baselines: slpa, demon, kclique, walkscan, conga, congo, lais2, lfm
    - highway: the vendored C++ Highway backend

To benchmark YOUR OWN algorithm, register it once and it shows up everywhere:

    from benchmark.algorithms import register_algorithm

    def my_ocd(G):
        # ... your method ...
        return [[1, 2, 3], [3, 4, 5]]     # overlapping communities

    register_algorithm("my_ocd", my_ocd)

External-binary baselines (COPRA, BigClam) are intentionally NOT built in — they
need compiled third-party binaries. Enable them by registering a thin wrapper that
shells out to your binary (see README "Optional external baselines").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, List, Optional

import networkx as nx

Node = Hashable
Cover = List[List[Node]]
AlgoFn = Callable[[nx.Graph], Cover]


# ---------------------------------------------------------------------------
# graph preprocessing (faithful to the original research harness)
# ---------------------------------------------------------------------------

def _make_seq_graph(G: nx.Graph) -> nx.Graph:
    """Deterministic copy with sorted nodes; keeps edge attributes (incl. weight)."""
    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes(), key=str))
    H.add_edges_from(G.edges(data=True))
    return H


def _as_unweighted(G: nx.Graph) -> nx.Graph:
    """Strip edge weights (some methods are unweighted-only) but keep topology."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from((u, v) for u, v in G.edges())
    return H


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmSpec:
    name: str
    fn: AlgoFn
    kind: str = "baseline"        # "baseline" | "highway" | "custom"
    note: str = ""


_REGISTRY: Dict[str, AlgorithmSpec] = {}


def register_algorithm(name: str, fn: AlgoFn, *, kind: str = "custom",
                       note: str = "", overwrite: bool = False) -> None:
    """Register an algorithm so the benchmark runners pick it up.

    ``fn`` takes an ``nx.Graph`` and returns a list of communities
    (each community a list of node labels). Communities may overlap.
    """
    if name in _REGISTRY and not overwrite:
        raise ValueError(
            f"algorithm '{name}' already registered; pass overwrite=True to replace it."
        )
    _REGISTRY[name] = AlgorithmSpec(name=name, fn=fn, kind=kind, note=note)


def unregister_algorithm(name: str) -> None:
    _REGISTRY.pop(name, None)


def available_algorithms() -> List[str]:
    return list(_REGISTRY.keys())


def get_algorithm(name: str) -> AlgorithmSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown algorithm '{name}'; available: {available_algorithms()}")
    return _REGISTRY[name]


# ---------------------------------------------------------------------------
# built-in cdlib baselines  (each: nx.Graph -> List[List[node]])
# ---------------------------------------------------------------------------

def _cdlib(method: str, *, seq: bool = False, unweighted: bool = False, **kwargs) -> AlgoFn:
    """Wrap a cdlib.algorithms.<method> into the uniform list-of-lists contract."""
    def run(G: nx.Graph) -> Cover:
        from cdlib import algorithms as A
        H = G
        if seq:
            H = _make_seq_graph(H)
        if unweighted:
            H = _as_unweighted(H)
        nc = getattr(A, method)(H, **kwargs)
        return nc.communities
    return run


def _highway(G: nx.Graph) -> Cover:
    from .highway import highway_nx, is_built, build
    if not is_built():
        build()  # compile the C++ backend on first use
    return highway_nx(G)


def _register_builtins() -> None:
    # cdlib baselines — defaults match the original research harness.
    register_algorithm("slpa",     _cdlib("slpa", t=100, r=0.05), kind="baseline",
                        note="Speaker-Listener LPA (cdlib)")
    register_algorithm("demon",    _cdlib("demon", epsilon=0.25, min_com_size=3), kind="baseline",
                        note="DEMON (cdlib)")
    register_algorithm("kclique",  _cdlib("kclique", k=3), kind="baseline",
                        note="k-clique percolation (cdlib)")
    register_algorithm("walkscan", _cdlib("walkscan", nb_steps=5), kind="baseline",
                        note="WalkSCAN (cdlib)")
    register_algorithm("conga",    _cdlib("conga", seq=True, number_communities=5), kind="baseline",
                        note="CONGA (cdlib); needs sequential graph")
    register_algorithm("congo",    _cdlib("congo", seq=True, number_communities=5, height=2), kind="baseline",
                        note="CONGO (cdlib); needs sequential graph")
    register_algorithm("lais2",    _cdlib("lais2", seq=True), kind="baseline",
                        note="LAIS2 (cdlib); needs sequential graph")
    register_algorithm("lfm",      _cdlib("lfm", unweighted=True, alpha=1.0), kind="baseline",
                        note="LFM (cdlib); unweighted")
    # Highway (vendored C++).
    register_algorithm("highway", _highway, kind="highway",
                        note="Highway overlapping community detection (vendored C++ backend)")


# Default baseline set used by the runners when none is specified.
DEFAULT_BASELINES: List[str] = [
    "slpa", "demon", "kclique", "walkscan", "conga", "congo", "lais2", "lfm",
]
HIGHWAY: str = "highway"

_register_builtins()
