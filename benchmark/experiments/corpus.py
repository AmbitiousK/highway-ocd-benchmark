# benchmark/experiments/corpus.py
"""Load a corpus of benchmark graphs and read each graph's sweep coordinates.

A corpus is just a directory tree of ``graph.gpickle`` files (the data_module
output layout). Each graph is self-describing: its generator parameters live in
``G.graph["params"]`` (e.g. ``muw`` for LFR, ``xi`` / ``eta`` / ``seed`` for
ABCD+o²), and ground truth lives in node attributes. So sweeps need no external
index — we read the x-axis value straight off the graph.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional

import networkx as nx


@dataclass
class CorpusGraph:
    graph_id: str
    path: Path
    G: nx.Graph
    params: Dict[str, object] = field(default_factory=dict)
    num_nodes: int = 0
    num_edges: int = 0

    def number(self, key: str) -> Optional[float]:
        """Read a numeric parameter (e.g. 'muw', 'xi', 'eta', 'seed') as float."""
        v = self.params.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None


def _flatten_params(raw: object) -> Dict[str, object]:
    """params may be a flat dict (LFR) or nested {'config': {...}, ...} (ABCD+o²).
    Flatten top-level scalar keys; nested 'config' values are merged underneath."""
    out: Dict[str, object] = {}
    if isinstance(raw, dict):
        cfg = raw.get("config")
        if isinstance(cfg, dict):
            out.update({k: v for k, v in cfg.items()})
        out.update({k: v for k, v in raw.items() if not isinstance(v, dict)})
    return out


def load_graph(path: Path) -> CorpusGraph:
    with Path(path).open("rb") as f:
        G = pickle.load(f)
    params = _flatten_params(G.graph.get("params", {}))
    gid = str(params.get("graph_id") or Path(path).parent.name)
    return CorpusGraph(graph_id=gid, path=Path(path), G=G, params=params,
                       num_nodes=G.number_of_nodes(), num_edges=G.number_of_edges())


def iter_corpus(root: Path, *, limit: Optional[int] = None) -> Iterator[CorpusGraph]:
    """Yield every graph under ``root`` (recursively), optionally capped at ``limit``."""
    paths = sorted(Path(root).rglob("graph.gpickle"))
    if limit is not None:
        paths = paths[:limit]
    for p in paths:
        yield load_graph(p)
