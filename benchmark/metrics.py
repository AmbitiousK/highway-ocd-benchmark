# benchmark/metrics.py
"""Evaluation metrics for overlapping community detection.

Thin, faithful public wrapper over the frozen research implementations
(``metrics_impl.py`` + ``make_full_coverage.py``). Two entry points:

    ground_truth_cover(G)                  -> List[List[node]]   # read GT from graph contract
    evaluate(predicted, G, ground_truth)   -> Dict[str, float]   # the standard metric panel

Metric panel (same order/names as the paper figures):

    extended_modularity   Q_ov   (unsupervised; no ground truth needed)
    fri                   Fuzzy Rand Index            (needs ground truth)
    cover_sim_czekanowski Dice                        (needs ground truth)
    fstar_wo              F*                           (needs ground truth)
    onmi                  ONMI (LFK)                   (needs ground truth)

Ground truth lives in the graph contract as a per-node attribute:
    G.nodes[v]["communities"] = [community_id, ...]   (a node may belong to many)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Sequence

import networkx as nx
from cdlib.classes import NodeClustering

from .make_full_coverage import make_full_coverage
from .metrics_impl import (
    extended_modularity_custom,
    fuzzy_rand_index_custom,
    cover_similarity_czekanowski_custom,
    fstar_wo_custom,
    onmi_lfk_custom,
)

Node = Hashable
Cover = List[List[Node]]

# Canonical panel order (matches the paper figures).
METRIC_ORDER: List[str] = [
    "extended_modularity",
    "fri",
    "cover_sim_czekanowski",
    "fstar_wo",
    "onmi",
]
# Metrics that require ground truth.
GT_METRICS = {"fri", "cover_sim_czekanowski", "fstar_wo", "onmi"}


def _val(x) -> Optional[float]:
    """Coerce a metric return (bare float OR MetricResult with .value) to float|None."""
    if x is None:
        return None
    score = getattr(x, "value", getattr(x, "score", x))
    try:
        f = float(score)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


def ground_truth_cover(G: nx.Graph, attr: str = "communities") -> Optional[Cover]:
    """Read the ground-truth overlapping cover from the graph's node attributes.

    Returns a list of communities (each a list of node labels), or ``None`` if the
    graph carries no ground truth (e.g. a real-world graph with no labels).
    """
    by_comm: Dict[object, List[Node]] = defaultdict(list)
    found = False
    for v, data in G.nodes(data=True):
        labels = data.get(attr)
        if labels is None:
            continue
        found = True
        if isinstance(labels, (list, tuple, set)):
            for c in labels:
                by_comm[c].append(v)
        else:
            by_comm[labels].append(v)
    if not found:
        return None
    return [by_comm[c] for c in sorted(by_comm, key=str)]


def _as_clustering(cover: Cover, G: nx.Graph, name: str) -> NodeClustering:
    return NodeClustering(communities=[list(c) for c in cover],
                          graph=G, method_name=name, overlap=True)


def evaluate(
    predicted: Cover,
    G: nx.Graph,
    ground_truth: Optional[Cover] = None,
    *,
    metrics: Optional[Sequence[str]] = None,
    full_coverage_policy: str = "singleton",
) -> Dict[str, Optional[float]]:
    """Compute the metric panel for one predicted cover on graph ``G``.

    Args:
        predicted: detected overlapping communities (list of node lists).
        G: the graph the cover is on (provides degrees + node set).
        ground_truth: optional GT cover. If ``None`` only unsupervised metrics
            (extended_modularity / Q_ov) are computed; GT metrics return ``None``.
        metrics: subset of METRIC_ORDER to compute (default: all available).
        full_coverage_policy: how Q_ov treats nodes left in no community
            ("singleton" | "null_community" | "drop_externals").

    Returns:
        dict {metric_name: float | None}. ``None`` means not-applicable
        (e.g. a GT metric with no ground truth) or a computation failure.
    """
    want = list(metrics) if metrics is not None else list(METRIC_ORDER)
    out: Dict[str, Optional[float]] = {m: None for m in want}

    pred_nc = _as_clustering(predicted, G, "predicted")
    gt_nc = _as_clustering(ground_truth, G, "ground_truth") if ground_truth else None

    if "extended_modularity" in want:
        try:
            part_full = make_full_coverage(pred_nc, G, policy=full_coverage_policy)
            out["extended_modularity"] = _val(extended_modularity_custom(part_full, G))
        except Exception:
            out["extended_modularity"] = None

    if gt_nc is not None:
        gt_fns = {
            "fri": fuzzy_rand_index_custom,
            "cover_sim_czekanowski": cover_similarity_czekanowski_custom,
            "fstar_wo": fstar_wo_custom,
            "onmi": onmi_lfk_custom,
        }
        for name, fn in gt_fns.items():
            if name not in want:
                continue
            try:
                out[name] = _val(fn(gt_nc, pred_nc))
            except Exception:
                out[name] = None

    return out
