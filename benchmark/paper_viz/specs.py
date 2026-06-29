# benchmark/paper_viz/specs.py
"""Metric / algorithm display specs for the paper-style figures and tables.

Self-contained source of truth for: metric display names + optimization
direction, the canonical panel order, and which algorithms render as the
highlighted "Highway" series. Decoupled from any path/context machinery so the
visualization layer is portable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class MetricSpec:
    column: str                      # column name in the metric table
    display: str                     # axis / header label (may be LaTeX)
    higher_is_better: bool = True
    requires_ground_truth: bool = True


METRICS: Dict[str, MetricSpec] = {
    "extended_modularity":   MetricSpec("extended_modularity", "Overlapping modularity",
                                        higher_is_better=True, requires_ground_truth=False),
    "fri":                   MetricSpec("fri", "FRI", higher_is_better=True),
    "cover_sim_czekanowski": MetricSpec("cover_sim_czekanowski", "Dice", higher_is_better=True),
    "fstar_wo":              MetricSpec("fstar_wo", r"$F^\ast$", higher_is_better=True),
    "onmi":                  MetricSpec("onmi", "ONMI", higher_is_better=True),
    "algo_runtime_sec":      MetricSpec("algo_runtime_sec", "Run time (s)",
                                        higher_is_better=False, requires_ground_truth=False),
}

# Canonical 1x5 performance panel (matches the paper figures).
PERF_PANEL_ORDER: List[str] = [
    "extended_modularity", "fri", "cover_sim_czekanowski", "fstar_wo", "onmi",
]
# Overlap sweep panel: same minus F* (not computed on that corpus historically).
OVERLAP_PANEL_ORDER: List[str] = [
    "extended_modularity", "fri", "cover_sim_czekanowski", "onmi",
]


def panel_specs(metric_ids: Optional[Sequence[str]] = None) -> List[MetricSpec]:
    ids = list(metric_ids) if metric_ids is not None else list(PERF_PANEL_ORDER)
    return [METRICS[m] for m in ids if m in METRICS]


# ---------------------------------------------------------------------------
# algorithm highlighting (the Highway series renders bold + crimson)
# ---------------------------------------------------------------------------
# Algorithms whose name starts with one of these render as highlighted.
HIGHLIGHT_ALGOS = {"highway", "highwayfull"}
# Fixed colors for the highlighted series.
HIGHLIGHT_COLORS = {"highway": "crimson", "highwayfull": "black"}


def is_highlight(algo: str) -> bool:
    return algo in HIGHLIGHT_ALGOS


def is_highway(algo: str) -> bool:
    """Whether an algorithm counts as a Highway variant (excluded from baselines)."""
    return algo in HIGHLIGHT_ALGOS


# Display names for the built-in algorithms (others fall back to the raw key).
DISPLAY_NAMES: Dict[str, str] = {
    "highway": "Highway", "highwayfull": "HighwayFull",
    "slpa": "SLPA", "demon": "Demon", "kclique": "Kclique", "walkscan": "Walkscan",
    "conga": "Conga", "congo": "Congo", "lais2": "Lais2", "lfm": "LFM",
}


def display_name(algo: str) -> str:
    return DISPLAY_NAMES.get(algo, algo)
