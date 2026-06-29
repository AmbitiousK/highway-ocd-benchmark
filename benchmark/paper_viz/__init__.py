# benchmark/paper_viz/__init__.py
"""Paper-style visualization + aggregation layer (vendored from performance_module,
decoupled from any path/context machinery).
"""
from __future__ import annotations

from .specs import (
    MetricSpec, METRICS, PERF_PANEL_ORDER, OVERLAP_PANEL_ORDER,
    panel_specs, is_highlight, is_highway,
)
from .style import metric_panel_grid, draw_metric_panel, save_figure, build_color_map
from .aggregation import aggregate_by_x, best_baseline_table, quantile_bins

__all__ = [
    "MetricSpec", "METRICS", "PERF_PANEL_ORDER", "OVERLAP_PANEL_ORDER",
    "panel_specs", "is_highlight", "is_highway",
    "metric_panel_grid", "draw_metric_panel", "save_figure", "build_color_map",
    "aggregate_by_x", "best_baseline_table", "quantile_bins",
]
