# benchmark/paper_viz/style.py
"""Single source of visual style for the benchmark figures.

Vendored and lightly adapted from the HighwayOCD ``performance_module`` so the
figures produced here use the same visual language as the paper: large fonts,
1xN metric-panel grids, the Highway series highlighted in crimson, seed error
bands, and pdf+png output.

Decoupled from any global algorithm registry: colors are assigned
deterministically from the *order of algorithms passed in*, so a newly
registered algorithm (e.g. a reviewer's own method) gets a stable color too.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .specs import MetricSpec, HIGHLIGHT_COLORS, is_highlight

# Global RC (inherits the paper's large font sizes).
RC_PARAMS = {
    "font.size": 20,
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
}
AXIS_LABEL_FONTSIZE = 36
TICK_LABELSIZE = 30
LEGEND_FONTSIZE = 30


def get_style(algo: str) -> dict:
    """Highway series: bold, high zorder, big marker. Baselines: semi-transparent."""
    if is_highlight(algo):
        return dict(alpha=1.0, linewidth=3.8, markersize=8.5, zorder=6)
    return dict(alpha=0.60, linewidth=2.5, markersize=6.8, zorder=2)


def build_color_map(order: Sequence[str]) -> Dict[str, object]:
    """Deterministic colors: highlighted series get fixed colors; the rest take
    ``tab20`` by position in ``order`` (so colors are stable for a given run)."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    colors: Dict[str, object] = {}
    i = 0
    for algo in order:
        if algo in HIGHLIGHT_COLORS:
            colors[algo] = HIGHLIGHT_COLORS[algo]
        else:
            colors[algo] = cmap(i % 20)
            i += 1
    return colors


def draw_metric_panel(
    ax,
    df,
    *,
    metric_col: str,
    y_label: str,
    x_col: str,
    x_label: Optional[str] = None,
    error_col: Optional[str] = None,
    algos: Optional[Sequence[str]] = None,
    display_map: Optional[dict] = None,
    color_map: Optional[dict] = None,
) -> None:
    """Plot one metric panel: one curve per algorithm (in ``algos`` order); if
    ``error_col`` is present draw a ±band (seed std / CI)."""
    order = list(algos) if algos is not None else sorted(df["algo"].unique())
    disp = display_map or {a: a for a in order}
    cmap = color_map or build_color_map(order)

    for algo in order:
        sub = df[df["algo"] == algo]
        if sub.empty or metric_col not in sub.columns:
            continue
        sub = sub.dropna(subset=[x_col, metric_col]).sort_values(x_col)
        if sub.empty:
            continue
        st = get_style(algo)
        c = cmap.get(algo, "gray")
        ax.plot(sub[x_col], sub[metric_col], marker="o",
                linewidth=st["linewidth"], markersize=st["markersize"],
                alpha=st["alpha"], zorder=st["zorder"], color=c, label=disp.get(algo, algo))
        if error_col and error_col in sub.columns:
            lo = sub[metric_col] - sub[error_col]
            hi = sub[metric_col] + sub[error_col]
            ax.fill_between(sub[x_col], lo, hi, color=c, alpha=0.12, zorder=st["zorder"] - 1)

    ax.set_xlabel(x_label if x_label is not None else x_col, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    ax.grid(alpha=0.3)


def metric_panel_grid(
    panel_specs: List[MetricSpec],
    df,
    *,
    x_col: str,
    x_label: str,
    error_col: Optional[str] = None,
    error_suffix: Optional[str] = None,
    algos: Optional[Sequence[str]] = None,
    display_map: Optional[dict] = None,
    figsize_per_panel: float = 9.0,
    panel_height: float = 7.8,
    log_y: bool = False,
    x_ticks: Optional[Sequence[float]] = None,
    x_ticklabels: Optional[Sequence[str]] = None,
    x_lim: Optional[tuple] = None,
):
    """1xN metric grid + one shared legend. Returns (fig, axes); does not save."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(RC_PARAMS)
    order = list(algos) if algos is not None else sorted(df["algo"].unique())
    cmap = build_color_map(order)
    n = len(panel_specs)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel * n, panel_height))
    if n == 1:
        axes = [axes]

    for ax, spec in zip(axes, panel_specs):
        ecol = error_col
        if error_suffix is not None:
            cand = f"{spec.column}{error_suffix}"
            ecol = cand if cand in getattr(df, "columns", []) else None
        draw_metric_panel(ax, df, metric_col=spec.column, y_label=spec.display,
                          x_col=x_col, x_label=x_label, error_col=ecol,
                          algos=order, display_map=display_map, color_map=cmap)
        if x_ticks is not None:
            ax.set_xticks(list(x_ticks))
            if x_ticklabels is not None:
                ax.set_xticklabels(list(x_ticklabels))
        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if log_y:
            ax.set_yscale("log")

    handles, labels, seen = [], [], set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h); labels.append(l); seen.add(l)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05),
               ncol=6, frameon=False, fontsize=LEGEND_FONTSIZE,
               handlelength=2.3, columnspacing=1.5)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    return fig, axes


def save_figure(fig, path: Path, *, also_png: bool = True) -> List[Path]:
    """Save pdf (+png). Returns the paths written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = [p]
    fig.savefig(p, bbox_inches="tight", pad_inches=0.14)
    if also_png and p.suffix == ".pdf":
        png = p.with_suffix(".png")
        fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.14)
        out.append(png)
    return out
