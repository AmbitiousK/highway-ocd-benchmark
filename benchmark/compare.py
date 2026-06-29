# benchmark/compare.py
"""Run many overlapping-community-detection algorithms on a graph and score them.

The single entry point most users need:

    from benchmark import compare_algorithms
    df = compare_algorithms(G)                       # all built-ins + your registered algos
    df = compare_algorithms(G, algos=["highway", "slpa", "my_ocd"])

Returns a tidy pandas DataFrame: one row per algorithm with status, runtime,
community count, and the metric panel (Q_ov + GT metrics when ground truth exists).

Ground truth is read automatically from the graph contract
(``G.nodes[v]["communities"]``); pass ``ground_truth=...`` to override, or
``ground_truth=None`` to force unsupervised-only (Q_ov).
"""
from __future__ import annotations

import time
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import networkx as nx
import pandas as pd

from .algorithms import (
    DEFAULT_BASELINES,
    HIGHWAY,
    available_algorithms,
    get_algorithm,
)
from .metrics import METRIC_ORDER, evaluate, ground_truth_cover

Node = Hashable
Cover = List[List[Node]]

_AUTO = object()  # sentinel: read ground truth from the graph


class _Timeout(Exception):
    pass


def _run_with_timeout(fn, G: nx.Graph, timeout: Optional[float]) -> Tuple[Optional[Cover], Optional[str], bool, float]:
    """Run ``fn(G)`` with an optional per-algorithm timeout.

    Uses a cheap in-process SIGALRM alarm (no subprocess, no re-import) so a
    hanging baseline (e.g. cdlib LFM on a low-mixing graph) is interrupted and
    recorded as a timeout instead of wedging the whole sweep.

    Returns (cover, error, timed_out, runtime_sec). The alarm needs the main
    thread on a POSIX system; if unavailable (Windows, worker thread) the call
    runs without a timeout.
    """
    import signal

    t0 = time.perf_counter()
    can_alarm = bool(timeout) and hasattr(signal, "SIGALRM")

    if not can_alarm:
        try:
            return fn(G), None, False, time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}", False, time.perf_counter() - t0

    def _handler(signum, frame):
        raise _Timeout()

    try:
        old = signal.signal(signal.SIGALRM, _handler)
    except ValueError:
        # not in the main thread -> can't arm the alarm; run without timeout
        try:
            return fn(G), None, False, time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}", False, time.perf_counter() - t0

    signal.setitimer(signal.ITIMER_REAL, float(timeout))
    try:
        cover = fn(G)
        return cover, None, False, time.perf_counter() - t0
    except _Timeout:
        return None, f"timed out after {timeout:.0f}s", True, time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}", False, time.perf_counter() - t0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def compare_algorithms(
    G: nx.Graph,
    *,
    algos: Optional[Sequence[str]] = None,
    ground_truth=_AUTO,
    metrics: Optional[Sequence[str]] = None,
    timeout: Optional[float] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Benchmark algorithms on ``G`` and return a scored results table.

    Args:
        G: input graph (undirected; may carry edge ``weight`` and node
            ``communities`` ground-truth attributes).
        algos: which algorithms to run. Default = all built-in baselines +
            Highway + anything you registered via ``register_algorithm``.
        ground_truth: ``"auto"`` (read from graph, default), an explicit cover
            (list of node lists), or ``None`` to force unsupervised-only.
        metrics: subset of the metric panel (default: all).
        timeout: optional per-algorithm hard timeout in seconds (real-world graphs).
        verbose: print a one-line progress note per algorithm.

    Returns:
        pandas.DataFrame, one row per algorithm.
    """
    if algos is None:
        builtins = [*DEFAULT_BASELINES, HIGHWAY]
        extras = [a for a in available_algorithms() if a not in builtins]
        algos = builtins + extras

    if ground_truth is _AUTO:
        gt = ground_truth_cover(G)
    else:
        gt = ground_truth

    want_metrics = list(metrics) if metrics is not None else list(METRIC_ORDER)

    rows: List[Dict] = []
    for name in algos:
        spec = get_algorithm(name)
        cover, err, timed_out, runtime = _run_with_timeout(spec.fn, G, timeout)

        row: Dict[str, object] = {
            "algo": name,
            "kind": spec.kind,
            "status": "ok" if err is None else ("timeout" if timed_out else "failed"),
            "num_communities": len(cover) if cover is not None else None,
            "runtime_sec": round(runtime, 4),
            "error": err,
        }
        if cover is not None:
            scores = evaluate(cover, G, gt, metrics=want_metrics)
            row.update(scores)
        else:
            row.update({m: None for m in want_metrics})
        rows.append(row)

        if verbose:
            if err is None:
                q = row.get("extended_modularity")
                qs = f"Q_ov={q:.4f}" if isinstance(q, float) else "Q_ov=NA"
                print(f"  [ok]    {name:<10} {row['num_communities']:>5} comms  {qs}  {runtime:.2f}s")
            else:
                print(f"  [{row['status']:<6}] {name:<10} {err[:80]}")

    col_order = ["algo", "kind", "status", "num_communities", "runtime_sec",
                 *want_metrics, "error"]
    df = pd.DataFrame(rows)
    return df[[c for c in col_order if c in df.columns]]
