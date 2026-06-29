from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Union

import networkx as nx
from cdlib.classes import NodeClustering


def make_full_coverage(
    clu: NodeClustering,
    G: nx.Graph,
    *,
    policy: str = "singleton",            # {"singleton", "null_community", "drop_externals"}
    null_comm_label: str = "__NULL__",     # only used if policy="null_community"
    dedup_within_community: bool = True,
    drop_nodes_not_in_G: bool = True,
    keep_original_graph: bool = False,     # if True, keep clu.graph if compatible
    sort_singletons: bool = True,
    method_suffix: str = "_full",
) -> NodeClustering:
    """
    Ensure full node coverage for evaluation: every node in G appears in at least one community.

    This function returns a NEW NodeClustering (does not mutate `clu`).

    Parameters
    ----------
    clu:
        Input NodeClustering (possibly overlapping).
    G:
        Target graph defining the node universe.
    policy:
        - "singleton": add one singleton community [v] for each uncovered node v (default; backward compatible)
        - "null_community": add ONE extra community containing all uncovered nodes (reduces community explosion)
        - "drop_externals": remove nodes from communities if they are not in G; do NOT add missing nodes
    null_comm_label:
        Only used for policy="null_community". (Note: label is for doc/meta; community is still a list of nodes.)
    dedup_within_community:
        If True, remove duplicates inside each community while preserving order.
    drop_nodes_not_in_G:
        If True, remove nodes from communities that are not in G.
    keep_original_graph:
        If True and clu.graph is compatible with G (same node set), keep clu.graph reference.
        Otherwise set graph=G.
    sort_singletons:
        If True, add singleton communities in sorted node order for reproducibility.
    method_suffix:
        Suffix appended to method_name to reflect transformation.

    Returns
    -------
    NodeClustering:
        New clustering with guaranteed full coverage under the chosen policy.

    Notes (research-grade)
    ----------------------
    - "singleton" policy can inflate community count and bias some cover metrics.
      Use "null_community" for large-scale performance evaluation unless singleton coverage is required.
    - This function deliberately does not attempt to infer "overlap" beyond setting overlap=True,
      because adding coverage nodes makes overlap semantics ambiguous; treat as cover by default.
    """

    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx.Graph (or subclass).")

    all_nodes: Set[object] = set(G.nodes())
    if len(all_nodes) == 0:
        # Empty graph: return empty communities safely
        return NodeClustering(
            communities=[],
            graph=G,
            method_name=(getattr(clu, "method_name", "unknown") or "unknown") + method_suffix,
            overlap=True,
        )

    # ---- 1) Normalize / sanitize communities ----
    new_comms: List[List[object]] = []
    covered: Set[object] = set()

    for com in clu.communities:
        # Convert community to list (cdlib may store list already)
        if com is None:
            continue
        com_list = list(com)

        if drop_nodes_not_in_G:
            com_list = [u for u in com_list if u in all_nodes]

        if dedup_within_community:
            # stable de-dup
            seen = set()
            deduped = []
            for u in com_list:
                if u in seen:
                    continue
                seen.add(u)
                deduped.append(u)
            com_list = deduped

        if len(com_list) == 0:
            continue

        new_comms.append(com_list)
        covered.update(com_list)

    # ---- 2) Determine missing nodes ----
    missing = list(all_nodes - covered)

    # ---- 3) Apply policy ----
    if policy == "singleton":
        if sort_singletons:
            try:
                missing = sorted(missing)
            except TypeError:
                # nodes not sortable (mixed types) -> keep arbitrary but deterministic-ish order
                missing = list(missing)

        for v in missing:
            new_comms.append([v])

    elif policy == "null_community":
        # Add ONE extra community containing all missing nodes
        if len(missing) > 0:
            if sort_singletons:
                try:
                    missing = sorted(missing)
                except TypeError:
                    missing = list(missing)
            new_comms.append(list(missing))

    elif policy == "drop_externals":
        # Do not add missing nodes; only sanitize
        pass
    else:
        raise ValueError(f"Unknown policy={policy}. Use 'singleton', 'null_community', or 'drop_externals'.")

    # ---- 4) Choose graph reference ----
    out_graph = G
    if keep_original_graph and getattr(clu, "graph", None) is not None:
        try:
            if set(clu.graph.nodes()) == all_nodes:
                out_graph = clu.graph
        except Exception:
            out_graph = G

    # ---- 5) Method name ----
    base_name = getattr(clu, "method_name", None) or "unknown"
    method_name = f"{base_name}{method_suffix}"

    # ---- 6) Return ----
    return NodeClustering(
        communities=new_comms,
        graph=out_graph,
        method_name=method_name,
        overlap=True,
    )
