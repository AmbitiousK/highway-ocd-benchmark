from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import networkx as nx
import scipy.sparse as sp
from cdlib import evaluation
from cdlib.classes import NodeClustering


# ============================================================
# Core utilities
# ============================================================

@dataclass(frozen=True)
class MetricResult:
    value: float
    meta: Dict[str, Union[int, float, str, bool]]


def _resolve_nodes(
    gt_nc: NodeClustering,
    pred_nc: NodeClustering,
    nodes: Optional[Sequence] = None,
    *,
    mode: str = "intersection",
) -> List:
    """
    Resolve evaluation node list.

    mode:
      - "intersection": only nodes present in both graphs (recommended)
      - "union": union of nodes (may include nodes missing in one clustering)
      - "gt": nodes from gt only
      - "pred": nodes from pred only
    """
    if nodes is not None:
        return list(nodes)

    gt_nodes = set(gt_nc.graph.nodes()) if getattr(gt_nc, "graph", None) is not None else set()
    pr_nodes = set(pred_nc.graph.nodes()) if getattr(pred_nc, "graph", None) is not None else set()

    if mode == "intersection":
        out = gt_nodes & pr_nodes
    elif mode == "union":
        out = gt_nodes | pr_nodes
    elif mode == "gt":
        out = gt_nodes
    elif mode == "pred":
        out = pr_nodes
    else:
        raise ValueError(f"Unknown mode={mode}")

    return sorted(out)


def _node_to_comms(nc: NodeClustering, nodes: Sequence) -> Dict:
    """
    Build node -> set(community_ids).
    Nodes not appearing in any community will map to empty set().
    """
    node_set = set(nodes)
    mapping: Dict[object, Set[int]] = {u: set() for u in nodes}
    for cid, comm in enumerate(nc.communities):
        for u in comm:
            if u in node_set:
                mapping[u].add(cid)
    return mapping


def _sample_pairs(
    n: int,
    sample_pairs: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample unordered pairs (i, j) with i < j.
    Uses rejection for collisions; good enough for large n with moderate sample_pairs.

    Returns:
      i_idx, j_idx arrays of length = sample_pairs
    """
    if n < 2 or sample_pairs <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    need = sample_pairs
    pairs: Set[Tuple[int, int]] = set()

    while len(pairs) < sample_pairs:
        batch = int(max(need * 1.5, 1024))
        a = rng.integers(0, n, size=batch, dtype=np.int64)
        b = rng.integers(0, n, size=batch, dtype=np.int64)
        for x, y in zip(a, b):
            if x == y:
                continue
            i, j = (x, y) if x < y else (y, x)
            pairs.add((i, j))
            if len(pairs) >= sample_pairs:
                break
        need = sample_pairs - len(pairs)

    ii = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=sample_pairs)
    jj = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=sample_pairs)
    return ii, jj


def _pairwise_membership_jaccard(
    a: Set[int],
    b: Set[int],
    *,
    empty_union_policy: str = "skip",
) -> Optional[float]:
    """
    Jaccard similarity over membership sets.
    empty_union_policy:
      - "skip": return None if both empty (recommended for unbiased eval)
      - "one":  return 1.0 if both empty
      - "zero": return 0.0 if both empty
    """
    if not a and not b:
        if empty_union_policy == "skip":
            return None
        if empty_union_policy == "one":
            return 1.0
        if empty_union_policy == "zero":
            return 0.0
        raise ValueError(f"Unknown empty_union_policy={empty_union_policy}")

    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union > 0 else None


# ============================================================
# 1) Custom "FRI" (actually pairwise Jaccard agreement)
# ============================================================

def _membership_matrix_from_nc(nc: NodeClustering, node_list):
    """
    根据 NodeClustering 构造一个 (n_nodes x n_communities) 的 membership 矩阵，
    元素为 {0, 1} 表示是否属于该社区。
    """
    n = len(node_list)
    k = len(nc.communities)
    M = np.zeros((n, k), dtype=float)

    node_index = {n: idx for idx, n in enumerate(node_list)}

    for cid, comm in enumerate(nc.communities):
        for node in comm:
            if node in node_index:
                M[node_index[node], cid] = 1.0

    return M


def fuzzy_rand_index_custom(gt_nc, part_nc):
    nodes = sorted(list(gt_nc.graph.nodes()))
    n = len(nodes)
    if n < 2:
        return 1.0

    U = _membership_matrix_from_nc(gt_nc, nodes)
    V = _membership_matrix_from_nc(part_nc, nodes)

    num_pairs = 0
    diff_sum = 0.0

    for i in range(n):
        ui = U[i]
        vi = V[i]
        for j in range(i + 1, n):
            uj = U[j]
            vj = V[j]

            inter_U = np.minimum(ui, uj).sum()
            union_U = np.maximum(ui, uj).sum()
            sU = inter_U / union_U if union_U > 0 else 1.0

            inter_V = np.minimum(vi, vj).sum()
            union_V = np.maximum(vi, vj).sum()
            sV = inter_V / union_V if union_V > 0 else 1.0

            diff_sum += abs(sU - sV)
            num_pairs += 1

    fri = 1.0 - diff_sum / num_pairs
    return float(max(0.0, min(1.0, fri)))


# ============================================================
# 2) Pairwise F1 (overlap-aware; supports sampling and affinity rules)
# ============================================================

def f1_score_custom(gt_nc: NodeClustering, part_nc: NodeClustering) -> float:
    """
    Pairwise F1 Score for overlapping community detection.
    使用 pairs 的同属关系（至少共享一个社区）来计算 Precision / Recall / F1。
    """
    G = gt_nc.graph
    nodes = sorted(G.nodes())
    n = len(nodes)

    gt_membership = {node: set() for node in nodes}
    for cid, comm in enumerate(gt_nc.communities):
        for node in comm:
            if node in gt_membership:
                gt_membership[node].add(cid)

    part_membership = {node: set() for node in nodes}
    for cid, comm in enumerate(part_nc.communities):
        for node in comm:
            if node in part_membership:
                part_membership[node].add(cid)

    TP = 0
    FP = 0
    FN = 0

    for i in range(n):
        u = nodes[i]
        gt_u = gt_membership[u]
        part_u = part_membership[u]

        for j in range(i + 1, n):
            v = nodes[j]
            gt_v = gt_membership[v]
            part_v = part_membership[v]

            gt_same = len(gt_u & gt_v) > 0
            part_same = len(part_u & part_v) > 0

            if part_same and gt_same:
                TP += 1
            elif part_same and not gt_same:
                FP += 1
            elif (not part_same) and gt_same:
                FN += 1

    if TP == 0:
        return 0.0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


# ============================================================
# 3) NID-style cover information distance (your "OVI" refined)
# ============================================================

def normalized_information_distance_cover_custom(
    gt_nc: NodeClustering,
    pred_nc: NodeClustering,
    *,
    nodes: Optional[Sequence] = None,
    nodes_mode: str = "intersection",
    log_base: float = np.e,
    include_null: bool = True,
    eps: float = 1e-12,
) -> MetricResult:
    """
    A normalized information-distance variant for overlapping covers.

    Construction:
      For each node v:
        - GT: uniform over its GT communities; if none and include_null -> assign to NULL.
        - Pred: uniform over its Pred communities; if none and include_null -> assign to NULL.
      This induces distributions p_X, p_Y, p_XY over community labels.

    Output:
      nid = 1 - I(X;Y) / ((H(X)+H(Y))/2)
    """
    eval_nodes = _resolve_nodes(gt_nc, pred_nc, nodes, mode=nodes_mode)
    n = len(eval_nodes)

    if n == 0:
        return MetricResult(float("nan"), {"n_nodes": 0, "reason": "n==0"})

    gt_map = {u: list(s) for u, s in _node_to_comms(gt_nc, eval_nodes).items()}
    pr_map = {u: list(s) for u, s in _node_to_comms(pred_nc, eval_nodes).items()}

    Kx = len(gt_nc.communities) + (1 if include_null else 0)
    Ky = len(pred_nc.communities) + (1 if include_null else 0)

    if Kx == 0 or Ky == 0:
        return MetricResult(float("nan"), {"n_nodes": n, "reason": "no communities"})

    null_x = Kx - 1 if include_null else None
    null_y = Ky - 1 if include_null else None

    p_x = np.zeros(Kx, dtype=float)
    p_y = np.zeros(Ky, dtype=float)
    p_xy = np.zeros((Kx, Ky), dtype=float)

    inv_n = 1.0 / n

    for v in eval_nodes:
        gx = gt_map[v]
        gy = pr_map[v]

        if len(gx) > 0:
            px_idx = gx
            px_prob = 1.0 / len(gx)
        else:
            if include_null:
                px_idx = [null_x]  # type: ignore[arg-type]
                px_prob = 1.0
            else:
                continue

        if len(gy) > 0:
            py_idx = gy
            py_prob = 1.0 / len(gy)
        else:
            if include_null:
                py_idx = [null_y]  # type: ignore[arg-type]
                py_prob = 1.0
            else:
                continue

        for ix in px_idx:
            p_x[ix] += inv_n * px_prob
        for iy in py_idx:
            p_y[iy] += inv_n * py_prob

        for ix in px_idx:
            for iy in py_idx:
                p_xy[ix, iy] += inv_n * (px_prob * py_prob)

    def _log(x: np.ndarray) -> np.ndarray:
        if log_base == np.e:
            return np.log(x)
        return np.log(x) / np.log(log_base)

    mask_x = p_x > eps
    mask_y = p_y > eps

    Hx = float(-np.sum(p_x[mask_x] * _log(p_x[mask_x])))
    Hy = float(-np.sum(p_y[mask_y] * _log(p_y[mask_y])))

    I = 0.0
    for ix in range(Kx):
        for iy in range(Ky):
            pij = p_xy[ix, iy]
            if pij > eps and p_x[ix] > eps and p_y[iy] > eps:
                I += float(
                    pij
                    * (
                        _log(np.array([pij]))[0]
                        - _log(np.array([p_x[ix] * p_y[iy]]))[0]
                    )
                )

    H_bar = 0.5 * (Hx + Hy)
    if H_bar < eps:
        return MetricResult(
            0.0,
            {"n_nodes": n, "Hx": Hx, "Hy": Hy, "I": I, "degenerate": True, "log_base": float(log_base)},
        )

    nid = float(np.clip(1.0 - (I / H_bar), 0.0, 1.0))
    return MetricResult(
        nid,
        {"n_nodes": n, "Hx": Hx, "Hy": Hy, "I": I, "H_bar": H_bar, "log_base": float(log_base)},
    )


def overlapping_variation_of_information_custom(
    gt_nc: NodeClustering,
    part_nc: NodeClustering,
    *,
    nodes: Optional[Sequence] = None,
    nodes_mode: str = "intersection",
) -> float:
    return normalized_information_distance_cover_custom(
        gt_nc, part_nc, nodes=nodes, nodes_mode=nodes_mode
    ).value


# ============================================================
# 4) Extended Modularity (Shen-style) — scalable implementation
# ============================================================

def extended_modularity_custom(
    part_nc: NodeClustering,
    G: nx.Graph,
    *,
    weight: str = "weight",
    nodes: Optional[Sequence] = None,
    nodes_mode: str = "intersection",
) -> MetricResult:
    """
    Shen-style Extended Modularity for overlapping covers.
    """
    if nodes is None:
        if getattr(part_nc, "graph", None) is not None:
            tmp = NodeClustering([], part_nc.graph, "")
            eval_nodes = _resolve_nodes(tmp, tmp, list(G.nodes()), mode="gt")
        else:
            eval_nodes = sorted(G.nodes())
    else:
        eval_nodes = list(nodes)

    if len(eval_nodes) == 0:
        return MetricResult(float("nan"), {"n_nodes": 0, "reason": "empty nodes"})

    k: Dict[object, float] = {}
    m2 = 0.0
    for u in eval_nodes:
        deg = 0.0
        for _, data in G[u].items():
            deg += float(data.get(weight, 1.0))
        k[u] = deg
        m2 += deg

    if m2 <= 0:
        return MetricResult(0.0, {"n_nodes": len(eval_nodes), "reason": "m2==0"})

    node_set = set(eval_nodes)

    node_to_comms = {u: set() for u in eval_nodes}
    comm_nodes: List[List[object]] = []
    for cid, comm in enumerate(part_nc.communities):
        lst = [u for u in comm if u in node_set]
        comm_nodes.append(lst)
        for u in lst:
            node_to_comms[u].add(cid)

    O: Dict[object, int] = {}
    for u in eval_nodes:
        Ou = len(node_to_comms[u])
        O[u] = Ou if Ou > 0 else 1

    sumA = 0.0
    for u, v, data in G.edges(data=True):
        if u not in node_set or v not in node_set:
            continue
        w = float(data.get(weight, 1.0))
        common = len(node_to_comms[u] & node_to_comms[v])
        if common == 0:
            continue
        S_uv = common / (O[u] * O[v])
        sumA += 2.0 * w * S_uv

    sumKK = 0.0
    for nodes_c in comm_nodes:
        if not nodes_c:
            continue
        s_c = 0.0
        for u in nodes_c:
            s_c += k[u] / O[u]
        sumKK += s_c * s_c

    Q = (sumA - (sumKK / m2)) / m2
    return MetricResult(
        float(Q),
        {"n_nodes": len(eval_nodes), "m2": float(m2), "sumA": float(sumA), "sumKK": float(sumKK)},
    )


# ============================================================
# 5) Cover similarity (Czekanowski / Dice) — accelerated candidates
# ============================================================

def cover_similarity_czekanowski_custom(
    gt_nc: NodeClustering,
    pred_nc: NodeClustering,
    *,
    min_comm_size: int = 1,
) -> MetricResult:
    """
    Cover similarity (Czekanowski / Sørensen–Dice) between two covers.
    """
    U = [set(comm) for comm in gt_nc.communities if len(comm) >= min_comm_size]
    V = [set(comm) for comm in pred_nc.communities if len(comm) >= min_comm_size]

    if len(U) == 0 or len(V) == 0:
        return MetricResult(0.0, {"U": len(U), "V": len(V), "reason": "empty cover"})

    inv: Dict[object, List[int]] = {}
    for j, D in enumerate(V):
        for x in D:
            inv.setdefault(x, []).append(j)

    def cz(C: Set, D: Set) -> float:
        inter = len(C & D)
        if inter == 0:
            return 0.0
        denom = len(C) + len(D)
        return float((2.0 * inter) / denom) if denom > 0 else 0.0

    sum_U = 0.0
    for C in U:
        cand = set()
        for x in C:
            cand.update(inv.get(x, []))
        best = 0.0
        for j in cand:
            best = max(best, cz(C, V[j]))
        sum_U += best
    avg_U = sum_U / len(U)

    invU: Dict[object, List[int]] = {}
    for i, C in enumerate(U):
        for x in C:
            invU.setdefault(x, []).append(i)

    sum_V = 0.0
    for D in V:
        cand = set()
        for x in D:
            cand.update(invU.get(x, []))
        best = 0.0
        for i in cand:
            best = max(best, cz(U[i], D))
        sum_V += best
    avg_V = sum_V / len(V)

    cs = float(np.clip(0.5 * (avg_U + avg_V), 0.0, 1.0))
    return MetricResult(cs, {"U": len(U), "V": len(V), "avg_U": float(avg_U), "avg_V": float(avg_V)})


# ============================================================
# 5b) CDlib Dice / Czekanowski / Sorensen (pair-counting F-measure)
# ============================================================

def cdlib_dice_custom(
    gt_nc: NodeClustering,
    pred_nc: NodeClustering,
) -> MetricResult:
    """
    CDlib / clusim Dice index between two clusterings.

    Note:
      - This is NOT the same as cover_similarity_czekanowski_custom.
      - CDlib's dice_index is a pair-counting clustering similarity metric
        implemented through clusim's fmeasure:
            F = 2*N11 / (2*N11 + N10 + N01)
    """
    result = evaluation.dice_index(gt_nc, pred_nc)

    score = None
    if hasattr(result, "score"):
        score = result.score
    elif hasattr(result, "value"):
        score = result.value
    else:
        score = result

    return MetricResult(
        float(score),
        {
            "impl": "cdlib.evaluation.dice_index",
            "family": "pair_counting_fmeasure",
        },
    )

# ============================================================
# 5c) ONMI / LFK overlapping normalized mutual information
# ============================================================

def onmi_lfk_custom(
    gt_nc: NodeClustering,
    pred_nc: NodeClustering,
) -> MetricResult:
    """
    LFK-style Overlapping Normalized Mutual Information (ONMI).

    This metric compares two overlapping clusterings using the
    overlapping NMI extension proposed by Lancichinetti et al.

    Notes:
      - Preferred path:
          gt_nc.overlapping_normalized_mutual_information_LFK(pred_nc)
      - Fallback path:
          cdlib.evaluation.overlapping_normalized_mutual_information_LFK(gt_nc, pred_nc)

    Returns:
      MetricResult(value=<score>, meta={...})
    """
    result = None
    impl = None

    # Preferred: NodeClustering bound method
    if hasattr(gt_nc, "overlapping_normalized_mutual_information_LFK"):
        result = gt_nc.overlapping_normalized_mutual_information_LFK(pred_nc)
        impl = "NodeClustering.overlapping_normalized_mutual_information_LFK"

    # Fallback: cdlib.evaluation function
    elif hasattr(evaluation, "overlapping_normalized_mutual_information_LFK"):
        result = evaluation.overlapping_normalized_mutual_information_LFK(gt_nc, pred_nc)
        impl = "cdlib.evaluation.overlapping_normalized_mutual_information_LFK"

    else:
        raise AttributeError(
            "LFK ONMI is not available in the installed cdlib version. "
            "Please upgrade cdlib or provide a manual implementation."
        )

    score = None
    if hasattr(result, "score"):
        score = result.score
    elif hasattr(result, "value"):
        score = result.value
    else:
        score = result

    return MetricResult(
        float(score),
        {
            "impl": impl,
            "family": "overlapping_nmi",
            "variant": "LFK",
            "range_min": 0.0,
            "range_max": 1.0,
        },
    )

# ============================================================
# 6) F*wo / Fstar metric (overlaps + outliers)
# ============================================================

def _nodeclustering_to_csr(
    nc: NodeClustering,
    node_list: Sequence,
) -> sp.csr_matrix:
    """
    Convert NodeClustering to CSR membership matrix of shape:
        (n_clusters, n_objects)

    c[i, j] = 1 if node_list[j] is in community i.
    Nodes not appearing in any community become outliers implicitly:
    their column will be all zeros.
    """
    n_clusters = len(nc.communities)
    n_objects = len(node_list)

    if n_clusters == 0:
        return sp.csr_matrix((0, n_objects), dtype=np.int8)

    node_index = {u: j for j, u in enumerate(node_list)}

    rows: List[int] = []
    cols: List[int] = []

    for cid, comm in enumerate(nc.communities):
        for u in comm:
            j = node_index.get(u)
            if j is not None:
                rows.append(cid)
                cols.append(j)

    data = np.ones(len(rows), dtype=np.int8)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_clusters, n_objects), dtype=np.int8)


def _fstar_compare_csr(c1: sp.csr_matrix, c2: sp.csr_matrix) -> float:
    """
    Paper-style F*wo comparison on two CSR membership matrices.

    Inputs:
      c1, c2: scipy csr matrices of shape (n_clusters, n_objects)
      where c[i, j] = 1 iff object j is in cluster i.
    """
    if c1.shape[1] != c2.shape[1]:
        raise ValueError(f"c1 and c2 must have same n_objects, got {c1.shape} vs {c2.shape}")

    n_objects = c1.shape[1]

    if c1.shape[0] == 0 and c2.shape[0] == 0:
        return 1.0

    c1_sizes = np.asarray(c1.sum(axis=1)).reshape(-1).astype(np.float64)
    c2_sizes = np.asarray(c2.sum(axis=1)).reshape(-1).astype(np.float64)

    sum_c1 = float(np.sum(c1_sizes))
    sum_c2 = float(np.sum(c2_sizes))

    c1_props = c1_sizes / sum_c1 if sum_c1 > 0 else np.zeros_like(c1_sizes, dtype=np.float64)
    c2_props = c2_sizes / sum_c2 if sum_c2 > 0 else np.zeros_like(c2_sizes, dtype=np.float64)

    if c1.shape[0] == 0 or c2.shape[0] == 0:
        intersect = sp.csr_matrix((c1.shape[0], c2.shape[0]), dtype=np.float64)
    else:
        intersect = c1.astype(np.float64) @ c2.transpose().astype(np.float64)

    row_sizes = sp.diags(c1_sizes) if c1.shape[0] > 0 else sp.csr_matrix((0, 0), dtype=np.float64)
    col_sizes = sp.diags(c2_sizes) if c2.shape[0] > 0 else sp.csr_matrix((0, 0), dtype=np.float64)

    nz = intersect.copy()
    if nz.nnz > 0:
        nz.data[:] = 1.0

    union = row_sizes @ nz + nz @ col_sizes - intersect if intersect.shape != (0, 0) else intersect.copy()

    if union.nnz > 0:
        union.data = 1.0 / union.data

    fs = intersect.multiply(union)

    fs_l = fs.max(axis=1).toarray().reshape(-1) if c1.shape[0] > 0 else np.array([], dtype=np.float64)
    fs_lw = float(np.sum(fs_l * c1_props)) if fs_l.size > 0 else 0.0

    fs_r = fs.max(axis=0).toarray().reshape(-1) if c2.shape[0] > 0 else np.array([], dtype=np.float64)
    fs_rw = float(np.sum(fs_r * c2_props)) if fs_r.size > 0 else 0.0

    # per-column nonzero count (objects covered by no community = outliers).
    # Use (M != 0).sum(axis=0) instead of count_nonzero(axis=0): the latter does
    # not accept an axis argument on sparse matrices in older/newer scipy alike.
    c1_o = np.asarray((c1 != 0).sum(axis=0)).reshape(-1) == 0
    c2_o = np.asarray((c2 != 0).sum(axis=0)).reshape(-1) == 0

    o_intersect = int(np.sum(c1_o & c2_o))
    o_fs = 0.0
    denom_o = int(np.sum(c1_o) + np.sum(c2_o) - o_intersect)
    if denom_o > 0:
        o_fs = float(o_intersect / denom_o)

    alpha = 1.0 - float(np.sum(c1_o)) / n_objects if n_objects > 0 else 0.0
    beta = 1.0 - float(np.sum(c2_o)) / n_objects if n_objects > 0 else 0.0

    fs_l_wo = (1.0 - alpha) * o_fs + alpha * fs_lw
    fs_r_wo = (1.0 - beta) * o_fs + beta * fs_rw

    score = 0.5 * (fs_l_wo + fs_r_wo)
    return float(np.clip(score, 0.0, 1.0))


def fstar_wo_custom(
    gt_nc: NodeClustering,
    pred_nc: NodeClustering,
    *,
    nodes: Optional[Sequence] = None,
    nodes_mode: str = "intersection",
) -> MetricResult:
    """
    F*wo similarity for clustering comparison with overlaps and outliers.
    """
    eval_nodes = _resolve_nodes(gt_nc, pred_nc, nodes, mode=nodes_mode)
    n = len(eval_nodes)

    if n == 0:
        return MetricResult(float("nan"), {"n_nodes": 0, "reason": "n==0"})

    c1 = _nodeclustering_to_csr(gt_nc, eval_nodes)
    c2 = _nodeclustering_to_csr(pred_nc, eval_nodes)

    value = _fstar_compare_csr(c1, c2)
    return MetricResult(
        value,
        {
            "n_nodes": n,
            "gt_n_clusters": int(c1.shape[0]),
            "pred_n_clusters": int(c2.shape[0]),
            "nodes_mode": nodes_mode,
        },
    )


def fstar_wo_score_custom(
    gt_nc: NodeClustering,
    pred_nc: NodeClustering,
    *,
    nodes: Optional[Sequence] = None,
    nodes_mode: str = "intersection",
) -> float:
    """
    Scalar-only wrapper for F*wo.
    """
    return fstar_wo_custom(gt_nc, pred_nc, nodes=nodes, nodes_mode=nodes_mode).value


# ============================================================
# Metric registry
# ============================================================

MetricCallable = Callable[..., Union[float, MetricResult]]

METRIC_REGISTRY: Dict[str, MetricCallable] = {
    "fri": fuzzy_rand_index_custom,
    "fuzzy_rand_index": fuzzy_rand_index_custom,
    "pairwise_f1": f1_score_custom,
    "f1_score": f1_score_custom,
    "nid_cover": normalized_information_distance_cover_custom,
    "normalized_information_distance_cover": normalized_information_distance_cover_custom,
    "ovi": overlapping_variation_of_information_custom,
    "overlapping_variation_of_information": overlapping_variation_of_information_custom,
    "extended_modularity": extended_modularity_custom,

    # old custom cover Dice / Czekanowski
    "cover_similarity_czekanowski": cover_similarity_czekanowski_custom,
    "cover_similarity_dice": cover_similarity_czekanowski_custom,

    # new CDlib pair-counting Dice
    "cdlib_dice": cdlib_dice_custom,
    "dice_cdlib": cdlib_dice_custom,
    "cdlib_pair_counting_dice": cdlib_dice_custom,

    "fstar_wo": fstar_wo_custom,
    "fstar": fstar_wo_custom,
    "fstar_wo_score": fstar_wo_score_custom,
    
    # LFK-style ONMI
    "onmi": onmi_lfk_custom,
    "onmi_lfk": onmi_lfk_custom,
    "overlapping_normalized_mutual_information": onmi_lfk_custom,
    "overlapping_normalized_mutual_information_lfk": onmi_lfk_custom,
    "lfk_onmi": onmi_lfk_custom,
}


def get_metric(name: str) -> MetricCallable:
    key = name.strip().lower()
    if key not in METRIC_REGISTRY:
        raise KeyError(f"Unknown metric: {name}. Available: {sorted(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[key]